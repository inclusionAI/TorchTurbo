"""NPU provider for the CANN mhc_pre_sinkhorn_premix ops.

Fused hyper-connection sinkhorn for DeepSeek V4.1: one call replaces the
``DeepseekV41HyperConnections.mixes`` + ``collapse_hc`` small-operator pair
(RMS-normalized projection, split sigmoid pre/post, softmax+sinkhorn over
the residual hc_mult x hc_mult logits, then the premix-weighted collapse
``hin``). The op pair ships as per-op vendor wheels (dsv41exp_mhc_pre and
its backward companion): their aclnn libs install under the vendor op_api
directories (must be on ``LD_LIBRARY_PATH``), and the wheels register the
ops as attributes of the master ``cann_ops_transformer`` module via entry
points.

arch22 (A2/A3) constraints, enforced here (backward op hard contract):
- x must be 4-dim BSND (TND is Ascend950 forward-only; the backward op
  takes BSND only).
- N (= hc_mult) is fixed to 4, D must be a multiple of 128, and the
  sinkhorn iteration count is fixed to 20.
- x must be bf16/fp16 (the kernel rejects fp32); phi/alpha/bias/premix
  stay fp32.
- premix and gradPre must be passed together or omitted together; the
  autograd wrapper materializes a zero gradPre when premix is passed but
  h_pre received no upstream grad (official autograd packaging).
"""

from __future__ import annotations

from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op

# The premix pair is vendor-only (not part of the bundled master wheel); the
# per-op wheels register the ops as attributes of the master module through
# entry points at import time, so presence of the attributes proves the
# vendor registration ran.
_MHC_OP_NAMES = ("mhc_pre_sinkhorn_premix", "mhc_pre_sinkhorn_premix_backward")

# Backward-op hard constraints (arch22): hc_mult=4, D a multiple of 128,
# 20 sinkhorn iters. The config-level triple check lives in the model patch
# (torchturbo.models.deepseek_v41.npu.patch_deepseek_v41_npu_mhc), which runs
# before installation; only the tensor-level contract is re-checked here.


def _load_ops():
    """Master cann_ops_transformer module with both mhc ops registered."""
    import cann_ops_transformer as master

    if all(callable(getattr(master, name, None)) for name in _MHC_OP_NAMES):
        return master
    raise ImportError(
        "mhc_pre_sinkhorn_premix ops not found: install the per-op vendor "
        "wheels cann_ops_transformer_dsv41exp_mhc_pre and "
        "cann_ops_transformer_dsv41exp_mhc_pre_bwd, which register onto the "
        "master module through entry points"
    )


def has_mhc_pre_sinkhorn_premix() -> bool:
    try:
        _load_ops()
        return True
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False


def _ensure_fp32(t: torch.Tensor | None) -> torch.Tensor | None:
    """Cast to fp32, no-op when already fp32 (``Tensor.float`` returns self).

    HC parameters may use the model parameter dtype, while the operator
    requires FP32 phi/alpha/bias/premix. Delayed pre_mix must retain FP32
    across decoder boundaries: converting a rounded BF16 input here cannot
    restore its precision. Parameter casts remain differentiable.
    """
    if t is None or t.dtype == torch.float32:
        return t
    return t.float()


def _check_backward_constraints(x: torch.Tensor) -> None:
    """Validate the op's tensor-level contract before calling it (the aclnn
    layer reports violations with kernel-side messages that hide the cause)."""
    if x.dim() != 4:
        raise ValueError(
            "mhc_pre_sinkhorn_premix requires a BSND (4-dim) x on arch22; "
            f"got shape {tuple(x.shape)}"
        )
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"mhc_pre_sinkhorn_premix requires bf16/fp16 x; got {x.dtype}"
        )


class MhcPreSinkhornPremixFn(torch.autograd.Function):
    """Autograd bridge over the fused forward + backward op pair.

    forward calls the fused op once with out_flag=True (the five backward
    caches are op outputs), returns (hin, h_post, h_res, h_pre) and keeps
    the caches for backward. backward hands the upstream grads plus the
    gradPre materialization contract to mhc_pre_sinkhorn_premix_backward.

    premix/gradPre contract (aclnn binding layer): both passed or both
    omitted. When premix is passed but h_pre received no upstream grad
    (grad_h_pre is None), gradPre is materialized as zeros, matching the
    official autograd packaging. When premix is omitted, gradPre must stay
    None and the fifth backward output (grad_pre_mix) is an empty tensor,
    mapped back to a None gradient.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        premix: torch.Tensor | None,
        hc_mult: int,
        num_iters: int,
        hc_eps: float,
        norm_eps: float,
    ):
        ops = _load_ops()
        hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            ops.mhc_pre_sinkhorn_premix(
                x,
                phi,
                alpha,
                bias,
                premix,
                hc_mult,
                num_iters,
                hc_eps,
                norm_eps,
                True,
            )
        )
        ctx.save_for_backward(
            x,
            phi,
            alpha,
            bias,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
            # save_for_backward cannot take None; an empty fp32 tensor is
            # the "no premix" sentinel (real premix always has numel > 0).
            premix if premix is not None else x.new_empty(0, dtype=torch.float32),
        )
        ctx.hc_eps = float(hc_eps)
        return hin, h_post, h_res, h_pre

    @staticmethod
    def backward(ctx, grad_hin, grad_h_post, grad_h_res, grad_h_pre):
        (
            x,
            phi,
            alpha,
            bias,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
            premix,
        ) = ctx.saved_tensors
        has_premix = premix.numel() > 0
        if has_premix:
            # premix and gradPre must be passed together; a missing upstream
            # h_pre grad is materialized as zeros (official autograd packaging).
            grad_pre = (
                grad_h_pre if grad_h_pre is not None else torch.zeros_like(premix)
            )
            op_premix = premix
        else:
            grad_pre = None
            op_premix = None
        ops = _load_ops()
        grad_x, grad_phi, grad_alpha, grad_bias, grad_pre_mix = (
            ops.mhc_pre_sinkhorn_premix_backward(
                grad_hin,
                grad_h_post,
                grad_h_res,
                x,
                phi,
                alpha,
                bias,
                h_pre,
                hc_before_norm,
                inv_rms,
                sum_out,
                norm_out,
                op_premix,
                grad_pre,
                ctx.hc_eps,
            )
        )
        grad_premix = grad_pre_mix if has_premix else None
        return (
            grad_x,
            grad_phi,
            grad_alpha,
            grad_bias,
            grad_premix,
            None,
            None,
            None,
            None,
        )


@register_op(
    op_name="mhc_pre_sinkhorn_premix",
    impl_name="npu_mhc_pre_sinkhorn_premix",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMhcPreSinkhornPremix(OpImpl):
    """Fused mhc sinkhorn + premix collapse with fused NPU backward.

    x: [B, S, N, D] bf16 (BSND); phi: [2N+N*N, N*D] fp32; alpha: [3] fp32;
    bias: [2N+N*N] fp32; premix: [B, S, N] fp32 or None. Returns
    (hin, h_post, h_res, h_pre): the collapsed stream, the post and
    residual coefficient sets, and the freshly computed pre coefficients
    (the delayed-HC hand-off to the next consumer). Gradients flow back to
    x/phi/alpha/bias/premix through MhcPreSinkhornPremixFn.
    """

    def available(self) -> bool:
        return has_mhc_pre_sinkhorn_premix()

    def __call__(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        premix: torch.Tensor | None,
        hc_mult: int,
        num_iters: int,
        hc_eps: float,
        norm_eps: float,
        **kwargs: Any,
    ) -> Any:
        _check_backward_constraints(x)
        # phi/alpha/bias/premix go through _ensure_fp32: FSDP2 mixed
        # precision may hand them over in bf16 while the op requires fp32
        # (see _ensure_fp32). x keeps its bf16/fp16 input dtype.
        return MhcPreSinkhornPremixFn.apply(
            x.contiguous(),
            _ensure_fp32(phi).contiguous(),
            _ensure_fp32(alpha).contiguous(),
            _ensure_fp32(bias).contiguous(),
            _ensure_fp32(premix).contiguous() if premix is not None else None,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
        )
