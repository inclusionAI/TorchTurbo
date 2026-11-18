"""NPU provider for the CANN mhc_post op.

Fused hyper-connection post update for DeepSeek V4.1: one call replaces the
``HcPost`` small-operator chain (broadcast multiply + reduce over the
hc_mult x hc_mult residual mixing matrix) that materializes a
[B, S, N, N, D] fp32 intermediate (2.7 GB at the production shape). The op
ships in the cann_ops_transformer wheel; forward and fused backward are two
separate torch.ops entries, so this provider bridges them through a
torch.autograd.Function.

Op contract (x_next[i] = h_post[i] * h_out + sum_j h_res[j, i] * x[j]):
- x (the residual streams) must be 4-dim BSND bf16/fp16.
- h_res [B, S, N, N] and h_post [B, S, N] stay fp32; they are forward
  intermediates (sinkhorn outputs), never FSDP-cast params.
"""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def has_mhc_post() -> bool:
    try:
        import cann_ops_transformer  # noqa: F401  (registers the torch.ops namespace)

        getattr(torch.ops.cann_ops_transformer, "mhc_post")
        getattr(torch.ops.cann_ops_transformer, "mhc_post_backward")
        return True
    except (ImportError, AttributeError):
        return False


class MhcPostFn(torch.autograd.Function):
    """Autograd bridge over the fused mhc_post / mhc_post_backward pair."""

    @staticmethod
    def forward(ctx, x, h_res, h_out, h_post):
        out = torch.ops.cann_ops_transformer.mhc_post(x, h_res, h_out, h_post)
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, h_res, h_out, h_post = ctx.saved_tensors
        grad_x, grad_h_res, grad_h_out, grad_h_post = (
            torch.ops.cann_ops_transformer.mhc_post_backward(
                grad_output.contiguous(), x, h_res, h_out, h_post
            )
        )
        return grad_x, grad_h_res, grad_h_out, grad_h_post


@register_op(
    op_name="mhc_post",
    impl_name="npu_mhc_post",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMhcPost(OpImpl):
    """Fused HC post update with fused NPU backward.

    x: [B, S, N, D] bf16 residual streams; h_res: [B, S, N, N] fp32 mixing
    matrix (comb); h_out: [B, S, D] bf16 sublayer output; h_post: [B, S, N]
    fp32 expansion coefficients. Returns the [B, S, N, D] updated streams in
    the x dtype. Gradients flow back to all four inputs through MhcPostFn.
    """

    def available(self) -> bool:
        return has_mhc_post()

    def __call__(
        self,
        x: torch.Tensor,
        h_res: torch.Tensor,
        h_out: torch.Tensor,
        h_post: torch.Tensor,
    ):
        # The aclnn layer reports dim/dtype violations with kernel-side
        # messages that hide the cause, so check the contract up front
        # (same rationale as the mhc_pre provider).
        if x.dim() != 4:
            raise ValueError(
                f"mhc_post requires a BSND (4-dim) x; got shape {tuple(x.shape)}"
            )
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"mhc_post requires bf16/fp16 x; got {x.dtype}")
        return MhcPostFn.apply(
            x.contiguous(),
            h_res.contiguous(),
            h_out.contiguous(),
            h_post.contiguous(),
        )
