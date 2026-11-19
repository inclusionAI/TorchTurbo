"""NPU provider for the CANN TND sparse_flash_mla op."""

from __future__ import annotations

from functools import cache
from importlib.metadata import entry_points

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op

_ORI_KV_TAIL_GUARD = 128

# The DSV4.1 vendor build ships as per-op wheels that override the bundled
# implementations through the "cann_ops_transformer.ops" entry-point group
# when the master module is imported. Probing the group (instead of the
# master attributes) tells the bundled and vendor builds apart, because the
# bundled master wheel also registers both ops. The vendor op_api libs must
# be on LD_LIBRARY_PATH; see the ops-transformer build doc.
_SFA_EP_GROUP = "cann_ops_transformer.ops"
_SFA_EP_REQUIREMENTS = (
    ("sparse_flash_mla", "cann_ops_transformer_dsv41_smla"),
    ("sparse_flash_mla_grad", "cann_ops_transformer_dsv41_smlag"),
)


def _vendor_registered() -> bool:
    """True when both per-op wheels override the ops through entry points."""
    for ep_name, dist in _SFA_EP_REQUIREMENTS:
        matches = entry_points(group=_SFA_EP_GROUP, name=ep_name)
        if not any(ep.value.startswith(dist) for ep in matches):
            return False
    return True


@cache
def _load_ops(custom_package: bool = False):
    """Load CANN ops once per mode; failed loads remain retriable.

    Default returns the CANN-bundled master module. With
    ``custom_package=True`` (DSV4.1 path), both per-op wheels must be
    installed: their entry points override the bundled implementations on
    import of the master module, so the master attributes then resolve to
    the vendor build (the same convention as the lightning_indexer_v2
    provider; no hardcoded deep package paths).
    """
    import cann_ops_transformer as master

    if not custom_package:
        return master
    if _vendor_registered() and callable(getattr(master, "sparse_flash_mla", None)):
        return master
    raise ImportError(
        "DSV4.1 vendor sparse_flash_mla ops not found: install the "
        f"{_SFA_EP_REQUIREMENTS[0][1]} and {_SFA_EP_REQUIREMENTS[1][1]} "
        "wheels (entry-point registration onto the master module) and make "
        "sure their op_api libs are on LD_LIBRARY_PATH."
    )


def _has_required_cann_ops() -> bool:
    try:
        ops = _load_ops()
        return (
            callable(ops.sparse_flash_mla)
            and callable(ops.sparse_flash_mla_metadata)
            and callable(ops.sparse_flash_mla_grad)
            and callable(ops.sparse_flash_mla_grad_metadata)
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False


def _max_seqlen(cu_seqlens: torch.Tensor | None) -> int:
    if cu_seqlens is None or cu_seqlens.numel() <= 1:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


# CANN exposes separate forward, metadata, and gradient kernels. Keep their
# coordination in one autograd bridge so the dispatcher exposes one operation.
class SparseFlashMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        cmp_residual_kv,
        ori_win_left,
        ori_win_right,
        custom_package,
        max_seqlens=None,
    ):
        ctx.custom_package = custom_package
        ops = _load_ops(custom_package)
        has_cmp = cmp_kv is not None
        runtime = {
            "cu_seqlens_q": cu_seq_lens_q,
            "cu_seqlens_ori_kv": cu_seq_lens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seq_lens_cmp_kv if has_cmp else None,
            "cmp_residual_kv": cmp_residual_kv if has_cmp else None,
            "cmp_ratio": cmp_ratio if has_cmp else 1,
            "ori_mask_mode": 4,
            # CANN 9.1 requires cmp_mask_mode=3 on A2/A3.
            "cmp_mask_mode": 3,
            "ori_win_left": ori_win_left,
            "ori_win_right": ori_win_right,
            "layout_q": "TND",
            "layout_kv": "TND",
        }
        if max_seqlens is None:
            max_q = _max_seqlen(cu_seq_lens_q)
            max_ori = _max_seqlen(cu_seq_lens_ori_kv)
            max_cmp = _max_seqlen(cu_seq_lens_cmp_kv) if has_cmp else 0
        else:
            max_q, max_ori, max_cmp = max_seqlens
            if not has_cmp:
                max_cmp = 0
        meta_only = {
            "max_seqlen_q": max_q,
            "max_seqlen_ori_kv": max_ori,
            "max_seqlen_cmp_kv": max_cmp,
            "ori_topk": 0,
            "cmp_topk": (
                cmp_sparse_indices.size(-1) if cmp_sparse_indices is not None else 0
            ),
            "has_ori_kv": True,
            "has_cmp_kv": has_cmp,
        }
        # TND batch size comes from cu_seqlens.
        metadata = ops.sparse_flash_mla_metadata(
            query.size(1), 1, query.size(2), **meta_only, **runtime
        )
        out, lse = ops.sparse_flash_mla(
            query,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **runtime,
        )
        ctx.save_for_backward(
            query,
            ori_kv,
            cmp_kv,
            out,
            lse,
            cmp_sparse_indices,
            cu_seq_lens_q,
            cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv,
            cmp_residual_kv,
            sinks,
        )
        ctx.softmax_scale = softmax_scale
        ctx.runtime = runtime
        ctx.meta_only = meta_only
        return out

    @staticmethod
    def backward(ctx, grad_output):
        ops = _load_ops(ctx.custom_package)
        (
            query,
            ori_kv,
            cmp_kv,
            out,
            lse,
            cmp_sparse_indices,
            _cu_seq_lens_q,
            _cu_seq_lens_ori_kv,
            _cu_seq_lens_cmp_kv,
            _cmp_residual_kv,
            sinks,
        ) = ctx.saved_tensors
        ori_kv_rows = ori_kv.size(0)
        # Pad a zero 128-row tail for the grad kernel without changing cu_seqlens.
        if ori_kv_rows:
            ori_kv_for_grad = torch.cat(
                [ori_kv, ori_kv.new_zeros((_ORI_KV_TAIL_GUARD, *ori_kv.shape[1:]))],
                dim=0,
            )
        else:
            ori_kv_for_grad = ori_kv
        grad_meta = ops.sparse_flash_mla_grad_metadata(
            query.size(1), 1, query.size(2), **ctx.meta_only, **ctx.runtime
        )
        dq, dori_kv, dcmp_kv, dsinks, _, _ = ops.sparse_flash_mla_grad(
            query,
            grad_output.contiguous(),
            out,
            lse,
            ori_kv=ori_kv_for_grad,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=sinks,
            metadata=grad_meta,
            softmax_scale=ctx.softmax_scale,
            **ctx.runtime,
        )
        dori_kv = dori_kv[:ori_kv_rows]
        # Autograd needs None when compressed KV is absent.
        if cmp_kv is None:
            dcmp_kv = None
        return (
            dq,
            dori_kv,
            dcmp_kv,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _run_sparse_flash_mla(
    query: torch.Tensor,
    ori_kv: torch.Tensor | None,
    cmp_kv: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    cmp_residual_kv: torch.Tensor | None,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: int,
    ori_win_left: int,
    ori_win_right: int,
    *,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_ori_kv: torch.Tensor | None = None,
    cu_seq_lens_cmp_kv: torch.Tensor | None = None,
    custom_package: bool = False,
    max_seqlens: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """Run the CANN TND sparse_flash_mla op with autograd."""
    query = query.contiguous()
    ori_kv = ori_kv.contiguous() if ori_kv is not None else None
    cmp_kv = cmp_kv.contiguous() if cmp_kv is not None else None
    if cmp_sparse_indices is not None:
        cmp_sparse_indices = cmp_sparse_indices.contiguous()
    if cmp_residual_kv is not None:
        cmp_residual_kv = cmp_residual_kv.to(torch.int32).contiguous()
    return SparseFlashMla.apply(
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        cmp_residual_kv,
        ori_win_left,
        ori_win_right,
        custom_package,
        max_seqlens,
    )


@register_op(
    op_name="sparse_flash_mla",
    impl_name="npu_sparse_flash_mla",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuSparseFlashMla(OpImpl):
    def available(self) -> bool:
        return _has_required_cann_ops()

    def __call__(
        self,
        query: torch.Tensor,
        ori_kv: torch.Tensor | None,
        cmp_kv: torch.Tensor | None,
        cmp_sparse_indices: torch.Tensor | None,
        cmp_residual_kv: torch.Tensor | None,
        sinks: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        ori_win_left: int,
        ori_win_right: int,
        *,
        cu_seq_lens_q: torch.Tensor | None = None,
        cu_seq_lens_ori_kv: torch.Tensor | None = None,
        cu_seq_lens_cmp_kv: torch.Tensor | None = None,
        custom_package: bool = False,
        max_seqlens: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        return _run_sparse_flash_mla(
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            softmax_scale,
            cmp_ratio,
            ori_win_left,
            ori_win_right,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_ori_kv=cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv=cu_seq_lens_cmp_kv,
            custom_package=custom_package,
            max_seqlens=max_seqlens,
        )
