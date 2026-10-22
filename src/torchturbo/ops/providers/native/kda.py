"""FLA implementation of chunked Kimi Delta Attention."""

from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...packed_params import PackedParams
from ...registry import register_op


@register_op(
    op_name="kda",
    impl_name="fla_kda",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class FlaKDA(OpImpl):
    def available(self) -> bool:
        try:
            from fla.ops.kda import chunk_kda  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        packed_params: PackedParams | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.kda import chunk_kda

        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            cu_seqlens=None if packed_params is None else packed_params.cu_seqlens,
        )
