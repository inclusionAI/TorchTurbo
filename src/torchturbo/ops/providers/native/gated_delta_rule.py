"""Native (fla) provider for the chunk gated-delta-rule op.

This is the reference implementation dense Qwen3.5 has always used on non-NPU
hosts (and as the fp32-correctness baseline). ``devices=()`` makes it a candidate
on every runtime; on NPU the AscendC custom provider (priority=100) is preferred
over this one (priority=0), so `auto` policy still routes NPU → AscendC.
"""

from __future__ import annotations

from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...packed_params import PackedParams
from ...registry import register_op


@register_op(
    op_name="gated_delta_rule",
    impl_name="fla_gated_delta_rule",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class FlaGatedDeltaRule(OpImpl):
    def available(self) -> bool:
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        initial_state=None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        packed_params=None,
        **kwargs,
    ):
        # fla's chunk_gated_delta_rule is an autograd function (head_first=False
        # convention); signature mirrors ascendc_chunk_gated_delta_rule.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        # ``packed_params`` is the provider API. Keep accepting the legacy
        # dispatcher keyword during the transition.
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        if packed_params is not None:
            if cu_seqlens is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            cu_seqlens = packed_params.cu_seqlens
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )


@register_op(
    op_name="gated_delta_rule",
    impl_name="torch_gated_delta_rule",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(TurboDevice.CPU,),
)
class TorchGatedDeltaRule(OpImpl):
    """Run the Transformers GDN reference on independent CPU training sequences."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        packed_params: PackedParams | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            torch_chunk_gated_delta_rule,
        )

        if initial_state is not None or output_final_state:
            raise NotImplementedError("CPU gated delta rule supports stateless training")
        boundaries = (
            packed_params.cu_seqlens_list
            if packed_params is not None
            else [0, q.size(1)]
        )
        outputs = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            output, _ = torch_chunk_gated_delta_rule(
                q[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                g[:, start:stop],
                beta[:, start:stop],
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), None
