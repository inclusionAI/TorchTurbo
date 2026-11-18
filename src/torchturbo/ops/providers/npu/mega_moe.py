"""HyperMulticore expert-parallel SwiGLU on Ascend A3."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import MegaMoeContext
from ...registry import register_op

if TYPE_CHECKING:
    from hyper_multicore import MegaMoeExperts


@register_op(
    op_name="mega_moe",
    impl_name="hmc_mega_moe",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class HmcMegaMoe(OpImpl):
    """Reuse one executor per model-bound context, with live external weights."""

    def __init__(self) -> None:
        self._executors: WeakKeyDictionary[MegaMoeContext, MegaMoeExperts] = WeakKeyDictionary()

    def available(self) -> bool:
        return find_spec("hyper_multicore") is not None

    def __call__(
        self,
        inputs: torch.Tensor,
        expert_indices: torch.Tensor,
        probs: torch.Tensor,
        experts: torch.nn.Module,
        *,
        context: MegaMoeContext,
    ) -> torch.Tensor:
        if experts.use_te:
            raise ValueError("The HMC MegaMoE provider requires GroupedMLP without Transformer Engine.")
        weight1, weight2 = experts.weight1, experts.weight2
        if inputs.dtype != torch.bfloat16:
            raise ValueError("MegaMoE requires BF16 expert activations.")
        if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (weight1, weight2)):
            raise ValueError("MegaMoE requires BF16 expert weights or FP32 master parameters.")

        executor = self._executors.get(context)
        if executor is None:
            executor = self._create_executor(context)
            self._executors[context] = executor

        local = experts.num_local_experts
        weights = (
            weight1.to(inputs.dtype).view(local, context.hidden_size, 2 * context.intermediate_size),
            weight2.to(inputs.dtype).view(local, context.intermediate_size, context.hidden_size),
        )
        if executor.training != experts.training:
            executor.train(experts.training)
        return executor(inputs, expert_indices.int(), probs, expert_weights=weights)

    @staticmethod
    def _create_executor(context: MegaMoeContext) -> MegaMoeExperts:
        """Register a model variant before its first resource allocation."""
        from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions

        return MegaMoeExperts(
            hidden_size=context.hidden_size,
            intermediate_size=context.intermediate_size,
            num_experts=context.num_experts,
            top_k=context.top_k,
            ep_group=context.ep_group,
            numerics_options=MegaMoeNumericsOptions(
                swiglu_limit=context.swiglu_limit,
                clamp_after_silu=context.clamp_after_silu,
                apply_probs_before_fc2=True,
            ),
        )
