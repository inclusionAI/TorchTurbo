"""HyperMulticore expert-parallel SwiGLU on Ascend A3."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...moe import RouteFormat
from ...registry import register_op

if TYPE_CHECKING:
    from hyper_multicore import MegaMoeExperts
    from torchturbo.models.modules.moe.experts import GroupedMLP
    from torchturbo.models.modules.moe.moe_layer import MoELayer


@register_op(
    op_name="mega_moe",
    impl_name="hmc_mega_moe",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class HmcMegaMoe(OpImpl):
    def available(self) -> bool:
        return find_spec("hyper_multicore") is not None

    def __call__(self) -> HmcRoutedExperts:
        return HmcRoutedExperts()


class HmcRoutedExperts:
    """Reuse executors across compatible layers of one model."""

    route_format = RouteFormat.TOPK

    def __init__(self) -> None:
        self._variants = {}
        self._layer_calls = WeakKeyDictionary()

    def forward_experts(
        self, layer: MoELayer, hidden_states: torch.Tensor, input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        run = self._layer_calls.get(layer)
        if run is None:
            from torchturbo.models.modules.moe.experts import GroupedMLP

            if not isinstance(layer.experts, GroupedMLP):
                raise ValueError("HMC MegaMoE requires TorchTurbo GroupedMLP experts.")
            experts = layer.experts
            cfg = layer.config
            if experts.use_te:
                raise ValueError("HMC MegaMoE requires GroupedMLP without Transformer Engine.")
            if layer.ep_group is None or layer.expert_parallel_size <= 1:
                raise ValueError("HMC MegaMoE requires expert parallelism with ep_size > 1.")
            if experts.num_local_experts * layer.expert_parallel_size != cfg.num_moe_experts:
                raise ValueError("MegaMoE expert partition does not match the EP process group.")
            if not cfg.gated_linear_unit or cfg.activation_func is not F.silu or cfg.moe_apply_probs_on_input:
                raise ValueError("MegaMoE requires SwiGLU with routing probabilities applied before FC2.")
            limit = experts.swiglu_limit or None
            key = (layer.ep_group, cfg.hidden_size, cfg.moe_ffn_hidden_size, cfg.num_moe_experts,
                   cfg.moe_router_topk, limit, bool(limit and cfg.moe_swiglu_clamp_gate_after_activation))
            run = self._variants.get(key)
            if run is None:
                run = _HmcExpertCall(key)
                self._variants[key] = run
            self._layer_calls[layer] = run

        probs, _, ids, logits = layer.route(hidden_states, input_ids)
        output, _ = layer.experts(hidden_states, None, probs, routed_forward=run, expert_indices=ids)
        return layer.add_shared_experts(hidden_states, output), logits


class _HmcExpertCall:
    """Construct the executor on the first call with unsharded weights."""

    def __init__(self, key: tuple) -> None:
        self._key = key
        self._executor: MegaMoeExperts | None = None

    def __call__(self, experts: GroupedMLP, inputs: torch.Tensor,
                 ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        weight1, weight2 = experts.weight1, experts.weight2
        if inputs.dtype != torch.bfloat16:
            raise ValueError("MegaMoE requires BF16 expert activations.")
        if any(w.dtype not in (torch.bfloat16, torch.float32) for w in (weight1, weight2)):
            raise ValueError("MegaMoE requires BF16 expert weights or FP32 master parameters.")
        executor = self._executor
        if executor is None:
            executor = self._create_executor(*self._key)
            self._executor = executor
        cfg = experts.config
        local = experts.num_local_experts
        weights = (
            weight1.to(inputs.dtype).view(local, cfg.hidden_size, 2 * cfg.moe_ffn_hidden_size),
            weight2.to(inputs.dtype).view(local, cfg.moe_ffn_hidden_size, cfg.hidden_size),
        )
        if executor.training != experts.training:
            executor.train(experts.training)
        return executor(inputs, ids.int(), probs, expert_weights=weights)

    @staticmethod
    def _create_executor(
        ep_group, hidden_size, intermediate_size, num_experts, top_k, limit, clamp_after_silu,
    ) -> MegaMoeExperts:
        from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions

        return MegaMoeExperts(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_experts=num_experts, top_k=top_k, ep_group=ep_group,
            numerics_options=MegaMoeNumericsOptions(
                swiglu_limit=limit, clamp_after_silu=clamp_after_silu, apply_probs_before_fc2=True,
            ),
        )
