"""Model-scoped context reuse for serial megakernel experts."""

from weakref import WeakKeyDictionary

import torch.distributed as dist
import torch.nn.functional as F

from torchturbo.ops.moe import MegaMoeContext

from .experts import GroupedMLP
from .moe_layer import MoELayer


class MegaKernelExperts:
    """Resolve each expert layer's geometry on first use within one prepared model."""

    def __init__(self, ep_group):
        if ep_group is None or dist.get_world_size(ep_group) <= 1:
            raise ValueError("MegaMoE requires expert parallelism with ep_size > 1.")
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.layers = WeakKeyDictionary()
        self._contexts = {}

    def attach(self, module):
        """Enable megakernel dispatch on an MoE layer returned by the parallel plan."""
        if isinstance(module, MoELayer):
            if not isinstance(module.experts, GroupedMLP):
                raise ValueError("MegaMoE requires TorchTurbo GroupedMLP experts.")
            module.megakernel_experts = self
            self.layers[module.experts] = None

    def context_for(self, experts):
        context = self.layers[experts]
        if context is None:
            cfg = experts.config
            if not cfg.gated_linear_unit or cfg.activation_func is not F.silu or cfg.moe_apply_probs_on_input:
                raise ValueError("MegaMoE requires SwiGLU with routing probabilities applied before FC2.")
            if experts.num_local_experts * self.ep_size != cfg.num_moe_experts:
                raise ValueError("MegaMoE expert partition does not match the EP process group.")
            limit = experts.swiglu_limit or None
            key = (cfg.hidden_size, cfg.moe_ffn_hidden_size, cfg.num_moe_experts,
                   cfg.moe_router_topk, limit, bool(limit and cfg.moe_swiglu_clamp_gate_after_activation))
            context = self._contexts.get(key)
            if context is None:
                context = MegaMoeContext(self.ep_group, *key)
                self._contexts[key] = context
            self.layers[experts] = context
        return context
