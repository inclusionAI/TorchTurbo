from sys import modules
import torch.distributed as dist

from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config


class QWen3MoeLayerWrapper(MoELayer):

    def __init__(
        self, original_module, hf_config, sp_info, ep_info, moe_submodules, **kwargs
    ):
        ep_group = ep_info["ep_group"]
        ep_size = ep_info["ep_size"]
        qwen3_config_kwargs = get_base_moe_config(
            hf_config=hf_config,
            ep_size=ep_size,
            num_moe_experts=hf_config.num_experts,
            params_dtype=hf_config.torch_dtype,
        )
        moe_config = MoEConfig(**qwen3_config_kwargs)

        super().__init__(
            config=moe_config,
            ep_group=ep_group,
            submodules=moe_submodules,
        )

        if hasattr(original_module, "shared_experts"):
            self.shared_experts = original.shared_experts

    def forward(self, hidden_states, input_ids=None):
        output, _ = super().forward(hidden_states, input_ids)
        return output


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=QWen3MoeLayerWrapper,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,
    )
