"""FSDP and expert parallelism for the flat Bailing VL checkpoint tree."""

from typing import Any

from torch import nn
from torch.distributed.tensor import Shard

from torchturbo.engine.config import EngineConfig
from torchturbo.parallel import (
    FSDPParallelPlan,
    FSDPPlacementSpec,
    FSDPPlanSpec,
    VisionBalanceSpec,
)

from ..bailing_hybrid.parallel_plan import _replace_moe_only
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.module import ModuleSpec
from ..modules.moe.moe_layer import MoESubmodules
from .configuration_bailing_moe_v3_vl import BailingMoeV3VLConfig
from .vision_balance import BailingVisionBalanceIO


def _replace_vl_moe(
    original_module: nn.Module,
    hf_config: BailingMoeV3VLConfig,
    sp_info: dict[str, Any],
    ep_info: dict[str, Any],
    moe_submodules: MoESubmodules,
    engine_config: EngineConfig,
) -> nn.Module:
    return _replace_moe_only(
        original_module,
        hf_config.text_config,
        sp_info,
        ep_info,
        moe_submodules,
        engine_config,
    )


def get_parallel_plan() -> FSDPParallelPlan:
    """Keep dense layers intact and shard routed experts over EDP."""
    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_vl_moe,
                module_kwargs={
                    "moe_submodules": MoESubmodules(
                        experts=ModuleSpec(module=GroupedMLP)
                    ),
                },
            ),
            "model.layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "model.visual.blocks.*": FSDPPlanSpec(activation_checkpoint=True),
            "model.visual": FSDPPlanSpec(
                standalone_world_shard=True,
                vision_balance=VisionBalanceSpec(io=BailingVisionBalanceIO()),
            ),
            "linear_proj": FSDPPlanSpec(standalone_world_shard=True),
            "model.word_embeddings": FSDPPlanSpec(standalone_world_shard=True),
            "lm_head": FSDPPlanSpec(standalone_world_shard=True),
        },
    )
