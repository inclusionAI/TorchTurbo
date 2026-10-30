from ...parallel import FSDPPlanSpec, FSDPParallelPlan, VisionBalanceSpec
from .vision_balance import Qwen3_5VisionBalanceIO


def get_parallel_plan() -> FSDPParallelPlan:
    return FSDPParallelPlan(
        plan={
            "mtp.layers.*": FSDPPlanSpec(activation_checkpoint=True),
            "model.language_model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.visual.blocks.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.visual": FSDPPlanSpec(
                standalone_world_shard=True,
                vision_balance=VisionBalanceSpec(io=Qwen3_5VisionBalanceIO()),
            ),
            "model.language_model.embed_tokens": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,  # dense: no experts → EP disabled
    )
