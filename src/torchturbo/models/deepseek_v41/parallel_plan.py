"""FSDP2 parallel plan for DeepSeek V4.1.

Mirrors the deepseek_v4 plan (MoE replacement + EP-sharded experts) and adds
the engram replacement: the full-table ``DeepseekV41Engram`` becomes a
``HostOffloadEngram`` whose table stays on the host (EP-sharded, rows
fetched on demand), so the 2x183GiB tables never enter FSDP2 (a sharded
parameter would be all-gathered on every forward) nor the generic weight
loader (full-tensor broadcasts).
"""

import torch

from ...parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec
from .configuration_deepseek_v41 import text_config_of
from ..modules.moe.config_converter import get_base_moe_config
from ..modules.moe.moe_layer import MoEConfig, MoELayer, MoESubmodules

# v4.1 has no hc_head params (delayed hyper-connections); the engram module
# is no_shard at module level, so its params are not listed here either.
_NO_SHARD_PARAM_NAMES = [
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
    "attn_sink",
]










def get_parallel_plan(**kwargs) -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard


    return FSDPParallelPlan(
        plan={
            # Delayed HC pre_mix crosses decoder boundaries in FP32. Casting
            # every input to the parameter dtype irreversibly rounds it to BF16.
            "layers.*": FSDPPlanSpec(
                activation_checkpoint=True, cast_forward_inputs=False,
            ),
            "layers.*.engram": FSDPPlanSpec(
                # The host-offload table is a CPU parameter and wkv/q/k read
                # their shard manually, so the whole module stays out of
                # FSDP; grads (once unfrozen) sync through the no_shard AVG
                # hooks for wkv/q/k and the table's own sparse path.
                # NOTE: engine cpu_offload FULL would try to move the host
                # table back to the device (unsupported combination).
                no_shard=True,
            ),
            **{name: FSDPPlanSpec(no_shard=True) for name in _NO_SHARD_PARAM_NAMES},
            **{
                f"**.{name}": FSDPPlanSpec(no_shard=True)
                for name in _NO_SHARD_PARAM_NAMES
            },
        },
        sp_size=1,
        ep_size=1,
    )
