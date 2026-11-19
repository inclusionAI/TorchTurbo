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


class DeepseekV41MoELayer(MoELayer):
    """Adapt the shared MoE result without changing its parameter hierarchy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .moe_bridge import SourceRouter, SourceTokenDispatcher

        self.config.moe_permute_fusion = False
        self.router = SourceRouter(config=self.config, layer_number=self.layer_number)
        self.token_dispatcher = SourceTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, self.config, self.token_dispatcher.ep_group,
        )

    def forward(self, hidden_states: torch.Tensor, input_ids=None) -> torch.Tensor:
        output, _ = super().forward(hidden_states, input_ids)
        return output


def _dtype_from_config(text_config):
    dtype = text_config.torch_dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype = dtype.removeprefix("torch.")
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return dtype_map[dtype]
    return torch.float32


def _replace_moe_only(
    original_module,
    hf_config,
    sp_info,
    ep_info,
    moe_submodules,
    **kwargs,
):
    from .modeling_deepseek_v41 import DeepseekV41MoE

    if not isinstance(original_module, DeepseekV41MoE):
        return original_module

    text_config = text_config_of(hf_config)
    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    swiglu_limit = text_config.swiglu_limit
    num_layers = text_config.num_hidden_layers
    moe_swiglu_limit = [swiglu_limit] * num_layers if swiglu_limit else None
    dsv41_config_kwargs = get_base_moe_config(
        hf_config=text_config,
        ep_size=ep_size,
        num_moe_experts=text_config.n_routed_experts,
        params_dtype=_dtype_from_config(text_config),
        moe_router_topk_scaling_factor=text_config.routed_scaling_factor,
        moe_router_score_function=text_config.scoring_func,
        moe_router_temperature=text_config.gate_temp,
        moe_router_norm_topk_prob=text_config.norm_topk_prob,
        moe_router_dtype="fp32",
        moe_swiglu_limit=moe_swiglu_limit,
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=True,
    )
    # The reference model applies router weights before the expert down projection.
    # MegaMoE applies them to expert outputs, which can change BF16 rounding.
    moe_config = MoEConfig(**dsv41_config_kwargs)
    layer_number = original_module.layer_idx
    moe_layer = DeepseekV41MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        layer_number=layer_number,
    )

    moe_layer.shared_experts = original_module.shared_experts

    return moe_layer




def get_parallel_plan(**kwargs) -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from .experts import DeepseekV41GroupedMLP as GroupedMLP
    from ..modules.moe.module import ModuleSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            # Delayed HC pre_mix crosses decoder boundaries in FP32. Casting
            # every input to the parameter dtype irreversibly rounds it to BF16.
            "layers.*": FSDPPlanSpec(
                activation_checkpoint=True, cast_forward_inputs=False,
            ),
            "layers.*.ffn.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "layers.*.ffn": FSDPPlanSpec(
                replace_module=_replace_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
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
