from ...parallel import FSDPPlanSpec, FSDPParallelPlan
from ..modules.moe.moe_layer import MoELayer, MoEConfig, MoESubmodules
from ..modules.moe.module import ModuleSpec
from ..modules.moe.experts import GroupedMLP
from ..modules.moe.config_converter import get_base_moe_config


def _replace_llada2_moe_only(
    original_module, hf_config, sp_info, ep_info, moe_submodules,
    engine_config, **kwargs
):
    from .modeling_llada2_moe import LLaDA2MoeSparseMoeBlock

    if not isinstance(original_module, LLaDA2MoeSparseMoeBlock):
        return original_module

    assert (
        engine_config.moe.aux_loss_coeff == 0.0
        and engine_config.moe.z_loss_coeff == 0.0
    ), "LLaDA2 diffusion does not support MoE aux/z losses; set both coefficients to 0."

    ep_group = ep_info["ep_group"]
    ep_size = ep_info["ep_size"]

    llada_config_kwargs = get_base_moe_config(
        hf_config=hf_config,
        ep_size=ep_size,
        num_moe_experts=hf_config.num_experts,
        params_dtype=hf_config.torch_dtype,
        moe_token_dispatcher_type=engine_config.moe.token_dispatcher,
        moe_router_score_function="sigmoid",
        # LLaDA2.1-mini uses topk=8 and group_topk=4, so the MoE layer's
        # topk // group_topk rule matches the native gate's top-2 group scores.
        moe_router_num_groups=hf_config.n_group,
        moe_router_group_topk=hf_config.topk_group,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
    )
    moe_config = MoEConfig(**llada_config_kwargs)

    moe_layer = MoELayer(
        config=moe_config,
        ep_group=ep_group,
        submodules=moe_submodules,
        sp_group=sp_info["sp_group"],
    )

    if hasattr(original_module, "shared_experts"):
        moe_layer.shared_experts = original_module.shared_experts

    return moe_layer


def get_parallel_plan() -> FSDPParallelPlan:
    from torch.distributed._tensor import Shard
    from ...parallel import FSDPPlacementSpec

    moe_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))

    return FSDPParallelPlan(
        plan={
            "model.layers.*": FSDPPlanSpec(
                activation_checkpoint=True,
            ),
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
            "model.layers.*.mlp": FSDPPlanSpec(
                replace_module=_replace_llada2_moe_only,
                module_kwargs={"moe_submodules": moe_submodules},
            ),
            "model.word_embeddings": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
            "lm_head": FSDPPlanSpec(
                standalone_world_shard=True,
            ),
        },
        sp_size=1,
        ep_size=1,
    )
