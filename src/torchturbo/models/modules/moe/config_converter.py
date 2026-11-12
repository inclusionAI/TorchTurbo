import torch.nn.functional as F
from transformers import PretrainedConfig


def get_base_moe_config(
    hf_config: PretrainedConfig, ep_size: int, **override_config_kwargs
) -> dict:
    base_config_kwargs = dict(
        # model arch
        hidden_size=hf_config.hidden_size,
        gated_linear_unit=True,
        activation_func=F.silu,
        # parallel
        expert_model_parallel_size=ep_size,
        # moe related params
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_load_balancing_type="none",
        moe_router_topk=hf_config.num_experts_per_tok,
        moe_token_dispatcher_type="alltoall",
    )
    base_config_kwargs.update(**override_config_kwargs)
    return base_config_kwargs
