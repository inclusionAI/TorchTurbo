import pytest
import torch
import transformers.integrations.flash_attention as flash_integration
import transformers.modeling_flash_attention_utils as hf_attention
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.engine.config import MuonOptimizerConfig, QKClipConfig
from torchturbo.engine.optimizers import (
    ExcludeFromMuon,
    MLAQKClipSpec,
    MuonPlan,
    MuonRecipe,
)
from torchturbo.models.bailing_hybrid.configuration_bailing_moe_v3 import BailingMoeV3Config
from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM
from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import BailingMoeV3MultiLatentAttention
from torchturbo.models.kimi_k25.configuration_deepseek import DeepseekV3Config
from torchturbo.models.kimi_k25.modeling_deepseek import DeepseekV3FlashAttention2
from torchturbo.models.kimi_k25.parallel_plan import _replace_self_attn
from torchturbo.models.modules.attention import interface
from torchturbo.models.modules.attention.qk_clip import get_qk_clip_state
from torchturbo.models.modules.attention.ulysses import TURBO_ULYSSES_ATTENTION
from torchturbo.ops import OpDispatcher, registry
from torchturbo.runtime.cpu import CpuRuntime


def _attention(model_name: str) -> nn.Module:
    config_kwargs = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": None,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "max_position_embeddings": 32,
    }
    if model_name == "kimi":
        config = DeepseekV3Config(
            **config_kwargs, rope_scaling={"type": "linear", "factor": 1.0}
        )
        config._attn_implementation = "flash_attention_2"
        return DeepseekV3FlashAttention2(config, layer_idx=0)
    config = BailingMoeV3Config(**config_kwargs)
    config._attn_implementation = "flash_attention_2"
    return BailingMoeV3MultiLatentAttention(config, layer_idx=0)


def _prepare_optimizer(attention: nn.Module) -> torch.optim.Optimizer:
    model = nn.Module()
    model.attention = attention
    recipe = MuonRecipe(
        {"**": ExcludeFromMuon()},
        qk_clip_specs={
            "attention": MLAQKClipSpec("q_proj", "kv_b_proj", 2, 1, 2, 2),
        },
    )
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(
            lr=0.1,
            weight_decay=0.0,
            adamw_fused=False,
            qk_clip=QKClipConfig(),
        ),
    )
    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = CpuRuntime()
    backend.fsdp_config = FSDPConfig()
    backend._mesh = {"edp": object()}
    backend._sp_group = None
    backend._sp_size = 1
    backend._sp_rank = 0
    return backend.prepare_optimizer(plan)


@pytest.mark.parametrize("model_name", ["kimi", "bailing"])
def test_prepared_attention_collects_only_training_logits(monkeypatch, model_name):
    monkeypatch.setattr(interface, "ops", OpDispatcher(device_runtime=CpuRuntime(), registry=registry))
    attention = _attention(model_name)
    original_state_keys = set(attention.state_dict())
    assert "qk_clip_state" not in vars(attention)
    assert get_qk_clip_state(attention) is None

    def patch() -> None:
        if model_name == "kimi":
            _replace_self_attn(attention, None, {"sp_group": None, "sp_size": 1}, None)
        else:
            # Production dispatch at SP=1: bailing's on_parallelized calls
            # register_sp_attention_interface, which registers the shared
            # interface at any SP size, and the forward resolves it from the
            # registry by config name.
            model = object.__new__(TurboBailingMoeV3ForCausalLM)
            model.config = attention.config
            model.register_sp_attention_interface(
                {"sp_group": None, "sp_size": 1}
            )

    query = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    query = query.reshape(1, 1, 2, 3).expand(1, 3, 2, 3)
    key = torch.ones_like(query)
    value = torch.ones(1, 3, 2, 2)
    positions = torch.arange(3).reshape(1, 3)

    def forward(module: nn.Module = attention) -> torch.Tensor:
        if model_name == "kimi":
            return module._flash_attention_forward(
                query, key, value, None, query_length=3, is_causal=True,
                softmax_scale=0.5, position_ids=positions,
            )
        return ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation](
            module, query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            None, position_ids=positions, scaling=0.5,
        )[0]

    hf_calls = []

    def hf_forward(query, key, value, *args, **kwargs):
        hf_calls.append(True)
        return torch.zeros_like(value)

    monkeypatch.setattr(hf_attention, "_flash_attention_forward", hf_forward)
    # Mirror the engine's startup binding so modules still dispatching to
    # flash_attention_2 reach the mock instead of the real flash-attn kernel.
    monkeypatch.setattr(flash_integration, "_flash_attention_forward", hf_forward)
    try:
        patch()
        assert "qk_clip_state" not in vars(attention)
        forward()
        assert len(hf_calls) == 1
        if model_name == "bailing":
            unprepared_attention = _attention(model_name)
            forward(unprepared_attention)
            assert "qk_clip_state" not in vars(unprepared_attention)
            assert len(hf_calls) == 2

        num_hf_calls = len(hf_calls)
        optimizer = _prepare_optimizer(attention)
        state = get_qk_clip_state(attention)
        assert state is not None
        output = forward()
        torch.testing.assert_close(output, torch.ones_like(output))
        torch.testing.assert_close(state.max_logits, torch.tensor([1.0, 1.5]))
        assert state.observed
        assert len(hf_calls) == num_hf_calls

        patch()
        assert get_qk_clip_state(attention) is state
        torch.testing.assert_close(state.max_logits, torch.tensor([1.0, 1.5]))
        assert state.observed
        torch.testing.assert_close(forward(), output)
        assert len(hf_calls) == num_hf_calls
        optimizer.step()
        assert not state.observed
        assert torch.isneginf(state.max_logits).all()
        assert set(attention.state_dict()) == original_state_keys

        attention.eval()
        forward()
        assert not state.observed
        assert torch.isneginf(state.max_logits).all()
        assert len(hf_calls) == num_hf_calls + 1
    finally:
        if model_name == "bailing":
            # The mask-table entry is a class-level global declaration; it
            # cannot be popped (delete only touches instance state) and is
            # intentionally left in place.
            ALL_ATTENTION_FUNCTIONS.pop(TURBO_ULYSSES_ATTENTION, None)
