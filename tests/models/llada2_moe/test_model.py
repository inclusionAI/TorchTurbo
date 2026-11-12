from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch
from torch import nn

import torchturbo.models.llada2_moe.model as model_module
import torchturbo.models.llada2_moe.modeling_llada2_moe as modeling_module
import torchturbo.models.llada2_moe.parallel_plan as parallel_plan_module
from torchturbo.engine.config import EngineConfig
from torchturbo.models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from torchturbo.models.llada2_moe.generate import _load_torchturbo_runtime
from torchturbo.models.llada2_moe.model import TurboLLaDA2MoeForCausalLM
from torchturbo.models.llada2_moe.modeling_llada2_moe import LLaDA2MoeRMSNorm
from torchturbo.models.registry import get_model
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


class _FakeRotaryEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.device("meta"):
            self.register_buffer("inv_freq", torch.empty(2), persistent=False)
            self.register_buffer(
                "original_inv_freq",
                torch.empty(2),
                persistent=False,
            )
        self.attention_scaling = None
        self.config = object()

    @staticmethod
    def compute_default_rope_parameters(config, device):
        return torch.ones(2, device=device), 2.0


class _MaterializedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(rotary_emb=_FakeRotaryEmbedding())


def test_llada2_model_is_registered():
    assert get_model("llada2_moe") is TurboLLaDA2MoeForCausalLM


def test_llada2_builds_with_sdpa_inside_init_context():
    config = LLaDA2MoeConfig(
        vocab_size=16,
        pad_token_id=0,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        first_k_dense_replace=1,
        partial_rotary_factor=1.0,
        attn_implementation="sdpa",
    )

    original_forward = LLaDA2MoeRMSNorm.forward
    try:
        with TurboLLaDA2MoeForCausalLM.init_context(config, EngineConfig()):
            with torch.device("meta"):
                model = TurboLLaDA2MoeForCausalLM._from_config(config)

        assert model.config._attn_implementation == "sdpa"
        assert model.model._use_sdpa is True
        assert all(parameter.is_meta for parameter in model.parameters())
    finally:
        LLaDA2MoeRMSNorm.forward = original_forward


def test_llada2_rms_norm_patch_uses_ops_dispatcher(monkeypatch):
    ops_module = importlib.import_module("torchturbo.ops.ops")
    monkeypatch.setattr(ops_module, "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())

    original_forward = LLaDA2MoeRMSNorm.forward
    try:
        TurboLLaDA2MoeForCausalLM._pre_init_patches(config=None)
        norm = LLaDA2MoeRMSNorm(hidden_size=4, eps=1e-6)
        hidden_states = torch.randn(2, 4)

        actual = norm(hidden_states)
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        expected = norm.weight * (
            hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)
        ).to(hidden_states.dtype)

        torch.testing.assert_close(actual, expected)
    finally:
        LLaDA2MoeRMSNorm.forward = original_forward


def test_on_materialized_rebuilds_rotary_buffers_on_runtime_device(monkeypatch):
    runtime = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(model_module, "get_device_runtime", lambda: runtime)
    model = _MaterializedModel()

    TurboLLaDA2MoeForCausalLM.on_materialized(model)

    rotary_emb = model.model.rotary_emb
    assert rotary_emb.inv_freq.is_meta is False
    assert rotary_emb.original_inv_freq.is_meta is False
    assert rotary_emb.inv_freq.device == runtime.device
    assert rotary_emb.attention_scaling == 2.0


def test_parallel_plan_replacement_uses_current_moe_layer_signature(monkeypatch):
    class FakeSparseMoeBlock:
        shared_experts = object()

    captured = {}

    class FakeMoELayer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.shared_experts = None

    monkeypatch.setattr(
        modeling_module,
        "LLaDA2MoeSparseMoeBlock",
        FakeSparseMoeBlock,
    )
    monkeypatch.setattr(parallel_plan_module, "MoELayer", FakeMoELayer)
    monkeypatch.setattr(
        parallel_plan_module,
        "get_base_moe_config",
        lambda **kwargs: {},
    )

    original = FakeSparseMoeBlock()
    result = parallel_plan_module._replace_llada2_moe_only(
        original,
        hf_config=SimpleNamespace(
            num_experts=2,
            torch_dtype=torch.float32,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
        ),
        sp_info={"sp_group": None},
        ep_info={"ep_group": object(), "ep_size": 1},
        moe_submodules=object(),
        engine_config=EngineConfig(),
    )

    assert isinstance(result, FakeMoELayer)
    assert "use_fused_swiglu" not in captured
    assert result.shared_experts is original.shared_experts


def test_standalone_generate_loader_uses_runtime_package():
    is_npu_available, get_device_name = _load_torchturbo_runtime()

    assert isinstance(is_npu_available(), bool)
    assert get_device_name() in {"cpu", "cuda", "npu"}
