"""DeepSeekV4 MoE unit tests.

There are two MoE paths in this model:

* native DeepseekV4Gate/DeepseekV4Expert, used by the base modeling class;
* generic MoELayer replacement, configured by the TorchTurbo parallel plan.

These tests keep the two paths separate so their different config fields are
not read as an output-equivalence claim.
"""

import os
import sys
import types
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeekV4 MoE tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from .test_alignment import TEST_CONFIG_MODELING  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution
    from test_alignment import TEST_CONFIG_MODELING  # noqa: E402


def make_config(**overrides):
    from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    config_dict = dict(TEST_CONFIG_MODELING)
    config_dict.update(
        {
            "hidden_size": 8,
            "moe_intermediate_size": 6,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 16,
            "num_hash_layers": 1,
            "route_scale": 1.0,
            "routed_scaling_factor": 99.0,
            "scoring_func": "sqrtsoftplus",
            "swiglu_limit": 2.0,
        }
    )
    config_dict.update(overrides)
    return DeepseekV4Config(**config_dict)


def test_generic_moe_plan_config_uses_routed_scaling_factor(monkeypatch):
    from torchturbo.engine.config import EngineConfig
    from torchturbo.models.deepseek_v4 import parallel_plan

    class DummyMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Module()
            self.experts = torch.nn.Module()
            self.layer_idx = 0

    class DummyMoELayer:
        def __init__(self, config, **_kwargs):
            self.router = types.SimpleNamespace(config=config)

    monkeypatch.setattr(parallel_plan, "MoELayer", DummyMoELayer)

    config = make_config(route_scale=1.0, routed_scaling_factor=7.0)
    moe_layer = parallel_plan._replace_moe_only(
        DummyMoE(),
        config,
        sp_info={"sp_group": None},
        ep_info={"ep_group": None, "ep_size": 1},
        moe_submodules=None,
        engine_config=EngineConfig(),
    )

    assert moe_layer.router.config.moe_router_topk_scaling_factor == config.routed_scaling_factor


def test_native_hash_gate_uses_token_id_lookup_and_route_scale():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Gate

    config = make_config(route_scale=1.5, routed_scaling_factor=7.0, num_hash_layers=1)
    gate = DeepseekV4Gate(config, layer_idx=0)
    gate.weight.data.copy_(
        torch.arange(config.n_routed_experts * config.hidden_size, dtype=torch.float32).view(
            config.n_routed_experts, config.hidden_size
        )
        / 100
    )
    gate.tid2eid.data.copy_(
        torch.tensor(
            [[0, 1], [2, 3], [3, 0], [1, 2]] * 4,
            dtype=torch.long,
        )
    )

    hidden = torch.arange(24, dtype=torch.float32).view(3, config.hidden_size) / 10
    input_ids = torch.tensor([0, 2, 3], dtype=torch.long)
    weights, indices = gate(hidden, input_ids)

    scores = F.softplus(F.linear(hidden.float(), gate.weight.float())).sqrt()
    expected_indices = gate.tid2eid[input_ids]
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)
    expected_weights = expected_weights * config.route_scale

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)
def test_native_score_gate_bias_selects_topk_but_weights_use_unbiased_scores():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Gate

    config = make_config(route_scale=2.0, num_hash_layers=0, scoring_func="sqrtsoftplus")
    gate = DeepseekV4Gate(config, layer_idx=1)
    gate.weight.data.zero_()
    gate.weight.data[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    gate.bias.data.copy_(torch.tensor([10.0, 0.0, 0.0, -10.0]))

    hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    weights, indices = gate(hidden)

    scores = F.softplus(F.linear(hidden.float(), gate.weight.float())).sqrt()
    biased_scores = scores + gate.bias
    expected_indices = biased_scores.topk(config.num_experts_per_tok, dim=-1)[1]
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * config.route_scale

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)
    assert indices.tolist() == [[0, 2]]


def test_expert_swiglu_limit_matches_fp32_reference():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Expert

    config = make_config(swiglu_limit=1.25)
    expert = DeepseekV4Expert(config).bfloat16()
    expert.w1.weight.data.copy_(
        torch.linspace(-2.0, 2.0, steps=expert.w1.weight.numel()).view_as(expert.w1.weight).bfloat16()
    )
    expert.w2.weight.data.copy_(
        torch.linspace(-1.0, 1.0, steps=expert.w2.weight.numel()).view_as(expert.w2.weight).bfloat16()
    )
    expert.w3.weight.data.copy_(
        torch.linspace(1.5, -1.5, steps=expert.w3.weight.numel()).view_as(expert.w3.weight).bfloat16()
    )

    x = torch.linspace(-3.0, 3.0, steps=16, dtype=torch.bfloat16).view(2, config.hidden_size)
    route_weights = torch.tensor([[0.5], [1.5]], dtype=torch.float32)
    actual = expert(x, route_weights)

    gate = expert.w1(x).float()
    up = expert.w3(x).float()
    gate = torch.clamp(gate, max=config.swiglu_limit)
    up = torch.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit)
    expected = F.silu(gate) * up
    expected = expected * route_weights
    expected = F.linear(expected.to(x.dtype), expert.w2.weight)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.bfloat16
