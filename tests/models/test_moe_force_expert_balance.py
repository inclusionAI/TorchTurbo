"""Benchmark-only forced MoE load balancing and configuration tests."""

from argparse import ArgumentTypeError
from unittest.mock import Mock

import pytest
import torch
import yaml
from torch.utils.checkpoint import checkpoint

from torchturbo.engine.args import parse_torchturbo_args
from torchturbo.engine.config import EngineConfig, build_from_args
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity


@pytest.mark.parametrize("score_function,pre_softmax", [
    ("softmax", False), ("softmax", True),
    ("sigmoid", True), ("sqrtsoftplus", True),
])
@pytest.mark.parametrize("tokens,experts,topk", [
    (23, 8, 2), (1, 8, 8), (19, 7, 3), (0, 8, 2), (2048, 512, 8),
])
def test_forced_routing_balances_assignments_and_retains_weights(
    score_function, pre_softmax, tokens, experts, topk,
):
    logits = torch.randn(tokens, experts)
    kwargs = dict(use_pre_softmax=pre_softmax, score_function=score_function,
                  scaling_factor=1.7)
    original = topk_softmax_with_capacity(logits, topk, **kwargs)
    routing = topk_softmax_with_capacity(
        logits, topk, force_expert_balance=True, **kwargs,
    )
    probs, routing_map = routing.probs, routing.routing_map
    counts = routing_map.sum(dim=0)

    assert probs.shape == logits.shape
    assert routing_map.dtype == torch.bool
    assert torch.all(routing_map.sum(dim=1) == topk)
    assert counts.sum().item() == tokens * topk
    assert (counts.max() - counts.min()).item() <= 1
    assert torch.all(probs[~routing_map] == 0)
    torch.testing.assert_close(
        probs[routing_map].reshape(tokens, topk).sort(dim=-1).values,
        original.selected_probs().sort(dim=-1).values,
        rtol=0, atol=0,
    )


def test_forced_routing_overrides_skew_and_group_restrictions():
    logits = torch.zeros(32, 8)
    logits[:, :2] = 10
    natural = topk_softmax_with_capacity(
        logits, 2, use_pre_softmax=True, num_groups=4, group_topk=1,
    )
    routing = topk_softmax_with_capacity(
        logits, 2, use_pre_softmax=True, num_groups=4, group_topk=1,
        force_expert_balance=True,
    )
    assert torch.all(natural.routing_map[:, :2])
    assert not torch.equal(routing.routing_map, natural.routing_map)
    torch.testing.assert_close(routing.routing_map.sum(dim=0), torch.full((8,), 8))


def test_forced_routing_overrides_hash_assignments():
    kwargs = dict(use_hash_routing=True, hash_table=torch.tensor([[0, 1]]),
                  input_ids=torch.zeros(16, dtype=torch.long))
    routing = topk_softmax_with_capacity(
        torch.randn(16, 8), 2, force_expert_balance=True, **kwargs,
    )
    assert not torch.all(routing.routing_map[:, :2])
    torch.testing.assert_close(routing.routing_map.sum(dim=0), torch.full((8,), 4))


def test_forced_routing_retains_probability_gradients():
    logits = torch.randn(19, 8, requires_grad=True)
    original = topk_softmax_with_capacity(
        logits, 2, use_pre_softmax=True,
    )
    routing = topk_softmax_with_capacity(
        logits, 2, use_pre_softmax=True, force_expert_balance=True,
    )
    original_grad, = torch.autograd.grad(original.selected_probs().square().sum(), logits)
    forced_grad, = torch.autograd.grad(routing.selected_probs().square().sum(), logits)
    torch.testing.assert_close(forced_grad, original_grad)
    assert torch.isfinite(forced_grad).all()
    assert forced_grad.abs().sum() > 0


def _router(force_expert_balance=False):
    return TopKRouter(MoEConfig(
        hidden_size=4, num_moe_experts=8, moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_force_expert_balance=force_expert_balance,
    ))


def test_forced_routing_is_deterministic_under_checkpoint_recomputation():
    router = _router(True)
    inputs = torch.randn(7, 3, 4, requires_grad=True)
    maps = []

    def forward(value):
        routing, _ = router(value)
        maps.append(routing.routing_map.detach().clone())
        return routing.probs.square().sum()

    checkpoint(forward, inputs, use_reentrant=True).backward()

    assert len(maps) == 2
    assert torch.equal(maps[0], maps[1])
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(router.weight.grad).all()
    assert router.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("enabled", [True, False, "true", "false"])
def test_force_expert_balance_yaml_round_trip(tmp_path, enabled):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"engine_config": {"backend_config": {
        "force_expert_balance": enabled,
    }}}))
    tt, _ = parse_torchturbo_args(["--tt_config", str(path)])
    config = build_from_args(tt)
    expected = enabled is True or enabled == "true"
    assert config.backend_config.force_expert_balance is expected
    restored = EngineConfig.from_dict({"backend_config": config.to_dict()["backend_config"]})
    assert restored.backend_config.force_expert_balance is expected


def test_force_expert_balance_defaults_to_disabled():
    assert EngineConfig().backend_config.force_expert_balance is False
    assert build_from_args(parse_torchturbo_args([])[0]).backend_config.force_expert_balance is False
    assert _router().config.moe_router_force_expert_balance is False


def test_force_expert_balance_rejects_invalid_boolean():
    with pytest.raises(ArgumentTypeError):
        EngineConfig.from_dict({"backend_config": {"force_expert_balance": "not-a-boolean"}})


@pytest.mark.parametrize("enabled,rank,warnings", [(True, 0, 3), (True, 1, 0), (False, 0, 0)])
def test_backend_configures_all_routers_and_warns_three_times(monkeypatch, enabled, rank, warnings):
    import torchturbo.engine.backends.fsdp2.backend as backend_module

    logger = Mock()
    monkeypatch.setattr(backend_module, "logger", logger)
    monkeypatch.setattr(backend_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend_module.dist, "get_rank", lambda: rank)
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = EngineConfig.from_dict({"backend_config": {"force_expert_balance": enabled}}).backend_config
    model = torch.nn.Sequential(_router(), _router())
    keys = list(model.state_dict())
    backend._configure_moe_routing(model)

    assert all(router.config.moe_router_force_expert_balance is enabled for router in model)
    assert list(model.state_dict()) == keys
    assert logger.warning.call_count == warnings
    for call in logger.warning.call_args_list:
        message = call.args[0]
        assert "!" in message
        assert "performance upper bound" in message
        assert "accuracy is INCORRECT" in message
    if warnings:
        assert len({call.args[0] for call in logger.warning.call_args_list}) == 1


def test_enabled_force_expert_balance_rejects_models_without_shared_router():
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = EngineConfig.from_dict({"backend_config": {"force_expert_balance": True}}).backend_config
    with pytest.raises(ValueError, match="requires TorchTurbo TopKRouter"):
        backend._configure_moe_routing(torch.nn.Linear(4, 4))


def test_old_moe_force_expert_balance_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "old_config.yaml"
    path.write_text("engine_config:\n  moe:\n    force_expert_balance: true\n")
    tt, _ = parse_torchturbo_args(["--tt_config", str(path)])
    with pytest.raises(ValueError, match="Unknown key: moe.force_expert_balance"):
        build_from_args(tt)


def test_old_backend_force_load_balance_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "old_backend_config.yaml"
    path.write_text("engine_config:\n  backend_config:\n    force_load_balance: true\n")
    tt, _ = parse_torchturbo_args(["--tt_config", str(path)])
    with pytest.raises(ValueError, match="Unknown key: backend_config.force_load_balance"):
        build_from_args(tt)


@pytest.mark.parametrize("zero_weights", [False, True])
def test_forced_expert_indices_and_selected_probs_match_mask_order(zero_weights):
    logits = torch.full((19, 7), -1000.0) if zero_weights else torch.randn(19, 7)
    logits.requires_grad_()
    routing = topk_softmax_with_capacity(
        logits, 3, score_function="sigmoid", force_expert_balance=True,
    )
    expected_indices = routing.routing_map.nonzero(as_tuple=True)[1].reshape(19, 3)
    torch.testing.assert_close(routing.expert_indices, expected_indices, rtol=0, atol=0)
    expected_probs = routing.probs.masked_select(routing.routing_map).reshape(19, 3)
    actual_probs = routing.selected_probs()
    torch.testing.assert_close(actual_probs, expected_probs, rtol=0, atol=0)
    if zero_weights:
        assert torch.count_nonzero(actual_probs) == 0
        assert routing.expert_indices.numel() == 57
    upstream = torch.randn_like(actual_probs)
    expected_grad, = torch.autograd.grad((expected_probs * upstream).sum(), logits, retain_graph=True)
    actual_grad, = torch.autograd.grad((actual_probs * upstream).sum(), logits)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
