"""Tests for restoring optimizer state when some parameters are unused."""

import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
from torchturbo.engine.config import MuonOptimizerConfig
from torchturbo.engine.optimizers import MuonPlan, MuonRecipe, OptimizersContainer


def _model() -> nn.ModuleDict:
    return nn.ModuleDict({"active": nn.Linear(3, 3), "unused": nn.Linear(3, 3)})


def _optimizer(model: nn.Module, container: bool) -> torch.optim.Optimizer:
    if not container:
        return torch.optim.AdamW(model.parameters(), lr=0.02)
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=0.02, adamw_lr=0.01, weight_decay=0.0),
    )
    groups = plan.get_param_groups_by_optimizer_name()
    optimizers = {
        "Muon": torch.optim.SGD(
            [group.get_optimizer_param_group() for group in groups["Muon"]],
            lr=0.02,
            momentum=0.9,
        ),
        "AdamW": torch.optim.AdamW(
            [group.get_optimizer_param_group() for group in groups["AdamW"]],
            lr=0.01,
        ),
    }
    return OptimizersContainer(plan, optimizers)


def _step(
    model: nn.ModuleDict, optimizer: torch.optim.Optimizer, *, include_unused: bool
) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
    loss = model["active"](inputs).square().sum()
    if include_unused:
        loss = loss + model["unused"](inputs).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_optimizer_state(
    actual: torch.optim.Optimizer, expected: torch.optim.Optimizer
) -> None:
    actual_children = tuple(actual) if isinstance(actual, OptimizersContainer) else (actual,)
    expected_children = (
        tuple(expected) if isinstance(expected, OptimizersContainer) else (expected,)
    )
    for actual_child, expected_child in zip(actual_children, expected_children, strict=True):
        actual_state = actual_child.state_dict()
        expected_state = expected_child.state_dict()
        assert actual_state["param_groups"] == expected_state["param_groups"]
        torch.testing.assert_close(
            actual_state["state"], expected_state["state"], rtol=0, atol=0
        )


@pytest.mark.parametrize("container", [False, True], ids=["adamw", "container"])
@pytest.mark.parametrize("target_used", [False, True], ids=["fresh", "used"])
def test_restore_partial_state_preserves_first_update(tmp_path, container, target_used):
    torch.manual_seed(7)
    reference = _model()
    reference_optimizer = _optimizer(reference, container)
    for _ in range(3):
        _step(reference, reference_optimizer, include_unused=False)

    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_optimizer(reference_optimizer, reference, str(tmp_path))

    resumed = _model()
    resumed_optimizer = _optimizer(resumed, container)
    if target_used:
        _step(resumed, resumed_optimizer, include_unused=True)
    resumed.load_state_dict(reference.state_dict())
    checkpoint.load_optimizer(resumed_optimizer, resumed, str(tmp_path))

    children = tuple(resumed_optimizer) if container else (resumed_optimizer,)
    for child in children:
        assert all(param not in child.state for param in resumed["unused"].parameters())
    _assert_optimizer_state(resumed_optimizer, reference_optimizer)

    _step(reference, reference_optimizer, include_unused=True)
    _step(resumed, resumed_optimizer, include_unused=True)
    torch.testing.assert_close(resumed.state_dict(), reference.state_dict(), rtol=0, atol=0)
    _assert_optimizer_state(resumed_optimizer, reference_optimizer)


@pytest.mark.parametrize("missing", ["exp_avg", "lr"])
def test_restore_rejects_missing_saved_state_leaves(tmp_path, missing):
    model = _model()
    optimizer = _optimizer(model, container=False)
    _step(model, optimizer, include_unused=True)
    state = get_optimizer_state_dict(model, optimizer)
    if missing == "exp_avg":
        del state["state"]["active.weight"][missing]
    else:
        del state["param_groups"][0][missing]
    dcp.save({"optimizer": state}, checkpoint_id=str(tmp_path))

    resumed = _model()
    resumed_optimizer = _optimizer(resumed, container=False)
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    with pytest.raises(CheckpointException, match=f"Missing key.*{missing}"):
        checkpoint.load_optimizer(resumed_optimizer, resumed, str(tmp_path))


def test_restore_stateless_optimizer(tmp_path):
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    _step(model, optimizer, include_unused=False)
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_optimizer(optimizer, model, str(tmp_path))

    resumed_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint.load_optimizer(resumed_optimizer, model, str(tmp_path))
    assert resumed_optimizer.state_dict() == optimizer.state_dict()


def test_restore_rejects_saved_state_without_a_target(tmp_path):
    reference = _model()
    reference_optimizer = _optimizer(reference, container=False)
    _step(reference, reference_optimizer, include_unused=False)
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_optimizer(reference_optimizer, reference, str(tmp_path))

    resumed = _model()
    resumed_optimizer = _optimizer(resumed, container=False)
    resumed["unused"](torch.ones(2, 3)).square().sum().backward()
    resumed_optimizer.step()
    resumed_optimizer.zero_grad(set_to_none=True)

    with pytest.raises(RuntimeError, match="missing target state.*active"):
        checkpoint.load_optimizer(resumed_optimizer, resumed, str(tmp_path))


def test_restore_ignores_state_outside_current_parameter_groups(tmp_path):
    reference = _model()
    reference_optimizer = _optimizer(reference, container=False)
    _step(reference, reference_optimizer, include_unused=True)
    checkpoint = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=object())
    checkpoint.save_optimizer(reference_optimizer, reference, str(tmp_path))

    resumed = _model()
    resumed["unused"].requires_grad_(False)
    resumed_optimizer = torch.optim.AdamW(resumed["active"].parameters(), lr=0.02)
    checkpoint.load_optimizer(resumed_optimizer, resumed, str(tmp_path))

    assert set(resumed_optimizer.state) == set(resumed["active"].parameters())
    for name, parameter in resumed["active"].named_parameters():
        expected_parameter = dict(reference["active"].named_parameters())[name]
        torch.testing.assert_close(
            resumed_optimizer.state[parameter],
            reference_optimizer.state[expected_parameter],
            rtol=0,
            atol=0,
        )
