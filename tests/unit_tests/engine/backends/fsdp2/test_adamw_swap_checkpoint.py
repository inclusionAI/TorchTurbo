"""Checkpoint round-trip coverage for AdamWSwap optimizer state."""

import copy
import os
import sys
import types

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch

from torchturbo.engine.backends.fsdp2.optimizers.adamw import AdamWSwap
from torchturbo.runtime.cpu import CpuRuntime


class SyncRuntime(CpuRuntime):
    """Plain CPU runtime for exercising host-side control flow."""


# torch mis-dispatches CPU fused AdamW to the MPS backend on MPS builds
# without CUDA (Apple silicon laptops): it raises "missing kernel for mps"
# or crashes the process inside torch._fused_adamw_. CUDA/NPU hosts are
# unaffected. Skip the tests that must run optimizer steps.
CPU_FUSED_ADAMW_BROKEN = (
    torch.backends.mps.is_built() and not torch.cuda.is_available()
)
requires_optimizer_step = pytest.mark.skipif(
    CPU_FUSED_ADAMW_BROKEN,
    reason="CPU fused AdamW mis-dispatches to the MPS backend on this torch build",
)


def _param_groups(params):
    return [
        {"params": list(params[:2]), "lr": 2e-3, "weight_decay": 0.1},
        {"params": [params[2]], "lr": 5e-4, "weight_decay": 0.0},
    ]


def _fresh_optimizer():
    torch.manual_seed(5)
    params = [
        torch.nn.Parameter(torch.randn(3, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(2, 3)),
    ]
    optimizer = AdamWSwap(
        _param_groups(params),
        device_runtime=SyncRuntime(),
        optimizer_memory_fraction=0,
    )
    return optimizer, params


def _step_all(optimizer, params, seed):
    generator = torch.Generator().manual_seed(seed)
    for param in params:
        param.grad = torch.randn(param.shape, generator=generator)
    optimizer.step()
    optimizer.zero_grad()


@requires_optimizer_step
def test_state_dict_exposes_standard_adamw_format():
    optimizer, params = _fresh_optimizer()
    _step_all(optimizer, params, seed=1)

    state = optimizer.state_dict()
    assert set(state) == {"state", "param_groups"}
    ids = [pid for group in state["param_groups"] for pid in group["params"]]
    assert all(isinstance(pid, int) for pid in ids)
    assert sorted(state["state"]) == sorted(ids)
    entry = state["state"][ids[0]]
    assert set(entry) == {"step", "exp_avg", "exp_avg_sq"}

    # The serialized tensors alias the live swap buffers (in-place loaders
    # write straight into them).
    param = params[0]
    assert entry["step"] is optimizer.param_to_step_map[param]
    assert entry["exp_avg"] is optimizer.param_to_cpu_states_map[param]["exp_avg"]
    assert entry["exp_avg_sq"] is optimizer.param_to_cpu_states_map[param]["exp_avg_sq"]


@requires_optimizer_step
def test_state_dict_round_trip_restores_swap_state():
    trained, trained_params = _fresh_optimizer()
    for step in range(4):
        _step_all(trained, trained_params, seed=10 + step)
    saved = trained.state_dict()

    resumed, resumed_params = _fresh_optimizer()
    buffer_ptr = resumed.param_to_cpu_states_map[resumed_params[0]]["exp_avg"].data_ptr()

    # A real resume restores the model weights from the same checkpoint
    # before the optimizer state arrives.
    with torch.no_grad():
        for trained_param, resumed_param in zip(trained_params, resumed_params):
            resumed_param.copy_(trained_param)
    resumed.load_state_dict(saved)

    # Loading copies into the existing buffers instead of adopting new ones.
    assert (
        resumed.param_to_cpu_states_map[resumed_params[0]]["exp_avg"].data_ptr()
        == buffer_ptr
    )
    for trained_param, resumed_param in zip(trained_params, resumed_params):
        torch.testing.assert_close(
            resumed.param_to_step_map[resumed_param],
            trained.param_to_step_map[trained_param],
        )
        for key in resumed.state_keys:
            torch.testing.assert_close(
                resumed.param_to_cpu_states_map[resumed_param][key],
                trained.param_to_cpu_states_map[trained_param][key],
                rtol=0,
                atol=0,
            )

    # Continuation equivalence: identical gradients drive identical updates.
    generator = torch.Generator().manual_seed(99)
    grads = [torch.randn(param.shape, generator=generator) for param in trained_params]
    for param, grad in zip(trained_params, grads):
        param.grad = grad.clone()
    for param, grad in zip(resumed_params, grads):
        param.grad = grad.clone()
    trained.step()
    resumed.step()

    for trained_param, resumed_param in zip(trained_params, resumed_params):
        torch.testing.assert_close(resumed_param, trained_param, rtol=0, atol=0)
        torch.testing.assert_close(
            resumed.param_to_step_map[resumed_param],
            trained.param_to_step_map[trained_param],
        )


@requires_optimizer_step
def test_state_dict_routes_moments_through_checkpoint_wrapper():
    optimizer, params = _fresh_optimizer()
    calls = []

    def record_wrap(tensor, param):
        calls.append((id(tensor), param))
        return tensor

    optimizer._wrap_checkpoint_tensor = record_wrap
    state = optimizer.state_dict()
    ids = [pid for group in state["param_groups"] for pid in group["params"]]

    assert len(calls) == 2 * len(params)
    wrapped_tensor_ids = {tensor_id for tensor_id, _ in calls}
    for param_id, param in zip(ids, params):
        step_id = id(state["state"][param_id]["step"])
        assert step_id not in wrapped_tensor_ids
        assert (
            id(state["state"][param_id]["exp_avg"]),
            param,
        ) in calls
        assert (
            id(state["state"][param_id]["exp_avg_sq"]),
            param,
        ) in calls


def test_state_dict_is_repeatable_and_does_not_mutate_runtime_state():
    optimizer, params = _fresh_optimizer()
    calls = []

    def replace_with_marker(tensor, param):
        marker = object()
        calls.append(marker)
        return marker

    optimizer._wrap_checkpoint_tensor = replace_with_marker
    first = optimizer.state_dict()
    second = optimizer.state_dict()
    ids = [pid for group in first["param_groups"] for pid in group["params"]]

    assert len(calls) == 4 * len(params)
    for param_id, param in zip(ids, params):
        live_state = optimizer.param_to_cpu_states_map[param]
        assert first["state"][param_id]["step"] is live_state["step"]
        assert second["state"][param_id]["step"] is live_state["step"]
        for key in optimizer.state_keys:
            assert first["state"][param_id][key] is not live_state[key]
            assert second["state"][param_id][key] is not live_state[key]


def test_checkpoint_wrapper_preserves_local_storage_and_metadata(monkeypatch):
    optimizer, _ = _fresh_optimizer()
    local_tensor = torch.randn(2, 3)
    source_mesh = object()
    cpu_mesh = object()
    calls = []

    class FakeDTensor:
        @staticmethod
        def from_local(*args, **kwargs):
            calls.append((args, kwargs))
            return "checkpoint dtensor"

    class FakeParam(FakeDTensor):
        device_mesh = source_mesh
        placements = ("placement",)
        shape = torch.Size((4, 6))

        @staticmethod
        def stride():
            return (6, 1)

    from torchturbo.engine.backends.fsdp2.optimizers import state as state_module

    monkeypatch.setattr(state_module, "DTensor", FakeDTensor)
    optimizer._checkpoint_meshes[source_mesh] = cpu_mesh

    result = optimizer._wrap_checkpoint_tensor(local_tensor, FakeParam())

    assert result == "checkpoint dtensor"
    args, kwargs = calls[0]
    assert args[0] is local_tensor
    assert args[1] is cpu_mesh
    assert args[2] == ("placement",)
    assert kwargs == {
        "run_check": False,
        "shape": torch.Size((4, 6)),
        "stride": (6, 1),
    }


def test_load_state_dict_refreshes_hyperparameters():
    optimizer, _ = _fresh_optimizer()
    saved = optimizer.state_dict()
    saved["param_groups"][0]["lr"] = 1e-5
    saved["param_groups"][1]["betas"] = (0.8, 0.99)

    optimizer.load_state_dict(saved)

    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["betas"] == (0.8, 0.99)


@requires_optimizer_step
def test_load_state_dict_with_empty_state_keeps_buffers():
    optimizer, params = _fresh_optimizer()
    _step_all(optimizer, params, seed=3)
    before = {
        param: optimizer.param_to_cpu_states_map[param]["exp_avg"].clone()
        for param in params
    }

    optimizer.load_state_dict(
        {"state": {}, "param_groups": optimizer.state_dict()["param_groups"]}
    )

    for param in params:
        torch.testing.assert_close(
            optimizer.param_to_cpu_states_map[param]["exp_avg"], before[param]
        )


def test_load_state_dict_ignores_unknown_state_entries():
    optimizer, params = _fresh_optimizer()
    # Deep-copy first: state_dict() hands out live buffer references, so
    # injecting the key in place would corrupt the optimizer under test.
    saved = copy.deepcopy(optimizer.state_dict())
    pid = saved["param_groups"][0]["params"][0]
    saved["state"][pid]["max_exp_avg_sq"] = torch.zeros_like(params[0])

    optimizer.load_state_dict(saved)  # must not raise

    assert "max_exp_avg_sq" not in optimizer.param_to_cpu_states_map[params[0]]


@requires_optimizer_step
def test_load_state_dict_accepts_python_step():
    optimizer, params = _fresh_optimizer()
    saved = copy.deepcopy(optimizer.state_dict())
    pid = saved["param_groups"][0]["params"][0]
    saved["state"][pid]["step"] = 7

    optimizer.load_state_dict(saved)

    assert optimizer.param_to_step_map[params[0]].item() == 7
    assert optimizer.state[params[0]]["step"] is optimizer.param_to_step_map[params[0]]


@pytest.mark.parametrize("key", ["step", "exp_avg"])
def test_load_state_dict_rejects_mismatched_state_shape(key):
    optimizer, params = _fresh_optimizer()
    saved = copy.deepcopy(optimizer.state_dict())
    pid = saved["param_groups"][0]["params"][0]

    if key == "step":
        saved["state"][pid][key] = torch.zeros(1)
    else:
        saved["state"][pid][key] = torch.zeros(3, 5)

    with pytest.raises(ValueError, match=f"'{key}'.*shape"):
        optimizer.load_state_dict(saved)


def test_load_state_dict_rejects_mismatched_groups():
    optimizer, _ = _fresh_optimizer()
    saved = optimizer.state_dict()
    saved["param_groups"] = saved["param_groups"][:-1]

    with pytest.raises(ValueError, match="parameter groups"):
        optimizer.load_state_dict(saved)

    saved = optimizer.state_dict()
    saved["param_groups"][0]["params"] = saved["param_groups"][0]["params"][:-1]
    with pytest.raises(ValueError, match="number of parameters"):
        optimizer.load_state_dict(saved)


def test_load_state_dict_rejects_unknown_parameter_state():
    optimizer, _ = _fresh_optimizer()
    saved = optimizer.state_dict()
    saved["state"]["nonexistent"] = {"step": 1}

    with pytest.raises(ValueError, match="unknown parameter"):
        optimizer.load_state_dict(saved)
