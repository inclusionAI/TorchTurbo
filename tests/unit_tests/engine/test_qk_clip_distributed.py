from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from datetime import timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.debug import CommDebugMode

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode
from torchturbo.engine.backends.fsdp2.qk_clip import install_qk_clip
from torchturbo.engine.config import EngineConfig, MuonOptimizerConfig, OptimizerConfig, QKClipConfig
from torchturbo.engine.distributed import assert_finite
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.engine.optimizers import (
    ExcludeFromMuon,
    MuonPlan,
    MuonRecipe,
    OptimizerPlan,
    OptimizersContainer,
    ParamGroup,
)
from torchturbo.engine.optimizers.qk_clip import (
    MLAQKClipSpec,
    QKClipLayer,
    QKClipPlan,
    QKClipSpec,
)
from torchturbo.models.modules.attention import interface
from torchturbo.models.modules.attention.qk_clip import get_qk_clip_state
from torchturbo.ops.providers.native.attention import TorchAttentionWithMaxLogits
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime


class _MLAAttention(nn.Module):
    def __init__(self, query: Tensor, key_value: Tensor) -> None:
        super().__init__()
        self.query_weight = nn.Parameter(query)
        self.key_value_weight = nn.Parameter(key_value)


def _make_spec(num_heads: int, nope: int, rope: int, value: int) -> MLAQKClipSpec:
    return MLAQKClipSpec(
        query_projection="query",
        key_value_projection="key_value",
        num_heads=num_heads,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=value,
    )


def _make_plan(
    modules: list[_MLAAttention], specs: list[MLAQKClipSpec], *, alpha: float = 0.5
) -> QKClipPlan:
    return cast(
        QKClipPlan,
        SimpleNamespace(
            config=QKClipConfig(threshold=100.0, alpha=alpha),
            layers=tuple(
                QKClipLayer(
                    module=module,
                    weights={
                        spec.query_projection: module.query_weight,
                        spec.key_value_projection: module.key_value_weight,
                    },
                    spec=spec,
                )
                for module, spec in zip(modules, specs, strict=True)
            ),
        ),
    )


def _reference_clip(
    query: Tensor, key_value: Tensor, maxima: Tensor, spec: MLAQKClipSpec, alpha: float
) -> None:
    for head in range(spec.num_heads):
        scale = min(1.0, 100.0 / max(100.0, maxima[head].item()))
        query_begin = head * (spec.qk_nope_head_dim + spec.qk_rope_head_dim)
        query_split = query_begin + spec.qk_nope_head_dim
        query_end = query_split + spec.qk_rope_head_dim
        query[query_begin:query_split].mul_(scale**alpha)
        query[query_split:query_end].mul_(scale)
        key_begin = head * (spec.qk_nope_head_dim + spec.v_head_dim)
        key_value[key_begin : key_begin + spec.qk_nope_head_dim].mul_(
            scale ** (1.0 - alpha)
        )


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_qk_clip_runs_after_optimizer_and_resets_stats(alpha: float) -> None:
    spec = _make_spec(3, 2, 1, 2)
    query = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10
    key_value = torch.arange(36, dtype=torch.float32).reshape(12, 3) / 10
    module = _MLAAttention(query.clone(), key_value.clone())
    assert get_qk_clip_state(module) is None
    assert "qk_clip_state" not in vars(module)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.2, momentum=0.9)
    controller = install_qk_clip(_make_plan([module], [spec], alpha=alpha), optimizer)
    module.qk_clip_state.record(torch.tensor([50.0, 200.0, 100.0]))
    module.qk_clip_state.record(torch.tensor([25.0, 100.0, 400.0]))
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    query.sub_(0.2)
    key_value.sub_(0.2)
    _reference_clip(query, key_value, torch.tensor([50.0, 200.0, 400.0]), spec, alpha)
    torch.testing.assert_close(module.query_weight, query)
    torch.testing.assert_close(module.key_value_weight, key_value)
    assert torch.isneginf(module.qk_clip_state.max_logits).all()
    assert not module.qk_clip_state.observed
    for parameter in module.parameters():
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], torch.ones_like(parameter)
        )

    module.qk_clip_state.record(torch.full((3,), 1000.0))
    controller.reset()
    assert torch.isneginf(module.qk_clip_state.max_logits).all()
    assert not module.qk_clip_state.observed
    assert set(module.state_dict()) == {"query_weight", "key_value_weight"}
    assert set(optimizer.state_dict()) == {"state", "param_groups"}
    assert all(
        set(state) == {"momentum_buffer"}
        for state in optimizer.state_dict()["state"].values()
    )


@pytest.mark.parametrize("nonfinite_kind", ["nan", "inf"])
def test_amp_skip_preserves_only_finite_head_statistics(nonfinite_kind: str) -> None:
    spec = _make_spec(3, 1, 1, 1)
    query = torch.tensor([
        [20.0, 0.0], [1.0, 0.0],
        [0.0, 2.0], [0.0, 2.0],
        [1.0, 0.0], [1.0, 0.0],
    ])
    key_scale = 0.0 if nonfinite_kind == "nan" else 2.0
    key_value = torch.tensor([
        [20.0, 0.0], [1.0, 0.0],
        [0.0, key_scale], [0.0, 1.0],
        [1.0, 0.0], [1.0, 0.0],
    ])
    module = _MLAAttention(query.clone(), key_value.clone())
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    install_qk_clip(_make_plan([module], [spec]), optimizer)
    completed_steps = []
    optimizer.register_step_post_hook(lambda *_: completed_steps.append(True))
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    attention = TorchAttentionWithMaxLogits()
    mask = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    mask[:, 2] = False

    def forward(inputs: Tensor) -> tuple[Tensor, Tensor]:
        inputs = inputs.view(1, 1, 2).expand(1, 2, 2)
        with torch.autocast("cpu", dtype=torch.float16):
            queries = nn.functional.linear(inputs, module.query_weight).view(1, 2, 3, 2)
            projected_kv = nn.functional.linear(inputs, module.key_value_weight).view(1, 2, 3, 2)
        keys = torch.stack(
            (projected_kv[..., 0], torch.ones_like(projected_kv[..., 0])), dim=-1
        )
        values = projected_kv[..., 1:]
        output, maxima = attention(queries, keys, values, mask, 1.0, 0.0, True)
        module.qk_clip_state.record(maxima)
        return output.float().square().sum(), maxima

    # One head stays finite, one overflows in FP16, and one is always masked.
    loss, maxima = forward(torch.tensor([1.0, 65504.0]))
    assert not torch.isfinite(loss)
    assert maxima[0] == 401.0
    if nonfinite_kind == "nan":
        assert torch.isnan(maxima[1])
    else:
        assert torch.isposinf(maxima[1])
    assert torch.isneginf(maxima[2])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not completed_steps
    assert scaler.get_scale() == 4.0
    torch.testing.assert_close(module.query_weight, query)
    torch.testing.assert_close(module.key_value_weight, key_value)
    retained_maxima = torch.tensor([401.0, -torch.inf, -torch.inf])
    torch.testing.assert_close(module.qk_clip_state.max_logits, retained_maxima)
    assert module.qk_clip_state.observed

    optimizer.zero_grad()
    loss, maxima = forward(torch.tensor([0.1, 1.0]))
    assert torch.isfinite(loss)
    assert torch.isfinite(maxima[:2]).all()
    assert (maxima[:2] < 100.0).all()
    assert torch.isneginf(maxima[2])
    accumulated_maxima = torch.maximum(retained_maxima, maxima)
    torch.testing.assert_close(module.qk_clip_state.max_logits, accumulated_maxima)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    query_after_update = module.query_weight.detach() - 0.1 * module.query_weight.grad
    kv_after_update = module.key_value_weight.detach() - 0.1 * module.key_value_weight.grad
    expected_query = query_after_update.clone()
    expected_kv = kv_after_update.clone()
    _reference_clip(expected_query, expected_kv, accumulated_maxima, spec, 0.5)

    scaler.step(optimizer)
    scaler.update()

    assert len(completed_steps) == 1
    torch.testing.assert_close(module.query_weight, expected_query)
    torch.testing.assert_close(module.key_value_weight, expected_kv)
    assert not torch.equal(module.query_weight[:2], query_after_update[:2])
    torch.testing.assert_close(module.query_weight[2:], query_after_update[2:])
    torch.testing.assert_close(module.key_value_weight[2:], kv_after_update[2:])
    assert torch.isneginf(module.qk_clip_state.max_logits).all()
    assert not module.qk_clip_state.observed


def test_qk_clip_runs_after_all_child_optimizers() -> None:
    spec = _make_spec(2, 2, 1, 2)
    module = _MLAAttention(torch.ones(6, 2), torch.ones(8, 2))
    groups = (
        ParamGroup(
            module,
            optimizer_name="query",
            param_names=["query_weight"],
            optimizer_kwargs={"lr": 0.2},
        ),
        ParamGroup(
            module,
            optimizer_name="key_value",
            param_names=["key_value_weight"],
            optimizer_kwargs={"lr": 0.4},
        ),
    )

    class SplitPlan(OptimizerPlan):
        def get_param_groups(self) -> tuple[ParamGroup, ...]:
            return groups

        def get_param_groups_by_optimizer_name(self) -> dict[str, tuple[ParamGroup, ...]]:
            return {group.optimizer_name: (group,) for group in groups}

    children = {
        group.optimizer_name: torch.optim.SGD([group.get_optimizer_param_group()])
        for group in groups
    }
    optimizer = OptimizersContainer(SplitPlan(), children)
    install_qk_clip(_make_plan([module], [spec]), optimizer)
    maxima = torch.tensor([400.0, 200.0])
    module.qk_clip_state.record(maxima)
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    updated_weights = []

    def record_update(child, args, kwargs):
        updated_weights.append(child.param_groups[0]["params"][0].detach().clone())

    for child in children.values():
        child.register_step_post_hook(record_update)
    optimizer.step()

    query = torch.full_like(module.query_weight, 0.8)
    key_value = torch.full_like(module.key_value_weight, 0.6)
    torch.testing.assert_close(updated_weights[0], query)
    torch.testing.assert_close(updated_weights[1], key_value)
    _reference_clip(query, key_value, maxima, spec, 0.5)
    torch.testing.assert_close(module.query_weight, query)
    torch.testing.assert_close(module.key_value_weight, key_value)
    assert not module.qk_clip_state.observed


def test_qk_clip_validates_all_layers_before_binding() -> None:
    spec = _make_spec(3, 2, 1, 2)
    modules = [
        _MLAAttention(torch.ones(9, 2), torch.ones(12, 2)),
        _MLAAttention(torch.ones(8, 2), torch.ones(12, 2)),
    ]
    optimizer = torch.optim.SGD(nn.ModuleList(modules).parameters(), lr=0.1)
    with pytest.raises(ValueError, match="weight shape"):
        install_qk_clip(_make_plan(modules, [spec, spec]), optimizer)
    assert all(get_qk_clip_state(module) is None for module in modules)
    assert all("qk_clip_state" not in vars(module) for module in modules)
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks
    for module in modules:
        for parameter in module.parameters():
            torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_qk_clip_rejects_unsupported_spec_before_binding() -> None:
    class SingleProjectionSpec(QKClipSpec):
        num_heads: int = 3

        @property
        def projection_rows(self) -> Mapping[str, int]:
            return {"query": 9}

    modules = [
        _MLAAttention(torch.ones(9, 2), torch.ones(12, 2)),
        _MLAAttention(torch.ones(9, 2), torch.ones(12, 2)),
    ]
    supported_plan = _make_plan([modules[0]], [_make_spec(3, 2, 1, 2)])
    unsupported_layer = QKClipLayer(
        module=modules[1],
        weights={"query": modules[1].query_weight},
        spec=SingleProjectionSpec(),
    )
    plan = cast(QKClipPlan, SimpleNamespace(
        config=supported_plan.config,
        layers=(*supported_plan.layers, unsupported_layer),
    ))
    optimizer = torch.optim.SGD(nn.ModuleList(modules).parameters(), lr=0.1)

    with pytest.raises(
        ValueError, match="FSDP2 QK clipping does not support SingleProjectionSpec"
    ):
        install_qk_clip(plan, optimizer)

    assert all(get_qk_clip_state(module) is None for module in modules)
    assert all("qk_clip_state" not in vars(module) for module in modules)
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks
    for module in modules:
        for parameter in module.parameters():
            torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_qk_clip_requires_optimizer_ownership_before_binding() -> None:
    spec = _make_spec(2, 2, 1, 2)
    module = _MLAAttention(torch.ones(6, 2), torch.ones(8, 2))
    optimizer = torch.optim.SGD([module.query_weight], lr=0.1)
    with pytest.raises(ValueError, match="own every Q/K weight"):
        install_qk_clip(_make_plan([module], [spec]), optimizer)
    assert get_qk_clip_state(module) is None
    assert "qk_clip_state" not in vars(module)
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks


def test_qk_clip_requires_materialized_weights_before_binding() -> None:
    spec = _make_spec(2, 2, 1, 2)
    module = _MLAAttention(
        torch.ones(6, 2, device="meta"), torch.ones(8, 2, device="meta")
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="materialized projection weights"):
        install_qk_clip(_make_plan([module], [spec]), optimizer)
    assert get_qk_clip_state(module) is None
    assert "qk_clip_state" not in vars(module)
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks


def test_qk_clip_binding_does_not_select_attention_backend(monkeypatch) -> None:
    spec = _make_spec(2, 2, 1, 2)
    module = _MLAAttention(torch.ones(6, 2), torch.ones(8, 2))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    def unexpected_backend_selection():
        pytest.fail("Optimizer preparation must not select an attention backend.")

    monkeypatch.setattr(interface, "ops", SimpleNamespace(
        attention=unexpected_backend_selection,
        attention_with_max_logits=unexpected_backend_selection,
    ))
    install_qk_clip(_make_plan([module], [spec]), optimizer)
    assert module.qk_clip_state is not None
    assert len(optimizer._optimizer_step_pre_hooks) == 1
    assert len(optimizer._optimizer_step_post_hooks) == 1


def test_qk_clip_rejects_duplicate_installation() -> None:
    spec = _make_spec(2, 2, 1, 2)
    module = _MLAAttention(torch.ones(6, 2), torch.ones(8, 2))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    plan = _make_plan([module], [spec])
    install_qk_clip(plan, optimizer)
    state = module.qk_clip_state
    with pytest.raises(ValueError, match="already installed"):
        install_qk_clip(plan, optimizer)
    assert module.qk_clip_state is state
    assert len(optimizer._optimizer_step_pre_hooks) == 1
    assert len(optimizer._optimizer_step_post_hooks) == 1


def test_qk_clip_validates_sp_before_binding() -> None:
    spec = _make_spec(2, 2, 1, 2)
    module = _MLAAttention(torch.ones(6, 2), torch.ones(8, 2))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="divisible"):
        install_qk_clip(_make_plan([module], [spec]), optimizer, sp_size=3)
    assert get_qk_clip_state(module) is None
    assert "qk_clip_state" not in vars(module)
    assert not optimizer._optimizer_step_pre_hooks
    assert not optimizer._optimizer_step_post_hooks


def test_qk_clip_rejects_missing_stats_before_optimizer_update() -> None:
    spec = _make_spec(3, 2, 1, 2)
    module = _MLAAttention(torch.ones(9, 2), torch.ones(12, 2))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
    install_qk_clip(_make_plan([module], [spec]), optimizer)
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="attention statistics"):
        optimizer.step()
    for parameter in module.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter))
    assert not optimizer.state

    with pytest.raises(ValueError, match="closures"):
        optimizer.step(closure=lambda: None)
    with pytest.raises(ValueError, match="closures"):
        optimizer.step(lambda: None)


@pytest.mark.parametrize("full_offload", [False, True])
@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_grad_norm_stops_before_optimizer_and_qk_clip(
    monkeypatch, full_offload, norm_type, invalid
) -> None:
    module = _MLAAttention(torch.ones(2, 2), torch.ones(2, 2))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
    install_qk_clip(_make_plan([module], [_make_spec(1, 1, 1, 1)]), optimizer)
    module.qk_clip_state.record(torch.tensor([400.0]))
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    module.query_weight.grad[0, 0] = invalid
    original_grad = module.query_weight.grad.clone()

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._device_runtime = CpuRuntime()
    backend._mesh = None
    backend.fsdp_config = SimpleNamespace(
        cpu_offload=SimpleNamespace(enabled=full_offload, mode=CPUOffloadMode.FULL)
    )
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, **kwargs: None)

    with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
        backend.clip_grad_norm_(module.parameters(), max_norm=float("inf"), norm_type=norm_type)
        optimizer.step()

    for parameter in module.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter))
    torch.testing.assert_close(module.query_weight.grad, original_grad, equal_nan=True)
    assert not optimizer.state
    assert module.qk_clip_state.observed
    torch.testing.assert_close(module.qk_clip_state.max_logits, torch.tensor([400.0]))


def _run_nonfinite_checks(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        dense_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        expert_mesh = init_device_mesh("cpu", (world_size, 1), mesh_dim_names=("ep", "edp"))
        invalid_values = [float("nan"), float("inf"), -float("inf")]
        for invalid in [1.0, *invalid_values]:
            loss = torch.tensor(invalid if rank == 1 else 1.0)
            distributed_loss = DTensor.from_local(loss, dense_mesh, (Replicate(),))
            for value in (loss, distributed_loss):
                if invalid == 1.0:
                    assert_finite(value, "loss")
                else:
                    with pytest.raises(RuntimeError, match="Non-finite loss"):
                        assert_finite(value, "loss")

        backend = FSDP2Backend.__new__(FSDP2Backend)
        backend._device_runtime = CpuRuntime()
        backend._mesh = expert_mesh
        backend.gradient_scaling_factor = 1.0 / world_size
        for full_offload in (False, True):
            backend.fsdp_config = SimpleNamespace(
                cpu_offload=SimpleNamespace(enabled=full_offload, mode=CPUOffloadMode.FULL)
            )
            for layout in ("plain", "dense", "moe", "empty"):
                if layout in ("dense", "moe"):
                    mesh = dense_mesh if layout == "dense" else expert_mesh["edp"]
                    parameter = nn.Parameter(distribute_tensor(torch.ones(2), mesh, (Shard(0),)))
                else:
                    parameter = nn.Parameter(torch.ones(2))
                optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
                for norm_type in (2.0, float("inf")):
                    for invalid in invalid_values:
                        parameter.grad = torch.ones_like(parameter)
                        local_grad = parameter.grad.to_local() if isinstance(parameter, DTensor) else parameter.grad
                        if rank == 1:
                            local_grad[0] = invalid
                        if layout == "empty" and rank == 0:
                            parameter.grad = None
                        with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
                            backend.clip_grad_norm_([parameter], max_norm=0.25, norm_type=norm_type)
                            optimizer.step()
                        assert not optimizer.state
                        local_param = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                        torch.testing.assert_close(local_param, torch.ones_like(local_param))
                        if rank == 0 and parameter.grad is not None:
                            expected_grad = backend.gradient_scaling_factor if layout == "moe" else 1.0
                            torch.testing.assert_close(local_grad, torch.full_like(local_grad, expected_grad))

            dense = nn.Parameter(distribute_tensor(torch.ones(2), dense_mesh, (Shard(0),)))
            expert = nn.Parameter(distribute_tensor(torch.ones(2), expert_mesh["edp"], (Shard(0),)))
            replicated = nn.Parameter(torch.ones(1))
            for norm_type in (2.0, float("inf")):
                expected_norm = (3.0**2 * 2 + 4.0**2 * 2 * world_size + 12.0**2) ** 0.5
                if norm_type == float("inf"):
                    expected_norm = 12.0
                for max_norm in (1.0, float("inf")):
                    dense.grad = torch.full_like(dense, 3.0)
                    expert.grad = torch.full_like(expert, 4.0 * world_size)
                    replicated.grad = torch.full_like(replicated, 12.0)
                    norm = backend.clip_grad_norm_([dense, expert, replicated], max_norm, norm_type)
                    torch.testing.assert_close(norm, torch.tensor(expected_norm))
                    scale = min(1.0, max_norm / (expected_norm + 1e-6))
                    for parameter, magnitude in ((dense, 3.0), (expert, 4.0), (replicated, 12.0)):
                        gradient = parameter.grad.to_local() if isinstance(parameter, DTensor) else parameter.grad
                        torch.testing.assert_close(gradient, torch.full_like(gradient, magnitude * scale))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
def test_nonfinite_loss_and_grad_norm_stop_all_ranks(tmp_path) -> None:
    mp.spawn(_run_nonfinite_checks, args=(2, str(tmp_path / "nonfinite")), nprocs=2, join=True)


def _run_distributed_qk_clip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        specs = [_make_spec(3, 2, 1, 2), _make_spec(2, 1, 1, 2), _make_spec(1, 1, 1, 1)]
        modules = []
        references = []
        maxima = []
        for index, spec in enumerate(specs):
            query_rows = spec.num_heads * (
                spec.qk_nope_head_dim + spec.qk_rope_head_dim
            )
            kv_rows = spec.num_heads * (spec.qk_nope_head_dim + spec.v_head_dim)
            query = (
                torch.arange(query_rows * 3, dtype=torch.float32).reshape(query_rows, 3)
                / 10
            )
            key_value = (
                torch.arange(kv_rows * 3, dtype=torch.float32).reshape(kv_rows, 3) / 10
            )
            references.append((query.clone(), key_value.clone()))
            modules.append(
                _MLAAttention(
                    distribute_tensor(query, mesh, (Shard(0),)),
                    distribute_tensor(key_value, mesh, (Shard(0),)),
                )
            )
            maxima.append(torch.arange(1, spec.num_heads + 1).float() * 200 + index)
        optimizer = torch.optim.SGD(nn.ModuleList(modules).parameters(), lr=0.0)
        controller = install_qk_clip(_make_plan(modules, specs), optimizer)
        storage = controller._max_logits.untyped_storage().data_ptr()
        assert all(
            module.qk_clip_state.max_logits.untyped_storage().data_ptr() == storage
            for module in modules
        )
        versions = [
            parameter._version
            for module in modules
            for parameter in module.parameters()
        ]
        for module, head_maxima in zip(modules, maxima, strict=True):
            # Each rank sees different heads, as it does after Ulysses attention.
            local_maxima = torch.full_like(head_maxima, -torch.inf)
            local_maxima[rank::world_size] = head_maxima[rank::world_size]
            module.qk_clip_state.record(local_maxima)

        with (
            CommDebugMode() as communication,
            patch.object(
                DTensor,
                "full_tensor",
                side_effect=AssertionError("QK clipping must not gather weights"),
            ),
            patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce_maxima,
        ):
            optimizer.step()
        assert reduce_maxima.call_count == 1
        assert reduce_maxima.call_args.args[0] is controller._max_logits
        counts = {
            str(op): count
            for op, count in communication.get_comm_counts().items()
            if count
        }
        assert sum(counts.values()) == 1, counts
        assert "allreduce" in next(iter(counts)).replace("_", "").lower(), counts

        for module, (query, key_value), head_maxima, spec in zip(
            modules, references, maxima, specs, strict=True
        ):
            _reference_clip(query, key_value, head_maxima, spec, 0.5)
            for parameter, expected in (
                (module.query_weight, query),
                (module.key_value_weight, key_value),
            ):
                chunks = expected.chunk(world_size)
                expected_local = chunks[rank] if rank < len(chunks) else expected[:0]
                torch.testing.assert_close(
                    parameter.to_local(), expected_local
                )
            assert torch.isneginf(module.qk_clip_state.max_logits).all()
            assert not module.qk_clip_state.observed
        updated_versions = [
            parameter._version
            for module in modules
            for parameter in module.parameters()
        ]
        assert all(
            after > before
            for before, after in zip(versions, updated_versions, strict=True)
        )

        # A fresh step with no threshold exceedance must preserve every shard.
        before = [
            parameter.to_local().clone()
            for module in modules
            for parameter in module.parameters()
        ]
        controller.clip()
        after = [
            parameter.to_local()
            for module in modules
            for parameter in module.parameters()
        ]
        for actual, expected in zip(after, before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        unsupported = _MLAAttention(
            distribute_tensor(torch.ones(9, 2), mesh, (Replicate(),)),
            distribute_tensor(torch.ones(12, 2), mesh, (Shard(0),)),
        )
        invalid_optimizer = torch.optim.SGD(unsupported.parameters(), lr=0.1)
        with pytest.raises(ValueError, match=r"Shard\(0\)"):
            install_qk_clip(_make_plan([unsupported], [specs[0]]), invalid_optimizer)
        assert get_qk_clip_state(unsupported) is None
        assert "qk_clip_state" not in vars(unsupported)
        assert not invalid_optimizer._optimizer_step_pre_hooks
        assert not invalid_optimizer._optimizer_step_post_hooks

        singleton_mesh = init_device_mesh(
            "cpu", (world_size, 1), mesh_dim_names=("outer", "inner")
        )["inner"]
        subgroup_module = _MLAAttention(
            distribute_tensor(torch.ones(9, 2), singleton_mesh, (Shard(0),)),
            distribute_tensor(torch.ones(12, 2), singleton_mesh, (Shard(0),)),
        )
        subgroup_optimizer = torch.optim.SGD(subgroup_module.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="dense world mesh"):
            install_qk_clip(
                _make_plan([subgroup_module], [specs[0]]), subgroup_optimizer
            )
        assert get_qk_clip_state(subgroup_module) is None
        assert "qk_clip_state" not in vars(subgroup_module)

        # Metadata must match physical shard rows, including empty shards.
        malformed_query = DTensor.from_local(
            torch.ones(1, 2),
            mesh,
            (Shard(0),),
            run_check=False,
            shape=torch.Size((9, 2)),
            stride=(2, 1),
        )
        malformed = _MLAAttention(
            malformed_query,
            distribute_tensor(torch.ones(12, 2), mesh, (Shard(0),)),
        )
        invalid_optimizer = torch.optim.SGD(malformed.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="local weight shape"):
            install_qk_clip(_make_plan([malformed], [specs[0]]), invalid_optimizer)
        assert get_qk_clip_state(malformed) is None
        assert "qk_clip_state" not in vars(malformed)
        assert not invalid_optimizer._optimizer_step_pre_hooks

        # Ulysses ranks fill distinct head ranges of the same packed reduction.
        sp_spec = _make_spec(world_size * 2, 1, 1, 1)
        num_rows = sp_spec.num_heads * 2
        sp_module = _MLAAttention(
            distribute_tensor(torch.ones(num_rows, 2), mesh, (Shard(0),)),
            distribute_tensor(torch.ones(num_rows, 2), mesh, (Shard(0),)),
        )
        sp_optimizer = torch.optim.SGD(sp_module.parameters(), lr=0.0)
        install_qk_clip(
            _make_plan([sp_module], [sp_spec]),
            sp_optimizer,
            sp_size=world_size,
        )
        sp_maxima = torch.arange(1, sp_spec.num_heads + 1).float() * 100
        sp_module.qk_clip_state.record(sp_maxima[2 * rank : 2 * rank + 2], 2 * rank)
        sp_optimizer.step()
        query, key_value = torch.ones(num_rows, 2), torch.ones(num_rows, 2)
        _reference_clip(query, key_value, sp_maxima, sp_spec, 0.5)
        torch.testing.assert_close(
            sp_module.query_weight.to_local(), query.chunk(world_size)[rank]
        )
        torch.testing.assert_close(
            sp_module.key_value_weight.to_local(), key_value.chunk(world_size)[rank]
        )

        local_module = _MLAAttention(torch.ones(num_rows, 2), torch.ones(num_rows, 2))
        local_optimizer = torch.optim.SGD(local_module.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="dense world mesh"):
            install_qk_clip(
                _make_plan([local_module], [sp_spec]),
                local_optimizer,
                sp_size=world_size,
            )
        assert get_qk_clip_state(local_module) is None
        assert "qk_clip_state" not in vars(local_module)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize(
    "world_size", [2, 4], ids=["split-head-shards", "empty-shards"]
)
def test_distributed_qk_clip_matches_mla_reference(tmp_path, world_size: int) -> None:
    mp.spawn(
        _run_distributed_qk_clip,
        args=(world_size, str(tmp_path / "qk-clip")),
        nprocs=world_size,
        join=True,
    )


def _run_replicated_qk_clip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        specs = (_make_spec(2, 1, 1, 1), _make_spec(1, 1, 1, 2))
        model = nn.Module()
        model.layers = nn.ModuleList()
        for spec in specs:
            attention = nn.Module()
            query_rows = spec.num_heads * (spec.qk_nope_head_dim + spec.qk_rope_head_dim)
            kv_rows = spec.num_heads * (spec.qk_nope_head_dim + spec.v_head_dim)
            attention.query = nn.Linear(2, query_rows, bias=False)
            attention.key_value = nn.Linear(2, kv_rows, bias=False)
            model.layers.append(attention)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.0)

        config = MuonOptimizerConfig(
            lr=0.0, weight_decay=0.0, adamw_fused=False, qk_clip=QKClipConfig()
        )
        backend = FSDP2Backend(
            EngineConfig(optimizer=OptimizerConfig(name="muon", muon=config)),
            CpuRuntime(),
        )
        parallel_plan = FSDPParallelPlan({"layers": FSDPPlanSpec(no_shard=True)})
        backend._init_mesh(parallel_plan)
        model_fqn_map = build_model_fqn_map(model)
        resolved_plan = backend._compile_parallel_plan(parallel_plan, model_fqn_map)
        ignored_params, _ = backend._collect_no_shard(model, resolved_plan)
        assert ignored_params == set(model.parameters())
        assert all(not isinstance(parameter, DTensor) for parameter in ignored_params)

        recipe = MuonRecipe(
            {"**": ExcludeFromMuon()},
            qk_clip_specs={f"layers.{index}": spec for index, spec in enumerate(specs)},
        )
        plan = MuonPlan(model, recipe=recipe, config=config, model_fqn_map=model_fqn_map)
        optimizer = backend.prepare_optimizer(plan)
        assert type(optimizer) is torch.optim.AdamW
        post_step_hook, = optimizer._optimizer_step_post_hooks.values()
        controller = post_step_hook.__self__
        assert controller._reduction_group is dist.group.WORLD
        assert controller._reduction_size == world_size

        maxima_by_rank = torch.tensor([
            [100.0 + 300.0 * worker, 400.0 - 200.0 * worker, 200.0 + 100.0 * worker]
            for worker in range(world_size)
        ])
        num_heads = tuple(spec.num_heads for spec in specs)
        for attention, maxima in zip(
            model.layers, maxima_by_rank[rank].split(num_heads), strict=True
        ):
            state = get_qk_clip_state(attention)
            assert state is not None
            assert (
                state.max_logits.untyped_storage().data_ptr()
                == controller._max_logits.untyped_storage().data_ptr()
            )
            state.record(maxima)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)

        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce_maxima:
            optimizer.step()
        assert reduce_maxima.call_count == int(world_size > 1)
        if world_size > 1:
            assert reduce_maxima.call_args.args[0] is controller._max_logits
            assert reduce_maxima.call_args.kwargs["op"] == dist.ReduceOp.MAX
            assert reduce_maxima.call_args.kwargs["group"] is dist.group.WORLD

        global_maxima = maxima_by_rank.amax(dim=0)
        for attention, spec, maxima in zip(
            model.layers, specs, global_maxima.split(num_heads), strict=True
        ):
            query = torch.ones_like(attention.query.weight)
            key_value = torch.ones_like(attention.key_value.weight)
            _reference_clip(query, key_value, maxima, spec, 0.5)
            torch.testing.assert_close(attention.query.weight, query)
            torch.testing.assert_close(attention.key_value.weight, key_value)
            kv_rows = spec.qk_nope_head_dim + spec.v_head_dim
            value_rows = attention.key_value.weight.view(spec.num_heads, kv_rows, -1)[
                :, spec.qk_nope_head_dim:
            ]
            torch.testing.assert_close(value_rows, torch.ones_like(value_rows))
            state = get_qk_clip_state(attention)
            assert torch.isneginf(state.max_logits).all()
            assert not state.observed

        if world_size > 1:
            parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
            replicas = [torch.empty_like(parameters) for _ in range(world_size)]
            dist.all_gather(replicas, parameters)
            for replica in replicas:
                torch.testing.assert_close(replica, parameters)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize(
    "world_size", [1, 2], ids=["single-rank-no-collective", "world-max"]
)
def test_replicated_qk_clip_uses_world_maximum(tmp_path, world_size: int) -> None:
    mp.spawn(
        _run_replicated_qk_clip,
        args=(world_size, str(tmp_path / "qk-clip-replicated")),
        nprocs=world_size,
        join=True,
    )
