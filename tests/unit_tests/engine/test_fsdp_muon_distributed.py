from __future__ import annotations

import copy
import os
import sys
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig
from torchturbo.engine.backends.fsdp2.checkpoint.dcp import DcpCheckpoint
from torchturbo.engine.backends.fsdp2.optimizers.memory import OptimizerMemoryController
from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon
from torchturbo.engine.config import MuonOptimizerConfig
from torchturbo.engine.optimizers import (
    AttentionPerHeadComputeView,
    ExpertUpComputeView,
    MuonParameterSpec,
    MuonPlan,
    MuonRecipe,
    Owned,
)
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.utils.torch_utils import torch_version

_HAS_GLOO = dist.is_available() and dist.is_gloo_available()
_HAS_FSDP_MUON = torch_version() >= (2, 10, 0)


class _DistributedMuonModel(nn.Module):
    def __init__(self, values: dict[str, Tensor]) -> None:
        super().__init__()
        for name, value in values.items():
            self.register_parameter(name, nn.Parameter(value))


class _StubStreamCpuRuntime(CpuRuntime):
    """CPU runtime exposing ``torch.cpu``'s device-agnostic stream stubs.

    ``CpuRuntime`` reports no asynchronous streams because CPU execution cannot
    overlap communication with compute. This subclass reports support anyway so
    the optimizer takes its asynchronous pipeline, which lets these tests check
    that the pipeline computes the same values as the synchronous path. Overlap
    itself is a performance property and is not asserted here.

    ``torch.cpu.Stream`` and ``torch.cpu.Event`` exist for device-agnostic code
    and do nothing, so every enqueued operation still runs inline on the calling
    thread in program order.
    """

    @property
    def supports_async_streams(self) -> bool:
        return True

    def stream(self, stream=None):
        selected = self.current_stream() if stream is None else stream
        return torch.cpu.stream(selected)

    def current_stream(self):
        return torch.cpu.current_stream()

    def new_stream(self, priority: int = 0):
        return torch.cpu.Stream(priority)

    def new_event(self):
        return torch.cpu.Event()

    def record_stream(self, tensor: Tensor, stream) -> None:
        # CPU has no stream-aware caching allocator, so no buffer lifetime needs
        # guarding and ``aten::record_stream`` registers no CPU kernel.
        return None

    def synchronize_stream(self, stream) -> None:
        return None


def _make_optimizer(
    model: _DistributedMuonModel,
    matrix_layouts: dict[str, tuple[int, int] | None],
    *,
    bucket_strategy: str,
    execution: str,
    offload_mode: str | None = None,
) -> FSDPDistributedMuon:
    specs = {}
    for name, layout in matrix_layouts.items():
        if layout is None:
            specs[name] = MuonParameterSpec(compute_placement=Owned())
        else:
            num_batches, column_groups = layout
            view = (
                AttentionPerHeadComputeView(num_heads=num_batches)
                if column_groups == 1
                else ExpertUpComputeView(num_experts=num_batches, fused_gate_up=True)
            )
            specs[name] = MuonParameterSpec(
                compute_placement=Shard(0), compute_view=view
            )
    config_kwargs = {}
    if bucket_strategy != "default":
        config_kwargs["bucket"] = {"strategy": bucket_strategy}
    plan = MuonPlan(
        model,
        recipe=MuonRecipe(specs),
        config=MuonOptimizerConfig(
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            adjust_lr_fn=None,
            overlap_comm_compute=execution == "async-overlap",
            **config_kwargs,
        ),
    )
    runtime = CpuRuntime() if execution == "sync" else _StubStreamCpuRuntime()
    return FSDPDistributedMuon(
        plan,
        device_runtime=runtime,
        cpu_offload=CPUOffloadConfig(
            enabled=offload_mode is not None,
            mode=offload_mode or "full",
            optimizer_memory_fraction=0,
        ),
    )


def _matrix_views(
    tensor: Tensor,
    num_batches: int,
    column_groups: int,
) -> tuple[Tensor, ...]:
    return tuple(
        matrix
        for row_block in tensor.chunk(num_batches, dim=0)
        for matrix in row_block.chunk(column_groups, dim=1)
    )


def _join_matrices(matrices: tuple[Tensor, ...], column_groups: int) -> Tensor:
    return torch.cat(
        [
            torch.cat(matrices[index : index + column_groups], dim=1)
            for index in range(0, len(matrices), column_groups)
        ],
        dim=0,
    )


def _run_distributed_muon_test(
    rank: int,
    world_size: int,
    scenario: str,
    bucket_strategy: str,
    execution: str,
    init_file: str,
    offload_mode: str | None = None,
    resumed_mode: str | None = None,
) -> None:
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
        mesh = init_device_mesh(
            "cpu",
            (world_size,),
            mesh_dim_names=("dp",),
        )
        generator = torch.Generator().manual_seed(42)
        if scenario == "balanced":
            shapes = {"weight_w": (6, 4), "weight_v": (6, 4)}
            matrix_layouts = {"weight_w": (2, 1), "weight_v": (2, 1)}
        elif scenario == "fused":
            shapes = {"fused": (6, 6)}
            matrix_layouts = {"fused": (2, 2)}
        else:
            num_local_heads = 2 * world_size
            if scenario == "single-matrix":
                split_shape, num_split_heads = (9, 4), 1
            elif world_size == 2:
                split_shape, num_split_heads = (9, 4), 3
            else:
                split_shape, num_split_heads = (10, 4), 2
            shapes = {
                "local_first": (2 * num_local_heads, 3),
                "owned": (5, 3),
                "local_middle": (2 * num_local_heads, 4),
                "split_heads": split_shape,
                "fused": (2, 6),
            }
            matrix_layouts = {
                "local_first": (num_local_heads, 1),
                "owned": None,
                "local_middle": (num_local_heads, 1),
                "split_heads": (num_split_heads, 1),
                "fused": (1, 2),
            }
        values = {
            name: torch.randn(shape, generator=generator)
            for name, shape in shapes.items()
        }
        reference_layouts = {
            name: (1, 1) if layout is None else layout
            for name, layout in matrix_layouts.items()
        }

        def distributed_parameter(value: Tensor) -> DTensor:
            return distribute_tensor(value.clone(), mesh, (Shard(0),))

        def make_model_and_optimizer(
            full_values: dict[str, Tensor],
        ) -> tuple[_DistributedMuonModel, FSDPDistributedMuon]:
            model = _DistributedMuonModel(
                {
                    name: distributed_parameter(value)
                    for name, value in full_values.items()
                }
            )
            optimizer = _make_optimizer(
                model,
                matrix_layouts,
                bucket_strategy=bucket_strategy,
                execution=execution,
                offload_mode=offload_mode,
            )
            return model, optimizer

        model, optimizer = make_model_and_optimizer(values)
        checkpoint_io = DcpCheckpoint(ep_rank=0, ep_size=1, edp_mesh=mesh)
        use_dcp = offload_mode is not None or resumed_mode is not None
        checkpoint_dir = init_file + "-checkpoint"
        if use_dcp:
            # A fresh optimizer already holds zero momentum, so saving it needs
            # no dummy step that could apply weight decay to the model.
            with patch.object(
                optimizer, "step", side_effect=AssertionError("dummy step")
            ):
                checkpoint_io.save_optimizer(
                    optimizer, model, checkpoint_dir
                )
            for state in optimizer.state.values():
                momentum = state["momentum_buffer"]
                local = (
                    momentum.to_local() if isinstance(momentum, DTensor) else momentum
                )
                assert torch.count_nonzero(local) == 0

        # A workspace released for inference is acquired again by the next
        # step. An allocation failure on one rank must stop every rank.
        optimizer.release_workspace()
        configure = optimizer._workspace.configure

        def fail_on_one_rank(buckets):
            configure(buckets)
            if rank == 0:
                raise RuntimeError("injected workspace allocation failure")

        with (
            patch.object(
                optimizer._workspace, "configure", side_effect=fail_on_one_rank
            ),
            pytest.raises(RuntimeError, match="before any update"),
        ):
            optimizer._acquire_workspace()
        assert not optimizer._workspace.configured
        assert not optimizer._workspace.slots

        workspace = optimizer._workspace
        assert workspace.async_enabled == (execution != "sync")
        assert workspace.overlap_comm_compute == (execution == "async-overlap")
        buckets = optimizer._compiled_buckets
        assert len(buckets) == (1 if bucket_strategy == "single" else len(shapes))
        plans = {
            item.fqn: plan
            for bucket in buckets
            for item, plan in zip(
                bucket.redistributed_items, bucket.redistributed_plans, strict=True
            )
        }
        for name, plan in plans.items():
            num_batches, column_groups = reference_layouts[name]
            indices = [
                index
                for partition in plan.compute_partitions
                for index in partition.matrix_indices
            ]
            assert sorted(indices) == list(range(num_batches * column_groups))
        if scenario in {"balanced", "fused"}:
            num_matrices_by_rank = [0] * world_size
            owners = {}
            for name, plan in plans.items():
                matrix_owners = [0] * len(plan.matrices)
                for partition in plan.compute_partitions:
                    num_matrices_by_rank[partition.group_rank] += len(
                        partition.matrix_indices
                    )
                    for index in partition.matrix_indices:
                        matrix_owners[index] = partition.group_rank
                owners[name] = tuple(matrix_owners)
            if scenario == "balanced":
                if bucket_strategy == "default":
                    assert num_matrices_by_rank == [2, 0, 2, 0]
                    assert owners == {"weight_w": (0, 2), "weight_v": (0, 2)}
                    expected_remote_bytes = 64
                else:
                    assert num_matrices_by_rank == [1, 1, 1, 1]
                    assert set(owners.values()) == {(0, 2), (1, 3)}
                    expected_remote_bytes = 112
                forward_remote_bytes = sum(
                    route.numel * 4
                    for plan in plans.values()
                    for route in plan.storage_to_compute_routes
                    if route.source_group_rank != route.destination_group_rank
                )
                assert forward_remote_bytes == expected_remote_bytes
            else:
                assert num_matrices_by_rank == [1, 1, 1, 1]
                assert owners["fused"][0] != owners["fused"][1]
                assert owners["fused"][2] != owners["fused"][3]
            # Empty storage does not prevent ownership, but separate small
            # buckets save traffic by reusing ranks that already hold rows.
            if rank == 3:
                assert all(
                    parameter.to_local().numel() == 0
                    for parameter in model.parameters()
                )
                computes_matrix = any(
                    plan.compute_partition(rank).numel > 0 for plan in plans.values()
                )
                assert computes_matrix == (
                    scenario == "fused" or bucket_strategy == "single"
                )
        expected_exchanges = 2 * sum(
            bucket.forward is not None and bucket.forward.has_remote_transfers
            for bucket in buckets
        )
        assert expected_exchanges > 0

        reference_parameters = {
            name: tuple(
                nn.Parameter(matrix.clone())
                for matrix in _matrix_views(value, *reference_layouts[name])
            )
            for name, value in values.items()
        }
        reference = torch.optim.Muon(
            [
                parameter
                for matrices in reference_parameters.values()
                for parameter in matrices
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            adjust_lr_fn=None,
        )

        expected_path = "_run_synchronous" if execution == "sync" else "_run_async"

        def step_and_check() -> None:
            for name, parameter in model.named_parameters():
                gradient = torch.randn(shapes[name], generator=generator)
                parameter.grad = distributed_parameter(gradient)
                for matrix, matrix_gradient in zip(
                    reference_parameters[name],
                    _matrix_views(gradient, *reference_layouts[name]),
                    strict=True,
                ):
                    matrix.grad = matrix_gradient.clone()

            # Wrapping the expected execution method keeps its real behaviour
            # while proving the step did not silently take the other path.
            with (
                patch.object(
                    dist, "all_to_all_single", wraps=dist.all_to_all_single
                ) as exchange,
                patch.object(
                    FSDPDistributedMuon,
                    expected_path,
                    autospec=True,
                    side_effect=(
                        FSDPDistributedMuon._run_synchronous
                        if execution == "sync"
                        else FSDPDistributedMuon._run_async
                    ),
                ) as execution_path,
            ):
                optimizer.step()
            assert execution_path.call_count == 1
            assert exchange.call_count == expected_exchanges
            # Only offloaded device execution releases buffers after a step.
            assert optimizer._workspace.configured != optimizer._step_scoped_workspace
            reference.step()

            for name, parameter in model.named_parameters():
                column_groups = reference_layouts[name][1]
                reference_value = _join_matrices(
                    tuple(matrix.detach() for matrix in reference_parameters[name]),
                    column_groups,
                )
                reference_momentum = _join_matrices(
                    tuple(
                        reference.state[matrix]["momentum_buffer"]
                        for matrix in reference_parameters[name]
                    ),
                    column_groups,
                )
                # Explicit slicing preserves the empty final DTensor shard.
                shard_rows = (shapes[name][0] + world_size - 1) // world_size
                shard = slice(rank * shard_rows, (rank + 1) * shard_rows)
                torch.testing.assert_close(
                    parameter.to_local(),
                    reference_value[shard],
                    rtol=0,
                    atol=0,
                )
                state = optimizer.state[parameter]
                assert set(state) == {"momentum_buffer"}
                momentum = state["momentum_buffer"]
                if optimizer._offload.enabled:
                    assert not isinstance(momentum, DTensor)
                    assert momentum.device.type == "cpu"
                    local_momentum = momentum
                else:
                    assert isinstance(momentum, DTensor)
                    assert momentum.placements == (Shard(0),)
                    local_momentum = momentum.to_local()
                assert local_momentum.shape == parameter.to_local().shape
                torch.testing.assert_close(
                    local_momentum,
                    reference_momentum[shard],
                    rtol=0,
                    atol=0,
                )

        step_and_check()
        step_and_check()

        optimizer.zero_grad(set_to_none=True)
        storage = [param.to_local().data_ptr() for param in model.parameters()]
        controller = OptimizerMemoryController(optimizer, CpuRuntime())
        for _ in range(2):
            controller.offload()
            controller.offload()
            assert not optimizer._workspace.slots
            assert not optimizer._workspace.local_buffers.compute_scratch
            with pytest.raises(RuntimeError, match="optimizer is suspended"):
                optimizer.step()
            controller.restore()
            controller.restore()
        assert storage == [param.to_local().data_ptr() for param in model.parameters()]
        step_and_check()

        checkpoint = copy.deepcopy(optimizer.state_dict())
        if use_dcp:
            checkpoint_io.save_optimizer(
                optimizer, model, checkpoint_dir
            )
        resumed_values = {
            name: parameter.full_tensor().detach()
            for name, parameter in model.named_parameters()
        }
        offload_mode = resumed_mode
        model, optimizer = make_model_and_optimizer(resumed_values)
        buckets_before_load = optimizer._compiled_buckets
        optimizer.load_state_dict(checkpoint)
        if use_dcp:
            # Load into a second fresh optimizer, whose state already exists, so
            # DCP fills the target mode's destinations without a dummy step.
            model, optimizer = make_model_and_optimizer(resumed_values)
            initial_buffers = {
                parameter: state["momentum_buffer"]
                for parameter, state in optimizer.state.items()
            }
            assert len(initial_buffers) == len(shapes)
            with patch.object(
                optimizer, "step", side_effect=AssertionError("dummy step")
            ):
                checkpoint_io.load_optimizer(
                    optimizer, model, checkpoint_dir
                )
            if optimizer._offload.enabled:
                # DCP fills the live CPU buffers, and loading keeps them.
                assert all(
                    optimizer.state[parameter]["momentum_buffer"] is buffer
                    for parameter, buffer in initial_buffers.items()
                )
        assert optimizer._compiled_buckets is not buckets_before_load
        for resumed_bucket, bucket in zip(
            optimizer._compiled_buckets, buckets, strict=True
        ):
            assert resumed_bucket.forward == bucket.forward
            assert resumed_bucket.reverse == bucket.reverse
            assert resumed_bucket.redistributed_plans == bucket.redistributed_plans

        step_and_check()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _HAS_GLOO,
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.skipif(
    not _HAS_FSDP_MUON,
    reason="requires PyTorch 2.10 or newer",
)
@pytest.mark.parametrize(
    ("world_size", "scenario", "bucket_strategy", "execution"),
    [
        (2, "mixed", "single", "sync"),
        (3, "mixed", "default", "sync"),
        (3, "mixed", "single", "async-overlap"),
        (3, "mixed", "default", "async"),
        (3, "single-matrix", "default", "async-overlap"),
        (4, "balanced", "single", "sync"),
        (4, "balanced", "default", "sync"),
        (4, "fused", "default", "async-overlap"),
    ],
    ids=[
        "mixed-uneven-shards",
        "mixed-default-buckets",
        "mixed-bucket-async-overlap",
        "default-buckets-async",
        "matrix-spanning-every-shard",
        "balanced-matrices-single-bucket",
        "balanced-matrices-default-buckets",
        "fused-column-owners-async-overlap",
    ],
)
def test_distributed_muon_matches_plain_muon_and_reloads_state(
    tmp_path, world_size, scenario, bucket_strategy, execution
):
    mp.spawn(
        _run_distributed_muon_test,
        args=(world_size, scenario, bucket_strategy, execution, str(tmp_path / "muon")),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    not _HAS_GLOO or not _HAS_FSDP_MUON, reason="requires Gloo and PyTorch 2.10"
)
@pytest.mark.parametrize(
    "mode,resumed_mode",
    [
        ("full", "optimizer"),
        ("optimizer", "full"),
        (None, "optimizer"),
        ("full", None),
    ],
)
def test_distributed_muon_cpu_offload_and_cross_mode_resume(
    tmp_path, mode, resumed_mode
):
    mp.spawn(
        _run_distributed_muon_test,
        args=(3, "mixed", "single", "sync", str(tmp_path / "muon"), mode, resumed_mode),
        nprocs=3,
        join=True,
    )
