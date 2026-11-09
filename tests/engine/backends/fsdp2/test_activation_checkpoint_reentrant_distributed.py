"""Reentrant checkpointing preserves optimizer state through nested FSDP."""

import os
import sys
import time
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import ActivationCheckpointingConfig, EngineConfig
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
import torchturbo.runtime.runtime as runtime_module


class _ScaledMoE(torch.nn.Module):
    """Keep a trainable FSDP ancestor outside the checkpointed MoE."""

    def __init__(self, moe: MoELayer) -> None:
        super().__init__()
        self.moe = moe
        self.output_scale = torch.nn.Parameter(torch.ones(moe.config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe(hidden_states)[0] * self.output_scale


def _check_optimizer_gradients(rank, dispatcher, freeze_inputs, runtime):
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp", "ep"))
    root_mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp",))
    ep_group = mesh["ep"].get_group()
    ep_rank = dist.get_rank(ep_group)
    torch.manual_seed(172 + ep_rank)
    config = MoEConfig(
        hidden_size=8,
        moe_ffn_hidden_size=4,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_router_topk=2,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        moe_router_dtype="fp32",
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    reference = _ScaledMoE(MoELayer(config, ep_group, submodules=submodules))
    candidate = _ScaledMoE(
        MoELayer(
            replace(
                config,
                moe_token_dispatcher_type=dispatcher,
                moe_virtual_expert_load_balance=dispatcher == "flex",
            ),
            ep_group,
            submodules=submodules,
        )
    )
    with torch.no_grad():
        for parameter in reference.moe.parameters():
            parameter.normal_(std=0.15)
        # The two input signs route to different expert owners, so each
        # recomputation contributes gradients that the optimizer must retain.
        reference.moe.router.weight.zero_()
        reference.moe.router.weight[0].fill_(0.2)
        reference.moe.router.weight[1].fill_(0.1)
    candidate.load_state_dict(reference.state_dict())
    if freeze_inputs:
        reference.moe.router.requires_grad_(False)
        candidate.moe.router.requires_grad_(False)

    backend = FSDP2Backend(
        EngineConfig(
            activation_checkpointing=ActivationCheckpointingConfig(
                enabled=True,
                use_reentrant=True,
            ),
        ),
        runtime,
    )
    plan = backend._compile_parallel_plan(
        FSDPParallelPlan(plan={"moe": FSDPPlanSpec(activation_checkpoint=True)}),
        build_model_fqn_map(candidate),
    )
    backend._apply_activation_checkpointing(candidate, plan)
    fully_shard(candidate.moe.experts, mesh=mesh["edp"])
    fully_shard(candidate.moe, mesh=root_mesh)
    fully_shard(candidate, mesh=root_mesh)
    backend.prefetch_manager.setup(candidate)

    # Capture the same objects that a training optimizer owns before forward.
    optimizer_parameters = dict(candidate.named_parameters())
    optimizer = torch.optim.SGD(optimizer_parameters.values(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    reference_parameters = dict(reference.named_parameters())
    torch.manual_seed(417 + rank)
    tolerance = {"atol": 2e-6, "rtol": 2e-5}
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        inputs = [
            ((torch.rand(1, length, 8) + 1) * sign).requires_grad_(not freeze_inputs)
            for length, sign in ((3 + ep_rank, 1), (5 + ep_rank, -1))
        ]
        reference_inputs = [
            x.detach().clone().requires_grad_(not freeze_inputs) for x in inputs
        ]
        actual = [candidate(x) for x in inputs]
        expected = [reference(x) for x in reference_inputs]
        context = (
            f"rank={rank}, dispatcher={dispatcher}, frozen_inputs={freeze_inputs}, "
            f"iteration={iteration}"
        )
        for result, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, wanted, **tolerance, msg=context)

        # A single outer backward includes two reentrant backward graph tasks.
        sum(result.square().sum() for result in actual).backward()
        sum(result.square().sum() for result in expected).backward()
        for x, reference_x in zip(inputs, reference_inputs, strict=True):
            torch.testing.assert_close(x.grad, reference_x.grad, **tolerance, msg=context)
            assert x.requires_grad == (not freeze_inputs)

        for name, parameter in optimizer_parameters.items():
            assert candidate.get_parameter(name) is parameter, (
                f"{context}, optimizer parameter replaced: {name}"
            )
            reference_parameter = reference_parameters[name]
            if not parameter.requires_grad:
                assert parameter.grad is None and reference_parameter.grad is None
                continue
            assert parameter.grad is not None, (
                f"{context}, missing optimizer gradient: {name}"
            )
            assert reference_parameter.grad is not None, (context, name)
            reduction_group = (
                mesh["edp"].get_group() if name.startswith("moe.experts.")
                else dist.group.WORLD
            )
            dist.all_reduce(reference_parameter.grad, group=reduction_group)
            reference_parameter.grad /= dist.get_world_size(reduction_group)
            torch.testing.assert_close(
                parameter.grad.full_tensor(),
                reference_parameter.grad,
                **tolerance,
                msg=f"{context}, optimizer gradient: {name}",
            )

        optimizer.step()
        reference_optimizer.step()
        for name, parameter in optimizer_parameters.items():
            torch.testing.assert_close(
                parameter.full_tensor(),
                reference_parameters[name],
                **tolerance,
                msg=f"{context}, optimizer update: {name}",
            )


def _run_optimizer_gradients(rank, init_file, dispatcher, freeze_inputs):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
        _check_optimizer_gradients(rank, dispatcher, freeze_inputs, runtime)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("dispatcher", ["alltoall", "flex"])
@pytest.mark.parametrize(
    "freeze_inputs", [False, True], ids=["trainable-inputs", "frozen-inputs"],
)
def test_reentrant_checkpoint_preserves_optimizer_gradients(
    dispatcher, freeze_inputs, tmp_path,
):
    context = mp.spawn(
        _run_optimizer_gradients,
        args=(str(tmp_path / "checkpoint-init"), dispatcher, freeze_inputs),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(
                    "Reentrant checkpoint workers did not finish within 120 seconds"
                )
    finally:
        # A collective mismatch may prevent a worker from finishing teardown.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
