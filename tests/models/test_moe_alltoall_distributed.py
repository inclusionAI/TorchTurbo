"""MoE outputs and gradients across real variable-size EP collectives."""

import os
import sys
from datetime import timedelta
from itertools import product
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.prefetch import FSDPPrefetchManager
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
import torchturbo.runtime.runtime as runtime_module


def _check_moe(
    rank: int, ep_size: int, empty_source: str,
    boundary: str, use_reentrant: bool, on_input: bool,
    *, freeze_inputs: bool = False, use_lora: bool = False,
    fsdp_mesh: DeviceMesh | None = None,
) -> None:
    ep_group = dist.group.WORLD if fsdp_mesh is None else fsdp_mesh["ep"].get_group()
    torch.manual_seed(930 + dist.get_rank(ep_group))
    config = MoEConfig(
        hidden_size=8, num_moe_experts=ep_size * 2,
        expert_model_parallel_size=ep_size, moe_ffn_hidden_size=4,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        moe_router_topk=1 if on_input else 2,
        moe_router_score_function="sigmoid" if on_input else "softmax",
        moe_router_dtype="fp32", moe_apply_probs_on_input=on_input,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    reference = MoELayer(config, ep_group, submodules=submodules)
    candidate = MoELayer(config, ep_group, submodules=submodules)
    if use_lora:
        for layer in (reference, candidate):
            layer.experts.requires_grad_(False)
            layer.experts = GroupedMLPLoRA(layer.experts, "default", r=2, lora_alpha=2)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.15)
        reference.router.weight.normal_(std=0.15, generator=torch.Generator().manual_seed(450))
    candidate.load_state_dict(reference.state_dict(), strict=True)
    if freeze_inputs:
        reference.router.requires_grad_(False)
        candidate.router.requires_grad_(False)

    backend = SimpleNamespace(
        config=EngineConfig.from_dict({
            "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
        }),
    )
    backend.prefetch_manager = FSDPPrefetchManager(backend.config.backend_config)
    key = "" if boundary == "whole" else "experts"
    target = candidate if boundary == "whole" else candidate.experts
    original_forward = target.forward
    forward_modes = []

    def observed_forward(*args):
        forward_modes.append(torch.is_grad_enabled())
        return original_forward(*args)

    target.forward = observed_forward
    plan = SimpleNamespace(modules={key: FSDPPlanSpec(activation_checkpoint=True)})
    FSDP2Backend._apply_activation_checkpointing(backend, candidate, plan)
    if fsdp_mesh is not None:
        fully_shard(candidate.experts, mesh=fsdp_mesh["edp"])
        fully_shard(candidate, mesh=fsdp_mesh)
        backend.prefetch_manager.setup(candidate)

    is_empty = empty_source == "all" or (empty_source == "one" and rank == 0)
    lengths = [0, 0] if is_empty else [3 + rank, 5 + rank]
    inputs = [torch.randn(1, length, 8, requires_grad=not freeze_inputs) for length in lengths]
    ref_inputs = [x.detach().clone().requires_grad_(not freeze_inputs) for x in inputs]
    # Finish both forwards before backward to expose overwritten routing metadata.
    outputs = [candidate(x)[0] for x in inputs]
    expected = [reference(x)[0] for x in ref_inputs]
    context = (
        f"rank={rank}, EP={ep_size}, empty_source={empty_source}, "
        f"boundary={boundary}, reentrant={use_reentrant}, on_input={on_input}, "
        f"freeze_inputs={freeze_inputs}, lora={use_lora}, fsdp={fsdp_mesh is not None}"
    )

    def failure_message(message: str) -> str:
        return f"{context}\n{message}"

    for actual, ref in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, ref, msg=failure_message)
    sum(y.square().sum() for y in outputs).backward()
    sum(y.square().sum() for y in expected).backward()
    if use_reentrant:
        assert forward_modes == [False, False, True, True], context
    for actual, ref in zip(inputs, ref_inputs, strict=True):
        assert actual.requires_grad == (not freeze_inputs), context
        torch.testing.assert_close(actual.grad, ref.grad, msg=failure_message)
    ref_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        actual_grad = parameter.grad
        ref_grad = ref_parameters[name].grad
        if fsdp_mesh is not None and parameter.requires_grad:
            assert actual_grad is not None, f"{context}, parameter={name}"
            assert ref_grad is not None, f"{context}, parameter={name}"
            assert isinstance(actual_grad, DTensor), f"{context}, parameter={name}, grad={type(actual_grad)}"
            actual_grad = actual_grad.full_tensor()
            # Experts differ across EP ranks; router weights are shared by all ranks.
            group = fsdp_mesh["edp"].get_group() if name.startswith("experts.") else dist.group.WORLD
            dist.all_reduce(ref_grad, group=group)
            ref_grad /= dist.get_world_size(group)
        torch.testing.assert_close(
            actual_grad, ref_grad,
            msg=lambda message: f"{context}, parameter={name}\n{message}",
        )


def _run_moe(rank: int, ep_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=ep_size,
        timeout=timedelta(seconds=60),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
        for on_input, empty_source, boundary, use_reentrant in product(
            (False, True), ("none", "one", "all"),
            ("whole", "experts"), (False, True),
        ):
            _check_moe(rank, ep_size, empty_source, boundary, use_reentrant, on_input)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_fsdp_moe(rank: int, world_size: int, init_file: str, freeze_inputs: bool) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("edp", "ep"))
        training_modes = ((True, False), (True, True)) if freeze_inputs else ((False, False), (True, False))
        for on_input, empty_source, boundary, (use_reentrant, use_lora) in product(
            (False, True), ("none", "one", "all"), ("whole", "experts"), training_modes,
        ):
            _check_moe(
                rank, 2, empty_source, boundary, use_reentrant, on_input,
                freeze_inputs=freeze_inputs, use_lora=use_lora, fsdp_mesh=mesh,
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize("freeze_inputs", [False, True], ids=["full", "frozen"])
def test_fsdp_alltoall_checkpoint_outputs_and_gradients(tmp_path, freeze_inputs) -> None:
    mp.spawn(
        _run_fsdp_moe, args=(4, str(tmp_path / "moe-fsdp-ep"), freeze_inputs), nprocs=4, join=True,
    )


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize("ep_size", [2, 4])
def test_alltoall_checkpoint_outputs_and_gradients(ep_size, tmp_path) -> None:
    mp.spawn(
        _run_moe, args=(ep_size, str(tmp_path / "moe-ep")),
        nprocs=ep_size, join=True,
    )
