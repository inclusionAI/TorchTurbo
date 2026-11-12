"""FSDP activation recomputation preserves forward and backward lifecycle."""

import copy
import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend, FSDPModule, _FSDPUnit
from torchturbo.engine.backends.fsdp2.prefetch import FSDPPrefetchManager
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.utils.torch_utils import torch_version


@pytest.fixture
def cpu_mesh(tmp_path, monkeypatch):
    if not dist.is_gloo_available():
        pytest.skip("CPU FSDP tests require Gloo")
    if torch_version() < (2, 10, 0):
        pytest.skip("CPU FSDP requires torch >= 2.10")
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1,
        )
        yield init_device_mesh("cpu", (1,))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class LastChildForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Linear(4, 4)

    def forward(self, x):
        # Non-reentrant early-stop can interrupt the child's post-forward hook.
        return self.child(x)


@pytest.fixture
def fsdp_parent(cpu_mesh):
    module = LastChildForward()
    module.unsharded = nn.Identity()
    fully_shard(module.child, mesh=cpu_mesh)
    fully_shard(module, mesh=cpu_mesh)
    return module


@pytest.fixture
def cpu_ops(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def mixed_precision_backend(use_reentrant=False):
    config = EngineConfig.from_dict({
        "mixed_precision": "bf16",
        "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
    })
    return SimpleNamespace(
        config=config,
        mixed_precision=config.mixed_precision,
        fsdp_config=config.backend_config,
        reshard_after_forward=True,
        prefetch_manager=FSDPPrefetchManager(config.backend_config),
    )


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("checkpoint_target", ["", "experts"], ids=["whole", "experts"])
def test_fsdp_checkpoint_preserves_expert_probability_dtype(
    cpu_mesh, cpu_ops, use_reentrant, use_lora, on_input, checkpoint_target,
):
    torch.manual_seed(731)
    config = MoEConfig(
        hidden_size=8, moe_ffn_hidden_size=4, num_moe_experts=2,
        params_dtype=torch.bfloat16, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, moe_router_dtype="fp32",
        moe_router_topk=1 if on_input else 2,
        moe_router_score_function="sigmoid", moe_apply_probs_on_input=on_input,
        moe_aux_loss_coeff=0.2, moe_z_loss_coeff=0.3,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    candidate = MoELayer(config, dist.group.WORLD, submodules=submodules)
    reference = MoELayer(config, dist.group.WORLD, submodules=submodules)
    for model in (candidate, reference):
        if use_lora:
            model.requires_grad_(False)
            model.experts = GroupedMLPLoRA(model.experts, "default", r=2, lora_alpha=2)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.15)
    candidate.load_state_dict(reference.state_dict())
    backend = mixed_precision_backend(use_reentrant)
    FSDP2Backend._apply_activation_checkpointing(
        backend, candidate,
        SimpleNamespace(modules={checkpoint_target: FSDPPlanSpec(activation_checkpoint=True)}),
    )
    fsdp_kwargs = FSDP2Backend._build_fsdp_kwargs(backend, set())
    FSDP2Backend._apply_fsdp(
        backend, candidate,
        [_FSDPUnit([candidate.experts], cpu_mesh), _FSDPUnit([candidate], cpu_mesh)],
        fsdp_kwargs,
    )
    # The reference has no checkpoint and explicitly preserves FP32 probabilities.
    fully_shard(reference.experts, mesh=cpu_mesh, mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
    ))
    fully_shard(reference, mesh=cpu_mesh, **fsdp_kwargs)
    expert_dtypes = []
    candidate.experts.register_forward_pre_hook(
        lambda module, args: expert_dtypes.append((args[0].dtype, args[2].dtype))
    )
    optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        inputs, ref_inputs, outputs, expected = [], [], [], []
        for length in (3, 5):
            x = torch.randn(1, length, 8, requires_grad=not use_lora)
            ref_x = x.detach().clone().requires_grad_(not use_lora)
            runtime_inputs = MoERuntimeInputs(
                padding_mask=torch.arange(length).remainder(2).bool().unsqueeze(0),
            )
            candidate.set_runtime_inputs(runtime_inputs)
            reference.set_runtime_inputs(runtime_inputs)
            inputs.append(x)
            ref_inputs.append(ref_x)
            outputs.append(candidate(x)[0])
            expected.append(reference(ref_x)[0])
        sum(y.float().square().sum() for y in outputs).backward()
        if use_reentrant and checkpoint_target == "":
            # Reentrant backward reduces one graph at a time; match BF16 rounding.
            for y in reversed(expected):
                y.float().square().sum().backward()
        else:
            sum(y.float().square().sum() for y in expected).backward()
        for output, ref_output in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
        for x, ref_x in zip(inputs, ref_inputs, strict=True):
            torch.testing.assert_close(x.grad, ref_x.grad)
        for (name, parameter), (_, ref_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True,
        ):
            # FP32 adapter gradients still come from BF16 expert computation.
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, rtol=1e-2, atol=1e-5, msg=name,
            )
        optimizer.step()
        ref_optimizer.step()
        for parameter, ref_parameter in zip(
            candidate.parameters(), reference.parameters(), strict=True,
        ):
            torch.testing.assert_close(parameter, ref_parameter)
        assert candidate._runtime_inputs is runtime_inputs
    assert expert_dtypes
    assert set(expert_dtypes) == {(torch.bfloat16, torch.float32)}


def test_fsdp_mixed_precision_keeps_regular_input_casting(cpu_mesh):
    model = nn.Linear(8, 4)
    backend = mixed_precision_backend()
    FSDP2Backend._apply_fsdp(
        backend, model, [_FSDPUnit([model], cpu_mesh)],
        FSDP2Backend._build_fsdp_kwargs(backend, set()),
    )
    output = model(torch.ones(2, 8, dtype=torch.float32))
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpointed_fsdp_clears_forward_root_across_accumulated_graphs(cpu_mesh, use_reentrant):
    torch.manual_seed(419)
    candidate = LastChildForward()
    reference = copy.deepcopy(candidate)
    config = EngineConfig.from_dict({
        "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
    })
    backend = SimpleNamespace(config=config, prefetch_manager=FSDPPrefetchManager(config.backend_config))
    FSDP2Backend._apply_activation_checkpointing(
        backend, candidate, SimpleNamespace(modules={"": FSDPPlanSpec(activation_checkpoint=True)}),
    )
    fully_shard(candidate.child, mesh=cpu_mesh)
    fully_shard(candidate, mesh=cpu_mesh)
    assert isinstance(candidate, FSDPModule)
    assert isinstance(candidate.child, FSDPModule)
    state = candidate._get_fsdp_state()

    for _ in range(2):
        candidate.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        inputs = [torch.randn(num_tokens, 4, requires_grad=True) for num_tokens in (3, 5)]
        reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        outputs = [candidate(x) for x in inputs]
        expected = [reference(x) for x in reference_inputs]
        for output, expected_output in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output, expected_output)
        sum(output.square().sum() for output in outputs).backward()
        sum(output.square().sum() for output in expected).backward()

        assert state._state_ctx.iter_forward_root is None
        assert candidate.child._get_fsdp_state()._state_ctx is state._state_ctx
        for (name, parameter), (reference_name, reference_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True,
        ):
            assert name == reference_name
            assert isinstance(parameter.grad, DTensor)
            torch.testing.assert_close(parameter.grad.to_local(), reference_parameter.grad, msg=name)
        for x, reference_x in zip(inputs, reference_inputs, strict=True):
            torch.testing.assert_close(x.grad, reference_x.grad)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_recompute_context_preserves_outer_hooks_on_nested_exception_exits(
    fsdp_parent, monkeypatch, use_reentrant,
):
    from torchturbo.engine.backends.fsdp2 import activation_checkpointing

    monkeypatch.setattr(activation_checkpointing, "torch_version", lambda: (2, 10, 0))
    module = fsdp_parent
    root_hooks = dict(module._forward_pre_hooks)
    original_hooks = dict(module.child._forward_pre_hooks)
    with pytest.raises(ValueError, match="outer exit"):
        with activation_checkpointing.fsdp_recompute_context(module, use_reentrant):
            outer_hooks = dict(module.child._forward_pre_hooks)
            assert len(outer_hooks) == len(original_hooks) + 1
            assert module._forward_pre_hooks == root_hooks
            assert not module.unsharded._forward_pre_hooks
            added_hook_ids = outer_hooks.keys() - original_hooks.keys()
            added_hook_id = added_hook_ids.pop()
            assert list(outer_hooks)[0 if use_reentrant else -1] == added_hook_id
            with pytest.raises(RuntimeError, match="inner exit"):
                with activation_checkpointing.fsdp_recompute_context(module, use_reentrant):
                    assert len(module.child._forward_pre_hooks) == len(outer_hooks) + 1
                    raise RuntimeError("inner exit")
            assert module.child._forward_pre_hooks == outer_hooks
            raise ValueError("outer exit")
    assert module.child._forward_pre_hooks == original_hooks
    assert module._forward_pre_hooks == root_hooks


@pytest.mark.parametrize("version,use_reentrant,extra_hooks", [
    ((2, 10, 0), False, 1),
    ((2, 11, 0), False, 0),
    ((2, 10, 0), True, 1),
    ((2, 11, 0), True, 1),
])
def test_recompute_context_limits_non_reentrant_hooks_to_older_torch(
    fsdp_parent, monkeypatch, version, use_reentrant, extra_hooks,
):
    from torchturbo.engine.backends.fsdp2 import activation_checkpointing

    monkeypatch.setattr(activation_checkpointing, "torch_version", lambda: version)
    child = fsdp_parent.child
    original_hooks = dict(child._forward_pre_hooks)
    with activation_checkpointing.fsdp_recompute_context(fsdp_parent, use_reentrant):
        assert len(child._forward_pre_hooks) == len(original_hooks) + extra_hooks
    assert child._forward_pre_hooks == original_hooks
