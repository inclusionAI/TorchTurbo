"""CPU checks for HMC provider selection and live expert parameter ownership."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpPolicy, OpRegistry
from torchturbo.ops.moe import MegaMoeContext
from torchturbo.ops.providers.npu import mega_moe as hmc_provider
from torchturbo.runtime import TurboDevice


@pytest.fixture
def hmc(monkeypatch):
    constructed = []

    class Executor(torch.nn.Module):
        def __init__(self, **configuration):
            super().__init__()
            self.configuration = configuration
            self.weights = []
            constructed.append(self)

        def forward(self, inputs, ids, probs, *, expert_weights):
            assert ids.dtype == torch.int32
            self.weights.append(expert_weights)
            scale = sum(weight.float().sum() for weight in expert_weights)
            return inputs * (scale * probs.sum(dim=-1, keepdim=True)).to(inputs.dtype)

    monkeypatch.setitem(
        sys.modules, "hyper_multicore",
        SimpleNamespace(MegaMoeExperts=Executor, MegaMoeNumericsOptions=SimpleNamespace),
    )
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: object())
    monkeypatch.setattr(
        OpDispatcher, "_load_provider_impl_classes",
        lambda self, device: {"hmc_mega_moe": hmc_provider.HmcMegaMoe},
    )
    return constructed


def dispatcher(policy=OpPolicy.AUTO, *, overrides=None, device=TurboDevice.NPU):
    registry = OpRegistry()
    registry.register(hmc_provider.HmcMegaMoe.op_spec())
    return OpDispatcher(
        device_runtime=SimpleNamespace(turbo_device=device),
        registry=registry, default_policy=policy, overrides=overrides,
    )


def inputs():
    context = MegaMoeContext(object(), 2, 1, 4, 2, 0.5, True)
    x = torch.ones(2, 2, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 3]])
    probs = torch.full((2, 2), 0.5, requires_grad=True)
    w1 = torch.ones(4, 2, requires_grad=True)
    w2 = torch.ones(2, 2, requires_grad=True)
    experts = torch.nn.Module()
    experts.num_local_experts = 2
    experts.use_te = False
    experts.weight1 = torch.nn.Parameter(w1)
    experts.weight2 = torch.nn.Parameter(w2)
    return (x, ids, probs, experts), context


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM])
def test_dispatch_keeps_live_weights_and_reuses_context(hmc, policy):
    op = dispatcher(policy)
    args, context = inputs()
    assert not hmc
    result = op.mega_moe(*args, context=context)
    result.float().sum().backward()
    assert all(v.grad is not None for v in (args[0], args[2], args[3].weight1, args[3].weight2))
    assert len(hmc) == 1
    assert hmc[0].configuration["ep_group"] is context.ep_group
    assert "expert_placement" not in hmc[0].configuration
    assert vars(hmc[0].configuration["numerics_options"]) == {
        "swiglu_limit": 0.5, "clamp_after_silu": True, "apply_probs_before_fc2": True,
    }
    with torch.no_grad():
        args[3].weight1.fill_(2)
    args[3].eval()
    second = op.mega_moe(*args, context=context)
    assert len(hmc) == 1 and not hmc[0].training
    torch.testing.assert_close(second, torch.full_like(second, 20))
    assert hmc[0].weights[-1][0].shape == (2, 2, 2)
    op.mega_moe(*args, context=replace(context))
    assert len(hmc) == 2


def test_native_policy_rejects_hmc(hmc):
    args, context = inputs()
    with pytest.raises(RuntimeError, match="policy=native"):
        dispatcher(OpPolicy.NATIVE).mega_moe(*args, context=context)
    assert not hmc


@pytest.mark.parametrize("device", [TurboDevice.CUDA, TurboDevice.CPU])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM, OpPolicy.NATIVE])
@pytest.mark.parametrize("selection", ["default", "explicit", "override"])
def test_unsupported_device_rejects_hmc_provider(hmc, device, policy, selection):
    args, context = inputs()
    overrides = {"mega_moe": "hmc_mega_moe"} if selection == "override" else None
    impl = "hmc_mega_moe" if selection == "explicit" else None
    with pytest.raises(RuntimeError, match=f"device={device.value}"):
        dispatcher(policy, overrides=overrides, device=device).mega_moe(
            *args, context=context, impl=impl,
        )
    assert not hmc


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM])
@pytest.mark.parametrize("selection", ["default", "explicit", "override"])
def test_cuda_dispatch_reports_unimplemented_on_call(policy, selection):
    from torchturbo.ops.registry import registry

    op = OpDispatcher(
        device_runtime=SimpleNamespace(turbo_device=TurboDevice.CUDA),
        registry=registry, default_policy=policy,
        overrides={"mega_moe": "cuda_mega_moe"} if selection == "override" else None,
    )
    args, context = inputs()
    with pytest.raises(NotImplementedError, match="not yet implemented on CUDA"):
        op.mega_moe(*args, context=context, impl="cuda_mega_moe" if selection == "explicit" else None)


def test_override_and_explicit_selection(hmc):
    args, context = inputs()
    dispatcher(OpPolicy.NATIVE, overrides={"mega_moe": "hmc_mega_moe"}).mega_moe(
        *args, context=context,
    )
    dispatcher(OpPolicy.NATIVE).mega_moe(*args, context=context, impl="hmc_mega_moe")
    assert len(hmc) == 2
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "missing"})


def test_unavailable_hmc_fails_without_initializing(hmc, monkeypatch):
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: None)
    args, context = inputs()
    with pytest.raises(RuntimeError, match="No implementation"):
        dispatcher().mega_moe(*args, context=context)
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "hmc_mega_moe"})
    assert not hmc


@pytest.mark.parametrize("invalid", ["activation", "weights"])
def test_invalid_inputs_fail_before_workspace_creation(hmc, invalid):
    args, context = inputs()
    args = list(args)
    if invalid == "activation":
        args[0] = args[0].float()
    else:
        args[3].double()
    with pytest.raises(ValueError):
        dispatcher().mega_moe(*args, context=context)
    assert not hmc


def test_hmc_rejects_te_before_accessing_grouped_weights(hmc):
    args, context = inputs()
    te = torch.nn.Module()
    te.use_te = True
    with pytest.raises(ValueError, match="Transformer Engine"):
        dispatcher().mega_moe(*args[:3], te, context=context)
    assert not hmc

def test_matching_serial_layers_share_a_context_only_within_one_model(monkeypatch):
    from torchturbo.models.modules.moe.mega_moe import MegaKernelExperts

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    group = object()
    kernels = MegaKernelExperts(group)
    other_model = MegaKernelExperts(group)
    cfg = SimpleNamespace(hidden_size=4, moe_ffn_hidden_size=8, num_moe_experts=4,
                          moe_router_topk=2, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                          moe_apply_probs_on_input=False, moe_swiglu_clamp_gate_after_activation=False)
    first, second = torch.nn.Module(), torch.nn.Module()
    for layer in (first, second):
        layer.config, layer.num_local_experts, layer.swiglu_limit = cfg, 2, None
        kernels.layers[layer] = None
    other_model.layers[first] = None
    context = kernels.context_for(first)
    assert kernels.context_for(first) is kernels.context_for(second) is context
    assert other_model.context_for(first) is not context
