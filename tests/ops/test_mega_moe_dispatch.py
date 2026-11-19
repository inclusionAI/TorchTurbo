"""CPU checks for HMC provider selection and live expert parameter ownership."""

import sys
from types import SimpleNamespace

import pytest
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.ops import OpDispatcher, OpPolicy, OpRegistry
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs
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


def inputs(group=None):
    cfg = MoEConfig(hidden_size=2, moe_ffn_hidden_size=1, num_moe_experts=4,
                    moe_router_topk=2, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                    moe_swiglu_limit=[0.5], moe_swiglu_clamp_gate_after_activation=True,
                    params_dtype=torch.float32)
    x = torch.ones(2, 2, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 3]])
    probs = torch.full((2, 2), 0.5, requires_grad=True)
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.shared_expert = layer.shared_experts = None
    layer.config = cfg
    layer.ep_group = object() if group is None else group
    layer.expert_parallel_size = 2
    layer.experts = GroupedMLP(2, False, cfg, layer_number=0)
    with torch.no_grad():
        layer.experts.weight1.fill_(1)
        layer.experts.weight2.fill_(1)

    class DenseRouter(torch.nn.Module):
        def forward(self, hidden, input_ids, padding_mask, accumulate):
            assert hidden is x
            dense = probs.new_zeros(2, 4).scatter(1, ids, probs)
            return dense, None, ids, probs

    layer.router = DenseRouter()
    layer._runtime_inputs = MoERuntimeInputs()
    FSDP2Backend._bind_megakernel(layer, hmc_provider.HmcRoutedExperts())
    return layer, x, probs


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM])
def test_dispatch_keeps_live_weights_and_reuses_executor(hmc, policy):
    op = dispatcher(policy)
    layer, x, probs = inputs()
    backend = op.mega_moe()
    assert not hmc
    result, logits = backend.forward_experts(layer, x, None)
    assert logits is probs
    result.float().sum().backward()
    assert all(v.grad is not None for v in (x, probs, layer.experts.weight1, layer.experts.weight2))
    assert len(hmc) == 1
    assert hmc[0].configuration["ep_group"] is layer.ep_group
    assert vars(hmc[0].configuration["numerics_options"]) == {
        "swiglu_limit": 0.5, "clamp_after_silu": True, "apply_probs_before_fc2": True,
    }
    with torch.no_grad():
        layer.experts.weight1.fill_(2)
    layer.experts.eval()
    second, _ = backend.forward_experts(layer, x, None)
    assert len(hmc) == 1 and not hmc[0].training
    torch.testing.assert_close(second, torch.full_like(second, 20))
    assert hmc[0].weights[-1][0].shape == (2, 2, 2)
    sibling, y, _ = inputs(layer.ep_group)
    backend.forward_experts(sibling, y, None)
    assert len(hmc) == 1
    op.mega_moe().forward_experts(sibling, y, None)
    assert len(hmc) == 2


def test_hmc_adds_each_shared_contribution_once(hmc):
    layer, x, probs = inputs()
    FSDP2Backend._bind_megakernel(layer, dispatcher().mega_moe())
    routed, _ = layer(x)
    layer.shared_expert = torch.nn.Identity()
    layer.shared_expert_gate = torch.nn.Linear(2, 1, bias=False, dtype=x.dtype)
    layer.shared_experts = torch.nn.Linear(2, 2, bias=False, dtype=x.dtype)
    with torch.no_grad():
        layer.shared_expert_gate.weight.zero_()
        layer.shared_experts.weight.fill_(1)
    output, logits = layer(x)
    torch.testing.assert_close(output, routed + 0.5 * x + 2 * x)
    torch.testing.assert_close(logits, probs)
    output.sum().backward()
    assert layer.shared_expert_gate.weight.grad is not None
    assert layer.shared_experts.weight.grad is not None


def test_fused_backend_returns_complete_output_without_layer_shared_call():
    layer, x, probs = inputs()

    class FusedBackend:
        def forward_experts(self, layer, hidden_states, input_ids):
            return hidden_states + 7, probs

    FSDP2Backend._bind_megakernel(layer, FusedBackend())
    layer.shared_experts = torch.nn.Identity()
    output, logits = layer(x)
    torch.testing.assert_close(output, x + 7)
    torch.testing.assert_close(logits, probs)


def test_native_policy_rejects_hmc(hmc):
    with pytest.raises(RuntimeError, match="policy=native"):
        dispatcher(OpPolicy.NATIVE).mega_moe()
    assert not hmc


@pytest.mark.parametrize("device", [TurboDevice.CUDA, TurboDevice.CPU])
@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM, OpPolicy.NATIVE])
@pytest.mark.parametrize("selection", ["default", "explicit", "override"])
def test_unsupported_device_rejects_hmc_provider(hmc, device, policy, selection):
    overrides = {"mega_moe": "hmc_mega_moe"} if selection == "override" else None
    impl = "hmc_mega_moe" if selection == "explicit" else None
    with pytest.raises(RuntimeError, match=f"device={device.value}"):
        dispatcher(policy, overrides=overrides, device=device).mega_moe(
            impl=impl,
        )
    assert not hmc


@pytest.mark.parametrize("policy", [OpPolicy.AUTO, OpPolicy.CUSTOM, OpPolicy.NATIVE])
def test_cuda_has_no_mega_moe_implementation(policy):
    from torchturbo.ops.registry import registry

    op = OpDispatcher(device_runtime=SimpleNamespace(turbo_device=TurboDevice.CUDA),
                      registry=registry, default_policy=policy)
    with pytest.raises(RuntimeError, match="No implementation for op='mega_moe', device=cuda"):
        op.mega_moe()


def test_override_and_explicit_selection(hmc):
    layer, x, _ = inputs()
    backend = dispatcher(OpPolicy.NATIVE, overrides={"mega_moe": "hmc_mega_moe"}).mega_moe()
    backend.forward_experts(layer, x, None)
    dispatcher(OpPolicy.NATIVE).mega_moe(impl="hmc_mega_moe").forward_experts(layer, x, None)
    assert len(hmc) == 2
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "missing"})


def test_unavailable_hmc_fails_without_initializing(hmc, monkeypatch):
    monkeypatch.setattr(hmc_provider, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="No implementation"):
        dispatcher().mega_moe()
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(overrides={"mega_moe": "hmc_mega_moe"})
    assert not hmc


@pytest.mark.parametrize("invalid", ["activation", "weights"])
def test_invalid_inputs_fail_before_workspace_creation(hmc, invalid):
    layer, x, _ = inputs()
    if invalid == "activation":
        x.data = x.data.float()
    else:
        layer.experts.double()
    with pytest.raises(ValueError):
        dispatcher().mega_moe().forward_experts(layer, x, None)
    assert not hmc


def test_hmc_rejects_te_before_accessing_grouped_weights(hmc):
    layer, x, _ = inputs()
    layer.experts.use_te = True
    with pytest.raises(ValueError, match="Transformer Engine"):
        dispatcher().mega_moe().forward_experts(layer, x, None)
    assert not hmc
