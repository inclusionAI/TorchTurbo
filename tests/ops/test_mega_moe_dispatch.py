"""CPU checks for HMC construction, dispatch, and live expert weights."""

import gc
import sys
import weakref
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import torch

from torchturbo.ops import OpDispatcher, OpPolicy, OpRegistry
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.ops.providers.npu import mega_moe as hmc_provider
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime


@pytest.fixture(autouse=True)
def expert_parallel_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)


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
        SimpleNamespace(MegaMoeExperts=Executor, MegaMoeNumericsOptions=SimpleNamespace, SwiGLUParameters=SimpleNamespace),
    )
    monkeypatch.setattr(
        OpDispatcher, "_load_provider_impl_classes",
        lambda self, device: {"hmc_mega_moe": hmc_provider.HmcMegaMoe},
    )
    return constructed


@pytest.fixture
def dispatcher(monkeypatch):
    def create(*, runtime=None, policy=OpPolicy.AUTO, overrides=None):
        registry = OpRegistry()
        registry.register(hmc_provider.HmcMegaMoe.op_spec())
        op = OpDispatcher(
            device_runtime=runtime or NpuRuntime(), registry=registry,
            default_policy=policy, overrides=overrides,
        )
        monkeypatch.setattr("torchturbo.models.modules.moe.experts.ops", op)
        return op

    return create


def inputs(group=None, submodules=None):
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
    layer.submodules = submodules or MoESubmodules()
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
            return dense, dense.bool(), ids, probs

    layer.router = DenseRouter()
    layer._runtime_inputs = MoERuntimeInputs()
    cfg.mega_moe.enable = True
    return layer, x, probs


def test_dispatch_keeps_live_weights_and_reuses_executor(hmc, dispatcher):
    op = dispatcher()
    layer, x, probs = inputs()
    assert not hmc
    result, logits = layer(x)
    torch.testing.assert_close(logits, probs)
    weights, mask, ids, _ = layer.router(x, None, None, False)
    dispatched = op.mega_moe(
        layer.experts, x, weights, mask, expert_indices=ids, ep_group=layer.ep_group,
    )
    assert isinstance(dispatched, torch.Tensor)
    torch.testing.assert_close(dispatched, result)
    result.float().sum().backward()
    assert all(v.grad is not None for v in (x, probs, layer.experts.weight1, layer.experts.weight2))
    assert len(hmc) == 1
    assert hmc[0].configuration["ep_group"] is layer.ep_group
    assert vars(hmc[0].configuration["numerics_options"]) == {
        "apply_probs_before_fc2": True,
    }
    assert vars(hmc[0].configuration["activation"]) == {
        "clamp_limit": 0.5, "clamp_after_silu": True,
    }
    with torch.no_grad():
        layer.experts.weight1.fill_(2)
    layer.experts.eval()
    second, _ = layer(x)
    assert len(hmc) == 1 and not hmc[0].training
    torch.testing.assert_close(second, torch.full_like(second, 20))
    assert hmc[0].weights[-1][0].shape == (2, 2, 2)
    sibling, y, _ = inputs(layer.ep_group, layer.submodules)
    sibling_output, _ = sibling(y)
    torch.testing.assert_close(sibling_output, result.detach())
    assert len(hmc) == 1 and hmc[0].training
    other, z, _ = inputs(layer.ep_group)
    with torch.no_grad():
        other.experts.weight1.fill_(3)
    other_output, _ = other(z)
    torch.testing.assert_close(other_output, torch.full_like(other_output, 28))
    sibling_output.float().sum().backward()
    assert other.experts.weight1.grad is None
    other_output.float().sum().backward()
    assert other.experts.weight1.grad is not None
    assert len(hmc) == 1
    different_group, z, _ = inputs()
    different_group(z)
    assert len(hmc) == 2


def test_hmc_adds_each_shared_contribution_once(hmc, dispatcher):
    layer, x, probs = inputs()
    dispatcher()
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


def test_hmc_provider_releases_unused_executors(hmc, dispatcher):
    op = dispatcher()
    layer, x, _ = inputs()
    layer(x)
    provider = op._impls["hmc_mega_moe"]
    assert len(provider._expert_executors) == len(provider._executors) == 1
    owner = weakref.ref(layer)
    del layer
    gc.collect()
    assert owner() is None
    assert not provider._expert_executors and not provider._executors


def test_model_adds_shared_experts_to_routed_output(monkeypatch):
    layer, x, probs = inputs()

    class FusedBackend:
        def mega_moe(self, experts, hidden_states, weights, mask, *, ep_group, expert_indices=None):
            return hidden_states + 7

    monkeypatch.setattr("torchturbo.models.modules.moe.experts.ops", FusedBackend())
    layer.shared_experts = torch.nn.Identity()
    output, logits = layer(x)
    torch.testing.assert_close(output, 2 * x + 7)
    torch.testing.assert_close(logits, probs)


@pytest.mark.parametrize("runtime", [CudaRuntime, CpuRuntime])
def test_mega_moe_requires_a_supported_device(dispatcher, runtime):
    dispatcher(runtime=runtime())
    layer, x, _ = inputs()
    with pytest.raises(RuntimeError, match="No implementation"):
        layer(x)


@pytest.mark.parametrize("package", [None, SimpleNamespace()])
def test_hmc_requires_a_compatible_package(hmc, dispatcher, monkeypatch, package):
    monkeypatch.setitem(sys.modules, "hyper_multicore", package)
    dispatcher()
    layer, x, _ = inputs()
    with pytest.raises(RuntimeError, match="No implementation"):
        layer(x)
    assert not hmc


@pytest.mark.parametrize("invalid", ["activation", "weights"])
def test_invalid_inputs_fail_before_workspace_creation(hmc, dispatcher, invalid):
    layer, x, _ = inputs()
    if invalid == "activation":
        x.data = x.data.float()
    else:
        layer.experts.double()
    with pytest.raises(ValueError):
        dispatcher()
        layer(x)
    assert not hmc


def test_hmc_rejects_te_before_accessing_grouped_weights(hmc, dispatcher):
    layer, x, _ = inputs()
    layer.experts.use_te = True
    with pytest.raises(ValueError, match="Transformer Engine"):
        dispatcher()
        layer(x)
    assert not hmc


def test_hot_replica_configures_and_prepares_its_executor(hmc, dispatcher, monkeypatch):
    planner = Mock()
    monkeypatch.setattr(sys.modules["hyper_multicore"], "MegaMoePlacementOptions", SimpleNamespace, raising=False)
    monkeypatch.setitem(sys.modules, "hyper_multicore.modules.hot_replica.planner",
                        SimpleNamespace(HotReplicaPlanner=planner))
    dispatcher()
    layer, x, _ = inputs()
    layer.config.mega_moe.expert_balance = "hot_replica"
    layer.config.mega_moe.source_token_bound = 32
    assert not hmc
    layer(x)
    planner.assert_called_once_with(source_token_bound=32)
    assert len(hmc) == 1
    options = hmc[0].configuration["expert_placement"]
    assert options.strategy == "adaptive" and options.planner is planner.return_value

    layer(x)
    assert len(hmc) == 1
    plain, y, _ = inputs(layer.ep_group, layer.submodules)
    plain.config.mega_moe.source_token_bound = 32
    assert len(hmc) == 1
    plain(y)
    assert len(hmc) == 2 and "expert_placement" not in hmc[1].configuration
    planner.assert_called_once()


def test_hmc_uses_mask_only_when_router_omits_ids(hmc, dispatcher, monkeypatch):
    extract_ids = Mock(wraps=hmc_provider.get_expert_indices)
    monkeypatch.setattr(hmc_provider, "get_expert_indices", extract_ids)
    layer, x, _ = inputs()
    weights, mask, ids, _ = layer.router(x, None, None, False)
    op = dispatcher()
    expected = op.mega_moe(
        layer.experts, x, weights, mask, expert_indices=ids,
        ep_group=layer.ep_group,
    )
    extract_ids.assert_not_called()
    actual = op.mega_moe(
        layer.experts, x, weights, mask, ep_group=layer.ep_group,
    )
    extract_ids.assert_called_once_with(mask)
    torch.testing.assert_close(actual, expected)


def test_hmc_rejects_invalid_balance_before_creating_executor(hmc, dispatcher):
    layer, x, _ = inputs()
    layer.config.mega_moe.expert_balance = "invalid"
    dispatcher()
    with pytest.raises(ValueError, match="HMC MegaMoE expert_balance"):
        layer(x)
    assert not hmc


@pytest.mark.parametrize(("policy", "overrides", "supported"), [
    (OpPolicy.NATIVE, None, False),
    (OpPolicy.CUSTOM, None, True),
    (OpPolicy.NATIVE, {"mega_moe": "hmc_mega_moe"}, True),
])
def test_hmc_dispatch_respects_operator_selection(hmc, dispatcher, policy, overrides, supported):
    dispatcher(policy=policy, overrides=overrides)
    layer, x, _ = inputs()
    assert not hmc
    if supported:
        output, _ = layer(x)
        assert isinstance(output, torch.Tensor)
        assert len(hmc) == 1
    else:
        with pytest.raises(RuntimeError, match="No implementation"):
            layer(x)
        assert not hmc
