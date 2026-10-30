"""CPU checks for dynamic balancing and expert parameter ownership."""

from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchturbo.engine.args import TorchTurboArgs
from torchturbo.engine.config import EngineConfig, build_from_args
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.optimizations.mega_moe import integration
from torchturbo.optimizations.mega_moe.moonep.backend import MoonEPMegaMoeBackend
from torchturbo.optimizations.mega_moe.moonep.routing import virtual_counts


def make_model() -> nn.Module:
    config = MoEConfig(
        hidden_size=4,
        moe_ffn_hidden_size=8,
        num_moe_experts=4,
        moe_router_topk=2,
        gated_linear_unit=True,
        activation_func=F.silu,
    )
    layers = []
    for _ in range(2):
        layer = MoELayer.__new__(MoELayer)
        nn.Module.__init__(layer)
        layer.config = config
        layer.experts = GroupedMLP(2, False, config)
        layers.append(layer)
    return nn.Sequential(*layers)


def test_mixed_clip_positions_share_backend(monkeypatch):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    model = make_model()
    for layer, after in zip(model, (False, True)):
        layer.config = replace(layer.config, moe_swiglu_clamp_gate_after_activation=after)
        layer.experts.config = layer.config
        layer.experts.swiglu_limit = 4.
    config = EngineConfig(mega_moe=True)
    config.extra["mega_moe_local_tokens"] = 128
    integration.install(model, config, None)
    backend = model[0]._routed_experts_backend
    assert backend is model[1]._routed_experts_backend
    assert set(backend.modes) == {("pre_activation", 4.), ("post_activation", 4.)}


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dropout", [None, 0., .25])
def test_lora_disables_mega_moe_before_model_replacement(monkeypatch, enabled, dropout):
    from torchturbo.engine.backends.fsdp2 import backend as fsdp_backend

    backend = fsdp_backend.FSDP2Backend.__new__(fsdp_backend.FSDP2Backend)
    backend.config = EngineConfig(mega_moe=enabled)
    peft_config = None if dropout is None else SimpleNamespace(r=4, lora_dropout=dropout)
    messages = []
    monkeypatch.setattr(fsdp_backend, "logger", SimpleNamespace(
        warning_rank0=lambda message: messages.append(message)
    ))

    def stop_before_mesh(plan):
        assert backend.config.mega_moe is (enabled and peft_config is None)
        raise RuntimeError("configuration checked before mesh and module replacement")

    monkeypatch.setattr(backend, "_init_mesh", stop_before_mesh)
    model = nn.Linear(4, 4)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="configuration checked before mesh"):
            backend.prepare_model(model, peft_config=peft_config)
    assert len(messages) == int(enabled and peft_config is not None)
    if messages:
        assert all(value in messages[0] for value in (
            "does not support LoRA", "Disabling mega_moe", "native MoE", "LoRA remains enabled"))


@pytest.mark.parametrize("enabled", [False, True])
def test_configuration_roundtrip(tmp_path, enabled):
    value = {"mega_moe": enabled}
    assert EngineConfig.from_dict(value).mega_moe is enabled
    path = tmp_path / "engine.yaml"
    path.write_text("engine_config:\n  mega_moe: " + str(enabled).lower() + "\n")
    config = build_from_args(TorchTurboArgs(config_path=str(path)))
    assert config.mega_moe is enabled
    assert "mega_moe" not in config.extra


@pytest.mark.parametrize("flag", ["--tuner_type", "--train_type"])
def test_lora_startup_does_not_require_native_dependencies(tmp_path, monkeypatch, flag):
    from pathlib import Path
    import runpy

    activate = runpy.run_path(str(Path(integration.__file__).with_name("bootstrap.py")))["activate"]
    config = tmp_path / "engine.yaml"
    config.write_text("engine_config:\n  mega_moe: true\n")

    def reject_native_activation(*args, **kwargs):
        pytest.fail("LoRA must not activate the MegaMoE runtime")

    monkeypatch.setattr(runpy, "run_path", reject_native_activation)
    assert activate(["--tt_config", str(config), flag, "lora"]) is False


def test_swift_lora_configuration_warns_and_skips_mega_moe_token_bounds(tmp_path, monkeypatch):
    from torchturbo.adapters.msswift import utils

    path = tmp_path / "engine.yaml"
    path.write_text("engine_config:\n  mega_moe: true\n")
    monkeypatch.setattr(utils, "get_torchturbo_args", lambda: TorchTurboArgs(config_path=str(path)))
    monkeypatch.setattr(utils, "_raise_if_unsupported", lambda args: None)
    monkeypatch.setattr(utils, "raise_if_evaluation_enabled", lambda *args, **kwargs: None)
    messages = []
    monkeypatch.setattr(utils, "logger", SimpleNamespace(warning_rank0=messages.append))
    args = SimpleNamespace(training_args=None, predict_with_generate=False, tuner_type="lora", bf16=True)
    config = utils.build_config_from_swift_args(args)
    assert config.mega_moe is False
    assert "mega_moe_local_tokens" not in config.extra
    assert len(messages) == 1 and "native MoE" in messages[0]


@pytest.mark.parametrize("enabled,use_reentrant", [(False, False), (True, False), (True, True)])
def test_checkpoint_modes_are_independent_of_backend(monkeypatch, enabled, use_reentrant):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    config = EngineConfig(mega_moe=True)
    config.activation_checkpointing.enabled = enabled
    config.activation_checkpointing.use_reentrant = use_reentrant
    config.extra["mega_moe_local_tokens"] = 128
    model = make_model()
    integration.install(model, config, None)
    assert model[0]._routed_experts_backend is model[1]._routed_experts_backend


def test_shared_backend_preserves_hooks_and_parameters(monkeypatch, tmp_path):
    from torchturbo.models.modules.moe import experts as expert_module
    from torchturbo.ops.providers.npu.mega_moe import MoonEPMegaMoe

    monkeypatch.setattr(expert_module, "ops", SimpleNamespace(mega_moe=MoonEPMegaMoe()))
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    config = EngineConfig(mega_moe=True)
    config.activation_checkpointing.enabled = True
    config.activation_checkpointing.use_reentrant = True
    config.extra["mega_moe_local_tokens"] = 128
    model = make_model()
    identities = {name: id(p) for name, p in model.named_parameters()}
    keys = set(model.state_dict())
    integration.install(model, config, None)
    assert model[0]._routed_experts_backend is model[1]._routed_experts_backend
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert keys == set(model.state_dict())
    calls = []
    model[0].experts.register_forward_pre_hook(lambda *_: calls.append("unshard"))
    model[0].experts.register_forward_hook(lambda *_: calls.append("reshard"))

    def execute(self, experts, hidden, indices, probs):
        calls.append("execute")
        assert experts.weight1 is model[0].experts.weight1
        assert indices.tolist() == [[1, 3]]
        assert probs.tolist() == [[0.0, 1.0]]
        return hidden * experts.weight1.mean()

    monkeypatch.setattr(MoonEPMegaMoeBackend, "execute", execute)
    out, _ = model[0]._routed_experts_backend.route(
        model[0].experts,
        torch.ones(1, 4),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[False, True, False, True]]),
    )
    out.sum().backward()
    assert calls == ["unshard", "execute", "reshard"]
    assert model[0].experts.weight1.grad is not None


@pytest.mark.parametrize("seed", range(8))
def test_dynamic_counts_conserve_every_source_and_expert(seed):
    # Independently partition each expert's global token stream across owners.
    torch.manual_seed(seed)
    ranks, experts = 3, 6
    counts = torch.randint(0, 20, (ranks, experts), dtype=torch.int32)
    allocation = torch.zeros(ranks, experts, dtype=torch.int32)
    expected = torch.zeros(ranks, ranks * experts, dtype=torch.int32)
    for expert in range(experts):
        source_ids = torch.repeat_interleave(
            torch.arange(ranks), counts[:, expert].long()
        )
        destinations = torch.randint(0, ranks, (source_ids.numel(),)).sort().values
        for source, destination in zip(source_ids, destinations):
            allocation[destination, expert] += 1
            expected[source, destination * experts + expert] += 1
    cumulative = allocation.cumsum(0, dtype=torch.int32).T.contiguous()
    table = torch.arange(experts).expand(ranks, experts)
    actual = virtual_counts(counts, cumulative, table, experts)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum(1), counts.sum(1), rtol=0, atol=0)


def test_router_probability_dtype_preserves_dispatcher_semantics():
    captured = {}

    class Experts:
        config = SimpleNamespace(moe_router_topk=2)

        def __call__(self, hidden, indices, weights):
            captured.update(indices=indices, weights=weights)
            return hidden * weights.sum(), None

    backend = integration.RoutedExpertsBackend(None, 128)
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    probs = torch.tensor([[0.0, 0.2, 0.0, 0.8]], requires_grad=True)
    mask = torch.tensor([[False, True, False, True]])
    output, _ = backend.route(Experts(), hidden, probs, mask)
    assert captured["weights"].dtype == torch.bfloat16
    torch.testing.assert_close(
        captured["weights"], probs[:, [1, 3]].to(torch.bfloat16), rtol=0, atol=0
    )
    output.sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[0.0, 4.0, 0.0, 4.0]]))


@pytest.mark.parametrize("path", ["fused", "deferred", "weighted"])
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_layer_routing_result_preserves_outputs_and_gradients(monkeypatch, path, score_function):
    from torchturbo.models.modules.moe.utils import RoutingResult

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setenv("TORCH_TURBO_DEBUG", "0")
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    probs = torch.tensor([[0., 0., 0., .5], [.1, 0., .2, 0.]], requires_grad=True)
    indices = torch.tensor([[1, 3], [0, 2]])
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, indices, True)
    routing = RoutingResult(probs, mask, indices)
    logits = torch.zeros(2, 1, 4)
    padding_mask = torch.tensor([False, True])

    class Router(nn.Module):
        def forward(self, inputs, input_ids, padding):
            assert padding is padding_mask
            return routing, logits

    class Experts(nn.Module):
        config = SimpleNamespace(moe_router_topk=2)

        def forward(self, inputs, assignments, weights):
            if path == "fused":
                torch.testing.assert_close(assignments, indices, rtol=0, atol=0)
                return inputs * weights.sum(-1, keepdim=True), None
            return inputs, None

    class Dispatcher:
        deferred_probs_to_unpermutation = path == "deferred"

        def token_permutation(self, inputs, result):
            assert isinstance(result, RoutingResult)
            assert result.routing_map is mask
            assert result.expert_indices is indices
            return inputs.repeat_interleave(2, dim=0), None, result.selected_probs().flatten()

        def token_unpermutation(self, inputs, bias, probs=None):
            if probs is not None:
                inputs = inputs * probs.flatten().unsqueeze(-1)
            return inputs.reshape(2, 2, 4).sum(1), bias

    layer = MoELayer.__new__(MoELayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(moe_router_score_function=score_function)
    layer._runtime_inputs = SimpleNamespace(padding_mask=padding_mask)
    layer.router = Router()
    layer.experts = Experts()
    layer.token_dispatcher = Dispatcher()
    layer._routed_experts_backend = (
        integration.RoutedExpertsBackend(None, 128) if path == "fused" else None
    )
    output, actual_logits = layer(hidden)
    final_probs = probs / probs.sum(-1, keepdim=True) if score_function == "softmax" else probs
    expected = hidden * final_probs.masked_select(mask).reshape(2, 2).sum(-1, keepdim=True)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(actual_logits, logits.reshape(2, 4))
    actual_grads = torch.autograd.grad(output.sum(), (hidden, probs), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, probs))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
    assert routing.probs is probs


@pytest.mark.parametrize("capacity,valid", [(128, True), (256, True), (127, False), (0, False), ("128", False)])
def test_adapter_validates_explicit_token_capacity(capacity, valid):
    config = EngineConfig(mega_moe=True)
    config.parallel_plan = SimpleNamespace(sp_size=1)
    config.extra["mega_moe_local_tokens"] = capacity
    args = SimpleNamespace(packing=False, max_length=100, per_device_train_batch_size=1)
    if valid:
        integration.configure_token_bound(config, args)
        assert config.extra["mega_moe_local_tokens"] == capacity
    else:
        with pytest.raises(ValueError, match="128-aligned and at least 128"):
            integration.configure_token_bound(config, args)
