"""MoE probability placement preserves model outputs, gradients, and wire dtype."""

import copy
import importlib
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from torchturbo.engine.peft.lora import GroupedMLPLoRA
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe import token_dispatcher as dispatch_module
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


def _layer(*, on_input, use_lora=False, dtype=torch.float32):
    config = MoEConfig(
        hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=2,
        gated_linear_unit=True, activation_func=F.silu,
        moe_router_topk=1 if on_input else 2,
        moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        moe_apply_probs_on_input=on_input, params_dtype=dtype,
    )
    layer = MoELayer(
        config, object(), submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    if use_lora:
        layer.experts = GroupedMLPLoRA(layer.experts, "default", r=2, lora_alpha=2)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.3)
    return layer


def test_completed_dispatch_stages_release_their_activation_buffers(monkeypatch):
    # Give each collective a distinct receive buffer, as in multi-rank EP.
    # Keep the real router, dispatcher stages, native operators, and experts.
    monkeypatch.setattr(
        dispatch_module, "all_to_all", lambda group, tensor, *splits: tensor.clone()
    )
    layer = _layer(on_input=False)
    dispatcher = layer.token_dispatcher
    references = {}

    def observe_stage(name, stage):
        def observed(*args):
            result = stage(*args)
            tensors = result if isinstance(result, tuple) else (result,)
            references[name] = [weakref.ref(tensor) for tensor in tensors]
            return result
        return observed

    for name, stage in (
        ("dispatch_preprocess", dispatcher.dispatch_preprocess),
        ("token_dispatch", dispatcher.token_dispatch),
        ("combine_preprocess", dispatcher.combine_preprocess),
        ("token_combine", dispatcher.token_combine),
    ):
        monkeypatch.setattr(dispatcher, name, observe_stage(name, stage))

    def assert_released(names):
        for name in names:
            assert all(ref() is None for ref in references[name]), name

    layer.experts.register_forward_pre_hook(
        lambda module, args: assert_released(("dispatch_preprocess", "token_dispatch"))
    )
    layer.shared_experts = torch.nn.Identity()
    layer.shared_experts.register_forward_pre_hook(
        lambda module, args: assert_released(("combine_preprocess", "token_combine"))
    )
    inputs = torch.randn(1, 5, 4, requires_grad=True)
    output, _ = layer(inputs)
    output.sum().backward()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("topk,pre_softmax,scale", [
    (1, True, None), (1, True, 2.0), (2, True, 2.0), (2, False, 2.0),
])
def test_softmax_layer_preserves_router_probabilities_and_gradients(topk, pre_softmax, scale):
    config = MoEConfig(
        hidden_size=1, moe_ffn_hidden_size=1, num_moe_experts=3,
        activation_func=torch.nn.Identity(), moe_router_topk=topk,
        moe_router_pre_softmax=pre_softmax, moe_router_topk_scaling_factor=scale,
    )
    layer = MoELayer(
        config, object(), submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.no_grad():
        layer.router.weight.copy_(torch.tensor([[2.0], [1.0], [0.0]]))
        layer.experts.weight1.fill_(1.0)
        layer.experts.weight2.copy_(torch.tensor([[1.0], [2.0], [3.0]]))
    reference_logits = layer.router.weight.detach().flatten().requires_grad_()
    if pre_softmax:
        reference_probs = reference_logits.softmax(-1)[:topk]
    else:
        reference_probs = reference_logits[:topk].softmax(-1)
    expected = (reference_probs * torch.arange(1, topk + 1)).sum()
    if scale is not None:
        expected = expected * scale

    output, _ = layer(torch.ones(1, 1, 1))

    torch.testing.assert_close(output.squeeze(), expected)
    output.sum().backward()
    expected.backward()
    assert torch.count_nonzero(layer.router.weight.grad) > 0
    torch.testing.assert_close(layer.router.weight.grad.flatten(), reference_logits.grad)


def _dense_reference(layer, hidden_states):
    config = layer.config
    experts = layer.experts
    base = experts.base_layer if isinstance(experts, GroupedMLPLoRA) else experts
    weight1 = base.weight1.view(2, 4, 6)
    weight2 = base.weight2.view(2, 3, 4)
    if isinstance(experts, GroupedMLPLoRA):
        weight1 = weight1 + (
            experts.lora_A["default"].weight @ experts.lora_B["default"].weight
        ) * experts.scaling["default"]
        weight2 = weight2 + (
            experts.lora_A2["default"].weight @ experts.lora_B2["default"].weight
        ) * experts.scaling["default"]
    tokens = hidden_states.reshape(-1, 4)
    logits = F.linear(tokens.float(), layer.router.weight.float())
    scores, indices = logits.sigmoid().topk(config.moe_router_topk, dim=-1)
    if config.moe_router_topk > 1:
        scores = scores / scores.sum(-1, keepdim=True)
    outputs = []
    for token, selected, probabilities in zip(tokens, indices, scores):
        contributions = []
        for expert, probability in zip(selected.tolist(), probabilities):
            probability = probability.unsqueeze(0)
            x = (token * probability).to(token.dtype) if config.moe_apply_probs_on_input else token
            gate, up = (x @ weight1[expert]).chunk(2, dim=-1)
            activation = F.silu(gate) * up
            if not config.moe_apply_probs_on_input:
                activation = (activation * probability).to(activation.dtype)
            contribution = activation @ weight2[expert]
            contributions.append(contribution)
        outputs.append(torch.stack(contributions).sum(0))
    return torch.stack(outputs).to(hidden_states.dtype).reshape_as(hidden_states)


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("on_input", [False, True])
def test_expert_weighting_matches_dense_outputs_and_gradients(use_lora, on_input):
    torch.manual_seed(547)
    layer = _layer(on_input=on_input, use_lora=use_lora)
    reference = copy.deepcopy(layer)
    x = torch.randn(1, 5, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, _ = layer(x)
    expected = _dense_reference(reference, ref_x)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for (name, parameter), (ref_name, ref_parameter) in zip(
        layer.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, msg=name)


@pytest.mark.parametrize("on_input", [False, True])
def test_megatron_weighting_keeps_bf16_tokens_and_fp32_probabilities(monkeypatch, on_input):
    torch.manual_seed(793)
    layer = _layer(on_input=on_input, dtype=torch.bfloat16)
    reference = copy.deepcopy(layer)
    payload_dtypes = []
    original_all_to_all = dispatch_module.all_to_all

    def observe_all_to_all(group, tensor, *splits):
        payload_dtypes.append(tensor.dtype)
        return original_all_to_all(group, tensor, *splits)

    monkeypatch.setattr(dispatch_module, "all_to_all", observe_all_to_all)
    x = torch.randn(1, 5, 4, dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, _ = layer(x)
    expected = _dense_reference(reference, ref_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert payload_dtypes == [torch.bfloat16, torch.float32, torch.bfloat16]
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(layer.router.weight.grad, reference.router.weight.grad)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_keeps_each_runtime_mask_with_frozen_inputs(use_reentrant):
    torch.manual_seed(17)
    reference = _layer(on_input=False)
    reference.config.moe_z_loss_coeff = 0.3
    candidate = copy.deepcopy(reference)
    backend = SimpleNamespace(
        config=EngineConfig.from_dict({
            "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
        }),
        prefetch_manager=SimpleNamespace(suspend_expert_forward_prefetch=lambda model: nullcontext()),
    )
    FSDP2Backend._apply_activation_checkpointing(
        backend, candidate, SimpleNamespace(modules={"": FSDPPlanSpec(activation_checkpoint=True)}),
    )
    inputs = torch.randn(1, 6, 4)
    outputs, expected = [], []
    for padded_index in (2, 1):
        mask = torch.zeros(1, 6, dtype=torch.bool)
        mask[:, [padded_index, 5]] = True
        runtime_inputs = MoERuntimeInputs(padding_mask=mask)
        candidate.set_runtime_inputs(runtime_inputs)
        reference.set_runtime_inputs(runtime_inputs)
        outputs.append(candidate(inputs)[0])
        expected.append(reference(inputs)[0])
    sum(output.sum() * 0 for output in outputs).backward()
    sum(output.sum() * 0 for output in expected).backward()
    torch.testing.assert_close(candidate.router.weight.grad, reference.router.weight.grad)
    assert candidate._runtime_inputs is runtime_inputs


@pytest.mark.parametrize("fail_exit", [False, True])
def test_runtime_input_context_restores_each_nested_callers_inputs(fail_exit):
    from torchturbo.engine.backends.fsdp2.activation_checkpointing import restore_moe_runtime_inputs

    layers = (_layer(on_input=False), _layer(on_input=True))
    original_inputs = tuple(layer._runtime_inputs for layer in layers)
    outer_snapshot = tuple((layer, MoERuntimeInputs()) for layer in layers)
    inner_snapshot = tuple((layer, MoERuntimeInputs()) for layer in reversed(layers))

    with pytest.raises(ValueError, match="outer exit") if fail_exit else nullcontext():
        with restore_moe_runtime_inputs(outer_snapshot):
            for layer, inputs in outer_snapshot:
                assert layer._runtime_inputs is inputs
            with pytest.raises(RuntimeError, match="inner exit") if fail_exit else nullcontext():
                with restore_moe_runtime_inputs(inner_snapshot):
                    for layer, inputs in inner_snapshot:
                        assert layer._runtime_inputs is inputs
                    if fail_exit:
                        raise RuntimeError("inner exit")
            for layer, inputs in outer_snapshot:
                assert layer._runtime_inputs is inputs
            if fail_exit:
                raise ValueError("outer exit")
    for layer, inputs in zip(layers, original_inputs, strict=True):
        assert layer._runtime_inputs is inputs


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("freeze_router", [False, True])
def test_expert_checkpoint_with_frozen_inputs_preserves_parameter_gradients(
    use_lora, use_reentrant, on_input, freeze_router,
):
    torch.manual_seed(491)
    layer = _layer(on_input=on_input, use_lora=use_lora)
    layer.router.requires_grad_(not freeze_router)
    if use_lora:
        layer.experts.base_layer.requires_grad_(False)
    reference = copy.deepcopy(layer)
    forward_modes = []
    original_forward = layer.experts.forward

    def observed_forward(hidden_states, tokens_per_expert, probs):
        forward_modes.append(torch.is_grad_enabled())
        assert not hidden_states.requires_grad
        return original_forward(hidden_states, tokens_per_expert, probs)

    layer.experts.forward = observed_forward
    backend = SimpleNamespace(
        config=EngineConfig.from_dict({
            "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
        }),
        prefetch_manager=SimpleNamespace(suspend_expert_forward_prefetch=lambda model: nullcontext()),
    )
    plan = SimpleNamespace(modules={"experts": FSDPPlanSpec(activation_checkpoint=True)})
    FSDP2Backend._apply_activation_checkpointing(backend, layer, plan)
    x = torch.randn(1, 5, 4)
    actual, _ = layer(x)
    expected = _dense_reference(reference, x)
    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    assert not x.requires_grad
    assert x.grad is None
    if use_reentrant:
        assert forward_modes == [False, True]
    for (name, parameter), (ref_name, ref_parameter) in zip(
        layer.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, msg=name)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_expert_checkpoint_with_disabled_adapters_preserves_router_gradients(use_reentrant):
    torch.manual_seed(492)
    layer = _layer(on_input=False, use_lora=True)
    layer.experts.base_layer.requires_grad_(False)
    layer.experts.enable_adapters(False)
    assert any(parameter.requires_grad for parameter in layer.experts.parameters())
    reference = copy.deepcopy(layer)
    reference.experts = reference.experts.base_layer
    backend = SimpleNamespace(
        config=EngineConfig.from_dict({
            "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
        }),
        prefetch_manager=SimpleNamespace(suspend_expert_forward_prefetch=lambda model: nullcontext()),
    )
    plan = SimpleNamespace(modules={"experts": FSDPPlanSpec(activation_checkpoint=True)})
    FSDP2Backend._apply_activation_checkpointing(backend, layer, plan)
    x = torch.randn(1, 5, 4)
    actual, _ = layer(x)
    expected = _dense_reference(reference, x)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    assert not x.requires_grad and x.grad is None
    assert all(parameter.grad is None for parameter in layer.experts.parameters())
    assert layer.router.weight.grad is not None
    assert torch.count_nonzero(layer.router.weight.grad) > 0
    torch.testing.assert_close(layer.router.weight.grad, reference.router.weight.grad)


@pytest.mark.parametrize("freeze_router", [False, True])
def test_frozen_experts_preserve_the_router_gradient_graph(freeze_router):
    torch.manual_seed(312)
    layer = _layer(on_input=False)
    layer.experts.requires_grad_(False)
    layer.router.requires_grad_(not freeze_router)
    reference = copy.deepcopy(layer)
    expert_output_requires_grad = []
    layer.experts.register_forward_hook(
        lambda module, args, output: expert_output_requires_grad.append(output[0].requires_grad)
    )
    x = torch.randn(1, 5, 4)
    actual, _ = layer(x)
    expected = _dense_reference(reference, x)
    torch.testing.assert_close(actual, expected)
    assert expert_output_requires_grad == [not freeze_router]
    assert actual.requires_grad == expected.requires_grad
    if not freeze_router:
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(layer.router.weight.grad, reference.router.weight.grad)
