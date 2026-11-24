"""MoonEP manager compatibility and runtime topology contracts."""

from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoERuntimeInputs, MoESubmodules
from torchturbo.models.modules.moe.token_dispatcher import (
    _HybridEPManager, _VirtualExpertHybridEPManager,
)
from torchturbo.models.modules.moe.virtual_expert_load_balancer import VirtualExpertLoadBalancer
from torchturbo.parallel import FSDPPlanSpec
from tests.models.test_virtual_expert_weights import _virtual_config, cpu_dispatch


@pytest.mark.parametrize("options", [{}, {"router_topk": None}, {"router_topk": 2}])
@pytest.mark.parametrize("manager_cls", [_HybridEPManager, _VirtualExpertHybridEPManager])
def test_manager_uses_megatron_constructor_order_and_config_topk(cpu_dispatch, options, manager_cls):
    config = replace(_virtual_config(), moe_router_topk=2)
    manager = manager_cls(None, 2, 2, config, **options)
    assert manager.router_topk == 2
    assert manager.num_experts == (4 if manager_cls is _VirtualExpertHybridEPManager else 2)


def test_manager_resolves_none_to_the_default_world_group(cpu_dispatch, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    config = replace(_virtual_config(), num_moe_experts=4, expert_model_parallel_size=2)
    manager = _VirtualExpertHybridEPManager(None, 2, 4, config)
    assert manager.ep_size == 2
    assert manager.num_experts == 8


@pytest.mark.parametrize("arguments,config_changes", [
    ({"num_local_experts": 1}, {}),
    ({"num_experts": 4}, {}),
    ({"router_topk": 2}, {}),
    ({}, {"num_moe_experts": 4}),
    ({}, {"expert_model_parallel_size": 2}),
    ({}, {"moe_router_topk": 2}),
])
def test_manager_rejects_runtime_layout_mismatches(cpu_dispatch, arguments, config_changes):
    config = replace(_virtual_config(), **config_changes)
    parameters = dict(group=None, num_local_experts=2, num_experts=2, router_topk=1, config=config)
    parameters.update(arguments)
    with pytest.raises(ValueError, match="runtime layout"):
        VirtualExpertLoadBalancer().initialize_virtual_expert_load_balancer(**parameters)


@pytest.mark.parametrize("all_padding", [False, True])
def test_metadata_masks_compact_probabilities_in_batch_sequence_order(cpu_dispatch, all_padding):
    config = replace(_virtual_config(), moe_router_topk=2)
    manager = _VirtualExpertHybridEPManager(None, 2, 2, config)
    indices = torch.tensor([[1, 0]] * 6)
    probabilities = torch.arange(12).reshape(6, 2).float().requires_grad_()
    mask = torch.tensor([[False, True, False], [True, False, True]])
    expected = torch.tensor([[0, 1], [0, 0], [4, 5], [0, 0], [8, 9], [0, 0]]).float()
    if all_padding:
        mask.fill_(True)
        expected.zero_()
    manager.wrap_layer_input(torch.ones(6, 4))
    manager.setup_metadata(indices, probabilities, padding_mask=mask)
    torch.testing.assert_close(manager.token_probs, expected)
    torch.testing.assert_close(manager._plan.routing.virtual_experts, indices)
    torch.testing.assert_close(probabilities.detach(), torch.arange(12).reshape(6, 2).float())
    manager.token_probs.sum().backward()
    torch.testing.assert_close(
        probabilities.grad, (~mask).reshape(-1, 1).expand(-1, 2).float(),
    )
    manager.release_routing_state()


def test_metadata_rejects_a_padding_mask_for_a_different_token_count(cpu_dispatch):
    manager = _VirtualExpertHybridEPManager(None, 2, 2, _virtual_config())
    manager.wrap_layer_input(torch.ones(2, 4))
    with pytest.raises(ValueError, match="padding_mask"):
        manager.setup_metadata(torch.tensor([[0], [1]]), torch.ones(2, 1), padding_mask=torch.ones(1))
    manager.release_routing_state()


@pytest.mark.parametrize("virtual", [False, True], ids=["plain", "virtual"])
def test_flex_post_topk_softmax_saves_no_full_width_probabilities(cpu_dispatch, virtual):
    config = replace(
        _virtual_config(), num_moe_experts=11, moe_router_topk=2,
        moe_virtual_expert_load_balance=virtual,
    )
    layer = MoELayer(config, None, submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
    shapes = []

    def save(tensor):
        shapes.append(tensor.shape)
        return tensor

    x = torch.randn(2, 3, config.hidden_size, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        output, _ = layer(x)
    assert (6, 11) not in shapes
    output.sum().backward()


@pytest.mark.parametrize("checkpoint_mode", [None, False, True], ids=["eager", "nonreentrant", "reentrant"])
@pytest.mark.parametrize("on_input", [False, True])
@pytest.mark.parametrize("virtual", [False, True], ids=["plain", "virtual"])
def test_flex_padding_matches_masked_outputs_and_gradients(cpu_dispatch, checkpoint_mode, on_input, virtual):
    torch.manual_seed(517)
    config = replace(
        _virtual_config(), moe_router_topk=1 if on_input else 2,
        moe_apply_probs_on_input=on_input, moe_virtual_expert_load_balance=virtual,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    candidate = MoELayer(config, None, submodules=submodules)
    reference = MoELayer(
        replace(config, moe_virtual_expert_load_balance=False, moe_token_dispatcher_type="alltoall"),
        None, submodules=submodules,
    )
    with torch.no_grad():
        for parameter in candidate.parameters():
            parameter.normal_(std=0.2)
    reference.load_state_dict(candidate.state_dict())
    if checkpoint_mode is not None:
        backend = SimpleNamespace(
            config=EngineConfig.from_dict({
                "activation_checkpointing": {"enabled": True, "use_reentrant": checkpoint_mode},
            }),
            prefetch_manager=SimpleNamespace(suspend_expert_forward_prefetch=lambda model: nullcontext()),
        )
        FSDP2Backend._apply_activation_checkpointing(
            backend, candidate, SimpleNamespace(modules={"": FSDPPlanSpec(activation_checkpoint=True)}),
        )
    inputs, ref_inputs, outputs, expected = [], [], [], []
    masks = [
        torch.tensor([[False, True, False], [True, False, True]]),
        torch.tensor([[True, False, True], [False, True, False]]),
        torch.ones((2, 3), dtype=torch.bool),
    ]
    for mask in masks:
        x = torch.randn(2, 3, 4, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        candidate.set_runtime_inputs(MoERuntimeInputs(padding_mask=mask))
        outputs.append(candidate(x)[0])
        expected.append(reference(ref_x)[0].masked_fill(mask.unsqueeze(-1), 0))
        inputs.append(x)
        ref_inputs.append(ref_x)
    for actual, wanted in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    sum(output.sum() for output in outputs).backward()
    sum(output.sum() for output in expected).backward()
    for actual, wanted in zip(inputs, ref_inputs, strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad)
    for actual, wanted in zip(candidate.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, wanted.grad)
