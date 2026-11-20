"""Contracts for the router's dense weights and compact expert IDs."""

import pytest
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity
from torchturbo.ops.moe import RouteFormat
from torchturbo.ops.providers.npu.mega_moe import HmcRoutedExperts


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("hash_routing", [False, True])
def test_topk_result_packs_weights_map_and_expert_ids(
    score_function: str, grouped: bool, hash_routing: bool
) -> None:
    generator = torch.Generator().manual_seed(71)
    logits = torch.randn(17, 512, generator=generator, requires_grad=True)
    hash_table = torch.stack([torch.randperm(512, generator=generator)[:8] for _ in range(17)])
    probs, routing_map, expert_indices = topk_softmax_with_capacity(
        logits, 8, use_pre_softmax=True,
        num_groups=8 if grouped else None,
        group_topk=4 if grouped else None,
        scaling_factor=2.5, score_function=score_function,
        expert_bias=torch.linspace(-0.01, 0.01, 512),
        use_hash_routing=hash_routing, hash_table=hash_table,
        input_ids=torch.arange(17),
    )
    assert probs.shape == routing_map.shape == (17, 512)
    assert routing_map.sum(dim=-1).tolist() == [8] * 17

    expected_indices = routing_map.nonzero(as_tuple=True)[1].reshape(17, 8)
    assert expert_indices.shape == (17, 8)
    if hash_routing:
        torch.testing.assert_close(
            expert_indices.sort(-1).values, hash_table.sort(-1).values,
        )

    selected_ids = expert_indices
    order = selected_ids.sort(-1).indices
    selected = probs.gather(1, selected_ids).gather(1, order)
    expected = probs.masked_select(routing_map).reshape(17, 8)
    torch.testing.assert_close(selected_ids.gather(1, order), expected_indices)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)

    upstream = torch.randn(17, 8, generator=generator)
    selected_grad = torch.autograd.grad((selected * upstream).sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((expected * upstream).sum(), logits)[0]
    torch.testing.assert_close(selected_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_router_selected_weights_preserve_aux_and_z_loss_gradients(score_function: str) -> None:
    torch.manual_seed(42)
    router = TopKRouter(MoEConfig(
        hidden_size=16, num_moe_experts=32, moe_router_topk=4,
        moe_router_score_function=score_function, moe_router_pre_softmax=True,
        moe_router_enable_expert_bias=True, moe_aux_loss_coeff=0.1,
        moe_z_loss_coeff=0.01, params_dtype=torch.float32,
    ))
    torch.nn.init.normal_(router.weight, std=0.1)
    hidden = torch.randn(3, 2, 16, requires_grad=True)
    probs, routing_map, expert_indices, logits = router(hidden)
    assert probs.shape == routing_map.shape == (6, 32)
    assert expert_indices.shape == (6, 4)
    assert logits.shape == (3, 2, 32)

    selected = probs.gather(1, expert_indices)
    upstream = torch.randn_like(selected)
    (selected * upstream).sum().backward()
    assert hidden.grad is not None
    assert router.weight.grad is not None


@pytest.mark.parametrize("expert_bias", [False, True])
def test_compact_router_matches_dense_gradients_and_bias_counts(expert_bias):
    import copy

    torch.manual_seed(31)
    dense = TopKRouter(MoEConfig(
        hidden_size=16, num_moe_experts=32, moe_router_topk=4,
        moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        moe_router_enable_expert_bias=expert_bias, moe_aux_loss_coeff=0.1,
        moe_z_loss_coeff=0.01, params_dtype=torch.float32,
    ))
    torch.nn.init.normal_(dense.weight, std=0.1)
    compact = copy.deepcopy(dense)
    x = torch.randn(4, 1, 16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    padded = torch.tensor([[False], [True], [False], [False]])
    probs, mask, ids, _ = dense(x, padding_mask=padded)
    selected, compact_mask, compact_ids, _ = compact(other, padding_mask=padded, compact=True)
    reference = probs.gather(1, ids)
    torch.testing.assert_close(compact_ids, ids)
    torch.testing.assert_close(selected, reference)
    if expert_bias:
        torch.testing.assert_close(compact_mask, mask)
        torch.testing.assert_close(compact.local_tokens_per_expert, dense.local_tokens_per_expert)
    else:
        assert compact_mask is None
    upstream = torch.randn_like(selected)
    reference.backward(upstream)
    selected.backward(upstream)
    torch.testing.assert_close(x.grad, other.grad)
    torch.testing.assert_close(dense.weight.grad, compact.weight.grad)


def test_topk_recompute_without_load_accounting_skips_assignment_mask():
    router = TopKRouter(MoEConfig(
        hidden_size=4, num_moe_experts=8, moe_router_topk=2,
        moe_router_score_function="sigmoid", moe_router_enable_expert_bias=True,
        params_dtype=torch.float32,
    ))
    hidden = torch.ones(3, 1, 4)
    probs, mask, ids, _ = router(hidden, accumulate_tokens_per_expert=False, compact=True)
    assert probs.shape == ids.shape == (3, 2)
    assert mask is None


class ForwardOverrideRouter(TopKRouter):
    def forward(self, hidden, input_ids, padding_mask, accumulate_tokens_per_expert):
        return super().forward(hidden, input_ids, padding_mask, accumulate_tokens_per_expert)


class RoutingOverrideRouter(TopKRouter):
    def routing(self, logits, input_ids, padding_mask, accumulate_tokens_per_expert):
        return super().routing(logits, input_ids, padding_mask, accumulate_tokens_per_expert)


class DenseRouterWrapper(torch.nn.Module):
    def __init__(self, router):
        super().__init__()
        self.inner = router

    def forward(self, hidden, input_ids, padding_mask, accumulate_tokens_per_expert):
        weights, mask, ids, logits = self.inner(hidden, input_ids, padding_mask, accumulate_tokens_per_expert)
        assert weights.shape == mask.shape
        return weights, mask, ids, logits


@pytest.mark.parametrize("custom_router", [
    None, ForwardOverrideRouter, RoutingOverrideRouter, DenseRouterWrapper, "forward", "routing",
])
def test_hmc_uses_replaced_router_without_rebinding(monkeypatch, custom_router):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    config = MoEConfig(hidden_size=4, moe_ffn_hidden_size=4, num_moe_experts=8, moe_router_topk=2,
                       gated_linear_unit=True, activation_func=torch.nn.functional.silu, params_dtype=torch.float32)
    backend = HmcRoutedExperts()
    layer = MoELayer(config, object(), MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
    FSDP2Backend._bind_megakernel(layer, backend)
    if isinstance(custom_router, str):
        original = getattr(layer.router, custom_router)

        def legacy_method(tensor, ids, mask, count):
            return original(tensor, ids, mask, count)

        setattr(layer.router, custom_router, legacy_method)
    elif custom_router is DenseRouterWrapper:
        layer.router = DenseRouterWrapper(layer.router)
    elif custom_router is not None:
        layer.router = custom_router(config)
    source = layer.router.inner if custom_router is DenseRouterWrapper else layer.router
    torch.nn.init.normal_(source.weight)
    hidden = torch.randn(3, 1, 4, requires_grad=True)
    observed_shapes = []
    layer.router.register_forward_hook(lambda module, args, result: observed_shapes.append(result[0].shape))

    reference = TopKRouter(config)
    reference.load_state_dict(source.state_dict())
    dense, _, ids, logits = reference(hidden)
    weights, _, actual_ids, actual_logits = layer.route(hidden, None)
    assert observed_shapes == [(3, 2 if custom_router is None else 8)]
    torch.testing.assert_close(actual_ids, ids)
    torch.testing.assert_close(actual_logits, logits)
    selected = dense.gather(1, ids)
    torch.testing.assert_close(weights, selected)
    upstream = torch.randn_like(weights)
    expected = torch.autograd.grad(selected, (hidden, reference.weight), upstream)
    actual = torch.autograd.grad(weights, (hidden, source.weight), upstream)
    for observed, wanted in zip(actual, expected):
        torch.testing.assert_close(observed, wanted)


@pytest.mark.parametrize("route_format", [RouteFormat.DENSE, RouteFormat.TOPK, None])
def test_other_provider_can_declare_or_omit_route_format(monkeypatch, route_format):
    class OtherBackend(torch.nn.Module):
        def forward_experts(self, layer, hidden, input_ids):
            if route_format is None:
                return hidden, hidden
            weights, _, ids, logits = layer.route(hidden, input_ids)
            assert weights.shape == (3, 2 if route_format is RouteFormat.TOPK else 8)
            return hidden, logits

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    config = MoEConfig(hidden_size=4, moe_ffn_hidden_size=4, num_moe_experts=8, moe_router_topk=2,
                       gated_linear_unit=True, activation_func=torch.nn.functional.silu, params_dtype=torch.float32)
    kernel = OtherBackend()
    if route_format is not None:
        kernel.route_format = route_format
    layer = MoELayer(config, object(), MoESubmodules(experts=ModuleSpec(module=GroupedMLP)))
    FSDP2Backend._bind_megakernel(layer, kernel)
    calls = []
    layer.router.register_forward_hook(lambda *args: calls.append(True))
    hidden = torch.randn(3, 1, 4)
    layer(hidden)
    assert len(calls) == int(route_format is not None)
    assert kernel in layer.children()
    layer.eval()
    assert not kernel.training
    calls.clear()
    output, _ = layer(hidden)
    torch.testing.assert_close(output, hidden)
    assert len(calls) == int(route_format is not None)
