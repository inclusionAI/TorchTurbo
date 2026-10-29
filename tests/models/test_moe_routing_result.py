from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.utils import RoutingResult, topk_softmax_with_capacity


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("hash_routing", [False, True])
def test_selected_probs_preserve_mask_order_and_logits_gradient(
    score_function: str, grouped: bool, hash_routing: bool
) -> None:
    generator = torch.Generator().manual_seed(71)
    logits = torch.randn(17, 512, generator=generator, requires_grad=True)
    hash_table = torch.stack([torch.randperm(512, generator=generator)[:8] for _ in range(17)])
    routing = topk_softmax_with_capacity(
        logits, 8, use_pre_softmax=True,
        num_groups=8 if grouped else None,
        group_topk=4 if grouped else None,
        scaling_factor=2.5, score_function=score_function,
        expert_bias=torch.linspace(-0.01, 0.01, 512),
        use_hash_routing=hash_routing, hash_table=hash_table,
        input_ids=torch.arange(17),
    )
    assert isinstance(routing, RoutingResult)
    expected_indices = routing.routing_map.nonzero(as_tuple=True)[1].reshape(17, 8)
    torch.testing.assert_close(routing.expert_indices, expected_indices, rtol=0, atol=0)
    if hash_routing:
        torch.testing.assert_close(routing.expert_indices, hash_table.sort(-1).values, rtol=0, atol=0)
    expected = routing.probs.masked_select(routing.routing_map).reshape(17, 8)
    actual = routing.selected_probs()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn(17, 8, generator=generator)
    old_grad = torch.autograd.grad((expected * upstream).sum(), logits, retain_graph=True)[0]
    new_grad = torch.autograd.grad((actual * upstream).sum(), logits)[0]
    torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)


def test_replace_uses_final_weights_without_detaching() -> None:
    probs = torch.tensor([[0.0, 0.3, 0.0, 0.7]], requires_grad=True)
    routing = RoutingResult(probs, probs.detach() != 0, torch.tensor([[1, 3]]))
    final = replace(routing, probs=routing.probs * 2.5)
    torch.testing.assert_close(final.selected_probs(), torch.tensor([[0.75, 1.75]]))
    final.selected_probs().sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[0.0, 2.5, 0.0, 2.5]]))
    assert final.routing_map is routing.routing_map
    assert final.expert_indices is routing.expert_indices
    with pytest.raises(FrozenInstanceError):
        routing.probs = probs.detach()


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
    routing, logits = router(hidden)
    assert isinstance(routing, RoutingResult)
    assert logits.shape == (3, 2, 32)
    if score_function == "softmax":
        routing = replace(routing, probs=routing.probs / routing.probs.sum(-1, keepdim=True))
    old = routing.probs.masked_select(routing.routing_map).reshape(6, 4)
    new = routing.selected_probs()
    upstream = torch.randn_like(old)
    old_grads = torch.autograd.grad((old * upstream).sum(), (hidden, router.weight), retain_graph=True)
    new_grads = torch.autograd.grad((new * upstream).sum(), (hidden, router.weight))
    for actual, expected in zip(new_grads, old_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
