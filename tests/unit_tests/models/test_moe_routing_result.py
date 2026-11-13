"""Contracts for the router's dense weights and compact expert IDs."""

import pytest
import torch

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity


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
