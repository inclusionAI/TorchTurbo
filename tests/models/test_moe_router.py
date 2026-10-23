"""Tests for shared MoE router behavior."""

import pytest
import torch
import torch.nn.functional as F

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.models.modules.moe.router import TopKRouter
from torchturbo.models.modules.moe.utils import topk_softmax_with_capacity


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_scores_do_not_change_selected_expert(score_function, dtype):
    logits = torch.tensor([[0.0, 0.0001]], dtype=dtype)
    routing = topk_softmax_with_capacity(
        logits, 1, use_pre_softmax=True, score_function=score_function,
    )

    torch.testing.assert_close(routing.routing_map, torch.tensor([[False, True]]))
    torch.testing.assert_close(routing.expert_indices, torch.tensor([[1]]))


@pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("hash_routing", [False, True])
def test_probability_normalization_and_scaling_use_fp32(score_function, hash_routing):
    logits = torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.bfloat16, requires_grad=True)
    reference_logits = logits.detach().float().requires_grad_()
    score_func = torch.sigmoid if score_function == "sigmoid" else lambda x: F.softplus(x).sqrt()
    selected_scores = score_func(reference_logits)[:, :2]
    expected = (selected_scores / selected_scores.sum(-1, keepdim=True) * 1.7).to(logits.dtype)
    kwargs = {}
    if hash_routing:
        kwargs = {
            "use_hash_routing": True,
            "hash_table": torch.tensor([[0, 1]]),
            "input_ids": torch.tensor([0]),
        }

    routing = topk_softmax_with_capacity(
        logits, 2, score_function=score_function, scaling_factor=1.7, **kwargs,
    )
    probs = routing.probs

    torch.testing.assert_close(probs[:, :2], expected, rtol=0, atol=0)
    gradient = torch.tensor([[1.0, 3.0]], dtype=logits.dtype)
    (probs[:, :2] * gradient).sum().backward()
    (expected * gradient).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad.to(logits.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_softmax_normalization_precedes_scaling(normalize):
    config = MoEConfig(
        hidden_size=1, num_moe_experts=3, moe_router_topk=2,
        moe_router_pre_softmax=True, moe_router_normalize_topk_prob=normalize,
        moe_router_topk_scaling_factor=2.0,
    )
    router = TopKRouter(config)
    logits = torch.tensor([[[2.0, 1.0, 0.0]]], requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    expected = reference_logits.softmax(-1)[..., :2]
    if normalize:
        expected = expected / expected.sum(-1, keepdim=True)
    expected = expected * 2.0

    probs = router.routing(logits).probs

    torch.testing.assert_close(probs[:, :2], expected.reshape(1, 2))
    (probs * torch.tensor([[1.0, 3.0, 0.0]])).sum().backward()
    (expected * torch.tensor([1.0, 3.0])).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
@pytest.mark.parametrize("layer_number", [0, 1], ids=["hash", "learned"])
@pytest.mark.parametrize("z_coeff", [0.0, 0.3])
def test_hash_layers_keep_main_and_z_gradients_without_learned_aux(
    monkeypatch, score_function, layer_number, z_coeff,
):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "loss_scale", 1.0)
    router = TopKRouter(
        MoEConfig(
            hidden_size=2, num_moe_experts=4, moe_router_topk=2,
            moe_router_score_function=score_function,
            moe_router_pre_softmax=True, moe_router_normalize_topk_prob=True,
            num_hash_layers=1, vocab_size=2,
            moe_aux_loss_coeff=0.2, moe_z_loss_coeff=z_coeff,
        ),
        layer_number=layer_number,
    )
    hash_table = torch.tensor([[0, 1], [2, 3]])
    with torch.no_grad():
        router.weight.copy_(torch.tensor([[0.5, 0.1], [0.2, -0.3], [-0.4, 0.7], [0.6, -0.5]]))
        if layer_number == 0:
            router.tid2eid.copy_(hash_table)
    inputs = torch.tensor([[[1., 2.], [2., -1.]]], requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    ref_weight = router.weight.detach().clone().requires_grad_()
    logits = F.linear(ref_inputs, ref_weight).reshape(-1, 4)
    if score_function == "softmax":
        scores = logits.softmax(-1)
    elif score_function == "sigmoid":
        scores = logits.sigmoid()
    else:
        scores = F.softplus(logits).sqrt()
    indices = hash_table if layer_number == 0 else scores.topk(2, dim=-1).indices
    selected = scores.gather(1, indices)
    expected_probs = torch.zeros_like(scores).scatter(
        1, indices, selected / selected.sum(-1, keepdim=True),
    )
    coefficients = torch.tensor([[1., 2., 4., 8.], [3., 5., 2., 1.]])
    expected_loss = (expected_probs * coefficients).sum()
    expected_loss = expected_loss + z_coeff * logits.logsumexp(-1).square().mean()
    if layer_number == 1:
        aux_scores = scores / scores.sum(-1, keepdim=True)
        counts = torch.bincount(indices.flatten(), minlength=4)
        expected_loss = expected_loss + 0.2 * 4 * (aux_scores.sum(0) * counts).sum() / (2 * 2**2)

    routing, _ = router(inputs, input_ids=torch.tensor([[0, 1]]))
    (routing.probs * coefficients).sum().backward()
    expected_loss.backward()

    torch.testing.assert_close(routing.probs, expected_probs)
    torch.testing.assert_close(routing.routing_map, expected_probs != 0)
    torch.testing.assert_close(router.weight.grad, ref_weight.grad)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)
    metrics = router.pop_metrics()
    assert ("aux_loss" in metrics) == (layer_number == 1)
    assert ("z_loss" in metrics) == (z_coeff > 0)


def _router_config() -> MoEConfig:
    return MoEConfig(
        hidden_size=2,
        num_moe_experts=3,
        perform_initialization=False,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_topk=1,
    )


def test_local_token_counts_are_absent_during_meta_construction():
    with torch.device("meta"):
        router = TopKRouter(_router_config())

    assert not hasattr(router, "local_tokens_per_expert")
    assert "local_tokens_per_expert" not in router._buffers
    assert "local_tokens_per_expert" not in router._non_persistent_buffers_set


def test_local_token_counts_are_absent_when_expert_bias_is_disabled():
    config = _router_config()
    config.moe_router_enable_expert_bias = False

    router = TopKRouter(config)

    assert not hasattr(router, "local_tokens_per_expert")


def test_local_token_counts_are_lazily_initialized_and_accumulated():
    router = TopKRouter(_router_config())
    assert not hasattr(router, "local_tokens_per_expert")
    assert "local_tokens_per_expert" not in router._buffers

    logits = torch.tensor(
        [
            [[3.0, 2.0, 1.0]],
            [[1.0, 2.0, 3.0]],
        ]
    )
    routing_map = router.routing(logits).routing_map
    expected = routing_map.sum(dim=0).to(torch.float32)

    assert router.local_tokens_per_expert is not None
    assert "local_tokens_per_expert" in router._non_persistent_buffers_set
    assert router.local_tokens_per_expert.device == routing_map.device
    torch.testing.assert_close(router.local_tokens_per_expert, expected)

    router.routing(logits)
    torch.testing.assert_close(router.local_tokens_per_expert, expected * 2)


def test_local_token_counts_exclude_padding_without_changing_dispatch_map():
    router = TopKRouter(_router_config())
    logits = torch.tensor(
        [
            [[3.0, 2.0, 1.0]],
            [[1.0, 2.0, 3.0]],
            [[2.0, 3.0, 1.0]],
        ]
    )
    padding_mask = torch.tensor([[False], [True], [False]])

    routing_map = router.routing(logits, padding_mask=padding_mask).routing_map

    expected_counts = routing_map[~padding_mask.reshape(-1)].sum(dim=0).float()
    torch.testing.assert_close(router.local_tokens_per_expert, expected_counts)
    assert routing_map[padding_mask.reshape(-1)].any()
