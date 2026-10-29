"""Tests for shared MoE router behavior."""

import torch

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter


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
