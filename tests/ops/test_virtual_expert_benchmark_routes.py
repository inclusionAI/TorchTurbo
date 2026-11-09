"""Validate the distributions used to compare MoE dispatch performance."""

import pytest
import torch


@pytest.mark.parametrize("ep_size,expected", [(2, 14), (8, 27)])
def test_matrix_covers_random_seeds_and_distinct_hotspot_layouts(ep_size, expected):
    from tests.ops.npu.benchmark_virtual_expert_routes import routing_cases

    cases = routing_cases(
        ep_size, ["balanced", "random", "hotspot"], [17, 29, 43],
        [1, 2, 4], [0.25, 0.5, 0.75, 0.9, 1.0], ["packed", "spread"],
        num_experts=ep_size * 2, topk=1,
    )
    assert len(cases) == expected
    assert len(set(cases)) == expected
    assert [case.seed for case in cases if case.routing == "random"] == [17, 29, 43]
    for case in cases:
        if case.routing == "hotspot":
            assert case.num_hot_experts < ep_size * 2
            assert case.hot_fraction > case.num_hot_experts / (ep_size * 2)


@pytest.mark.parametrize("ep_size", [2, 8])
@pytest.mark.parametrize("tokens", [1024, 4096])
def test_uniform_and_full_single_hotspot_preserve_previous_inputs(ep_size, tokens):
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    uniform = RoutingCase("balanced").route_ids(tokens, ep_size, rank=0, num_experts=ep_size * 2, topk=1).squeeze(1)
    torch.testing.assert_close(uniform, torch.arange(tokens) % (2 * ep_size))
    hotspot = RoutingCase("hotspot", 1, 1.0, "packed", 17).route_ids(tokens, ep_size, rank=0, num_experts=ep_size * 2, topk=1).squeeze(1)
    assert hotspot.tolist() == [0] * tokens


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.parametrize("num_hot", [1, 2, 4])
def test_hot_fraction_counts_only_the_selected_group(fraction, num_hot):
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    case = RoutingCase("hotspot", num_hot, fraction, "packed", 17)
    tokens = 1024
    ids = case.route_ids(tokens, ep_size=8, rank=3, num_experts=16, topk=1).squeeze(1)
    assert ids.dtype == torch.int64 and ids.shape == (tokens,)
    assert ids.min() >= 0 and ids.max() < 16
    counts = torch.bincount(ids, minlength=16)
    assert counts[:num_hot].sum() == round(tokens * fraction)
    assert counts[:num_hot].max() - counts[:num_hot].min() <= 1
    assert counts[num_hot:].max() - counts[num_hot:].min() <= 1


def test_hotspot_placement_changes_rank_load_without_changing_expert_loads():
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    packed = RoutingCase("hotspot", 2, 0.75, "packed", 17)
    spread = RoutingCase("hotspot", 2, 0.75, "spread", 17)
    assert packed.hot_ids(8, num_experts=16) == [0, 1]
    assert spread.hot_ids(8, num_experts=16) == [0, 2]
    packed_counts = torch.bincount(packed.route_ids(1024, 8, 0, num_experts=16, topk=1).squeeze(1), minlength=16)
    spread_counts = torch.bincount(spread.route_ids(1024, 8, 0, num_experts=16, topk=1).squeeze(1), minlength=16)
    assert packed_counts[:2].sum() == 768
    assert spread_counts[[0, 2]].sum() == 768
    assert packed_counts.reshape(8, 2).sum(1).max() > spread_counts.reshape(8, 2).sum(1).max()
    assert RoutingCase("hotspot", 4, 1., "spread", 17).hot_ids(8, num_experts=16) == [0, 2, 4, 6]


def test_random_routes_are_reproducible_and_vary_by_seed_and_rank():
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    case = RoutingCase("random", seed=17)
    ids = case.route_ids(4096, 8, 0, num_experts=16, topk=1).squeeze(1)
    torch.testing.assert_close(ids, case.route_ids(4096, 8, 0, num_experts=16, topk=1).squeeze(1))
    assert not torch.equal(ids, case.route_ids(4096, 8, 1, num_experts=16, topk=1).squeeze(1))
    assert not torch.equal(ids, RoutingCase("random", seed=29).route_ids(4096, 8, 0, num_experts=16, topk=1).squeeze(1))
    assert torch.bincount(ids, minlength=16).min() > 0
    assert not torch.equal(torch.bincount(ids, minlength=16), torch.full((16,), 256))


@pytest.mark.parametrize("ep_size", [2, 8])
def test_bailing_uniform_routes_keep_256_experts_and_eight_distinct_choices(ep_size):
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    ids = RoutingCase("balanced").route_ids(1024, ep_size, 0, num_experts=256, topk=8)
    assert ids.shape == (1024, 8)
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
    torch.testing.assert_close(torch.bincount(ids.flatten(), minlength=256), torch.full((256,), 32))


@pytest.mark.parametrize("ep_size,expected_ids", [(2, [0, 128, 1, 129]), (8, [0, 32, 64, 96])])
def test_bailing_spread_hotspots_follow_actual_expert_ownership(ep_size, expected_ids):
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    case = RoutingCase("hotspot", 4, 0.0625, "spread", 17)
    assert case.hot_ids(ep_size, num_experts=256) == expected_ids


@pytest.mark.parametrize("num_hot", [1, 2, 4])
@pytest.mark.parametrize("multiplier,expected_per_hot", [(2, 64), (4, 128), (8, 256), (32, 1024)])
def test_bailing_hotspot_multipliers_count_routes_without_repeating_experts(num_hot, multiplier, expected_per_hot):
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    case = RoutingCase("hotspot", num_hot, multiplier * num_hot / 256, "packed", 17)
    ids = case.route_ids(1024, 8, 3, num_experts=256, topk=8)
    assert ids.shape == (1024, 8)
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
    counts = torch.bincount(ids.flatten(), minlength=256)
    torch.testing.assert_close(counts[:num_hot], torch.full((num_hot,), expected_per_hot))
    assert counts[num_hot:].max() - counts[num_hot:].min() <= 1
    assert counts.sum() == 8192


def test_bailing_random_routes_sample_without_replacement_and_record_full_configuration():
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase

    case = RoutingCase("random", seed=17)
    ids = case.route_ids(1024, 8, 0, num_experts=256, topk=8)
    assert ids.shape == (1024, 8)
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
    assert torch.bincount(ids.flatten(), minlength=256).min() > 0
    torch.testing.assert_close(ids, case.route_ids(1024, 8, 0, num_experts=256, topk=8))
    assert not torch.equal(ids, case.route_ids(1024, 8, 1, num_experts=256, topk=8))
    metadata = RoutingCase("hotspot", 2, 0.03125, "spread", 17).metadata(8, num_experts=256, topk=8)
    assert metadata["num_experts"] == 256
    assert metadata["topk"] == 8
    assert metadata["local_experts"] == 32
    assert metadata["hot_multiplier"] == 4


def test_topk_limits_hotspot_shares_to_distinct_experts():
    from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase, routing_cases

    with pytest.raises(ValueError, match="distinct"):
        RoutingCase("hotspot", 1, 0.25, "packed", 17).route_ids(1024, 8, 0, num_experts=256, topk=8)
    cases = routing_cases(
        8, ["hotspot"], [17], [1, 2, 4], [], ["packed", "spread"],
        num_experts=256, topk=8, hot_multipliers=[2, 4, 32, 64],
    )
    assert len(cases) == 15
    assert all(case.hot_fraction <= case.num_hot_experts / 8 for case in cases)
