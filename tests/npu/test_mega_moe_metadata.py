"""NPU regression coverage for shape-independent MegaMoE route statistics."""
import os
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.metadata import route_summary


@pytest.fixture(scope="module")
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    device = int(os.environ.get("MEGA_MOE_TEST_DEVICE", "0"))
    torch.npu.set_device(device)
    return torch.device("npu", device)


def reference(counts):
    counts = counts.to(torch.int64)
    ranks, experts = counts.shape
    totals = counts.sum(0)
    return totals.reshape(ranks, experts // ranks).sum(1).tolist() + [
        totals.max().item(), counts.max().item(), (totals > 0).sum().item()
    ]


@pytest.mark.parametrize("ranks,experts", [
    (2, 6), (3, 99), (8, 512), (16, 512), (32, 256),
    (64, 128), (64, 320), (64, 384), (64, 448),
    (128, 384), (128, 8192), (256, 16384), (257, 1799),
])
@pytest.mark.parametrize("pattern", ["random", "empty", "hot"])
def test_route_summary_shapes(npu_device, ranks, experts, pattern):
    generator = torch.Generator().manual_seed(ranks + experts)
    counts = torch.randint(0, 4096, (ranks, experts), dtype=torch.int32, generator=generator)
    if pattern == "empty":
        counts.zero_()
    elif pattern == "hot":
        counts.zero_()
        counts[:, 0] = 4096
    assert route_summary(counts.to(npu_device)) == reference(counts)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
def test_route_summary_strides(npu_device, dtype, layout):
    counts = torch.arange(6 * 198, dtype=dtype).reshape(6, 198)
    if layout == "transpose":
        actual = counts.T.contiguous().to(npu_device).T
    elif layout == "slice":
        storage = torch.zeros((12, 396), dtype=dtype)
        storage[::2, ::2] = counts
        actual = storage.to(npu_device)[::2, ::2]
    else:
        counts = counts[:1].expand(6, -1)
        actual = counts[:1].to(npu_device).expand(6, -1)
    assert route_summary(actual) == reference(counts)


@pytest.mark.parametrize("dtype,value", [(torch.int32, 2**30), (torch.int64, 2**34)])
def test_route_summary_integer_overflow(npu_device, dtype, value):
    counts = torch.full((8, 24), value, dtype=dtype)
    assert route_summary(counts.to(npu_device)) == reference(counts)


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (4, 0), (3, 8), (8,), (2, 4, 1)])
def test_route_summary_invalid_shape(npu_device, shape):
    with pytest.raises(ValueError):
        route_summary(torch.zeros(shape, device=npu_device, dtype=torch.int32))


def test_route_summary_invalid_dtype(npu_device):
    with pytest.raises(TypeError, match="int32 or int64"):
        route_summary(torch.zeros((2, 4), device=npu_device))


def test_route_summary_cpu_rejected():
    with pytest.raises(ValueError, match="NPU"):
        route_summary(torch.zeros((2, 4), dtype=torch.int32))

def test_ds41_distinct_topk(npu_device):
    counts = torch.zeros((64, 384), dtype=torch.int32)
    for rank in range(64):
        counts[rank, (rank % 8) * 6:(rank % 8 + 1) * 6] = 4096
    assert route_summary(counts.to(npu_device)) == reference(counts)


@pytest.mark.parametrize("ranks,experts", [(2, 6), (3, 99), (8, 512), (64, 384), (128, 1536)])
@pytest.mark.parametrize("pattern", ["random", "empty", "hot"])
def test_communication_metadata(npu_device, ranks, experts, pattern):
    from types import SimpleNamespace
    from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.metadata import compute_metadata

    counts = torch.randint(0, 101, (ranks, experts), dtype=torch.int32)
    if pattern != "random":
        counts.zero_()
        if pattern == "hot":
            counts[:, 0] = 4096
    device_counts = counts.to(npu_device)
    local = experts // ranks
    row_offsets = counts.long().cumsum(1) - counts
    source_offsets = counts.long().cumsum(0) - counts
    totals = counts.long().sum(0)
    local_offsets = (totals.reshape(ranks, local).cumsum(1) - totals.reshape(ranks, local)).flatten()
    for rank in {0, ranks - 1}:
        start, end = rank * local, (rank + 1) * local
        source = (source_offsets[:, start:end] + local_offsets[start:end]).flatten() * 2560
        target = row_offsets[:, start:end].flatten() * 2560
        sizes = counts[:, start:end].flatten() * 2560
        for mode in ("push", "pull"):
            spec = SimpleNamespace(ep_size=ranks, num_experts=experts, local_experts=local,
                                   rank_id=rank, hidden_size=2560, dispatch_mode=mode)
            actual = compute_metadata(device_counts[rank], device_counts,
                                      device_counts[:, start:end], spec, 4096)
            expected = {
                "combine_src_off": source, "combine_target_off": target, "combine_size": sizes,
                "dispatch_src_off": target if mode == "pull" else row_offsets[rank] * 2560,
                "dispatch_target_off": source if mode == "pull" else (local_offsets + source_offsets[rank]) * 2560,
                "dispatch_size": sizes if mode == "pull" else counts[rank] * 2560,
                "group_list": totals[start:end].cumsum(0),
            }
            for name, value in expected.items():
                torch.testing.assert_close(getattr(actual, name).cpu(), value, rtol=0, atol=0, check_dtype=False)


@pytest.mark.parametrize("ranks,experts", [(2, 6), (3, 12), (8, 64)])
@pytest.mark.parametrize("copied", [False, True, None])
@pytest.mark.parametrize("native", [False, True])
def test_virtual_route_render(npu_device, monkeypatch, ranks, experts, copied, native):
    from torchturbo.optimizations.mega_moe.moonep import render

    if native:
        monkeypatch.setattr(render, "_device_ub_bytes", lambda _: 1)
    generator = torch.Generator().manual_seed(714)
    counts = torch.randint(0, 20, (ranks, experts), dtype=torch.int32, generator=generator)
    local = experts // ranks
    allocation = torch.zeros_like(counts)
    copies = torch.full((ranks, local), -1, dtype=torch.int32)
    for expert in range(experts):
        home = expert // local
        total = counts[:, expert].sum().item()
        transfer = total // 2 if copied else 0
        allocation[home, expert] = total - transfer
        if copied:
            destination = (home + 1) % ranks
            allocation[destination, expert] = transfer
            copies[destination, expert % local] = expert
    cumulative = allocation.cumsum(0, dtype=torch.int32).T.contiguous()
    extra = local if copied is not False else 0
    width = local + extra
    source_starts = counts.cumsum(0, dtype=torch.int32) - counts
    for rank in (0, ranks - 1):
        flat = torch.repeat_interleave(torch.arange(experts), counts[rank])
        flat = flat[torch.randperm(flat.numel(), generator=generator)]
        inverse = torch.empty_like(flat)
        inverse[flat.argsort(stable=True)] = torch.arange(flat.numel())
        ids = flat.int().reshape(-1, 1)
        actual, actual_counts = render.render_route(
            ids.to(npu_device), inverse.int().to(npu_device), counts[rank].to(npu_device),
            counts.to(npu_device), cumulative.to(npu_device), copies.to(npu_device), rank, extra,
        )
        expected = torch.empty_like(ids)
        expected_counts = torch.zeros((ranks, ranks * width), dtype=torch.int32)
        for source in range(ranks):
            for expert in range(experts):
                begin = source_starts[source, expert].item()
                end = begin + counts[source, expert].item()
                for destination in range(ranks):
                    left = cumulative[expert, destination - 1].item() if destination else 0
                    right = cumulative[expert, destination].item()
                    overlap = max(0, min(end, right) - max(begin, left))
                    if overlap:
                        slot = expert % local if expert // local == destination else local + expert % local
                        expected_counts[source, destination * width + slot] = overlap
        for expert in range(experts):
            rows = (flat == expert).nonzero().flatten()
            for ordinal, row in enumerate(rows):
                global_ordinal = source_starts[rank, expert].item() + ordinal
                destination = int((global_ordinal >= cumulative[expert]).sum())
                slot = expert % local if expert // local == destination else local + expert % local
                expected[row, 0] = destination * width + slot
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_counts.cpu(), expected_counts, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_zero_empty_gradients_preserves_active_rows(npu_device, dtype):
    from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.gradient import zero_empty_expert_gradients

    first = torch.full((5, 7, 9), float("nan"), dtype=dtype, device=npu_device)
    second = torch.full((5, 5, 11), float("nan"), dtype=dtype, device=npu_device)
    groups = torch.tensor([0, 3, 3, 4, 4], dtype=torch.int64, device=npu_device)
    zero_empty_expert_gradients(first, second, groups)
    for gradient in (first.cpu(), second.cpu()):
        assert (gradient[[0, 2, 4]] == 0).all()
        assert gradient[[1, 3]].isnan().all()
