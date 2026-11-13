"""Verify arbitrary packed segment lengths are preserved."""

import pytest

torch = pytest.importorskip("torch")

from torchturbo.models.deepseek_v4.npu import (
    _get_compressed_kv_tnd_data,
    _get_li_query_tnd_data,
    _get_sfa_compressed_kv_tnd_data,
    _get_sfa_query_kv_tnd_data,
)
from torchturbo.models.deepseek_v4.packing_utils import (
    _segment_spans,
    compress_packed_kv,
    packed_compress_kv_ids_and_positions,
    packed_compress_mask,
)


def test_compression_keeps_unaligned_segment_boundaries():
    mask = torch.tensor([[1] * 127 + [2] * 129], dtype=torch.int32)
    positions = torch.tensor([list(range(127)) + list(range(129))])

    block_ids, block_positions = packed_compress_kv_ids_and_positions(
        mask, 128, positions
    )

    assert block_ids.tolist() == [[2, 0]]
    assert block_positions.tolist() == [[0, 0]]


def test_compression_metadata_handles_padding_and_multiple_rows():
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2],
            [0, 0, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    positions = torch.arange(mask.size(1)).unsqueeze(0).expand_as(mask)

    block_ids, block_positions = packed_compress_kv_ids_and_positions(
        mask, 4, positions
    )

    assert block_ids.tolist() == [[1, 2, 0], [3, 0, 0]]
    assert block_positions.tolist() == [[0, 7, 0], [2, 0, 0]]


@pytest.mark.parametrize("sp_rank", [0, 1])
def test_compressed_mask_uses_physical_block_ends_across_sp_shards(sp_rank):
    segment_ids = torch.tensor([[1] * 3 + [2] * 9 + [0] * 2])
    positions = torch.arange(14).unsqueeze(0)
    block_ids, block_positions = packed_compress_kv_ids_and_positions(
        segment_ids, 4, positions
    )
    local_slice = slice(sp_rank * 7, (sp_rank + 1) * 7)

    mask = packed_compress_mask(
        segment_ids[:, local_slice],
        block_ids.unsqueeze(1),
        block_positions.unsqueeze(1),
        4,
        positions[:, local_slice],
    )

    # The second segment starts at offset 3; its blocks end at offsets 6 and
    # 10. Its visibility cannot be rounded to multiples of 4 in the full row.
    expected_visible = torch.tensor(
        [[False, False, False]] * 6
        + [[True, False, False]] * 4
        + [[True, True, False]] * 2
        + [[False, False, False]] * 2
    ).unsqueeze(0)
    torch.testing.assert_close(~mask, expected_visible[:, local_slice])


def _reference_compress_packed_kv(kv, score, ape, mask, ratio, head_dim, overlap):
    rows = []
    for batch_idx, row_mask in enumerate(mask):
        segments = []
        starts, ends, _ = _segment_spans(row_mask)
        for start, end in zip(starts.tolist(), ends.tolist()):
            cutoff = start + ((end - start) // ratio) * ratio
            if cutoff == start:
                continue
            segment_kv = kv[batch_idx : batch_idx + 1, start:cutoff].unflatten(
                1, (-1, ratio)
            )
            segment_score = (
                score[batch_idx : batch_idx + 1, start:cutoff].unflatten(
                    1, (-1, ratio)
                )
                + ape
            )
            if overlap:
                num_blocks = segment_kv.size(1)
                overlap_kv = segment_kv.new_zeros(
                    (1, num_blocks, 2 * ratio, head_dim)
                )
                overlap_score = segment_score.new_full(
                    (1, num_blocks, 2 * ratio, head_dim), float("-inf")
                )
                overlap_kv[:, :, ratio:] = segment_kv[:, :, :, head_dim:]
                overlap_kv[:, 1:, :ratio] = segment_kv[:, :-1, :, :head_dim]
                overlap_score[:, :, ratio:] = segment_score[:, :, :, head_dim:]
                overlap_score[:, 1:, :ratio] = segment_score[
                    :, :-1, :, :head_dim
                ]
                segment_kv, segment_score = overlap_kv, overlap_score
            segments.append(
                (segment_kv * segment_score.softmax(dim=2)).sum(dim=2).squeeze(0)
            )
        rows.append(
            torch.cat(segments, dim=0)
            if segments
            else kv.new_empty((0, head_dim))
        )

    capacity = mask.size(1) // ratio
    output = kv.new_zeros((kv.size(0), capacity, head_dim))
    for batch_idx, row in enumerate(rows):
        output[batch_idx, : row.size(0)] = row
    return output


@pytest.mark.parametrize("ratio,overlap", [(4, True), (128, False)])
def test_vectorized_compressor_matches_segment_loop(ratio, overlap):
    if ratio == 4:
        rows = [
            [1] * 7 + [0] * 2 + [2] * 9 + [0] * 2,
            [0] + [3] * 12 + [0] * 2 + [4] * 5,
        ]
    else:
        rows = [
            [1] * 127 + [2] * 129 + [0] * 8,
            [0] * 3 + [3] * 260 + [0],
        ]
    mask = torch.tensor(rows, dtype=torch.long)
    positions = torch.arange(mask.size(1)).unsqueeze(0).expand_as(mask)
    head_dim = 4
    projected_dim = head_dim * (2 if overlap else 1)
    generator = torch.Generator().manual_seed(ratio)
    kv = torch.randn(
        mask.size(0),
        mask.size(1),
        projected_dim,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    score = torch.randn(
        kv.shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    ape = torch.randn(
        ratio,
        projected_dim,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )

    actual, _ = compress_packed_kv(
        kv, score, ape, mask, positions, ratio, head_dim, overlap
    )
    expected = _reference_compress_packed_kv(
        kv, score, ape, mask, ratio, head_dim, overlap
    )
    torch.testing.assert_close(actual, expected)

    grad_output = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    expected_grads = torch.autograd.grad(
        expected, (kv, score, ape), grad_output, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual, (kv, score, ape), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_compression_static_capacity_masks_rows_without_complete_blocks():
    mask = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long)
    positions = torch.arange(mask.size(1)).unsqueeze(0)
    kv = torch.randn(1, mask.size(1), 4, requires_grad=True)
    score = torch.randn_like(kv, requires_grad=True)
    ape = torch.randn(4, 4, requires_grad=True)

    compressed, block_positions = compress_packed_kv(
        kv, score, ape, mask, positions, 4, 4, overlap=False
    )

    assert compressed.shape == (1, 1, 4)
    assert block_positions.tolist() == [[0]]
    assert not compressed.any()


def test_li_query_metadata_ignores_zero_id_spans():
    mask = torch.tensor([[0, 1, 1, 0, 2, 2, 2]], dtype=torch.long)

    indices, cu_seqlens = _get_li_query_tnd_data(mask)

    assert indices.tolist() == [1, 2, 4, 5, 6]
    assert cu_seqlens.tolist() == [0, 2, 5]


@pytest.mark.parametrize(
    ("sp_rank", "expected_q", "expected_kv"),
    [
        (0, [0, 127, 128], [0, 127, 128]),
        (1, [0, 128], [0, 129]),
    ],
)
def test_sfa_tnd_boundaries_cross_sp_shards(sp_rank, expected_q, expected_kv):
    mask = torch.tensor([[1] * 127 + [2] * 129], dtype=torch.int32)
    query = torch.empty(1, 128, 1, 8)
    kv = torch.empty(1, 256, 8)

    _, _, cu_seqlens_q, cu_seqlens_kv, _ = _get_sfa_query_kv_tnd_data(
        query, kv, mask, sp_rank
    )

    assert cu_seqlens_q.tolist() == expected_q
    assert cu_seqlens_kv.tolist() == expected_kv


@pytest.mark.parametrize(
    ("sp_rank", "expected_cu_seqlens", "expected_residuals"),
    [
        (0, [0, 0, 0], [127, 1]),
        (1, [0, 1], [1]),
    ],
)
def test_sfa_preserves_partial_compression_blocks(
    sp_rank, expected_cu_seqlens, expected_residuals
):
    mask = torch.tensor([[1] * 127 + [2] * 129], dtype=torch.int32)
    local_mask = mask[:, sp_rank * 128 : (sp_rank + 1) * 128]
    kv_compress = torch.empty(1, 1, 8)
    topk = torch.empty(1, 128, 0, dtype=torch.long)

    _, cu_seqlens, _, residuals = _get_sfa_compressed_kv_tnd_data(
        kv_compress,
        topk,
        mask,
        local_mask,
        sp_rank * 128,
        256,
        128,
    )

    assert cu_seqlens.tolist() == expected_cu_seqlens
    assert residuals.tolist() == expected_residuals


def _multirow_segment_ids(ratio):
    return torch.tensor(
        [
            [1] * (ratio + 1) + [2] * (2 * ratio + 1) + [0] * 2,
            [0] * 2 + [3] * (2 * ratio + 1) + [4] * (ratio + 1),
        ],
        dtype=torch.long,
    )


@pytest.mark.parametrize(
    "sp_rank,expected_q,expected_kv,expected_cu_q,expected_cu_kv",
    [
        (0, [*range(8), *range(10, 16)], [*range(8), *range(18, 24)], [0, 5, 8, 14], [0, 5, 8, 14]),
        (1, [*range(6), *range(8, 16)], [*range(5, 14), *range(18, 32)], [0, 6, 9, 14], [0, 9, 18, 23]),
    ],
)
def test_sfa_tnd_indices_preserve_multiple_rows_and_sp_history(
    sp_rank, expected_q, expected_kv, expected_cu_q, expected_cu_kv
):
    segment_ids = _multirow_segment_ids(4)
    query = torch.empty(2, 8, 1, 2)
    kv = torch.empty(2, 16, 2)

    query_indices, kv_indices, cu_q, cu_kv, local_ids = _get_sfa_query_kv_tnd_data(
        query, kv, segment_ids, sp_rank
    )

    assert query_indices.tolist() == expected_q
    assert kv_indices.tolist() == expected_kv
    assert cu_q.tolist() == expected_cu_q
    assert cu_kv.tolist() == expected_cu_kv
    torch.testing.assert_close(local_ids, segment_ids[:, sp_rank * 8 : (sp_rank + 1) * 8])


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("sp_rank", [0, 1])
def test_compressed_tnd_values_preserve_multiple_rows_and_sp_history(ratio, sp_rank):
    segment_ids = _multirow_segment_ids(ratio)
    global_len = segment_ids.size(1)
    local_len = global_len // 2
    local_start = sp_rank * local_len
    local_ids = segment_ids[:, local_start : local_start + local_len]
    compressed = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    topk = torch.tensor([global_len, global_len + 1, global_len + 2, -1])
    topk = topk.view(1, 1, 4).expand(2, local_len, -1)

    li_kv, li_cu, block_indices, li_residuals = _get_compressed_kv_tnd_data(
        compressed, local_ids, segment_ids, local_start, ratio
    )
    sfa_kv, sfa_cu, csi, sfa_residuals = _get_sfa_compressed_kv_tnd_data(
        compressed, topk, segment_ids, local_ids, local_start, global_len, ratio
    )

    if sp_rank == 0:
        expected_kv = torch.stack([compressed[0, 0], compressed[1, 0]])
        expected_cu = [0, 1, 1, 2]
        expected_blocks = [0, 0]
        expected_residuals = [1, ratio // 2 + 1, ratio // 2]
        sparse_rows = (
            [[0, -1, -1, -1]] * (ratio + 1)
            + [[-1, -1, -1, -1]] * (ratio // 2 + 1)
            + [[0, -1, -1, -1]] * (local_len - 2)
        )
    else:
        expected_kv = torch.cat([compressed[0, 1:], compressed[1]], dim=0)
        expected_cu = [0, 2, 4, 5]
        expected_blocks = [1, 2, 0, 1, 2]
        expected_residuals = [1, 1, 1]
        sparse_rows = (
            [[-1, 0, 1, -1]] * (local_len - 2)
            + [[0, 1, -1, -1]] * (ratio // 2 + 1)
            + [[-1, -1, 0, -1]] * (ratio + 1)
        )

    torch.testing.assert_close(li_kv, expected_kv)
    torch.testing.assert_close(sfa_kv, expected_kv.unsqueeze(1))
    assert li_cu.tolist() == sfa_cu.tolist() == expected_cu
    assert block_indices.tolist() == expected_blocks
    assert li_residuals.tolist() == sfa_residuals.tolist() == expected_residuals
    if ratio == 4:
        assert csi[:, 0].tolist() == sparse_rows
    else:
        assert csi is None
