"""NPU fused DeepSeek-V4 operators."""

from functools import partialmethod

import torch

from torchturbo.ops import ops

from .packing_utils import _segment_spans


def patch_deepseek_v4_npu_sfa_li(sp_rank: int, sp_group, sp_size: int):
    """Install fused NPU sparse-attention and lightning-indexer forwards."""
    from . import modeling_deepseek_v4 as dsv4

    dsv4.SparseAttention.forward = partialmethod(_npu_sfa, sp_rank=sp_rank)
    dsv4.LiCompute.forward = partialmethod(
        _npu_li,
        sp_rank=sp_rank,
        sp_group=sp_group,
        sp_size=sp_size,
    )


def _get_li_query_tnd_data(segment_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = []
    cu_seqlens = [0]
    seq_len = segment_ids.size(1)
    for batch_idx in range(segment_ids.size(0)):
        # _segment_spans already keeps positive-ID spans only.
        starts, ends, _ = _segment_spans(segment_ids[batch_idx])
        for start, end in zip(starts.tolist(), ends.tolist()):
            indices.append(
                torch.arange(
                    batch_idx * seq_len + start,
                    batch_idx * seq_len + end,
                    device=segment_ids.device,
                )
            )
            cu_seqlens.append(cu_seqlens[-1] + end - start)
    if not indices:
        return segment_ids.new_empty((0,), dtype=torch.long), segment_ids.new_tensor(cu_seqlens, dtype=torch.int32)
    return (
        torch.cat(indices),
        segment_ids.new_tensor(cu_seqlens, dtype=torch.int32),
    )


def _segment_block_info(row: torch.Tensor, ratio: int) -> dict[int, tuple[int, int, int, int]]:
    starts, ends, segment_ids = _segment_spans(row)
    block_counts = ((ends - starts) // ratio).long()
    block_starts = torch.cumsum(block_counts, dim=0) - block_counts
    return {
        int(segment_id): (int(start), int(end), int(block_start), int(block_count))
        for start, end, segment_id, block_start, block_count in zip(
            starts.tolist(),
            ends.tolist(),
            segment_ids.tolist(),
            block_starts.tolist(),
            block_counts.tolist(),
        )
    }


def _get_compressed_kv_tnd_data(
    kv_compress: torch.Tensor,
    query_segment_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    local_start: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = []
    cu_seqlens = [0]
    residuals = []
    tnd_to_block = []
    for batch_idx in range(query_segment_ids.size(0)):
        block_info = _segment_block_info(segment_ids[batch_idx], ratio)
        _, ends, span_ids = _segment_spans(query_segment_ids[batch_idx])
        for end, segment_id in zip(ends.tolist(), span_ids.tolist()):
            if segment_id > 0:
                segment_start, _, block_start, _ = block_info[segment_id]
                visible_blocks = (local_start + end - segment_start) // ratio
                block_indices = torch.arange(
                    block_start,
                    block_start + visible_blocks,
                    device=segment_ids.device,
                )
                packed.append(kv_compress[batch_idx][block_indices])
                tnd_to_block.append(block_indices)
                cu_seqlens.append(cu_seqlens[-1] + visible_blocks)
                residuals.append((local_start + end - segment_start) % ratio)
    if not packed:
        return (
            kv_compress.new_empty((0, *kv_compress.shape[2:])),
            segment_ids.new_tensor(cu_seqlens, dtype=torch.int32),
            torch.empty(0, dtype=torch.long, device=segment_ids.device),
            segment_ids.new_empty((0,), dtype=torch.int32),
        )
    return (
        torch.cat(packed, dim=0),
        segment_ids.new_tensor(cu_seqlens, dtype=torch.int32),
        torch.cat(tnd_to_block),
        segment_ids.new_tensor(residuals, dtype=torch.int32),
    )


def _get_sfa_query_kv_tnd_data(
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    segment_ids: torch.Tensor,
    sp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_indices = []
    kv_indices = []
    cu_seqlens_q = [0]
    cu_seqlens_kv = [0]
    bsz, local_len = query_states.shape[:2]
    global_len = kv_states.size(1)
    local_start = sp_rank * local_len
    local_segment_ids = segment_ids[:, local_start : local_start + local_len]

    for batch_idx in range(bsz):
        block_info = _segment_block_info(segment_ids[batch_idx], 1)
        starts, ends, span_ids = _segment_spans(local_segment_ids[batch_idx])
        for start, end, segment_id in zip(
            starts.tolist(), ends.tolist(), span_ids.tolist()
        ):
            segment_start, _, _, _ = block_info[segment_id]
            kv_pos = torch.arange(
                segment_start, local_start + end, device=segment_ids.device
            )
            query_indices.append(
                torch.arange(
                    batch_idx * local_len + start,
                    batch_idx * local_len + end,
                    device=segment_ids.device,
                )
            )
            kv_indices.append(batch_idx * global_len + kv_pos)
            cu_seqlens_q.append(cu_seqlens_q[-1] + end - start)
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + kv_pos.numel())

    if not query_indices:
        empty = segment_ids.new_empty((0,), dtype=torch.long)
        return (
            empty,
            empty,
            segment_ids.new_tensor(cu_seqlens_q, dtype=torch.int32),
            segment_ids.new_tensor(cu_seqlens_kv, dtype=torch.int32),
            local_segment_ids,
        )
    return (
        torch.cat(query_indices),
        torch.cat(kv_indices),
        segment_ids.new_tensor(cu_seqlens_q, dtype=torch.int32),
        segment_ids.new_tensor(cu_seqlens_kv, dtype=torch.int32),
        local_segment_ids,
    )


def _get_sfa_compressed_kv_tnd_data(
    kv_compress: torch.Tensor,
    compress_topk_idxs: torch.Tensor,
    segment_ids: torch.Tensor,
    local_segment_ids: torch.Tensor,
    local_start: int,
    offset: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    build_csi = ratio == 4
    packed = []
    sparse_indices = []
    residuals = []
    cu_seqlens = [0]
    for batch_idx in range(local_segment_ids.size(0)):
        block_info = _segment_block_info(segment_ids[batch_idx], ratio)
        starts, ends, span_ids = _segment_spans(local_segment_ids[batch_idx])
        for start, end, segment_id in zip(
            starts.tolist(), ends.tolist(), span_ids.tolist()
        ):
            segment_start, _, block_start, _ = block_info[segment_id]
            visible_tokens = local_start + end - segment_start
            visible_blocks = visible_tokens // ratio
            block_indices = torch.arange(
                block_start,
                block_start + visible_blocks,
                device=segment_ids.device,
            )
            packed.append(kv_compress[batch_idx][block_indices])
            if build_csi:
                block_to_tnd = segment_ids.new_full(
                    (kv_compress.size(1),), -1, dtype=torch.long
                )
                block_to_tnd[block_indices] = torch.arange(
                    block_indices.numel(), device=block_indices.device
                )
                cmp_idx = compress_topk_idxs[batch_idx, start:end] - offset
                valid = cmp_idx >= 0
                cmp_idx = block_to_tnd[cmp_idx.clamp(0, block_to_tnd.size(0) - 1)]
                sparse_indices.append(torch.where(valid & (cmp_idx >= 0), cmp_idx, -1))
            cu_seqlens.append(cu_seqlens[-1] + visible_blocks)
            residuals.append(visible_tokens % ratio)

    return (
        torch.cat(packed, dim=0).unsqueeze(1).contiguous(),
        segment_ids.new_tensor(cu_seqlens, dtype=torch.int32),
        torch.cat(sparse_indices, dim=0).unsqueeze(1).int().contiguous() if build_csi else None,
        segment_ids.new_tensor(residuals, dtype=torch.int32),
    )


def _npu_sfa(
    self,
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    q_global_positions: torch.LongTensor,
    segment_ids: torch.Tensor,
    kv_compress: torch.Tensor = None,
    compress_topk_idxs: torch.Tensor = None,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Fused SFA forward."""
    query_indices, kv_indices, cu_seqlens_q, cu_seqlens_kv, local_segment_ids = (
        _get_sfa_query_kv_tnd_data(query_states, kv_states, segment_ids, sp_rank)
    )
    if query_indices.numel() == 0:
        output = query_states * 0
        output = output + (kv_states.sum() + attn_sink.sum()) * 0
        if kv_compress is not None:
            output = output + kv_compress.sum() * 0
        return output
    _, local_len, num_heads, head_dim = query_states.shape
    query_tnd = query_states.reshape(-1, num_heads, head_dim).index_select(0, query_indices).contiguous()
    kv_tnd = kv_states.reshape(-1, head_dim).index_select(0, kv_indices).unsqueeze(1).contiguous()
    cmp_kv_tnd = None
    cmp_sparse_indices = None
    cu_seqlens_cmp_kv = None
    cmp_residual_kv = None
    if kv_compress is not None and self.compress_ratio in (4, 128):
        offset = kv_states.size(1)
        # Only CSA needs LI indices; HCA attends all visible compressed blocks.
        if self.compress_ratio == 4 and compress_topk_idxs is None:
            compress_topk_idxs = self.get_compress_topk_idxs(
                query_states,
                offset,
                q_global_positions=q_global_positions,
                segment_ids=segment_ids,
            )
        cmp_kv_tnd, cu_seqlens_cmp_kv, cmp_sparse_indices, cmp_residual_kv = (
            _get_sfa_compressed_kv_tnd_data(
                kv_compress,
                compress_topk_idxs,
                segment_ids,
                local_segment_ids,
                sp_rank * local_len,
                offset,
                self.compress_ratio,
            )
        )
        if cmp_sparse_indices is not None:
            # The gradient kernel requires unused CSI slots to remain -1.
            if cmp_sparse_indices.size(-1) < 512:
                cmp_sparse_indices = torch.nn.functional.pad(
                    cmp_sparse_indices,
                    (0, 512 - cmp_sparse_indices.size(-1)),
                    value=-1,
                )
            elif cmp_sparse_indices.size(-1) > 512:
                cmp_sparse_indices = cmp_sparse_indices[..., :512]
    # CSA (ratio=4) passes CSI; HCA uses all visible compressed KV.
    scene_csi = cmp_sparse_indices
    output_tnd = ops.sparse_flash_mla(
        query=query_tnd,
        ori_kv=kv_tnd,
        cmp_kv=cmp_kv_tnd,
        cmp_sparse_indices=scene_csi,
        cmp_residual_kv=cmp_residual_kv,
        sinks=attn_sink.float(),
        softmax_scale=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
        ori_win_left=self.window_size - 1,
        ori_win_right=0,
        cu_seq_lens_q=cu_seqlens_q,
        cu_seq_lens_ori_kv=cu_seqlens_kv,
        cu_seq_lens_cmp_kv=cu_seqlens_cmp_kv,
    )
    output = query_states.new_zeros(query_states.shape)
    output.reshape(-1, num_heads, head_dim)[query_indices] = output_tnd
    return output


def _npu_li(
    self,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    seqlen: int,
    offset: int,
    q_global_positions: torch.LongTensor,
    segment_ids: torch.Tensor,
    sp_rank: int = 0,
    sp_group=None,
    sp_size: int = 1,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Fused lightning-indexer forward."""
    bsz, local_len, num_heads, head_dim = q_indexer.shape
    local_start = sp_rank * local_len
    local_segment_ids = segment_ids[:, local_start : local_start + local_len]
    query_indices, cu_seqlens_q = _get_li_query_tnd_data(local_segment_ids)
    if query_indices.numel() == 0:
        return None, None
    n_seqs = cu_seqlens_q.numel() - 1
    query_tnd = (
        q_indexer.reshape(-1, num_heads, head_dim)
        .index_select(0, query_indices)
        .detach()
        .to(torch.bfloat16)
        .contiguous()
    )
    weights_tnd = (
        weights.reshape(-1, num_heads)
        .index_select(0, query_indices)
        .detach()
        .float()
        .contiguous()
    )
    k_tnd, cu_seqlens_k, tnd_to_block, cmp_residual_k = _get_compressed_kv_tnd_data(
        k_indexer,
        local_segment_ids,
        segment_ids,
        local_start,
        self.ratio,
    )
    sparse_count = min(self.index_topk, k_indexer.size(1))
    compress_topk_idxs = segment_ids.new_full(
        (bsz, local_len, sparse_count), -1, dtype=torch.long
    )
    if k_tnd.size(0) == 0:
        return compress_topk_idxs, None
    k_tnd = k_tnd.detach().to(torch.bfloat16).unsqueeze(1).contiguous()
    max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if sp_size > 1:
        # The CANN SP prefix view uses the total local query length.
        max_seqlen_q = query_tnd.size(0)
    # TND batch size equals the number of segments.
    metadata = ops.lightning_indexer_metadata(
        num_heads_q=num_heads,
        num_heads_k=1,
        head_dim=head_dim,
        topk=sparse_count,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=n_seqs,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()),
        layout_q="TND",
        layout_k="TND",
        mask_mode=3,
        cmp_ratio=self.ratio,
    )
    indices, _ = ops.lightning_indexer(
        query_tnd,
        k_tnd,
        weights_tnd,
        sparse_count,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        max_seqlen_q=-1,
        layout_q="TND",
        layout_k="TND",
        mask_mode=3,
        cmp_ratio=self.ratio,
        return_value=0,
    )

    # Convert segment-local indices to global compressed-block indices.
    local = indices[:, 0, :]
    segment_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    key_offsets = torch.repeat_interleave(cu_seqlens_k[:-1], segment_lengths)
    rows = local.long() + key_offsets.unsqueeze(1)
    valid = local >= 0
    valid = valid & (rows < tnd_to_block.size(0))
    safe_rows = rows.clamp(max=tnd_to_block.size(0) - 1)
    mapped_block_indices = tnd_to_block[safe_rows]
    valid = valid & (mapped_block_indices >= 0)
    block_indices = torch.where(valid, mapped_block_indices + offset, -1)
    compress_topk_idxs.reshape(-1, sparse_count)[query_indices] = block_indices
    return compress_topk_idxs, None
