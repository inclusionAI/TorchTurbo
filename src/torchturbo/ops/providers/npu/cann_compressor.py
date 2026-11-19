"""NPU (cann_ops_transformer) provider for the packed-KV compressor op.

Bridges the vendored Huawei CSA2 fused compressor (``cann_ops_transformer``
wheel: aclnn forward + backward fused kernels with autograd) into the
dispatcher. Any mask shape runs as a single TH (BS-merged) call: the valid
tokens are flattened into a ``(t, hidden)`` stream with an int32
``cu_seqlens`` prefix sum, and the vendor kernel walks every segment inside
one launch, forward and backward. Back-to-back packed segments skip the
flattening: the row-major view of x already is that TH stream. The kernel
fuses the bf16 wkv/wgate
projections and the per-channel softmax pooling; the RMSNorm tail (npu_rms_norm
with the fp32 weight) runs here because the vendor op stops at the pooled
latent. Gradients come
from the paired backward kernel through the vendor autograd Function.

The page-cache side of the vendor contract is scratch here: whole-segment
prefill never reads prior state, so the provider reproduces the vendor
harness page allocation (cache_mode=1) with a zero state sized to just the
pages the table references.

Auto selection prefers this provider on NPU;
``TORCH_TURBO_DISABLE_V41_FUSED_COMPRESSOR`` drops it from auto selection
(explicit ``impl=`` requests still work).
"""

from __future__ import annotations

import os

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op
from .compressor import _cu_seqlens, _segment_starts

# Vendor page-cache geometry. block_size matches the state layout the kernel
# indexes through the block table.
_VENDOR_BLOCK_SIZE = 128

# The all-zero positional bias is a per-(ratio, head_dim) constant the kernel
# only reads; cache it instead of reallocating on every call.
_ZERO_APE: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _zero_ape(ratio: int, head_dim: int, device: torch.device):
    key = (ratio, head_dim, device)
    ape = _ZERO_APE.get(key)
    if ape is None:
        ape = torch.zeros(ratio, head_dim, dtype=torch.float32, device=device)
        _ZERO_APE[key] = ape
    return ape


def _official_block_table(segment_lengths, ratio: int, block_size: int):
    """Vendor-harness page allocation (golden.py, cache_mode=1, start_pos=0).

    Whole-segment prefill never reads prior state, but the kernel's verified
    input domain is the harness allocation: pages assigned around each
    segment's (empty) read window and its write window at the last full block
    boundary. Returns the table (CPU) and the page count it references.
    """
    num_seg = len(segment_lengths)
    max_blocks = (max(segment_lengths) + block_size - 1) // block_size
    table = torch.zeros(num_seg, max_blocks, dtype=torch.int32)
    next_id = 1
    page_cap = num_seg * max_blocks
    for seg, end in enumerate(segment_lengths):
        # Read window around start_pos=0 (empty unless the segment is tiny).
        read_start = max(-ratio, 0)
        read_end = min(ratio, end)
        for col in range(read_start // block_size,
                         (read_end - 1) // block_size + 1):
            if next_id < page_cap:
                table[seg][col] = next_id
                next_id += 1
        # Write window at the last full pooling boundary.
        write_start = max(end // ratio * ratio - ratio, 0)
        write_end = end // ratio * ratio + ratio
        if end % ratio == 0:
            write_end = end
        write_end = min(write_end, end)
        for col in range(write_start // block_size,
                         (write_end - 1) // block_size + 1):
            if next_id < page_cap and table[seg][col] == 0:
                table[seg][col] = next_id
                next_id += 1
    return table, next_id


def _load_cot():
    # cann_ops_transformer is a container-side wheel; importing it lazily
    # keeps the provider scan working everywhere else.
    import cann_ops_transformer

    return cann_ops_transformer


class _Flatten(torch.autograd.Function):
    """Flatten packed rows for the vendor TH call with a one-kernel backward.

    The forward concatenates per-segment row slices: index-select kernels
    running before the vendor call corrupt it (aicore exception or NaN), so
    the flatten cannot be an index_select. Plain autograd through this cat
    emits one narrow-copy per segment on the backward pass; at production
    segment counts those small kernels dominate the op on NPU, so the
    backward is a single index_copy instead. Index kernels are safe after
    the vendor call: backward graphs run gathers past it every step.
    """

    @staticmethod
    def forward(ctx, x, segments):
        bsz, seq_len, hidden = x.shape
        flat = torch.cat(
            [x[row, start:start + length] for row, start, length in segments],
            dim=0,
        )
        # arange/cat stay in the index-free set that is safe before the
        # vendor call; this table maps flat rows onto rows of x.view(-1, h).
        pos = torch.cat([
            torch.arange(row * seq_len + start, row * seq_len + start + length)
            for row, start, length in segments
        ])
        ctx.x_shape = x.shape
        ctx.save_for_backward(pos.to(x.device))
        return flat

    @staticmethod
    def backward(ctx, grad_flat):
        (pos,) = ctx.saved_tensors
        bsz, seq_len, hidden = ctx.x_shape
        grad_x = torch.zeros(
            bsz, seq_len, hidden, dtype=grad_flat.dtype, device=grad_flat.device
        )
        grad_x.view(-1, hidden).index_copy_(0, pos, grad_flat)
        return grad_x, None


def _is_gap_free(segments, seq_len: int) -> bool:
    """True when segments sit back-to-back at their in-place row offsets.

    Segment k must start at its running token offset in the row-major view
    of the packed tensor, which is exactly where it starts in x. Interior
    rows are then fully packed; a trailing gap in the last row still
    qualifies because no segment references it.
    """
    offset = 0
    for row, start, length in segments:
        if row * seq_len + start != offset:
            return False
        offset += length
    return True


def _run_cann(
    cot,
    x: torch.Tensor,
    wkv_weight: torch.Tensor,
    wgate_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    attention_mask: torch.Tensor,
    ratio: int,
    head_dim: int,
    eps: float,
):
    """Run the fused TH call and scatter the pooled rows to packed layout.

    Mirrors the PyPTO provider's segment handling (one host sync for the
    segment list, compact output scattered back per row). Back-to-back
    segments skip the flattening entirely: the row-major view of x already
    is the TH stream, so nothing gathers before the vendor call and the
    backward is a free reshape-back.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    device = x.device
    bsz, seq_len, _ = x.shape
    # The vendor kernel wants bf16 weights (it computes the projections in
    # the input dtype). The checkpoint stores fp32; the cast node carries the
    # gradient back to fp32 on the backward pass.
    wkv_weight = wkv_weight.to(torch.bfloat16)
    wgate_weight = wgate_weight.to(torch.bfloat16)

    mask_cpu = attention_mask.detach().to("cpu", non_blocking=False)
    segments = _segment_starts(mask_cpu)
    n_seg = len(segments)
    blocks_per_seg = [length // ratio for _, _, length in segments]
    total_blocks = sum(blocks_per_seg)

    # Packed layout is [b, max_blocks, head_dim] over the largest per-row
    # block total; a row's segments concatenate their blocks from slot 0.
    blocks_per_row = [0] * bsz
    for (row, _, _), seg_blocks in zip(segments, blocks_per_seg):
        blocks_per_row[row] += seg_blocks
    max_blocks = max(blocks_per_row, default=0)
    if max_blocks == 0:
        return None

    # Gap-free fast path: back-to-back segments make the row-major view of
    # x the TH stream itself (a trailing gap only leaves rows past the end
    # unreferenced), so the flatten is a view and its backward a free
    # reshape-back instead of the cat/index_copy pair.
    gap_free = _is_gap_free(segments, seq_len)
    if gap_free:
        total_tokens = sum(length for _, _, length in segments)
        x_flat = x.reshape(-1, x.shape[-1])[:total_tokens]
    else:
        x_flat = _Flatten.apply(x, segments)
    cu_seqlens = _cu_seqlens(segments, device)

    # Page-cache provisioning (see _official_block_table): the harness
    # allocation for cache_mode=1 and a zero scratch state sized to exactly
    # the pages the table references. An all-zero table also produced exact
    # values in isolation, but it sits outside the kernel's verified input
    # domain and intermittently faults (MTE invalid GM) depending on the
    # allocator layout around it.
    segment_lengths = [length for _, _, length in segments]
    block_table, num_pages = _official_block_table(
        segment_lengths, ratio, _VENDOR_BLOCK_SIZE
    )
    block_table = block_table.to(device)
    state_cache = torch.zeros(
        num_pages, _VENDOR_BLOCK_SIZE, 2 * head_dim, dtype=torch.float32,
        device=device,
    )
    ape = _zero_ape(ratio, head_dim, device)

    # start_pos=None means position 0, the same intent as the zero tensor
    # it replaces, without the per-call allocation and H2D copy.
    out = cot.compressor(
        x_flat, wkv_weight, wgate_weight, state_cache, ape, ratio,
        state_block_table=block_table, cu_seqlens=cu_seqlens, seqused=None,
        start_pos=None, coff=1, cache_mode=1,
    )
    # The TH output is the upper-bound shape; the leading total_blocks rows
    # hold the compact stream (floor semantics: trailing tokens pool later).
    latent = out[:total_blocks]

    # RMSNorm tail on the compact stream, before the scatter. npu_rms_norm
    # takes the bf16 latent with the fp32 weight (kept fp32 inside, matching
    # the model-wide fused-norm math) and its native backward replaces the
    # eager chain that dominated this op's backward pass.
    from torch_npu import npu_rms_norm

    latent = npu_rms_norm(latent, norm_weight, eps)[0]

    # Scatter back: row-major segment order means row r's blocks sit
    # contiguously in the compact stream starting at the running total of
    # the previous rows' blocks. When every row holds the same count the
    # compact stream already is the result layout, so a view replaces the
    # per-row copies.
    if gap_free and min(blocks_per_row) == max_blocks:
        return latent.reshape(bsz, max_blocks, head_dim)
    result = x.new_zeros(bsz, max_blocks, head_dim)
    row_base = 0
    for row, row_blocks in enumerate(blocks_per_row):
        if row_blocks == 0:
            continue
        result[row, :row_blocks] = latent[row_base : row_base + row_blocks]
        row_base += row_blocks
    return result


@register_op(
    op_name="compressor",
    impl_name="cann_compressor",
    kind=OpImplKind.CUSTOM,
    # Above pypto_compressor (100) so auto selection picks the aclnn fused
    # path on NPU; the env flag below drops it back for A/B comparisons.
    priority=110,
    devices=(TurboDevice.NPU,),
)
class CannCompressor(OpImpl):
    def available(self) -> bool:
        if os.environ.get("TORCH_TURBO_DISABLE_V41_FUSED_COMPRESSOR", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            return False
        try:
            _load_cot()
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning_rank0(
                "cann_compressor unavailable (explicit requests will fail; "
                "auto stays on the next provider): %r\n%s",
                exc,
                traceback.format_exc(),
            )
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        wkv_weight: torch.Tensor,
        wgate_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        ratio: int,
        head_dim: int,
        eps: float,
    ):
        if ratio == 1:
            # Plain projection has no pooling; the module keeps the
            # ratio == 1 path inline and torch_compressor owns dispatched
            # ratio == 1 calls, same contract as pypto_compressor.
            raise ValueError(
                "cann_compressor serves the ratio > 1 softmax-gated path; "
                "ratio == 1 stays on torch_compressor."
            )
        cot = _load_cot()
        return _run_cann(
            cot,
            x,
            wkv_weight,
            wgate_weight,
            norm_weight,
            attention_mask,
            ratio,
            head_dim,
            eps,
        )
