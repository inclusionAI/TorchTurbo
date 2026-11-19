"""NPU (PyPTO) provider for the packed-KV compressor op.

Bridges the vendored Huawei CSA2 compressor (PyPTO forward + backward fused
kernels, ``CSA2_compressor`` package beside this file) into the dispatcher.
Uniform masks (one full-length segment per row) run as a single dense BSH
call; packed masks run as a single TH (BS-merged) call: the valid tokens are
gathered into a flat ``(t, hidden)`` stream with an int32 ``cu_seqlens``
prefix sum, and the vendor kernel walks every segment inside one launch,
forward and backward. The kernel fuses the fp32 wkv/wgate projections, the
per-channel softmax pooling, and the RMSNorm; gradients come from the paired
backward kernel, so no reference recompute is needed in training mode.

Auto selection prefers this provider on NPU when its dependencies are
available. Explicit ``impl="pypto_compressor"`` requires the fused path;
``torch_compressor`` remains available for reference comparisons.
"""

from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _load_csa2():
    # pypto is a container-side dependency (pto-isa run package + pypto
    # wheel); importing it lazily keeps the provider scan working everywhere
    # else.
    from .CSA2_compressor import csa2_compressor

    return csa2_compressor


def _segment_starts(mask_cpu: torch.Tensor):
    """Host-side packed-segment list (row, start, length) from the CPU mask.

    A segment is a maximal run of equal positive ids inside a row (ids
    restart at 1 in every row, so detection must be per-row before any
    flattening; adjacent same-id segments across a row boundary stay
    separate). Fully vectorized: a start is a valid token whose predecessor
    is invalid or carries a different id; an end is a valid token whose
    successor is invalid or carries a different id.
    """
    ids = mask_cpu
    positive = ids > 0
    seq_len = ids.size(1)

    start_mask = positive.clone()
    if seq_len > 1:
        start_mask[:, 1:] = positive[:, 1:] & (ids[:, 1:] != ids[:, :-1])
    end_mask = positive.clone()
    if seq_len > 1:
        end_mask[:, :-1] = positive[:, :-1] & (
            ~positive[:, 1:] | (ids[:, 1:] != ids[:, :-1])
        )

    starts = start_mask.nonzero(as_tuple=False)
    ends = end_mask.nonzero(as_tuple=False)
    # Every segment contributes exactly one start and one end, both in
    # row-major order, so the two lists pair index-wise.
    assert starts.shape == ends.shape
    rows = starts[:, 0].tolist()
    begins = starts[:, 1].tolist()
    lengths = (ends[:, 1] - starts[:, 1] + 1).tolist()
    return list(zip(rows, begins, lengths))


def _gather_tokens(x: torch.Tensor, segments, seq_len: int):
    """Flatten the segments' tokens into a (t, hidden) stream, device-side.

    The gather index is built with vectorized ops from three small host
    lists (one entry per segment), so no token-sized host work happens.
    Tokens map 1:1 into the stream, so the autograd backward of the gather
    is a scatter of unique writes and stays deterministic.
    """
    device = x.device
    n_seg = len(segments)
    lengths = [length for _, _, length in segments]
    lens = torch.tensor(lengths, device=device)
    rows = torch.tensor([row for row, _, _ in segments], device=device)
    begins = torch.tensor([start for _, start, _ in segments], device=device)

    seg_of_tok = torch.repeat_interleave(
        torch.arange(n_seg, device=device), lens
    )
    # Position of each stream token inside its own segment: stream offset
    # minus the segment's exclusive-cumsum start offset. total comes from
    # the host list (no device read-back).
    total = sum(lengths)
    seg_starts = torch.zeros(n_seg + 1, dtype=torch.int64, device=device)
    torch.cumsum(lens, 0, out=seg_starts[1:])
    intra = torch.arange(total, device=device) - seg_starts[seg_of_tok]
    flat_pos = rows[seg_of_tok] * seq_len + begins[seg_of_tok] + intra
    return x.reshape(-1, x.shape[-1]).index_select(0, flat_pos)


def _cu_seqlens(segments, device):
    """int32 (n_seg+1,) prefix sum of segment lengths for the TH kernel."""
    n_seg = len(segments)
    lens = torch.tensor(
        [length for _, _, length in segments], dtype=torch.int64, device=device
    )
    cu = torch.zeros(n_seg + 1, dtype=torch.int64, device=device)
    torch.cumsum(lens, 0, out=cu[1:])
    return cu.to(torch.int32)


def _run_pypto(
    csa2_compressor,
    x: torch.Tensor,
    wkv_weight: torch.Tensor,
    wgate_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    attention_mask: torch.Tensor,
    ratio: int,
    head_dim: int,
    eps: float,
):
    """Run the fused kernels and scatter the pooled rows to packed layout.

    One host sync moves the mask to the CPU to build the segment list (the
    TH kernel needs the segment count on the host for its ``cu_seqlens``
    shape). A uniform all-valid mask takes the dense BSH fast path; anything
    else becomes a single TH call over the flattened valid tokens. In both
    cases complete blocks land contiguously in the owning row's block slots,
    matching the torch path's block ordering: segments come out of the
    detector in row-major order, so the TH kernel's compact output holds
    every row's blocks in one contiguous run and the scatter back is one
    slice copy per batch row. Incomplete trailing tokens have no block and
    no gradient path in either implementation.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    device = x.device
    bsz, _, _ = x.shape
    seq_len = attention_mask.size(1)
    # The kernel contract wants fp32 weights. The checkpoint stores fp32,
    # but the FSDP2 mixed-precision policy may hand the norm parameter over
    # as bf16 at runtime; upcast at the boundary (no-op for fp32 tensors).
    # The torch path does the same upcast inside DeepseekV41RMSNorm.forward.
    wkv_weight = wkv_weight.float()
    wgate_weight = wgate_weight.float()
    norm_weight = norm_weight.float()

    mask_cpu = attention_mask.detach().to("cpu", non_blocking=False)
    segments = _segment_starts(mask_cpu)
    blocks_per_seg = [length // ratio for _, _, length in segments]

    # The packed layout is [b, max_blocks, head_dim] where max_blocks is the
    # largest PER-ROW total: a row's segments concatenate their blocks from
    # slot 0, so sum per owning row before taking the max.
    blocks_per_row = [0] * bsz
    for (row, _, _), seg_blocks in zip(segments, blocks_per_seg):
        blocks_per_row[row] += seg_blocks
    max_blocks = max(blocks_per_row, default=0)
    if max_blocks == 0:
        return None

    start_pos = torch.zeros(1, dtype=torch.int32, device=device)

    # Dense fast path: every row is one full-length segment (uniform all-
    # valid mask) -- the whole batch runs as a single dense BSH call.
    uniform = len(segments) == bsz and all(
        start == 0 and length == seq_len for _, start, length in segments
    )
    if uniform:
        kv_state = torch.zeros(bsz, ratio, head_dim, dtype=torch.float32, device=device)
        score_state = torch.full(
            (bsz, ratio, head_dim), float("-inf"), dtype=torch.float32, device=device
        )
        return csa2_compressor(
            x, wkv_weight, wgate_weight, norm_weight,
            kv_state, score_state, start_pos, None, ratio, eps,
        )

    # Single TH call over the flattened valid tokens. The states are pure
    # scratch here: whole-segment input never reads prior state, only the
    # trailing remainder tokens write back (and nothing consumes them).
    n_seg = len(segments)
    x_flat = _gather_tokens(x, segments, seq_len)
    cu_seqlens = _cu_seqlens(segments, device)
    kv_state = torch.zeros(n_seg, ratio, head_dim, dtype=torch.float32, device=device)
    score_state = torch.full(
        (n_seg, ratio, head_dim), float("-inf"), dtype=torch.float32, device=device
    )
    compact = csa2_compressor(
        x_flat, wkv_weight, wgate_weight, norm_weight,
        kv_state, score_state, start_pos, cu_seqlens, ratio, eps,
    )

    # Scatter back: row-major segment order means row r's blocks sit
    # contiguously in the compact stream starting at the running total of
    # the previous rows' blocks.
    out = x.new_zeros(bsz, max_blocks, head_dim)
    row_base = 0
    for row, row_blocks in enumerate(blocks_per_row):
        if row_blocks == 0:
            continue
        out[row, :row_blocks] = compact[row_base : row_base + row_blocks]
        row_base += row_blocks
    return out


@register_op(
    op_name="compressor",
    impl_name="pypto_compressor",
    kind=OpImplKind.CUSTOM,
    # Default on NPU for the THD/SP development line: the fused path is the
    # integration target those features build on. Explicit impl= /
    # dispatcher overrides (torch_compressor) still reach the reference
    # provider for A/B comparisons.
    priority=100,
    devices=(TurboDevice.NPU,),
)
class PyptoCompressor(OpImpl):
    def available(self) -> bool:
        try:
            _load_csa2()
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning_rank0(
                "pypto_compressor unavailable (explicit requests will fail; "
                "auto stays on torch_compressor): %r\n%s",
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
            # Plain projection branch has no pooling; the module keeps the
            # ratio == 1 path inline, and the torch reference owns any
            # dispatched ratio == 1 call.
            raise ValueError(
                "pypto_compressor serves the ratio > 1 softmax-gated path; "
                "ratio == 1 stays on torch_compressor."
            )
        csa2_compressor = _load_csa2()
        return _run_pypto(
            csa2_compressor,
            x,
            wkv_weight,
            wgate_weight,
            norm_weight,
            attention_mask,
            ratio,
            head_dim,
            eps,
        )
