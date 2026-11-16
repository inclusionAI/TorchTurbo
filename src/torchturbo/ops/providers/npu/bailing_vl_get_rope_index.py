"""NPU Triton provider for the Bailing VL mRoPE index op.

Five small Triton kernels replace the CPU token scan: parallel per-tile
classification and run compaction, a tiny sequential prefix and greedy grid
decode, and the parallel position fill. One small stats readback sizes the
fill grid and selects the CPU-decoder error message on bad inputs.
"""

from __future__ import annotations

from typing import Any

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


_TRITON_MROPE_KERNELS = None
# Import triton.language lazily and bind it to module globals so
# CPU-only installs keep working.
tl: Any = None
# Fixed token block keeps fill-kernel buffers within UB and tiles the
# decode stream.
_TRITON_MROPE_BLOCK = 1024


def _mrope_triton_kernels():
    """Return the cached mRoPE Triton kernels, importing Triton lazily.

    The decode is split by parallelism: token tiles are classified and
    compacted into run records by parallel per-tile programs, while the
    inherently sequential parts (tile prefix, greedy grid consumption,
    segment position counters) run in tiny single-program kernels. On Ascend
    NPUs one program serializes every vector reduction and store it issues,
    so a monolithic single-program decode costs orders of magnitude more than
    the same work spread over parallel programs.
    """
    global _TRITON_MROPE_KERNELS, tl
    if _TRITON_MROPE_KERNELS is None:
        import triton
        import triton.language as tl

        # Skip value specialization so changing shapes reuse one compiled binary.
        @triton.jit(do_not_specialize=["num_tokens", "seq_length"])
        def _count_kernel(
            tokens_ptr,
            seg_ptr,
            counts_ptr,
            num_tokens,
            seq_length,
            img_token: tl.constexpr,
            vid_token: tl.constexpr,
            HAS_SEGMENTS: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            """Count maximal-run starts per tile, one program per tile."""
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < num_tokens
            # Clamp masked load offsets; npu still translates masked-off lanes.
            load_off = tl.minimum(offs, num_tokens - 1)
            prev_off = tl.minimum(tl.where(offs > 0, offs - 1, 0), num_tokens - 1)
            tok = tl.load(tokens_ptr + load_off, mask=m, other=-1)
            kind = tl.where(tok == img_token, 1, tl.where(tok == vid_token, 2, 0))
            tok_prev = tl.load(tokens_ptr + prev_off, mask=m & (offs > 0), other=-1)
            kind_prev = tl.where(
                tok_prev == img_token, 1, tl.where(tok_prev == vid_token, 2, 0)
            )
            boundary = (offs == 0) | ((offs % seq_length) == 0)
            if HAS_SEGMENTS:
                seg = tl.load(seg_ptr + load_off, mask=m, other=1)
                seg_prev = tl.load(seg_ptr + prev_off, mask=m & (offs > 0), other=1)
                boundary = boundary | (seg != seg_prev)
            run_start = m & (boundary | (kind != kind_prev))
            tl.store(counts_ptr + pid, tl.sum(run_start.to(tl.int32)))

        @triton.jit(do_not_specialize=["num_tiles"])
        def _prefix_kernel(counts_ptr, bases_ptr, num_tiles):
            """Exclusive prefix of per-tile run counts; bases[num_tiles] is the
            total run count."""
            acc = 0
            for t in range(num_tiles):
                c = tl.load(counts_ptr + t)
                tl.store(bases_ptr + t, acc)
                acc += c
            tl.store(bases_ptr + num_tiles, acc)

        @triton.jit(do_not_specialize=["num_tokens", "seq_length"])
        def _compact_kernel(
            tokens_ptr,
            seg_ptr,
            bases_ptr,
            rstart_ptr,
            rend_ptr,
            rkind_ptr,
            rbflag_ptr,
            rblabel_ptr,
            num_tokens,
            seq_length,
            img_token: tl.constexpr,
            vid_token: tl.constexpr,
            HAS_SEGMENTS: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            """Compact maximal runs into ordered per-run arrays, per tile.

            The per-tile exclusive scan is shifted by the tile prefix, so run
            ids stay ordered across tiles. The token ending a run that started
            in an earlier tile gets id ``base - 1``, which is exactly that
            open run, so run ends scatter into the right record too.
            """
            pid = tl.program_id(0)
            base_id = tl.load(bases_ptr + pid)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            m = offs < num_tokens
            # Clamp masked load/store offsets; npu still translates masked-off lanes.
            load_off = tl.minimum(offs, num_tokens - 1)
            prev_off = tl.minimum(tl.where(offs > 0, offs - 1, 0), num_tokens - 1)
            tok = tl.load(tokens_ptr + load_off, mask=m, other=-1)
            kind = tl.where(tok == img_token, 1, tl.where(tok == vid_token, 2, 0))
            tok_prev = tl.load(tokens_ptr + prev_off, mask=m & (offs > 0), other=-1)
            kind_prev = tl.where(
                tok_prev == img_token, 1, tl.where(tok_prev == vid_token, 2, 0)
            )
            boundary = (offs == 0) | ((offs % seq_length) == 0)
            seg_prev = tok_prev
            if HAS_SEGMENTS:
                seg = tl.load(seg_ptr + load_off, mask=m, other=1)
                seg_prev = tl.load(seg_ptr + prev_off, mask=m & (offs > 0), other=1)
                boundary = boundary | (seg != seg_prev)
            run_start = m & (boundary | (kind != kind_prev))
            starts_i = run_start.to(tl.int32)
            incl = tl.cumsum(starts_i, axis=0)
            ids = base_id + incl - 1
            # Clamp masked run-array store offsets.
            start_store_off = tl.where(run_start, ids, 0)
            tl.store(rstart_ptr + start_store_off, offs, mask=run_start)
            tl.store(rkind_ptr + start_store_off, kind, mask=run_start)
            tl.store(
                rbflag_ptr + start_store_off,
                boundary.to(tl.int32),
                mask=run_start,
            )
            if HAS_SEGMENTS:
                tl.store(
                    rblabel_ptr + start_store_off,
                    seg_prev.to(tl.int32),
                    mask=run_start,
                )
            next_off = tl.minimum(offs + 1, num_tokens - 1)
            tok_next = tl.load(
                tokens_ptr + next_off, mask=m & (offs + 1 < num_tokens), other=-2
            )
            kind_next = tl.where(
                tok_next == img_token, 1, tl.where(tok_next == vid_token, 2, 0)
            )
            bnext = ((offs + 1) % seq_length == 0) & (offs + 1 < num_tokens)
            if HAS_SEGMENTS:
                seg_next = tl.load(
                    seg_ptr + next_off, mask=m & (offs + 1 < num_tokens), other=1
                )
                bnext = bnext | (seg != seg_next)
            run_end = m & ((offs == num_tokens - 1) | bnext | (kind != kind_next))
            end_store_off = tl.where(run_end, ids, 0)
            tl.store(rend_ptr + end_store_off, offs + 1, mask=run_end)

        @triton.jit(
            do_not_specialize=[
                "num_tokens",
                "seq_length",
                "n_img",
                "n_vid",
                "num_tiles",
            ]
        )
        def _decode_kernel(
            tokens_ptr,
            seg_ptr,
            bases_ptr,
            rstart_ptr,
            rend_ptr,
            rkind_ptr,
            rbflag_ptr,
            rblabel_ptr,
            img_grid_ptr,
            vid_grid_ptr,
            out_start,
            out_num,
            out_base,
            out_h,
            out_w,
            out_is_text,
            deltas_ptr,
            stats_ptr,
            num_tokens,
            seq_length,
            n_img,
            n_vid,
            num_tiles,
            img_token: tl.constexpr,
            vid_token: tl.constexpr,
            merge_size: tl.constexpr,
            interleaved: tl.constexpr,
            HAS_SEGMENTS: tl.constexpr,
        ):
            """Sequential decode over the ordered runs (single program).

            Mirrors the CPU decoder: segment boundaries reset the position
            counter, media runs consume grid entries greedily in token order
            (interleaved videos expand into one frame entry per frame), and
            text runs are emitted directly. Writes the flat fill records,
            per-batch deltas and the error bits used to raise the same
            ValueErrors as the CPU decoder.
            """
            num_runs = tl.load(bases_ptr + num_tiles)
            count = 0
            max_num = 0
            pos = 0
            seg_start_flat = 0
            img_e = 0
            vid_e = 0
            vid_t_rem = 0
            vid_h = 0
            vid_w = 0
            error = 0
            vid_avail = 0
            if interleaved:
                # Bound the interleaved greedy loop by the remaining video frames.
                for i in range(n_vid):
                    vid_avail += tl.load(vid_grid_ptr + 3 * i)
            for r in range(num_runs):
                s = tl.load(rstart_ptr + r)
                e = tl.load(rend_ptr + r)
                k = tl.load(rkind_ptr + r)
                bf = tl.load(rbflag_ptr + r)
                if HAS_SEGMENTS:
                    lbl = tl.load(rblabel_ptr + r)
                else:
                    lbl = 1
                if (bf != 0) & (s > 0):
                    # Record the delta and reset the counter when the segment ends.
                    row0 = seg_start_flat // seq_length
                    if lbl != 0:
                        tl.store(
                            deltas_ptr + row0,
                            (pos - (s - seg_start_flat)).to(
                                deltas_ptr.dtype.element_ty
                            ),
                        )
                    pos = 0
                    seg_start_flat = s
                if k == 0:
                    n = e - s
                    tl.store(out_start + count, s)
                    tl.store(out_num + count, n)
                    tl.store(out_base + count, pos)
                    tl.store(out_h + count, 1)
                    tl.store(out_w + count, 1)
                    tl.store(out_is_text + count, 1)
                    count += 1
                    if n > max_num:
                        max_num = n
                    pos += n
                else:
                    rem = e - s
                    cur = s
                    if k == 1:
                        greedy_bound = n_img - img_e + 1
                    else:
                        if interleaved:
                            greedy_bound = vid_avail + 1
                        else:
                            greedy_bound = n_vid - vid_e + 1
                    for it in range(greedy_bound):
                        if rem > 0:
                            if k == 1:
                                avail = img_e < n_img
                            else:
                                if interleaved:
                                    avail = (vid_e < n_vid) | (vid_t_rem > 0)
                                else:
                                    avail = vid_e < n_vid
                            if avail:
                                if k == 1:
                                    t_ = tl.load(img_grid_ptr + 3 * img_e)
                                    h_ = tl.load(img_grid_ptr + 3 * img_e + 1) // merge_size
                                    w_ = tl.load(img_grid_ptr + 3 * img_e + 2) // merge_size
                                    size = t_ * h_ * w_
                                    dlt = 2 * t_ - 1
                                    img_e += 1
                                else:
                                    if interleaved:
                                        if vid_t_rem == 0:
                                            vid_t_rem = tl.load(vid_grid_ptr + 3 * vid_e)
                                            vid_h = tl.load(
                                                vid_grid_ptr + 3 * vid_e + 1
                                            ) // merge_size
                                            vid_w = tl.load(
                                                vid_grid_ptr + 3 * vid_e + 2
                                            ) // merge_size
                                            vid_e += 1
                                        size = vid_h * vid_w
                                        h_ = vid_h
                                        w_ = vid_w
                                        dlt = 1
                                        vid_t_rem -= 1
                                        vid_avail -= 1
                                    else:
                                        t_ = tl.load(vid_grid_ptr + 3 * vid_e)
                                        h_ = tl.load(vid_grid_ptr + 3 * vid_e + 1) // merge_size
                                        w_ = tl.load(vid_grid_ptr + 3 * vid_e + 2) // merge_size
                                        size = t_ * h_ * w_
                                        dlt = 2 * t_ - 1
                                        vid_e += 1
                                if size <= 0:
                                    error = error | 4
                                    rem = 0
                                elif size > rem:
                                    error = error | 4
                                    rem = 0
                                else:
                                    tl.store(out_start + count, cur)
                                    tl.store(out_num + count, size)
                                    tl.store(out_base + count, pos)
                                    tl.store(out_h + count, h_)
                                    tl.store(out_w + count, w_)
                                    tl.store(out_is_text + count, 0)
                                    count += 1
                                    if size > max_num:
                                        max_num = size
                                    pos += dlt
                                    cur += size
                                    rem -= size
                            else:
                                error = error | 2
                                rem = 0
            rowf = seg_start_flat // seq_length
            if HAS_SEGMENTS:
                last_lbl = tl.load(seg_ptr + num_tokens - 1)
            else:
                last_lbl = 1
            if last_lbl != 0:
                tl.store(
                    deltas_ptr + rowf,
                    (pos - (num_tokens - seg_start_flat)).to(
                        deltas_ptr.dtype.element_ty
                    ),
                )
            if img_e < n_img:
                error = error | 1
            if interleaved:
                if (vid_e < n_vid) | (vid_t_rem > 0):
                    error = error | 1
            else:
                if vid_e < n_vid:
                    error = error | 1
            tl.store(stats_ptr + 0, count)
            tl.store(stats_ptr + 1, max_num)
            tl.store(stats_ptr + 2, error)

        @triton.jit(do_not_specialize=["row_stride"])
        def _fill_kernel(
            run_start,
            run_num,
            run_base,
            run_h,
            run_w,
            run_is_text,
            positions,
            row_stride,
            seg_ptr,
            HAS_SEGMENTS: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            """Fill T/H/W positions for every run record in parallel."""
            run_pid = tl.program_id(0)
            block_pid = tl.program_id(1)
            num = tl.load(run_num + run_pid)
            start = tl.load(run_start + run_pid)
            base = tl.load(run_base + run_pid)
            h = tl.load(run_h + run_pid)
            w = tl.load(run_w + run_pid)
            is_text = tl.load(run_is_text + run_pid)
            offs = block_pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < num
            dest = start + offs
            # Clamp masked store offsets below the row end so masked-off lanes
            # keep a valid address without an extra select.
            store_dest = tl.minimum(dest, row_stride - 1)
            if is_text == 1:
                # Text runs keep all three rows identical and skip media frame decomposition.
                row = base + offs
                row0 = row
                row1 = row
                row2 = row
            else:
                hw = h * w
                # Single-frame media runs skip the frame division: num == h*w
                # means the grid temporal dimension is exactly one.
                if num == hw:
                    rem = offs
                    temporal = base + 0 * offs
                else:
                    frame = offs // hw
                    rem = offs - frame * hw
                    temporal = base + frame * 2
                h_idx = rem // w
                w_idx = rem - h_idx * w
                row0 = temporal
                row1 = temporal + h_idx - (h - 1) // 2
                row2 = temporal + w_idx - (w - 1) // 2
            if HAS_SEGMENTS:
                # Segment-0 tokens keep position 0, as in the CPU builder.
                keep = tl.load(seg_ptr + store_dest, mask=mask, other=0) != 0
                row0 = tl.where(keep, row0, 0)
                row1 = tl.where(keep, row1, 0)
                row2 = tl.where(keep, row2, 0)
            tl.store(
                positions + 0 * row_stride + store_dest,
                row0.to(positions.dtype.element_ty),
                mask=mask,
            )
            tl.store(
                positions + 1 * row_stride + store_dest,
                row1.to(positions.dtype.element_ty),
                mask=mask,
            )
            tl.store(
                positions + 2 * row_stride + store_dest,
                row2.to(positions.dtype.element_ty),
                mask=mask,
            )

        _TRITON_MROPE_KERNELS = (
            _count_kernel,
            _prefix_kernel,
            _compact_kernel,
            _decode_kernel,
            _fill_kernel,
        )
    return _TRITON_MROPE_KERNELS


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except ImportError:
        return False


def _triton_build_rope(
    input_ids: torch.Tensor,
    segment_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
    use_interleaved_frame_timestamp: bool,
    batch_size: int,
    seq_length: int,
    total_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    i32 = torch.int32
    # Token ids are compared against small constants; i32 keeps those vector
    # comparisons on the vector core instead of scalarized i64 stores.
    tokens = input_ids.reshape(-1).to(i32)
    if segment_ids is not None:
        seg_flat = segment_ids.reshape(-1).to(i32)
    else:
        seg_flat = tokens  # Unused dummy pointer; HAS_SEGMENTS gates loads.
    empty_grid = torch.empty((0, 3), dtype=i32, device=device)
    img_grid = (
        image_grid_thw.to(device=device, dtype=i32).contiguous()
        if image_grid_thw is not None and image_grid_thw.numel()
        else empty_grid
    )
    vid_grid = (
        video_grid_thw.to(device=device, dtype=i32).contiguous()
        if video_grid_thw is not None and video_grid_thw.numel()
        else empty_grid
    )
    block = _TRITON_MROPE_BLOCK
    num_tiles = (total_tokens + block - 1) // block
    # Text and greedily split media runs emit at most two records per token.
    max_records = 2 * total_tokens + 2
    counts = torch.empty(num_tiles, dtype=i32, device=device)
    bases = torch.empty(num_tiles + 1, dtype=i32, device=device)
    rstart = torch.empty(total_tokens, dtype=i32, device=device)
    rend = torch.empty(total_tokens, dtype=i32, device=device)
    rkind = torch.empty(total_tokens, dtype=i32, device=device)
    rbflag = torch.empty(total_tokens, dtype=i32, device=device)
    rblabel = torch.empty(total_tokens, dtype=i32, device=device)
    out_start = torch.empty(max_records, dtype=i32, device=device)
    out_num = torch.empty(max_records, dtype=i32, device=device)
    out_base = torch.empty(max_records, dtype=i32, device=device)
    out_h = torch.empty(max_records, dtype=i32, device=device)
    out_w = torch.empty(max_records, dtype=i32, device=device)
    out_is_text = torch.empty(max_records, dtype=i32, device=device)
    stats = torch.empty(3, dtype=i32, device=device)
    deltas = torch.zeros((batch_size, 1), dtype=dtype, device=device)

    count_k, prefix_k, compact_k, decode_k, fill_k = _mrope_triton_kernels()
    has_segments = segment_ids is not None
    common = dict(
        img_token=image_token_id,
        vid_token=video_token_id,
        HAS_SEGMENTS=has_segments,
        BLOCK=block,
        num_warps=1,
    )
    count_k[(num_tiles,)](tokens, seg_flat, counts, total_tokens, seq_length, **common)
    prefix_k[(1,)](counts, bases, num_tiles, num_warps=1)
    compact_k[(num_tiles,)](
        tokens,
        seg_flat,
        bases,
        rstart,
        rend,
        rkind,
        rbflag,
        rblabel,
        total_tokens,
        seq_length,
        **common,
    )
    decode_k[(1,)](
        tokens,
        seg_flat,
        bases,
        rstart,
        rend,
        rkind,
        rbflag,
        rblabel,
        img_grid,
        vid_grid,
        out_start,
        out_num,
        out_base,
        out_h,
        out_w,
        out_is_text,
        deltas,
        stats,
        total_tokens,
        seq_length,
        img_grid.shape[0],
        vid_grid.shape[0],
        num_tiles,
        img_token=image_token_id,
        vid_token=video_token_id,
        merge_size=spatial_merge_size,
        interleaved=use_interleaved_frame_timestamp,
        HAS_SEGMENTS=has_segments,
        num_warps=1,
    )
    # One host stat readback sizes the fill grid and picks the ValueError for
    # bad inputs.
    count, max_len, error_bits = stats.cpu().tolist()
    if error_bits & 4:
        raise ValueError("Media patch token count does not match grid_thw.")
    if error_bits & 2:
        raise ValueError("Media patch tokens require a matching grid_thw entry.")
    if error_bits & 1:
        raise ValueError("Unused media grid entries do not match the token batch.")
    # Fill writes every token, including segment-0 zeros, so positions skip
    # zero init.
    positions = torch.empty((3, total_tokens), dtype=dtype, device=device)
    num_blocks = (max_len + block - 1) // block
    fill_k[(count, num_blocks)](
        out_start,
        out_num,
        out_base,
        out_h,
        out_w,
        out_is_text,
        positions,
        total_tokens,
        seg_flat,
        HAS_SEGMENTS=has_segments,
        BLOCK=block,
        num_warps=1,
    )
    return positions.view(3, batch_size, seq_length), deltas




@register_op(
    op_name="bailing_vl_get_rope_index",
    impl_name="triton_bailing_vl_get_rope_index",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class TritonBailingVlGetRopeIndex(OpImpl):
    """Build Bailing T/H/W positions fully on the accelerator."""

    def available(self) -> bool:
        return _triton_available()

    def __call__(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        *,
        image_token_id: int,
        video_token_id: int,
        spatial_merge_size: int,
        use_interleaved_frame_timestamp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu  # noqa: F401

        batch_size, seq_length = input_ids.shape
        total_tokens = batch_size * seq_length
        device = input_ids.device
        dtype = input_ids.dtype
        if total_tokens == 0:
            return (
                torch.zeros((3, batch_size, seq_length), dtype=dtype, device=device),
                torch.zeros((batch_size, 1), dtype=dtype, device=device),
            )
        # Enter the input tensors' device so Triton launches on the right NPU even
        # when the current device differs.
        with torch.npu.device(device):
            return _triton_build_rope(
                input_ids,
                segment_ids,
                image_grid_thw,
                video_grid_thw,
                image_token_id=image_token_id,
                video_token_id=video_token_id,
                spatial_merge_size=spatial_merge_size,
                use_interleaved_frame_timestamp=use_interleaved_frame_timestamp,
                batch_size=batch_size,
                seq_length=seq_length,
                total_tokens=total_tokens,
                device=device,
                dtype=dtype,
            )
