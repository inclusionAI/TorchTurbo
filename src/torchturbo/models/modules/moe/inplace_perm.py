"""In-place chunk-permutation reorder for the MoE dispatch/combine path.

The dropless EP=64 routing-skew spike OOM is a *genuine* HBM shortage at the
reorder step: the sorted copy of the received tokens ([T, H], multi-GiB) is
allocated while the unsorted buffer is still alive and the sum exceeds free
HBM by ~2GB.  This module performs the reorder IN PLACE: chunk contents are
bit-exactly moved from their source ranges to their destination ranges via
cycle rotation with one spare slot sized at the largest chunk (typically a
few MB -- orders of magnitude below the coexistence copy it eliminates).

Layout model
------------
The received buffer is a concatenation of n variable-size chunks.  In the
source layout chunk j sits at [soff[j], soff[j]+sizes[j]); the goal layout is
a reordering: goal slot k must hold chunk perm[k] at
[goff[k], goff[k]+sizes[perm[k]]), where goff are the cumulative offsets of
sizes[perm[:k]].  For every cycle (c0, c1=perm[c0], ...):

    spare      <- buf[soff[c0] : +sizes[c0]]
    for j = 0..L-2:
        buf[goff[c_j] : +sizes[perm[c_j]]] <- buf[soff[c_{j+1}] : +sizes[c_{j+1}]]
    buf[goff[c_{L-1}] : +sizes[perm[c_{L-1}]]] <- spare

Each chunk is read exactly once from its pristine source range and written
exactly once to its goal range (a chunk's goal write always happens after its
source read within the cycle).  Cost: n + (#nontrivial cycles) slice copies
(n <= a few hundred for MoE reorders).

Autograd: wrapped in a Function whose backward applies the inverse
permutation the same way on the incoming grad (a private engine buffer at
that point).  Numerics are unchanged -- this moves whole rows bit-exactly,
exactly like torch.cat/split or runs_copy.

Disabled by default; enable with engine_config.moe.inplace_perm.

Implementation notes (block/vector-level cycle rotation)
--------------------------------------------------------
The original row-level cycle walk was O(T) pure Python per reorder (~240
reorders/step x 110k-660k rows => +65-70 s/it).  The hot path now does the
cycle decomposition with vectorized pointer doubling ON DEVICE:

  1. cycle heads:  m = running min of f^0..f^(2^j-1)(i) with f = row_perm;
     after ceil(log2 T) doublings m[i] is the minimal element of i's cycle.
  2. cycle rank:   break the edge out of every head (succ[head]=head,
     weight 0) and list-rank the remaining paths by pointer doubling;
     r[i] = distance from the cycle head.
  3. sort nodes by (head, rank) -- concatenates every cycle in rotation
     order; consecutive pairs (nodes[k+1] -> nodes[k]) restricted to equal
     heads are exactly the rotation moves "new[dst] = old[src]".
  4. stream the pairs in CHUNK-row blocks through one pipe buffer
     (gather chunk, then scatter chunk); per-cycle head rows are saved to a
     small spare before the stream and written to the cycle-tail positions
     after it.

Safety: a node's content is gathered at its sorted position-1 and scattered
at its sorted position; gather of any earlier-or-equal position strictly
precedes the scatter (per-chunk gather-before-scatter, chunks sequential), so
no position is ever read after being overwritten.  Heads are never read in
the stream (their only read pair is cross-cycle and masked out); tails are
only written from the spare after the stream.  Numerics identical to the
row-level walk and to torch.cat: every row is bit-exactly moved once.

Fallbacks: T < 4096 uses the original row-level walk.
"""

import numpy as np
import torch

PIPE = 8 * 1024

# Replication factor for the list-ranking tail: once the doubling indices
# converge onto the few head positions, a plain index_select serializes on
# one cache line (~50ms/iter); gathering from a table replicated _REP times
# with spread indices restores throughput.
_REP = 32

_VEC_MIN_T = 4 * 1024
_CHUNK = 64 * 1024

_SPARE = {}


def _get_spare(device, dtype, rows, cols, tag=""):
    dev = device.index if (hasattr(device, "index") and device.index is not None) else 0
    key = (tag, dev, dtype, cols)
    buf = _SPARE.get(key)
    if buf is None or buf.size(0) < rows:
        buf = torch.empty(rows, cols, dtype=dtype, device=device)
        _SPARE[key] = buf
    return buf


def _permute_rows_inplace_cycles(buf, row_perm):
    """Row-level cycle rotation executed in PIPE-row chunks through one small
    pipe buffer (and a one-row spare): new[q[j]] = old[q[j+1]] along each
    cycle q, with the cycle head restored from the spare.  Extra HBM is one
    pipe buffer (default 8192 rows); each row is copied exactly twice."""
    T = len(row_perm)
    if T <= 1:
        return
    dev = buf.device
    pipe = _get_spare(dev, buf.dtype, PIPE, buf.size(1), tag="pipe")
    one = _get_spare(dev, buf.dtype, 1, buf.size(1), tag="one")
    perm = row_perm
    visited = np.zeros(T, dtype=bool)
    for q0 in range(T):
        if visited[q0]:
            continue
        cyc = [q0]
        visited[q0] = True
        k = int(perm[q0])
        while k != q0:
            visited[k] = True
            cyc.append(k)
            k = int(perm[k])
        L = len(cyc)
        if L == 1:
            continue
        cq = np.asarray(cyc, dtype=np.int64)
        one.copy_(buf[int(cq[0]) : int(cq[0]) + 1])
        # new[cq[:-1]] = old[cq[1:]] chunked with offset alignment so a
        # scatter never clobbers the next chunk's gather.
        src_all = cq[1:]
        dst_all = cq[:-1]
        for c0 in range(0, L - 1, PIPE):
            c1 = min(c0 + PIPE, L - 1)
            sidx = torch.from_numpy(src_all[c0:c1]).to(dev, non_blocking=True)
            didx = torch.from_numpy(dst_all[c0:c1]).to(dev, non_blocking=True)
            torch.index_select(buf, 0, sidx, out=pipe[: c1 - c0])
            buf.index_put_((didx,), pipe[: c1 - c0])
        buf[int(cq[L - 1]) : int(cq[L - 1]) + 1].copy_(one)


def _all_same_cycle_pairs(p, ar, T, dev):
    """(src_all, dst_all, head_rows, last_rows) on device: pair stream plus
    per-cycle spare endpoints.

    p: int64 [T] device tensor, dst->src permutation (buf[i] <- old buf[p[i]]).

    Safety argument: the pair for node x is (src=x, dst=inv[x]) where
    inv = p^-1; x is READ at its own pair and WRITTEN at the pair of f(x)
    (rank+1).  Ordering pairs by rank ascending therefore always reads a
    position before writing it (equal ranks only occur across different
    cycles, where read/write constraints are independent), and within a
    chunk gather-then-scatter keeps the chunk self-consistent.

    All doubling gathers use u = p^(2^j) as the index tensor.  u stays a
    permutation (never converges), which avoids the NPU slow path where
    index_select with many indices on the same cache line serializes.
    """
    sel = torch.index_select
    arf = ar.float()
    n_it = max(1, T.bit_length())
    # One doubling pass tracking, over windows [0, 2^j):
    #   m[i] = min(orbit)          -> converged value = cycle head M[i]
    #   c[i] = # orbit points != M -> converged value = rank[i] (0 for heads)
    # Early-exit once m stops changing (window >= longest cycle length);
    # then c is exact as well.
    # Phase 1: window-min doubling until convergence -> cycle heads.
    u = p
    m = arf
    for _ in range(n_it):
        m_new = torch.minimum(m, sel(m, 0, u))
        u = sel(u, 0, u)
        if torch.equal(m_new, m):
            m = m_new
            break
        m = m_new
    head = m
    # Phase 2: rank = distance to the head along p edges (0 for heads).
    # Break edges out of heads and list-rank by pointer doubling.  Early
    # exit once converged (succ fixed points are heads with weight 0, so
    # dropping the last w update is exact).  Plain gathers are fast while
    # succ stays spread; once indices converge onto the few head cache
    # lines a plain index_select serializes, so switch to replicated
    # tables with spread indices.
    succ = torch.where(head == arf, ar, p)
    w = (head != arf).float()
    salt = torch.remainder(ar, _REP)
    rep_w = rep_s = None
    if _REP > 1:
        rep_w = torch.empty(_REP * T, dtype=torch.float32, device=dev)
        rep_s = torch.empty(_REP * T, dtype=torch.int64, device=dev)
    use_rep = False
    for _ in range(n_it):
        if use_rep:
            # Indices converged onto the few head cache lines make a plain
            # index_select serialize per line; gather from tables replicated
            # _REP-fold (interleaved) with salt-spread indices instead.
            idx = succ * _REP + salt
            rep_w.view(T, _REP).copy_(w.unsqueeze(1))
            rep_s.view(T, _REP).copy_(succ.unsqueeze(1))
            w_new = w + sel(rep_w, 0, idx)
            s_new = sel(rep_s, 0, idx)
        else:
            w_new = w + sel(w, 0, succ)
            s_new = sel(succ, 0, succ)
        if torch.equal(s_new, succ):
            break
        w, succ = w_new, s_new
        if not use_rep and _REP > 1:
            # switch to the replicated path once the plain path starts
            # hitting the same-line serialization regime
            d = (s_new != succ).sum()
            if d.item() < T // 2:
                use_rep = True
    rank = w
    # inv (one scatter; p is a permutation)
    inv = torch.empty_like(p)
    inv.index_put_((p,), ar)
    # stream = non-head nodes (rank > 0) ordered by rank DESCENDING: node y
    # is read at its own pair (rank r) and written at the pair of p[y]
    # (rank r-1), so reads must be scheduled before writes.  Equal ranks
    # only occur across different cycles, where constraints are independent,
    # so an unstable sort is safe.
    order = torch.argsort(rank, descending=True)
    rko = sel(rank, 0, order)
    stream = sel(order, 0, torch.nonzero(rko > 0).view(-1))
    src_all = stream
    dst_all = sel(inv, 0, stream)
    # spare endpoints: non-fixed heads and their cycle tails (successor has
    # rank 0), aligned per cycle via one small CPU transfer + sort.
    is_head = (rank == 0) & (p != ar)
    rk_of_p = sel(rank, 0, p)
    is_tail = (rk_of_p == 0) & (rank > 0)
    hr = torch.nonzero(is_head).view(-1)
    tr = torch.nonzero(is_tail).view(-1)
    stacked = torch.stack([
        sel(head, 0, hr), hr.to(arf.dtype),
        sel(head, 0, tr), tr.to(arf.dtype),
    ]).cpu().numpy().astype(np.int64)
    nh = int(hr.numel())
    h_np = stacked[:2].T
    t_np = stacked[2:].T
    h_np = h_np[np.argsort(h_np[:, 0], kind="stable")]
    t_np = t_np[np.argsort(t_np[:, 0], kind="stable")]
    head_rows = torch.as_tensor(h_np[:, 1]).to(dev)
    last_rows = torch.as_tensor(t_np[:, 1]).to(dev)
    return src_all, dst_all, head_rows, last_rows


def permute_rows_inplace(buf, row_perm):
    """In place: buf[i] <- old buf[row_perm[i]]. row_perm: np int64 [T]."""
    T = len(row_perm)
    if T <= 1:
        return
    if T < _VEC_MIN_T:
        return _permute_rows_inplace_cycles(buf, row_perm)
    dev = buf.device
    p = torch.as_tensor(np.asarray(row_perm, dtype=np.int64)).to(dev).view(-1)
    ar = torch.arange(T, dtype=torch.int64, device=dev)
    src_all, dst_all, head_rows, last_rows = _all_same_cycle_pairs(p, ar, T, dev)

    pipe = _get_spare(dev, buf.dtype, _CHUNK, buf.size(1), tag="pipe")
    nh = int(head_rows.numel())
    hspare = None
    if nh:
        hspare = _get_spare(dev, buf.dtype, nh, buf.size(1), tag="head")
        torch.index_select(buf, 0, head_rows, out=hspare[:nh])
    n_pairs = int(src_all.numel())
    for c0 in range(0, n_pairs, _CHUNK):
        c1 = min(c0 + _CHUNK, n_pairs)
        torch.index_select(buf, 0, src_all[c0:c1], out=pipe[: c1 - c0])
        buf.index_put_((dst_all[c0:c1],), pipe[: c1 - c0])
    if nh:
        buf.index_put_((last_rows,), hspare[:nh])


def rotate_chunks_inplace(buf, sizes, perm, soff=None, goff=None):
    """Back-compat wrapper: build the row-level permutation (goal slot k gets
    chunk perm[k]) and rotate rows in place."""
    sizes_a = np.asarray(sizes, dtype=np.int64)
    perm_a = np.asarray(perm, dtype=np.int64)
    if soff is None:
        soff = np.concatenate(([0], np.cumsum(sizes_a)[:-1])) if len(sizes_a) else np.empty(0, dtype=np.int64)
    parts = [
        np.arange(int(o), int(o) + int(l), dtype=np.int64)
        for o, l in zip(np.asarray(soff, dtype=np.int64)[perm_a], sizes_a[perm_a])
    ]
    permute_rows_inplace(buf, np.concatenate(parts) if parts else np.empty(0, dtype=np.int64))


class InplaceRowPermuteFn(torch.autograd.Function):
    """Autograd wrapper: forward applies the row permutation in place on
    `buf` and returns it; backward applies the inverse permutation to the
    incoming grad in place (a private engine buffer at that point)."""

    @staticmethod
    def forward(ctx, buf, row_perm, inv_row_perm):
        ctx.inv_row_perm = inv_row_perm
        ctx.mark_dirty(buf)
        permute_rows_inplace(buf, row_perm)
        return buf

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.contiguous()
        permute_rows_inplace(g, ctx.inv_row_perm)
        return g, None, None


def make_row_perm(src_offs, lens):
    """dst->src row permutation from per-block source offsets and lengths
    listed in DESTINATION order (helper for diagnostics/tests)."""
    parts = [
        np.arange(o, o + l, dtype=np.int64) for o, l in zip(src_offs, lens) if l
    ]
    if parts:
        return np.concatenate(parts)
    return np.empty(0, dtype=np.int64)


def run_perm_from_tables(run_src, run_dst, run_len):
    """(sizes, perm, soff, goff) for the mc2 runs reorder: recv buffer
    (src-major runs) -> expand buffer (expert-major runs).  Tables: numpy or
    CPU torch (padded tables OK; zero-length runs are dropped)."""
    if isinstance(run_src, torch.Tensor):
        run_src = run_src.cpu().numpy()
        run_dst = run_dst.cpu().numpy()
        run_len = run_len.cpu().numpy()
    run_src = np.asarray(run_src, dtype=np.int64).ravel()
    run_dst = np.asarray(run_dst, dtype=np.int64).ravel()
    run_len = np.asarray(run_len, dtype=np.int64).ravel()
    keep = run_len > 0
    run_src, run_dst, run_len = run_src[keep], run_dst[keep], run_len[keep]
    order = np.argsort(run_dst, kind="stable")  # goal position order
    perm = order.astype(np.int64).tolist()  # perm[k] = run id at goal position k
    sizes = run_len.tolist()
    soff = run_src  # recv layout offsets
    goff = run_dst[order]  # expand layout offsets in goal order
    return sizes, perm, soff, goff


def chunk_perm_from_sort(sizes, idxs):
    """(sizes, perm, soff, goff) for sort_chunks_by_idxs: sizes = chunk
    lengths in input order, idxs = input chunk id per output position."""
    sizes_l = [int(x) for x in sizes]
    n = len(sizes_l)
    if n == 0:
        return [], [], [], []
    soff = np.concatenate(([0], np.cumsum(sizes_l)[:-1])).astype(np.int64)
    return sizes_l, [int(i) for i in idxs], soff, None


def inverse_tables(sizes, perm, soff, goff):
    """Inverse-permutation tables for the autograd backward."""
    n = len(sizes)
    perm_a = np.asarray(perm, dtype=np.int64)
    pos = np.empty(n, dtype=np.int64)
    pos[perm_a] = np.arange(n, dtype=np.int64)
    sizes_a = np.asarray(sizes, dtype=np.int64)
    # inverse: buffer currently in goal layout (chunk perm[k] at goff[k]);
    # restore source layout (chunk j at soff[j]).
    inv_sizes = sizes_a[perm_a].tolist()  # chunk order = goal layout order
    inv_perm = pos.tolist()  # goal slot k (of inverse) = chunk at goal pos k
    inv_soff = np.asarray(goff, dtype=np.int64)
    inv_goff = np.asarray(soff, dtype=np.int64)
    return inv_sizes, inv_perm, inv_soff, inv_goff


def inverse_perm(row_perm):
    """Inverse of a row-level dst->src permutation."""
    inv = np.empty_like(row_perm)
    inv[row_perm] = np.arange(len(row_perm), dtype=row_perm.dtype)
    return inv


def sort_row_perm(sizes, idxs):
    """(row_perm, inv) for sort_chunks_by_idxs: sizes = chunk lengths in
    input order, idxs = input chunk id per output position."""
    sizes_l = [int(x) for x in sizes]
    n = len(sizes_l)
    if n == 0 or sum(sizes_l) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    offs = np.concatenate(([0], np.cumsum(sizes_l)[:-1]))
    rp = np.concatenate(
        [np.arange(offs[i], offs[i] + sizes_l[i], dtype=np.int64) for i in idxs]
    )
    return rp, inverse_perm(rp)


def runs_row_perm(offs_b, offs_a, run_len):
    """(row_perm, inv) mapping layout B rows to layout A rows from per-run
    offsets in both layouts: buf_b[i] = buf_a[row_perm[i]].  Tables: numpy or
    CPU torch; zero-length runs (incl. block padding) are dropped."""
    if isinstance(offs_b, torch.Tensor):
        offs_b = offs_b.cpu().numpy()
        offs_a = offs_a.cpu().numpy()
        run_len = run_len.cpu().numpy()
    offs_b = np.asarray(offs_b, dtype=np.int64).ravel()
    offs_a = np.asarray(offs_a, dtype=np.int64).ravel()
    run_len = np.asarray(run_len, dtype=np.int64).ravel()
    keep = run_len > 0
    offs_b, offs_a, run_len = offs_b[keep], offs_a[keep], run_len[keep]
    if len(run_len) == 0 or int(run_len.sum()) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    order = np.argsort(offs_b, kind="stable")
    rp = np.concatenate(
        [
            np.arange(int(offs_a[r]), int(offs_a[r]) + int(run_len[r]), dtype=np.int64)
            for r in order
        ]
    )
    return rp, inverse_perm(rp)
