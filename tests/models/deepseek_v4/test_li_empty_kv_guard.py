"""Regression: ``_npu_li`` must not crash when no compressed block is visible.

When every covered segment is shorter than ``compress_ratio`` (e.g. a
zero-loss placeholder micro-batch sliced by sequence parallelism so a rank
sees fewer than ``ratio`` tokens), ``_get_compressed_kv_tnd_data`` returns an
empty key tensor and ``lightning_indexer`` would raise
``RuntimeError: Tensor k is empty``. The guard in ``_npu_li`` must instead
return an all -1 topk (no compressed-KV hit), which is a no-op for the
zero-loss placeholder slot.

CPU-only: the guard returns before the NPU ``lightning_indexer`` call, so no
``cann_ops_transformer`` / NPU runtime is needed.
"""

import pytest
import torch

from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import LiCompute
from torchturbo.models.deepseek_v4.npu import _npu_li


def _fake_self(ratio: int = 4, index_topk: int = 512):
    """A LiCompute instance supplies ``ratio``/``index_topk`` to _npu_li."""
    return LiCompute(ratio=ratio, index_topk=index_topk)


class TestEmptyKvGuard:
    def test_short_segment_returns_all_minus_one(self):
        # One positive-id span of length 2 (< ratio=4), then padding. The
        # span yields 2 query rows but ceil(2/4)=0 compressed blocks, so the
        # compressed-kv selection is empty and the guard must fire.
        segment_ids = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        q_indexer = torch.randn(1, 4, 8, 16)       # [bsz, seqlen, heads, head_dim]
        k_indexer = torch.randn(1, 0, 16)          # empty compressed layout
        weights = torch.randn(1, 4, 8)
        q_global_positions = torch.arange(4).unsqueeze(0)

        out, score = _npu_li(
            _fake_self(ratio=4, index_topk=512),
            q_indexer,
            k_indexer,
            weights,
            seqlen=4,
            offset=0,
            q_global_positions=q_global_positions,
            segment_ids=segment_ids,
            sp_rank=0,
            sp_group=None,
            sp_size=1,
        )

        # The guard returns before the NPU op, so it must produce an all -1
        # topk of shape [bsz, seqlen, sparse_count] with no score.
        assert out is not None
        assert out.shape == (1, 4, 512)
        assert out.dtype == torch.int64
        assert (out == -1).all()
        assert score is None

    def test_non_empty_path_not_taken_on_cpu(self):
        # Sanity: a segment long enough to form a block does NOT hit the guard.
        # We only assert the guard is skipped (no early return) by checking the
        # function proceeds past it — on CPU the real op is unavailable, so we
        # just confirm k_tnd is non-empty here (the guard condition is False).
        from torchturbo.models.deepseek_v4.npu import _get_compressed_kv_tnd_data

        segment_ids = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
        k_indexer = torch.randn(1, 2, 16)          # 8 // ratio(4) = 2 blocks
        k_tnd, _, _, _ = _get_compressed_kv_tnd_data(
            k_indexer, segment_ids, segment_ids, 0, ratio=4
        )
        assert k_tnd.numel() > 0


if __name__ == "__main__":
    t = TestEmptyKvGuard()
    t.test_short_segment_returns_all_minus_one()
    t.test_non_empty_path_not_taken_on_cpu()
    print("empty-kv guard checks passed.")
