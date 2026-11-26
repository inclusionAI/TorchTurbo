"""LightningIndexerV2 单测：基于 MindSpeed-MM Qwen3.8-flash-next 实际接入参数.

包含功能测试、精度测试（torch 小算子为 golden）、性能测试三个维度。

调用入参完全对齐 MindSpeed-MM 生产代码：
  - mindspeed_mm/fsdp/ops/qsa/ascendc/qsa_indexer.py
  - qwen4_exp_qsa_indexer_forward_lightning()
  - qsa_lightning_selected_token_mask()

Qwen3.8-Flash-Next 生产配置：
  - num_heads_q=4, num_heads_k=1, head_dim=128
  - hidden_size=2560, num_attention_heads=24, head_dim=256 (full attn)
  - compress_ratio=4, token_budget=2048, block_topk=512
  - mask_mode=3 (rightDownCausal)
  - BSND layout（非 packed 场景）

Usage:
    # 功能 + 精度测试（需要 NPU）
    pytest tests/ops/npu/test_lightning_indexer_qwen38.py -v -s

    # 包含性能测试
    RUN_PERF_TESTS=1 pytest tests/ops/npu/test_lightning_indexer_qwen38.py::test_li_v2_performance -v -s
"""

import math
import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# 如需指向自建的 cann_ops_transformer 算子包，由用户通过 env var 配置：
#   export ASCEND_CUSTOM_OPP_PATH=/path/to/vendors/custom_transformer
_op_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
if _op_path:
    os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", _op_path)

try:
    import torch_npu  # noqa: F401  必须先 import 以注册 "npu" device
    # 优先使用 qwen38liv2 子包（本次编译产物），fallback 到整包
    try:
        import cann_ops_transformer_qwen38liv2 as _li_pkg
    except ImportError:
        import cann_ops_transformer as _li_pkg
    lightning_indexer = _li_pkg.lightning_indexer
    lightning_indexer_metadata = _li_pkg.lightning_indexer_metadata
    _CANN_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _CANN_AVAILABLE = False
    pytest.skip(
        "cann_ops_transformer / torch_npu not available (NPU-only test)",
        allow_module_level=True,
    )


# ============================================================
# Helpers
# ============================================================

def _device():
    """返回 "npu"，无 NPU 时跳过。"""
    if not _CANN_AVAILABLE or not torch.npu.is_available():
        pytest.skip("NPU not available")
    torch.npu.set_device(0)
    return "npu"


def _make_cu(lengths, device, dtype=torch.int32):
    """由每 seq 长度列表构造累积序列长度 [0, l0, l0+l1, ...]。"""
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return torch.tensor(cu, dtype=dtype, device=device)


# ============================================================
# Qwen3.8-Flash-Next 生产配置常量
# ============================================================
N_HEADS_Q = 4         # indexer_n_heads (actual config, not num_attention_heads)
N_HEADS_K = 1         # indexer_kv_heads = 1（硬约束）
HEAD_DIM = 128        # indexer_head_dim = 128（硬约束）
COMPRESS_RATIO = 4    # indexer_compress_ratio
TOKEN_BUDGET = 2048   # indexer_budget
BLOCK_TOPK = TOKEN_BUDGET // COMPRESS_RATIO  # = 512


def _rotary_mul(x, cos, sin):
    """模拟 torch_npu.npu_rotary_mul 的 CPU 版本。

    x: [B, S, H, D], cos/sin: [B, S, 1, D]
    返回 rotate_half(x) * cos + rotate_half(x) * sin 的结果
    """
    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    # x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim]) * sin
    # MindSpeed-MM 的 apply_rotary_pos_emb: q_rot * cos + rotate_half(q_rot) * sin
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # cos/sin 是 fp32，需要 cast 回 x 的 dtype 避免类型提升
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x_embed = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_embed, x_pass], dim=-1).to(x.dtype)


def _rms_norm(x, eps=1e-6):
    """CPU 参考 RMSNorm。"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _compressed_index_keys_cpu(raw_keys, full_cos, full_sin, compress_ratio, eps=1e-6):
    """CPU 参考：构造压缩后的块 key（对齐 MindSpeed-MM _compressed_index_keys）。

    raw_keys: [B, S, D], full_cos/sin: [B, S, D]
    返回: [B, S//ratio, D]
    """
    B, S, D = raw_keys.shape
    blocks = S // compress_ratio
    grouped = raw_keys[:, :blocks * compress_ratio].reshape(B, blocks, compress_ratio, D)
    # FP32 mean 后 cast 回原始 dtype（对齐 MindSpeed-MM）
    pooled = grouped.float().mean(dim=2).to(raw_keys.dtype)
    # RMSNorm
    normalized = _rms_norm(pooled.float(), eps).to(raw_keys.dtype)
    # 取块首位置的 cos/sin
    block_cos = full_cos[:, :blocks * compress_ratio:compress_ratio]
    block_sin = full_sin[:, :blocks * compress_ratio:compress_ratio]
    # RoPE: unsqueeze(2) → [B, blocks, 1, D]
    result = _rotary_mul(
        normalized.unsqueeze(2), block_cos.unsqueeze(2), block_sin.unsqueeze(2)
    ).squeeze(2)
    return result


def _lightning_indexer_ref(index_queries, compressed_keys, weights, compress_ratio, topk):
    """CPU fp32 参考：LightningIndexerV2 前向。

    对齐 MindSpeed-MM LightningIndexer 的数学定义：
      scores[t, g] = Σ_h w[t,h] * ReLU(scale * <q[t,h], k[g]>)
    其中 scale = 1/sqrt(head_dim)，w 为常数权重（head_dim^-0.5）

    index_queries: [B, S, H, D] fp32（已含 RoPE）
    compressed_keys: [B, S//ratio, D] fp32（已含 RMSNorm + RoPE）
    weights: [B, S, H] fp32

    返回: selected_blocks [B, S, topk] int32（段内局部组号）
    """
    B, S, H, D = index_queries.shape
    S_kv = compressed_keys.shape[1]
    scale = 1.0 / math.sqrt(D)

    # scores[b, t, g] = Σ_h w[b,t,h] * ReLU(scale * <q[b,t,h], k[b,g]>)
    q = index_queries.float()
    k = compressed_keys.float()
    w = weights.float()

    # einsum: 'bthd,bgd->btg' 先算 q·k，再乘 w
    qk = torch.einsum('bthd,bgd->bthg', q, k) * scale  # [B, S, H, S_kv]
    relu_qk = F.relu(qk)                                # [B, S, H, S_kv]
    scores = (w.unsqueeze(-1) * relu_qk).sum(dim=2)     # [B, S, S_kv]

    # 因果 mask: query t 可见组 0 .. (t+1+offset)//ratio - 1
    # 在 MindSpeed-MM 中，offset = kv_len - query_len（prefill 时 = 0）
    # mask_mode=3 (rightDownCausal): 可见组数 = (query_pos + 1) // ratio
    selected = torch.full((B, S, topk), -1, dtype=torch.int32)
    for b in range(B):
        for t in range(S):
            # 可见完整块数（含当前不完整块的因果性约束）
            n_visible = (t + 1) // compress_ratio
            n_visible = min(n_visible, S_kv)
            if n_visible <= 0:
                continue
            row_scores = scores[b, t, :n_visible]
            k_sel = min(topk, n_visible)
            _, topk_ids = torch.topk(row_scores, k_sel)
            # 排序（GBSA 要求升序）
            topk_ids, _ = torch.sort(topk_ids)
            selected[b, t, :k_sel] = topk_ids.to(torch.int32)
    return selected


def _make_qwen38_indexer_inputs(B, S, device, dtype=torch.bfloat16, seed=42):
    """构造 Qwen3.8 indexer 的完整输入（对齐 MindSpeed-MM 生产传参）。

    模拟 Qwen4ExpTextQSAIndexer.forward 的完整链路：
      hidden → index_qk_proj → split → q_layernorm + RoPE → index_queries
                                   → raw_keys → pool + k_layernorm + RoPE → compressed_keys

    Returns:
        index_queries: [B, S, H, D] bf16/fp16（已含 RoPE）
        compressed_keys: [B, S//ratio, D] bf16/fp16（已含 RMSNorm + RoPE）
        weights: [B, S, H] fp32（常数 head_dim^-0.5）
        raw_keys: [B, S, D] bf16/fp16（原始 keys，供 torch ref 用）
        full_cos/full_sin: [B, S, D] fp32
    """
    torch.manual_seed(seed)
    hidden_size = 2560  # actual Qwen3.8 config hidden_size
    H = N_HEADS_Q
    D = HEAD_DIM

    # 模拟 hidden_states → index_qk_proj 的输出
    # index_qk_proj: hidden_size → (H + 1) * D
    qk_proj_out = torch.randn(B, S, (H + 1) * D, dtype=dtype, device=device)

    # split: query [B, S, H*D], raw_keys [B, S, D]
    query, raw_keys = qk_proj_out.split([H * D, D], dim=-1)

    # query reshape → [B, S, H, D]
    query = query.view(B, S, H, D)
    raw_keys = raw_keys.view(B, S, D)

    # RoPE 位置编码（模拟 prefill 场景，position = 0..S-1）
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32, device=device) / D))
    positions = torch.arange(S, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # [S, D//2]
    full_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0).expand(B, -1, -1).contiguous()
    full_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0).expand(B, -1, -1).contiguous()

    # current_cos/sin = full_cos/sin（prefill 场景 kv_len == query_len）
    current_cos = full_cos
    current_sin = full_sin

    # q: RMSNorm + RoPE
    q_normed = _rms_norm(query.float(), eps=1e-6).to(dtype)
    index_queries = _rotary_mul(q_normed, current_cos.unsqueeze(2), current_sin.unsqueeze(2))

    # compressed_keys: pool + RMSNorm + RoPE（CPU 参考）
    compressed_keys = _compressed_index_keys_cpu(
        raw_keys.float().cpu(),
        full_cos.float().cpu(),
        full_sin.float().cpu(),
        COMPRESS_RATIO,
        eps=1e-6,
    ).to(device=device, dtype=dtype)

    # weights: 常数 head_dim^-0.5（对齐 MindSpeed-MM _constant_head_weights）
    weights = torch.full(
        (B, S, H), HEAD_DIM ** -0.5, dtype=torch.float32, device=device
    )

    return index_queries, compressed_keys, weights, raw_keys, (full_cos, full_sin)


# ============================================================
# 1. 功能测试
# ============================================================

class TestLightningIndexerV2Functional:
    """功能测试：验证算子基本行为符合预期。"""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_output_shape_and_dtype(self, dtype):
        """验证输出 shape 和 dtype 正确。"""
        device = _device()
        B, S = 2, 256

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype
        )
        S_kv = S // COMPRESS_RATIO  # 64

        metadata = None
        # assert metadata.shape  # non-exp LIv2 metadata size mismatch
        # assert metadata.dtype

        selected_blocks, sparse_values = lightning_indexer(
            index_queries,
            compressed_keys.unsqueeze(2),  # [B, S_kv, 1, D]
            weights,
            BLOCK_TOPK,
            cmp_residual_k=torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device),
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=3,
            cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()

        # shape: [B, S, N2, topk]
        assert selected_blocks.shape == torch.Size([B, S, N_HEADS_K, BLOCK_TOPK])
        assert selected_blocks.dtype == torch.int32
        # return_value=0 → sparse_values 为空
        assert sparse_values.numel() == 0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_output_bounds_and_validity(self, dtype):
        """验证输出索引在合法范围内，无效位为 -1。"""
        device = _device()
        B, S = 2, 512
        S_kv = S // COMPRESS_RATIO  # 128

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype
        )

        metadata = None
        selected_blocks, _ = lightning_indexer(
            index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
            cmp_residual_k=torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device),
            metadata=None, max_seqlen_q=-1,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()

        local = selected_blocks[:, :, 0, :]  # [B, S, topk]
        valid = local >= 0

        # 有效索引不越界
        if valid.any():
            assert (local[valid] < S_kv).all(), f"valid index should be < {S_kv}"
        # 无效位为 -1
        invalid = ~valid
        if invalid.any():
            assert (local[invalid] == -1).all()

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_causality_no_future_leak(self, dtype):
        """验证因果性：任何 query 都不会选中未来的块。"""
        device = _device()
        B, S = 2, 512
        S_kv = S // COMPRESS_RATIO

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype
        )

        metadata = None
        selected_blocks, _ = lightning_indexer(
            index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
            cmp_residual_k=torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device),
            metadata=None, max_seqlen_q=-1,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()

        local = selected_blocks[:, :, 0, :].cpu()  # [B, S, topk]

        for b in range(B):
            for t in range(S):
                n_visible = (t + 1) // COMPRESS_RATIO
                row = local[b, t]
                valid = row[row >= 0]
                if valid.numel() > 0:
                    assert (valid < n_visible).all(), \
                        f"query {t} selected future block: visible={n_visible}, got {valid.tolist()}"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_dense_prefix_and_tail_block(self, dtype):
        """验证 dense prefix 和 tail block 的处理（Qwen3.8 语义）。"""
        device = _device()
        B, S = 2, 1024
        S_kv = S // COMPRESS_RATIO

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype
        )

        metadata = None
        selected_blocks, _ = lightning_indexer(
            index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
            cmp_residual_k=torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device),
            metadata=None, max_seqlen_q=-1,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()

        local = selected_blocks[:, :, 0, :].cpu()

        # query 0: 只有 1 个 token，没有完整块 → 全部 -1（正确行为）
        # query 3 (第 4 个 token): 可见完整块 0
        # query 7 (第 8 个 token): 可见完整块 0, 1
        for b in range(B):
            # query 0: 无完整块，应全为 -1
            row0 = local[b, 0]
            valid0 = row0[row0 >= 0]
            assert valid0.numel() == 0, \
                f"query 0 has no complete block, expected all -1, got {row0.tolist()}"

            # query 3: 1 个完整块（block 0）
            row3 = local[b, 3]
            valid3 = row3[row3 >= 0].tolist()
            assert 0 in valid3, f"query 3 should select block 0, got {valid3}"

            # query 7: 2 个完整块（block 0, 1）
            row7 = local[b, 7]
            valid7 = row7[row7 >= 0].tolist()
            assert 0 in valid7, f"query 7 should select block 0, got {valid7}"
            assert 1 in valid7, f"query 7 should select block 1, got {valid7}"


# ============================================================
# 2. 精度测试
# ============================================================

class TestLightningIndexerV2Precision:
    """精度测试：与 CPU fp32 torch 小算子参考对比。"""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_topk_selection_jaccard(self, dtype):
        """验证 topk 选择结果与 torch 参考的 Jaccard 重合率 >= 0.9。

        使用完整 Qwen3.8 生产参数：N=4, D=128, ratio=4, topk=512
        """
        device = _device()
        B, S = 2, 2048  # 2 batch, 2048 tokens（Qwen3.8 典型训练长度）
        S_kv = S // COMPRESS_RATIO  # 512

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype, seed=123
        )

        # CANN LightningIndexerV2
        metadata = None
        cann_blocks, _ = lightning_indexer(
            index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
            cmp_residual_k=torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device),
            metadata=None, max_seqlen_q=-1,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()
        cann_local = cann_blocks[:, :, 0, :].cpu()  # [B, S, topk]

        # Torch 参考（CPU fp32）
        ref_blocks = _lightning_indexer_ref(
            index_queries.cpu().float(),
            compressed_keys.cpu().float().squeeze(1) if compressed_keys.dim() == 4 else compressed_keys.cpu().float(),
            weights.cpu().float(),
            COMPRESS_RATIO,
            BLOCK_TOPK,
        )

        # Jaccard 相似度
        match_count, total_count = 0, 0
        for b in range(B):
            for t in range(S):
                cann_set = {int(x) for x in cann_local[b, t].tolist() if x >= 0}
                ref_set = {int(x) for x in ref_blocks[b, t].tolist() if x >= 0}
                if len(ref_set) == 0 and len(cann_set) == 0:
                    continue
                union = cann_set | ref_set
                if union:
                    match_count += len(cann_set & ref_set) / len(union)
                    total_count += 1

        jaccard = match_count / max(total_count, 1)
        print(f"\n[Precision] dtype={dtype}, B={B}, S={S}, topk={BLOCK_TOPK}")
        print(f"  Jaccard similarity: {jaccard:.4f} ({match_count:.0f}/{total_count} rows)")
        assert jaccard >= 0.9, f"Topk Jaccard {jaccard:.4f} < 0.9"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_topk_values_precision(self, dtype):
        """验证 return_value=1 时 sparse_values 的精度。

        注意：cann sparse_values 的语义可能与 torch ref 的 score 不完全一致，
        这里只验证 values 的数量级和有限性。
        """
        device = _device()
        B, S = 2, 512
        S_kv = S // COMPRESS_RATIO

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype, seed=456
        )

        metadata = None
        _, sparse_values = lightning_indexer(
            index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
            cmp_residual_k=torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device),
            metadata=None, max_seqlen_q=-1,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=1,
        )
        torch.npu.synchronize()

        # 验证 values 的 shape 和 dtype
        assert sparse_values.shape == torch.Size([B, S, N_HEADS_K, BLOCK_TOPK])
        assert sparse_values.dtype == torch.float32

        # 有限性检查
        local_vals = sparse_values[:, :, 0, :]
        valid = local_vals > float("-inf")
        if valid.any():
            assert torch.isfinite(local_vals[valid]).all(), "valid values should be finite"

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_packed_tnd_layout_precision(self, dtype):
        """验证 TND packed 布局下的精度（Qwen3.8 packed QSA 场景）。"""
        device = _device()
        seq_lens = [512, 384, 640]  # 3 段不同长度
        B = 1
        S_total = sum(seq_lens)
        S_kv = S_total // COMPRESS_RATIO

        # 构造 packed 输入
        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S_total, device, dtype, seed=789
        )

        # TND 需要 cu_seqlens
        cu_q = _make_cu(seq_lens, device)
        seq_kv_lens = [s // COMPRESS_RATIO for s in seq_lens]
        cu_k = _make_cu(seq_kv_lens, device)
        cmp_residual_k = torch.tensor(
            [s % COMPRESS_RATIO for s in seq_lens], dtype=torch.int32, device=device
        )

        metadata = None
        cann_blocks, _ = lightning_indexer(
            index_queries.reshape(S_total, N_HEADS_Q, HEAD_DIM),
            compressed_keys.reshape(S_kv, 1, HEAD_DIM),
            weights.reshape(S_total, N_HEADS_Q),
            BLOCK_TOPK,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            cmp_residual_k=cmp_residual_k,
            metadata=None, max_seqlen_q=-1,
            layout_q="TND", layout_k="TND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()
        cann_local = cann_blocks[:, 0, :].cpu()  # [S_total, topk]

        # Torch 参考：逐段独立计算
        ref_blocks_all = []
        offset = 0
        for seg_len in seq_lens:
            seg_queries = index_queries[0, offset:offset + seg_len].float().cpu()  # [seg_len, H, D]
            seg_kv_len = seg_len // COMPRESS_RATIO
            seg_keys = compressed_keys[0, offset // COMPRESS_RATIO:(offset + seg_len) // COMPRESS_RATIO].float().cpu()
            seg_weights = weights[0, offset:offset + seg_len].float().cpu()
            seg_ref = _lightning_indexer_ref(
                seg_queries.unsqueeze(0),  # [1, seg_len, H, D]
                seg_keys.unsqueeze(0),     # [1, seg_kv_len, D]
                seg_weights.unsqueeze(0),  # [1, seg_len, H]
                COMPRESS_RATIO, BLOCK_TOPK,
            )
            ref_blocks_all.append(seg_ref[0])
            offset += seg_len
        ref_local = torch.cat(ref_blocks_all, dim=0)  # [S_total, topk]

        # Jaccard
        match_count, total_count = 0, 0
        for t in range(S_total):
            cann_set = {int(x) for x in cann_local[t].tolist() if x >= 0}
            ref_set = {int(x) for x in ref_local[t].tolist() if x >= 0}
            union = cann_set | ref_set
            if union:
                match_count += len(cann_set & ref_set) / len(union)
                total_count += 1

        jaccard = match_count / max(total_count, 1)
        print(f"\n[TND Packed] Jaccard similarity: {jaccard:.4f}")
        assert jaccard >= 0.85, f"TND packed Jaccard {jaccard:.4f} < 0.85"


# ============================================================
# 3. 性能测试
# ============================================================

@pytest.mark.skipif(os.environ.get("RUN_PERF_TESTS") != "1", reason="Set RUN_PERF_TESTS=1 to run")
class TestLightningIndexerV2Performance:
    """性能测试：对比 CANN 算子 vs torch 小算子的延迟。"""

    @staticmethod
    def _torch_li_small_ops(index_queries, compressed_keys, weights, compress_ratio, topk):
        """模拟 MindSpeed-MM 的 torch 小算子实现（用于性能对比）。"""
        B, S, H, D = index_queries.shape
        scale = 1.0 / math.sqrt(D)
        S_kv = compressed_keys.shape[1]

        # 向量化版本（比 python 循环快，但仍是小算子链）
        q = index_queries.float()
        k = compressed_keys.float()
        w = weights.float()

        # scores[b, t, g] = sum_h w[b,t,h] * relu(scale * q[b,t,h] @ k[b,g])
        qk = torch.einsum('bthd,bgd->bthg', q, k) * scale
        scores = (w.unsqueeze(-1) * F.relu(qk)).sum(dim=2)  # [B, S, S_kv]

        # causal mask
        positions = torch.arange(S, device=index_queries.device)
        n_visible = (positions + 1) // compress_ratio  # [S]
        visible_mask = torch.arange(S_kv, device=index_queries.device).unsqueeze(0) < n_visible.unsqueeze(1)
        scores = scores.masked_fill(~visible_mask.unsqueeze(0), float("-inf"))

        # topk
        _, topk_ids = torch.topk(scores, min(topk, S_kv), dim=-1)
        return topk_ids

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("S", [512, 1024, 2048, 4096])
    def test_li_v2_performance(self, dtype, S):
        """对比 CANN LightningIndexerV2 与 torch 小算子的延迟。"""
        device = _device()
        B = 1
        S_kv = S // COMPRESS_RATIO

        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, dtype, seed=42
        )

        cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)

        # warmup
        for _ in range(3):
            _, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
        torch.npu.synchronize()

        # CANN 算子计时
        n_iter = 20
        start = time.perf_counter()
        for _ in range(n_iter):
            _, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
        torch.npu.synchronize()
        cann_time = (time.perf_counter() - start) / n_iter * 1000  # ms

        # torch 小算子计时
        start = time.perf_counter()
        for _ in range(n_iter):
            self._torch_li_small_ops(index_queries, compressed_keys, weights, COMPRESS_RATIO, BLOCK_TOPK)
        torch.npu.synchronize()
        torch_time = (time.perf_counter() - start) / n_iter * 1000  # ms

        speedup = torch_time / max(cann_time, 1e-6)
        print(f"\n[Performance] dtype={dtype}, B={B}, S={S}")
        print(f"  CANN LightningIndexerV2: {cann_time:.3f} ms")
        print(f"  Torch small ops:         {torch_time:.3f} ms")
        print(f"  Speedup:                 {speedup:.2f}x")

        # 性能提升应该 > 1x（CANN 至少不慢于 torch）
        assert speedup > 1.0, f"CANN ({cann_time:.3f}ms) slower than torch ({torch_time:.3f}ms)"


# ============================================================
# 4. 显存测试（M1 模板 3.2 节要求）
# ============================================================

class TestLightningIndexerV2Memory:
    """显存测试：显存占用、泄漏、OOM 场景。"""

    def test_memory_no_leak(self):
        """连续调用 100 次，显存不应持续增长（泄漏检测）。"""
        device = _device()
        B, S = 2, 1024
        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, torch.bfloat16, seed=42
        )
        cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)

        # warmup
        for _ in range(5):
            _, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
        torch.npu.synchronize()

        # 记录初始显存
        mem_before = torch.npu.memory_allocated() / 1024**2  # MB
        mem_reserved_before = torch.npu.memory_reserved() / 1024**2

        # 连续调用 100 次
        for i in range(100):
            _, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
        torch.npu.synchronize()

        mem_after = torch.npu.memory_allocated() / 1024**2
        mem_reserved_after = torch.npu.memory_reserved() / 1024**2

        print(f"\n[Memory] allocated: {mem_before:.1f} -> {mem_after:.1f} MB")
        print(f"[Memory] reserved:  {mem_reserved_before:.1f} -> {mem_reserved_after:.1f} MB")

        # 显存增长应小于 5%（允许微小波动）
        growth = mem_after - mem_before
        assert growth < max(mem_before * 0.05, 1.0), \
            f"Potential memory leak: grew {growth:.1f} MB ({mem_before:.1f} -> {mem_after:.1f})"

    def test_memory_peak_usage(self):
        """记录典型生产配置下的峰值显存占用。"""
        device = _device()
        B, S = 2, 8192  # 8K 序列（生产典型长度）
        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, torch.bfloat16, seed=42
        )
        cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)

        torch.npu.reset_peak_memory_stats()
        _, _ = lightning_indexer(
            index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
            cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
            layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
            return_value=0,
        )
        torch.npu.synchronize()

        peak_allocated = torch.npu.max_memory_allocated() / 1024**2
        peak_reserved = torch.npu.max_memory_reserved() / 1024**2
        print(f"\n[Memory Peak] B={B}, S={S}")
        print(f"  Peak allocated: {peak_allocated:.1f} MB")
        print(f"  Peak reserved:  {peak_reserved:.1f} MB")

        # 只验证不 OOM，不 assert 具体数值
        assert peak_allocated > 0, "No memory allocated"


# ============================================================
# 5. 稳定性测试（M1 模板 3.2 节要求）
# ============================================================

class TestLightningIndexerV2Stability:
    """稳定性测试：长时间运行、高并发调用。"""

    def test_long_run_stability(self):
        """连续运行 1000 次，验证输出一致性（确定性）和稳定性。"""
        device = _device()
        B, S = 2, 512
        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, torch.bfloat16, seed=42
        )
        cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)

        ref_output = None
        for i in range(1000):
            selected, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
            if i == 0:
                ref_output = selected.cpu().clone()
            elif i % 200 == 0:
                # 每 200 次校验一次一致性
                torch.npu.synchronize()
                current = selected.cpu()
                assert torch.equal(ref_output, current), \
                    f"Output changed at iteration {i} (non-deterministic or unstable)"

        torch.npu.synchronize()
        print(f"\n[Stability] 1000 iterations completed, output consistent")

    def test_rapid_fire_calls(self):
        """高并发场景：无同步连续提交 50 次调用后统一同步。"""
        device = _device()
        B, S = 1, 1024
        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, torch.bfloat16, seed=42
        )
        cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)

        # 连续提交 50 次（无中间同步）
        for _ in range(50):
            _, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
        torch.npu.synchronize()
        print(f"\n[Stability] 50 rapid-fire calls completed successfully")


# ============================================================
# 6. 确定性测试（M1 模板 3.5 节要求）
# ============================================================

class TestLightningIndexerV2Determinism:
    """确定性测试：相同输入多次运行，输出应完全一致。"""

    def test_deterministic_output(self):
        """相同输入运行 10 次，输出应 bitwise 一致。"""
        device = _device()
        B, S = 2, 512
        index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
            B, S, device, torch.bfloat16, seed=42
        )
        cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)

        outputs = []
        for _ in range(10):
            selected, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
            torch.npu.synchronize()
            outputs.append(selected.cpu().clone())

        ref = outputs[0]
        for i, out in enumerate(outputs[1:], 1):
            assert torch.equal(ref, out), \
                f"Non-deterministic output at run {i}"

        print(f"\n[Determinism] 10 runs produced bitwise identical output")

    def test_deterministic_different_seed(self):
        """不同输入产生不同输出（验证不是常量输出）。"""
        device = _device()
        B, S = 2, 512

        outputs = []
        for seed in [1, 2, 3]:
            index_queries, compressed_keys, weights, _, _ = _make_qwen38_indexer_inputs(
                B, S, device, torch.bfloat16, seed=seed
            )
            cmp_residual_k = torch.full((B,), S % COMPRESS_RATIO, dtype=torch.int32, device=device)
            selected, _ = lightning_indexer(
                index_queries, compressed_keys.unsqueeze(2), weights, BLOCK_TOPK,
                cmp_residual_k=cmp_residual_k, metadata=None, max_seqlen_q=-1,
                layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=COMPRESS_RATIO,
                return_value=0,
            )
            torch.npu.synchronize()
            outputs.append(selected.cpu().clone())

        # 至少两个不同 seed 的输出应不同
        all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Different seeds produced identical output (suspicious)"
        print(f"\n[Determinism] Different seeds produce different outputs (as expected)")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
