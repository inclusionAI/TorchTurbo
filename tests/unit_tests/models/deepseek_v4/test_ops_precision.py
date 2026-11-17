"""DeepSeekV4 op-level precision checks against the infer reference.

This complements test_alignment.py:

* test_alignment.py compares end-to-end logits.
* this file compares the smaller attention/indexing/compression building blocks.

Usage:
    python tests/models/deepseek_v4/test_ops_precision.py --device cpu
"""

import argparse
import importlib.util
import math
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        pytest.skip("torch is required for DeepSeekV4 op precision tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
INFER_REFERENCE_DIR = Path(__file__).resolve().parent / "inference"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def act_quant_patched(x, block_size, scale_fmt=None, scale_dtype=torch.float32, in_place=False):
    return x, torch.ones(1, dtype=scale_dtype, device=x.device)


def fp4_act_quant_patched(x, fp4_block_size, in_place=False):
    return x


def fp8_gemm_patched(x, s, weight, weight_scale, scale_dtype):
    return F.linear(x, weight)


def fp4_gemm_patched(x, s, weight, weight_scale, scale_dtype):
    return F.linear(x.float(), weight.float())


def sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale):
    bsz, seqlen, n_heads, _head_dim = q.size()
    q = q.transpose(1, 2)
    kv = kv.unsqueeze(1)
    attn_weights = torch.matmul(q, kv.transpose(2, 3)) * softmax_scale

    index_mask = torch.full(
        (bsz, 1, seqlen, kv.shape[2] + 1),
        fill_value=torch.finfo(attn_weights.dtype).min,
        dtype=attn_weights.dtype,
        device=q.device,
    )
    valid_mask = topk_idxs >= 0
    if valid_mask.any():
        b_indices, s_indices, k_indices = torch.where(valid_mask)
        valid_topk = topk_idxs[b_indices, s_indices, k_indices]
        index_mask[b_indices, 0, s_indices, valid_topk] = 0

    attn_weights = attn_weights + index_mask[..., :-1]
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(bsz, -1, seqlen, -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits.float(), dim=-1, dtype=attn_weights.dtype)
    out = torch.matmul(probs[..., :-1], kv)
    return out.transpose(1, 2).contiguous()


def hc_split_sinkhorn_patched(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def install_reference_mocks():
    kernel = types.ModuleType("kernel")
    kernel.act_quant = act_quant_patched
    kernel.fp4_act_quant = fp4_act_quant_patched
    kernel.fp8_gemm = fp8_gemm_patched
    kernel.fp4_gemm = fp4_gemm_patched
    kernel.sparse_attn = sparse_attn_ref
    kernel.hc_split_sinkhorn = hc_split_sinkhorn_patched
    sys.modules["kernel"] = kernel

    fast_hadamard_transform = types.ModuleType("fast_hadamard_transform")
    fast_hadamard_transform.hadamard_transform = lambda x, scale=1.0: hadamard_transform_impl(x, scale)
    fast_hadamard_transform.__spec__ = importlib.util.spec_from_loader(
        "fast_hadamard_transform", loader=None
    )
    sys.modules["fast_hadamard_transform"] = fast_hadamard_transform


def load_infer_reference():
    install_reference_mocks()
    model_path = INFER_REFERENCE_DIR / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"infer reference model.py not found: {model_path}")
    module_name = "deepseek_v4_infer_reference"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.world_size = 1
    module.rank = 0
    module.default_dtype = torch.float32
    module.scale_fmt = None
    module.scale_dtype = torch.float32
    module.rotate_activation = lambda x: hadamard_transform_impl(x, scale=x.size(-1) ** -0.5)
    return module


def patch_modeling_hadamard(modeling_module):
    modeling_module.hadamard_transform_ref = lambda x, hadamard_mat, scale=1.0: hadamard_transform_impl(
        x, scale
    )
    modeling_module.rotate_activation = lambda x, hadamard_mat: hadamard_transform_impl(
        x, scale=x.size(-1) ** -0.5
    )


def hadamard_transform_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    x_shape = x.shape
    dim = x.shape[-1]
    x_flat = x.float().reshape(-1, dim)
    dim_padded = 2 ** math.ceil(math.log2(dim))
    if dim != dim_padded:
        x_flat = F.pad(x_flat, (0, dim_padded - dim))
    h = torch.tensor([[1.0]], dtype=torch.float32, device=x.device)
    for _ in range(int(math.log2(dim_padded))):
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    out = F.linear(x_flat, h) * scale
    return out[..., :dim].reshape(*x_shape).to(x.dtype)


def modeling_config_dict():
    return {
        "vocab_size": 1024,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "qk_rope_head_dim": 16,
        "q_lora_rank": 64,
        "o_lora_rank": 64,
        "o_groups": 2,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sqrtsoftplus",
        "route_scale": 1.0,
        "aux_loss_alpha": 0.0,
        "sliding_window": 8,
        "compress_ratios": [4, 128],
        "index_head_dim": 32,
        "index_n_heads": 4,
        "index_topk": 4,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 0,
        "compress_rope_theta": 40000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 0,
        "swiglu_limit": 10.0,
        "hc_eps": 1e-6,
        "initializer_range": 0.02,
        "use_cache": False,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def infer_config(reference_module):
    return reference_module.ModelArgs(
        vocab_size=1024,
        dim=128,
        n_layers=2,
        n_heads=4,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        moe_inter_dim=128,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=1.0,
        score_func="sqrtsoftplus",
        swiglu_limit=10.0,
        window_size=8,
        compress_ratios=(4, 128),
        index_head_dim=32,
        index_n_heads=4,
        index_topk=4,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=128,
        original_seq_len=0,
        compress_rope_theta=40000.0,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        n_hash_layers=1,
        n_mtp_layers=0,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        scale_dtype="fp32",
        max_batch_size=4,
    )


def assert_close(name, actual, expected, atol=1e-5, rtol=1e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    diff = (actual.float() - expected.float()).abs()
    print(f"{name}: max_diff={diff.max().item():.6e}, mean_diff={diff.mean().item():.6e}")


def make_modules(device):
    from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as modeling_module
    from torchturbo.models.deepseek_v4 import packing_utils

    patch_modeling_hadamard(modeling_module)
    reference_module = load_infer_reference()
    config = DeepseekV4Config(**modeling_config_dict())
    return modeling_module, reference_module, config, infer_config(reference_module)


# The packed op forwards live in TurboDeepseekV4ForCausalLM, which installs
# them as class-level patches during parallelization. Op precision checks must
# run against those packed forwards, not the plain modeling defaults, and must
# restore the originals afterwards so other tests in the same process keep the
# unpatched modeling behavior.
_PATCHED_OP_CLASSES = (
    "DeepseekV4Compressor",
    "DeepseekV4Indexer",
    "GetWindowTopkIdxs",
    "GetCompressTopkIdxs",
    "LiCompute",
    "InnerAttentionOps",
    "PreAttentionOps",
    "PostAttentionOps",
    "SparseAttention",
    "DeepseekV4Attention",
)


@contextmanager
def packed_attention_ops():
    from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as modeling_module
    from torchturbo.models.deepseek_v4.model import TurboDeepseekV4ForCausalLM

    saved_forwards = {
        name: getattr(modeling_module, name).forward for name in _PATCHED_OP_CLASSES
    }
    saved_scaler = modeling_module.DSAIndexerLossAutoScaler
    TurboDeepseekV4ForCausalLM.patch_deepseek_v4_attention_for_sp(None, 1, 0)
    try:
        yield
    finally:
        for name, forward in saved_forwards.items():
            getattr(modeling_module, name).forward = forward
        modeling_module.DSAIndexerLossAutoScaler = saved_scaler


@pytest.fixture(autouse=True)
def _packed_attention_ops():
    with packed_attention_ops():
        yield


def copy_compressor_weights(src, dst):
    dst.ape.data.copy_(src.ape.data)
    dst.wkv.weight.data.copy_(src.wkv.weight.data)
    dst.wgate.weight.data.copy_(src.wgate.weight.data)
    dst.norm.weight.data.copy_(src.norm.weight.data)


def copy_indexer_weights(src, dst):
    dst.wq_b.weight.data.copy_(src.wq_b.weight.data)
    dst.weights_proj.weight.data.copy_(src.weights_proj.weight.data)
    copy_compressor_weights(src.compressor, dst.compressor)


def test_topk_helpers(device="cpu"):
    modeling_module, reference_module, _config, _args = make_modules(device)
    bsz, seqlen, window_size, ratio, offset = 2, 16, 8, 4, 16
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones_like(position_ids)

    actual_window = modeling_module.GetWindowTopkIdxs()(
        window_size, bsz, seqlen, torch.device(device), segment_ids=attention_mask
    )
    expected_window = reference_module.get_window_topk_idxs(window_size, bsz, seqlen, start_pos=0).to(device)
    torch.testing.assert_close(actual_window, expected_window)

    x = torch.empty(bsz, seqlen, 1, device=device)
    actual_compress = modeling_module.GetCompressTopkIdxs(ratio)(
        x,
        offset,
        q_global_positions=position_ids,
        segment_ids=attention_mask,
    )
    expected_compress = reference_module.get_compress_topk_idxs(
        ratio, bsz, seqlen, start_pos=0, offset=offset
    ).to(device)
    torch.testing.assert_close(actual_compress, expected_compress)
    print("topk_helpers: matched")


def test_sparse_attention(device="cpu"):
    torch.manual_seed(7)
    modeling_module, reference_module, config, _args = make_modules(device)
    bsz, seqlen = 2, 16
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones_like(position_ids)
    n_heads, head_dim = config.num_attention_heads, config.head_dim
    offset = seqlen

    q = torch.randn(bsz, seqlen, n_heads, head_dim, device=device)
    kv = torch.randn(bsz, seqlen, head_dim, device=device)
    kv_compress = torch.randn(bsz, seqlen // 4, head_dim, device=device)
    attn_sink = torch.randn(n_heads, device=device)

    window_topk = reference_module.get_window_topk_idxs(config.sliding_window, bsz, seqlen, 0).to(device)
    compress_topk = reference_module.get_compress_topk_idxs(4, bsz, seqlen, 0, offset).to(device)
    topk = torch.cat([window_topk, compress_topk], dim=-1).int()

    actual = modeling_module.SparseAttention(config, layer_idx=0)(
        q, kv, attn_sink, position_ids, attention_mask, kv_compress=kv_compress, compress_topk_idxs=compress_topk
    )
    expected = reference_module.sparse_attn(
        q, torch.cat([kv, kv_compress], dim=1), attn_sink, topk, head_dim**-0.5
    )
    assert_close("sparse_attention", actual, expected)


def test_compressor_prefill(device="cpu"):
    torch.manual_seed(11)
    modeling_module, reference_module, config, args = make_modules(device)
    bsz, seqlen = 2, 16
    x = torch.randn(bsz, seqlen, config.hidden_size, device=device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones_like(position_ids)
    freqs_cis = modeling_module.DeepseekV4Attention(config, layer_idx=0).freqs_cis.to(device)

    for rotate in (False, True):
        actual_module = modeling_module.DeepseekV4Compressor(
            config, compress_ratio=4, head_dim=config.head_dim, rotate=rotate
        ).to(device).float()
        # Uninitialized torch.empty weights carry garbage values that can
        # overflow the softmax to NaN; seed-bounded init keeps the run stable.
        actual_module.init_weights(0.02)
        expected_module = reference_module.Compressor(
            args, compress_ratio=4, head_dim=args.head_dim, rotate=rotate
        ).to(device).float()
        copy_compressor_weights(actual_module, expected_module)
        expected_module.kv_cache = torch.empty(bsz, seqlen // 4, args.head_dim, device=device)
        expected_module.freqs_cis = freqs_cis

        actual = actual_module(x, freqs_cis, attention_mask, position_ids)
        if rotate:
            actual = hadamard_transform_impl(actual, scale=actual.size(-1) ** -0.5)
        expected = expected_module(x, start_pos=0)
        assert_close(f"compressor_prefill_rotate_{rotate}", actual, expected)


def test_indexer_prefill_topk(device="cpu"):
    torch.manual_seed(13)
    modeling_module, reference_module, config, args = make_modules(device)
    bsz, seqlen = 2, 16
    offset = seqlen
    x = torch.randn(bsz, seqlen, config.hidden_size, device=device)
    qr = torch.randn(bsz, seqlen, config.q_lora_rank, device=device)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    attention_mask = torch.ones_like(position_ids)
    attn = modeling_module.DeepseekV4Attention(config, layer_idx=0).to(device)
    freqs_cis = attn.freqs_cis

    actual_indexer = modeling_module.DeepseekV4Indexer(config, compress_ratio=4).to(device).float()
    actual_indexer.init_weights(0.02)
    expected_indexer = reference_module.Indexer(args, compress_ratio=4).to(device).float()
    copy_indexer_weights(actual_indexer, expected_indexer)
    expected_indexer.freqs_cis = freqs_cis

    q_indexer, k_indexer, weights = actual_indexer(
        x, qr, freqs_cis, attn.hadamard_mat, offset, position_ids, attention_mask
    )
    actual_topk, _index_score = modeling_module.LiCompute(4, config.index_topk)(
        q_indexer,
        k_indexer,
        weights,
        seqlen,
        offset,
        q_global_positions=position_ids,
        segment_ids=attention_mask,
    )
    expected_topk = expected_indexer(x, qr, start_pos=0, offset=offset)
    torch.testing.assert_close(actual_topk, expected_topk)
    print("indexer_prefill_topk: matched")


def test_packed_compress_metadata():
    from torchturbo.models.deepseek_v4 import packing_utils

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2],
            [0, 0, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    position_ids = torch.arange(attention_mask.size(1), dtype=torch.long).unsqueeze(0).expand_as(attention_mask)
    modeling_module, _reference_module, _config, _args = make_modules("cpu")

    for ratio, expected_ids, expected_positions in (
        (
            2,
            [[1, 1, 2, 2, 0, 0], [3, 3, 3, 4, 0, 0]],
            [[0, 2, 7, 9, 0, 0], [2, 4, 6, 9, 0, 0]],
        ),
        (
            4,
            [[1, 2, 0], [3, 0, 0]],
            [[0, 7, 0], [2, 0, 0]],
        ),
        (
            5,
            [[1, 2, 0], [3, 0, 0]],
            [[0, 7, 0], [2, 0, 0]],
        ),
    ):
        actual_ids, actual_positions = packing_utils.packed_compress_kv_ids_and_positions(
            attention_mask,
            ratio,
            position_ids,
        )
        assert torch.equal(actual_ids, torch.tensor(expected_ids))
        assert torch.equal(actual_positions, torch.tensor(expected_positions))
    print("packed_compress_metadata: matched")


def main():
    parser = argparse.ArgumentParser(description="DeepSeekV4 op-level precision checks")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    args = parser.parse_args()

    print("=" * 70)
    print("DeepSeek V4 Op Precision Test")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Infer reference: {INFER_REFERENCE_DIR / 'model.py'}")

    with packed_attention_ops():
        test_topk_helpers(args.device)
        test_sparse_attention(args.device)
        test_packed_compress_metadata()
        test_compressor_prefill(args.device)
        test_indexer_prefill_topk(args.device)
    print("All op precision tests passed")


if __name__ == "__main__":
    main()
