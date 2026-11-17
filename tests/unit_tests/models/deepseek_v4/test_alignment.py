"""Align DeepSeekV4 v3 modeling with the infer-style reference.

The infer reference is the original DeepSeek V4 inference model, kept under:

    tests/models/deepseek_v4/inference/model.py

The modeling side uses the v3 implementation under:

    src/torchturbo/models/deepseek_v4/modeling_deepseek_v4.py

Usage:
    python tests/models/deepseek_v4/test_alignment.py --device cpu --seq-len 128

    DEVICE=cpu SEQ_LEN=128 python tests/models/deepseek_v4/test_alignment.py
"""

import argparse
import os
import sys
import math
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeekV4 alignment tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

# __file__ = <repo>/tests/models/deepseek_v4/test_alignment.py
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
INFER_REFERENCE_DIR = Path(__file__).resolve().parent / "inference"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Small model config for testing
TEST_CONFIG_MODELING = {
    "vocab_size": 1024,
    "hidden_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "qk_rope_head_dim": 32,
    "q_lora_rank": 256,
    "o_lora_rank": 256,
    "o_groups": 4,
    "intermediate_size": 1024,
    "moe_intermediate_size": 512,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 1.0,
    "scoring_func": "sqrtsoftplus",
    "route_scale": 1.0,
    "aux_loss_alpha": 0.0,
    "sliding_window": 64,
    "compress_ratios": [4, 128],
    "index_head_dim": 64,
    "index_n_heads": 8,
    "index_topk": 16,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1024,
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

TEST_CONFIG_MODEL = {
    "vocab_size": 1024,
    "dim": 512,
    "n_layers": 2,
    "n_heads": 8,
    "head_dim": 64,
    "rope_head_dim": 32,
    "q_lora_rank": 256,
    "o_lora_rank": 256,
    "o_groups": 4,
    "moe_inter_dim": 512,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
    "route_scale": 1.0,
    "score_func": "sqrtsoftplus",
    "swiglu_limit": 10.0,
    "window_size": 64,
    "compress_ratios": (4, 128),
    "index_head_dim": 64,
    "index_n_heads": 8,
    "index_topk": 16,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_seq_len": 1024,
    "original_seq_len": 0,
    "compress_rope_theta": 40000.0,
    "rope_factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "n_hash_layers": 1,
    "n_mtp_layers": 0,
    "dtype": "bf16",
    "scale_fmt": None,
    "expert_dtype": None,
    "scale_dtype": "fp32",
    "max_batch_size": 4,
}


def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


# ==============================================================================
# Monkey-patch kernel functions for model.py
# ==============================================================================

def act_quant_patched(x, block_size, scale_fmt=None, scale_dtype=torch.float32, in_place=False):
    """Replace act_quant with identity for alignment testing."""
    # Return dummy scale tensor
    return x, torch.ones(1, dtype=scale_dtype, device=x.device)


def fp4_act_quant_patched(x, fp4_block_size, in_place=False):
    """Replace fp4_act_quant with identity for alignment testing."""
    return x


def fp8_gemm_patched(x, s, weight, weight_scale, scale_dtype):
    """Replace fp8_gemm with standard linear."""
    return F.linear(x, weight)


def fp4_gemm_patched(x, s, weight, weight_scale, scale_dtype):
    """Replace fp4_gemm with standard linear."""
    # FP4 weights are packed, need special handling
    return F.linear(x.float(), weight.float())


def sparse_attn_ref_patched(q, kv, attn_sink, topk_idxs, softmax_scale):
    """
    Reference sparse attention using native PyTorch.
    Matches the implementation in modeling_deepseek_v4.py SparseAttention.forward()

    Args:
        q: [bsz, seqlen, n_heads, head_dim] - query states
        kv: [bsz, kv_len, head_dim] - key/value states (sliding window + compressed)
        attn_sink: [n_heads] - attention sink parameter
        topk_idxs: [bsz, seqlen, topk] - indices to attend to (window + compressed)
        softmax_scale: float - softmax scaling factor (head_dim ** -0.5)

    Returns:
        attn_output: [bsz, seqlen, n_heads, head_dim]
    """
    # Handle both prefill and decode phases
    if q.dim() == 4:
        bsz, seqlen, n_heads, head_dim = q.shape
    else:
        # Decode phase: [bsz_seqlen, n_heads, head_dim]
        bsz_seqlen, n_heads, head_dim = q.shape
        bsz = 1
        seqlen = bsz_seqlen
        q = q.unsqueeze(0)
        kv = kv.unsqueeze(0) if kv.dim() == 2 else kv
        topk_idxs = topk_idxs.unsqueeze(0) if topk_idxs.dim() == 2 else topk_idxs

    kv_len = kv.size(1)
    device = q.device
    dtype = q.dtype

    # Step 1: Transpose q for attention computation
    # [bsz, seqlen, n_heads, head_dim] -> [bsz, n_heads, seqlen, head_dim]
    q_transposed = q.transpose(1, 2)

    # Step 2: Expand kv for broadcasting
    # [bsz, kv_len, head_dim] -> [bsz, 1, kv_len, head_dim]
    kv_expanded = kv.unsqueeze(1)

    # Step 3: Compute attention weights
    # [bsz, n_heads, seqlen, head_dim] @ [bsz, 1, head_dim, kv_len]
    # -> [bsz, n_heads, seqlen, kv_len]
    attn_weights = torch.matmul(q_transposed, kv_expanded.transpose(2, 3)) * softmax_scale

    # Step 4: Apply top-k mask (sparse attention)
    # Create mask with -inf for positions NOT in topk_idxs
    index_mask = torch.full(
        (bsz, 1, seqlen, kv_len + 1),  # +1 for attention sink
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    # Set valid indices to 0 (allow attention)
    for b in range(bsz):
        for s in range(seqlen):
            valid_indices = topk_idxs[b, s][topk_idxs[b, s] >= 0]
            if len(valid_indices) > 0:
                index_mask[b, 0, s, valid_indices] = 0

    # Add mask to attention weights (exclude last dim which is for sink)
    attn_weights = attn_weights + index_mask[..., :-1]

    # Step 5: Add attention sink
    # attn_sink: [n_heads] -> expand to [bsz, n_heads, seqlen, 1]
    sinks = attn_sink.reshape(1, -1, 1, 1).expand(bsz, -1, seqlen, 1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # Step 6: Subtract max for numerical stability
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

    # Step 7: Softmax
    probs = F.softmax(combined_logits.float(), dim=-1).to(dtype)

    # Step 8: Remove sink dimension probabilities
    scores = probs[..., :-1]

    # Step 9: Apply scores to values
    # [bsz, n_heads, seqlen, kv_len] @ [bsz, 1, kv_len, head_dim]
    # -> [bsz, n_heads, seqlen, head_dim]
    attn_output = torch.matmul(scores, kv_expanded)

    # Step 10: Transpose back
    # [bsz, n_heads, seqlen, head_dim] -> [bsz, seqlen, n_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


def hc_split_sinkhorn_ref_patched(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """Reference HC split sinkhorn using native PyTorch.
    Note: hc_mult, hc_sinkhorn_iters, hc_eps are passed but not used - they come from self."""
    # Get actual values from shape if not provided (they should always be provided from model.py)
    if hc_mult is None:
        hc_mult = (2 + int((-3 + math.sqrt(9 + 4 * (mixes.size(-1) - 2))) / 2))

    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * mixes.size(-1) // mix_hc

    # Split into pre, post, comb
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    post = 2 * F.sigmoid(
        post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0)
    )
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)

    return pre, post, comb


# Mock kernel module
class MockKernel:
    """Mock kernel module with patched functions."""
    act_quant = staticmethod(act_quant_patched)
    fp4_act_quant = staticmethod(fp4_act_quant_patched)
    fp8_gemm = staticmethod(fp8_gemm_patched)
    fp4_gemm = staticmethod(fp4_gemm_patched)
    sparse_attn = staticmethod(sparse_attn_ref_patched)
    hc_split_sinkhorn = staticmethod(hc_split_sinkhorn_ref_patched)


# Install mock
sys.modules['kernel'] = MockKernel()

# ==============================================================================
# Shared Hadamard transform — identical implementation used by BOTH model.py
# and modeling_deepseek_v4.py to eliminate numerical discrepancies.
#
# model.py calls:  rotate_activation(x)  ->  hadamard_transform(x, scale)
# modeling calls:  rotate_activation(x, hadamard_mat)  ->  hadamard_transform_ref(x, hadamard_mat, scale)
#
# We provide a single canonical implementation and inject it into both sides.
# ==============================================================================

def _generate_hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Generate Hadamard matrix (Sylvester construction, float32, no normalization)."""
    log_dim = int(math.log2(dim))
    H = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    for _ in range(log_dim):
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def hadamard_transform_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Canonical Hadamard transform matching modeling_deepseek_v4.py exactly.
    Computes in float32 for numerical consistency, returns in original dtype."""
    x_shape = x.shape
    dim = x.shape[-1]
    x_flat = x.float().reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x_flat = F.pad(x_flat, (0, dim_padded - dim))
    H = _generate_hadamard_matrix(dim_padded, x.device)
    out = F.linear(x_flat, H) * scale
    return out[..., :dim].reshape(*x_shape).to(x.dtype)


# Mock for model.py's fast_hadamard_transform.hadamard_transform
def hadamard_transform_mock(x, scale=1.0):
    return hadamard_transform_impl(x, scale)


import types as _types

_mock_fht = _types.ModuleType("fast_hadamard_transform")
_mock_fht.hadamard_transform = hadamard_transform_mock
# Set __spec__ so importlib.util.find_spec() doesn't crash
_mock_fht.__spec__ = __import__('importlib.util').util.spec_from_loader("fast_hadamard_transform", loader=None)
sys.modules['fast_hadamard_transform'] = _mock_fht


def get_modeling_model(config_dict, device='cpu', dtype=torch.bfloat16):
    """Create modeling_deepseek_v4 model with small config."""
    from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as modeling_module
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    # Monkey-patch modeling's hadamard functions to use the shared implementation,
    # ensuring both model.py and modeling use identical Hadamard transform.
    def _modeling_hadamard_transform_ref(x, hadamard_mat, scale=1.0):
        """Drop-in replacement matching the shared hadamard_transform_impl logic."""
        return hadamard_transform_impl(x, scale)

    def _modeling_rotate_activation(x, hadamard_mat):
        """Drop-in replacement for modeling's rotate_activation."""
        return hadamard_transform_impl(x, scale=x.size(-1) ** -0.5)

    modeling_module.hadamard_transform_ref = _modeling_hadamard_transform_ref
    modeling_module.rotate_activation = _modeling_rotate_activation

    config = DeepseekV4Config(**config_dict)
    model = DeepseekV4ForCausalLM(config)
    # Convert all parameters to float32 for alignment testing
    model = model.float()
    model.eval()
    return model, config


def get_model_model(config_dict, device='cpu', dtype=torch.bfloat16):
    """Create infer reference model.py model with small config."""
    if not (INFER_REFERENCE_DIR / "model.py").is_file():
        raise FileNotFoundError(f"infer reference model.py not found: {INFER_REFERENCE_DIR}")
    if str(INFER_REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(INFER_REFERENCE_DIR))
    from model import ModelArgs, Transformer

    config = ModelArgs(**config_dict)
    # Set globals for model.py
    import model as model_module
    model_module.world_size = 1
    model_module.rank = 0
    model_module.default_dtype = dtype
    model_module.scale_fmt = None
    model_module.scale_dtype = torch.float32

    # Monkey-patch rotate_activation to remove bf16 assert for fp32 alignment testing
    model_module.rotate_activation = lambda x: hadamard_transform_impl(x, scale=x.size(-1) ** -0.5)

    # model.py uses torch.empty() without explicit dtype for some layers
    # (e.g. ParallelEmbedding), so we must set the default dtype before
    # constructing the model to ensure all weights are created in bfloat16.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    model = Transformer(config)
    torch.set_default_dtype(prev_dtype)
    # # Convert all parameters to float32 for alignment testing
    model = model.float()
    model = model.to(device=device)
    model.eval()
    return model, config


def _collect_all_tensors(model):
    """Collect all named tensors (parameters + buffers + plain tensor attributes).

    This handles edge cases that named_parameters/named_buffers miss:
    - model.py uses @lru_cache for precompute_freqs_cis, causing shared tensor
      objects that PyTorch's named_buffers() deduplicates. We recover these by
      scanning module attributes directly.
    - Some tensors are plain Python attributes (not registered as buffers).
    """
    all_tensors = {}
    # Start with parameters and buffers
    for name, param in model.named_parameters():
        all_tensors[name] = param
    for name, buf in model.named_buffers():
        all_tensors[name] = buf

    # Scan for tensor attributes not captured by named_parameters/named_buffers.
    # This catches freqs_cis and hadamard_mat that may be plain attributes
    # or deduplicated by named_buffers() due to shared data_ptr.
    for module_name, module in model.named_modules():
        for attr_name in ('freqs_cis', 'hadamard_mat'):
            attr = getattr(module, attr_name, None)
            if attr is not None and isinstance(attr, torch.Tensor):
                # Build full name: "layers.0.attn.freqs_cis"
                full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                if full_name not in all_tensors:
                    all_tensors[full_name] = attr

    return all_tensors


def align_weights(source_model, target_model):
    """Copy weights from source model to target model (parameters + buffers).

    Handles cases where a tensor may be a buffer in one model but a parameter
    in the other (e.g. tid2eid is register_buffer in modeling but nn.Parameter
    in model.py).

    Also handles:
    - freqs_cis deduplication: model.py uses @lru_cache, causing shared tensor
      objects that named_buffers() skips. We recover them via attribute scanning.
    - dtype mismatches: source int64 vs target int32, source float32 vs target bfloat16.
    """
    source_all = _collect_all_tensors(source_model)
    target_all = _collect_all_tensors(target_model)

    copied = 0
    skipped = []
    for name, src_tensor in source_all.items():
        if name in target_all:
            tgt_tensor = target_all[name]
            # Allow shape match with dtype conversion
            if src_tensor.shape == tgt_tensor.shape:
                tgt_tensor.data.copy_(src_tensor.data.to(tgt_tensor.dtype))
                copied += 1
            else:
                skipped.append(f"{name}: shape mismatch {src_tensor.shape} vs {tgt_tensor.shape}")
        else:
            # Non-critical: hadamard_mat only exists in modeling (model.py uses C++ lib)
            # kv_state/score_state/kv_cache are runtime state, not needed for alignment
            if 'hadamard_mat' not in name and 'kv_state' not in name and 'score_state' not in name and 'kv_cache' not in name:
                skipped.append(f"{name}: not found in target")

    if skipped:
        print(f"  Skipped {len(skipped)} tensors:")
        for s in skipped:
            print(f"    - {s}")

    print(f"  Copied {copied} parameters/buffers (total source: {len(source_all)}, target: {len(target_all)})")
    return copied


def create_test_input(batch_size=2, seq_len=64, vocab_size=1024, device='cpu', seed=42):
    """Create test input."""
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids


def tensor_stats(t):
    """Get tensor statistics for comparison."""
    if t is None:
        return None
    t = t.detach().float()
    return {
        'shape': list(t.shape),
        'min': t.min().item(),
        'max': t.max().item(),
        'mean': t.mean().item(),
        'std': t.std().item(),
        'norm': t.norm().item(),
    }


def compare_outputs(out1, out2, name="output"):
    """Compare two tensors and print stats."""
    if out1 is None or out2 is None:
        print(f"  {name}: One output is None")
        return float('inf')

    if out1.shape != out2.shape:
        print(f"  {name}: Shape mismatch {out1.shape} vs {out2.shape}")
        return float('inf')

    diff = (out1.float() - out2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}", end="")
    if max_diff < 1e-5:
        print(" ✓")
    elif max_diff < 1e-3:
        print(" ~")
    else:
        print(" ✗")

    return max_diff


def run_forward_modeling(model, input_ids):
    """Run forward pass for modeling implementation."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        return outputs.logits


def run_forward_model(model, input_ids):
    """Run forward pass for model.py implementation.

    Monkey-patch ParallelHead.get_logits to compute logits for ALL positions
    (not just the last token) so the output shape matches modeling_deepseek_v4.
    """
    import model as model_module

    # Save original method
    original_get_logits = model_module.ParallelHead.get_logits

    # Patch: compute logits for all positions instead of just the last token
    def get_logits_all_positions(self, x):
        return F.linear(x.float(), self.weight)

    model_module.ParallelHead.get_logits = get_logits_all_positions

    with torch.no_grad():
        logits = model(input_ids, start_pos=0)

    # Restore original method
    model_module.ParallelHead.get_logits = original_get_logits

    return logits


def test_model_alignment_cpu():
    seed = 42
    device = "cpu"
    batch_size = 1
    seq_len = 128

    seed_all(seed)
    input_ids = create_test_input(batch_size, seq_len, TEST_CONFIG_MODELING["vocab_size"], device, seed)
    modeling_model, _ = get_modeling_model(TEST_CONFIG_MODELING, device)
    model_model, _ = get_model_model(TEST_CONFIG_MODEL, device)
    align_weights(modeling_model, model_model)

    modeling_out = run_forward_modeling(modeling_model, input_ids)
    model_out = run_forward_model(model_model, input_ids)
    torch.testing.assert_close(modeling_out, model_out, atol=1e-5, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser(description="DeepSeekV4 modeling vs infer-reference alignment")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", "1024")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    seq_len = args.seq_len
    seed = args.seed

    print("="*70)
    print("DeepSeek V4 Model Alignment Test")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Seed: {seed}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Infer reference: {INFER_REFERENCE_DIR / 'model.py'}")
    print()

    # Set seed
    seed_all(seed)

    # Create test input
    print("Creating test input...")
    input_ids = create_test_input(batch_size, seq_len, TEST_CONFIG_MODELING['vocab_size'], device, seed)
    print(f"  Input shape: {input_ids.shape}")
    print()

    # Create modeling implementation
    print("Creating modeling_deepseek_v4.py model...")
    try:
        modeling_model, modeling_config = get_modeling_model(TEST_CONFIG_MODELING, device)
        print(f"  Parameters: {sum(p.numel() for p in modeling_model.parameters())}")
        print("  ✓ Model created")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        modeling_model = None
    print()

    # Create model.py implementation
    print("Creating model.py model...")
    try:
        model_model, model_config = get_model_model(TEST_CONFIG_MODEL, device)
        print(f"  Parameters: {sum(p.numel() for p in model_model.parameters())}")
        print("  ✓ Model created")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        model_model = None
    print()

    # Copy weights from modeling to model
    if modeling_model is not None and model_model is not None:
        print("Aligning weights from modeling to model.py...")
        copied = align_weights(modeling_model, model_model)
        print()

    # Run forward pass
    modeling_out = None
    model_out = None

    if modeling_model is not None:
        print("Running modeling_deepseek_v4.py forward...")
        try:
            modeling_out = run_forward_modeling(modeling_model, input_ids)
            print(f"  Output shape: {modeling_out.shape}")
            print(f"  Output stats: {tensor_stats(modeling_out)}")
            print("  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
        print()

    if model_model is not None:
        print("Running model.py forward...")
        try:
            model_out = run_forward_model(model_model, input_ids)
            print(f"  Output shape: {model_out.shape}")
            print(f"  Output stats: {tensor_stats(model_out)}")
            print("  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
        print()

    # Compare outputs
    print("="*70)
    print("Comparison Results")
    print("="*70)

    if modeling_out is not None and model_out is not None:
        max_diff = compare_outputs(modeling_out, model_out, "logits")

        if max_diff < 1e-5:
            print("\n✓ Models are aligned!")
        elif max_diff < 1e-3:
            print("\n~ Models are approximately aligned (small numerical differences)")
        else:
            print("\n✗ Models have significant differences")

        # Detailed comparison
        if max_diff > 1e-5:
            print("\nDetailed comparison:")
            print(f"  modeling output first 10: {modeling_out.flatten()[:10].tolist()}")
            print(f"  model output first 10: {model_out.flatten()[:10].tolist()}")
    else:
        print("Cannot compare - one or both models failed")


if __name__ == "__main__":
    main()
