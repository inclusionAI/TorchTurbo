"""DeepSeekV4 post-attention and hyper-connection unit tests."""

import os
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeekV4 post-attention/HC tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from .test_alignment import TEST_CONFIG_MODELING  # noqa: E402
    from .test_ops_precision import packed_attention_ops  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution
    from test_alignment import TEST_CONFIG_MODELING  # noqa: E402
    from test_ops_precision import packed_attention_ops  # noqa: E402


def make_config(**overrides):
    from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    config_dict = dict(TEST_CONFIG_MODELING)
    config_dict.update(
        {
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "o_lora_rank": 6,
            "o_groups": 2,
            "hc_mult": 3,
            "hc_sinkhorn_iters": 4,
            "hc_eps": 1e-5,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "max_position_embeddings": 32,
        }
    )
    config_dict.update(overrides)
    return DeepseekV4Config(**config_dict)


def hcsplit_sinkhorn_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].view(1, 1, -1)) + eps
    post = 2 * torch.sigmoid(
        post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult].view(1, 1, -1)
    )
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(1, 1, hc_mult, hc_mult)

    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def test_post_attention_ops_matches_reference_formula():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention,
        PostAttentionOps,
        apply_rotary_emb,
    )

    torch.manual_seed(101)
    config = make_config()
    bsz, seqlen = 2, 5
    n_local_groups = config.o_groups

    attn = DeepseekV4Attention(config, layer_idx=0).float()
    post = PostAttentionOps(config)
    o = torch.randn(bsz, seqlen, config.num_attention_heads, config.head_dim)
    freqs_cis = attn.freqs_cis
    position_ids = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)

    with packed_attention_ops():
        actual = post(
            o, freqs_cis, bsz, seqlen, n_local_groups, attn.wo_a, attn.wo_b, position_ids
        )

    o_nope, o_rope = torch.split(
        o, [config.head_dim - config.qk_rope_head_dim, config.qk_rope_head_dim], dim=-1
    )
    # apply_rotary_emb broadcasts freqs across the batch, so gather per-token
    # freqs from one row; both rows share the same arange positions.
    o_rope = apply_rotary_emb(o_rope, freqs_cis[position_ids[0]], inverse=True)
    expected_o = torch.cat([o_nope, o_rope], dim=-1)
    expected_o = expected_o.view(bsz, seqlen, n_local_groups, -1)
    wo_a_weight = attn.wo_a.weight.view(n_local_groups, config.o_lora_rank, -1)
    expected_o = torch.einsum("bsgd,grd->bsgr", expected_o, wo_a_weight)
    expected = attn.wo_b(expected_o.reshape(bsz, seqlen, -1))

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_hc_split_sinkhorn_matches_reference_formula():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCSplitSinkhorn

    torch.manual_seed(102)
    config = make_config()
    mix_hc = (2 + config.hc_mult) * config.hc_mult
    mixes = torch.randn(2, 4, mix_hc)
    hc_scale = torch.tensor([0.7, 1.3, -0.4])
    hc_base = torch.linspace(-0.5, 0.5, steps=mix_hc)

    op = DeepseekV4HCSplitSinkhorn(config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps)
    actual = op(mixes, hc_scale, hc_base)
    expected = hcsplit_sinkhorn_reference(
        mixes, hc_scale, hc_base, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def test_hc_pre_and_post_match_reference_formula():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import HcPost, HcPre

    torch.manual_seed(103)
    config = make_config()
    bsz, seqlen = 2, 3
    mix_hc = (2 + config.hc_mult) * config.hc_mult
    hc_dim = config.hc_mult * config.hidden_size

    x = torch.randn(bsz, seqlen, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, hc_dim)
    hc_scale = torch.tensor([0.5, 1.1, -0.8])
    hc_base = torch.randn(mix_hc)

    hc_pre = HcPre(config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps)
    y, post, comb = hc_pre(x, hc_fn, hc_scale, hc_base)

    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    mixes = F.linear(flat, hc_fn) * rsqrt
    pre_ref, post_ref, comb_ref = hcsplit_sinkhorn_reference(
        mixes, hc_scale, hc_base, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
    )
    y_ref = torch.sum(pre_ref.unsqueeze(-1) * flat.view_as(x.float()), dim=2).to(x.dtype)

    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)
    torch.testing.assert_close(post, post_ref, atol=0, rtol=0)
    torch.testing.assert_close(comb, comb_ref, atol=0, rtol=0)

    residual = torch.randn_like(x)
    hc_post = HcPost()
    actual_post = hc_post(y, residual, post, comb)
    expected_post = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    expected_post = expected_post.type_as(y)
    torch.testing.assert_close(actual_post, expected_post, atol=0, rtol=0)


def test_hc_head_matches_reference_formula():
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Head,
        DeepseekV4RMSNorm,
    )

    torch.manual_seed(104)
    config = make_config()
    bsz, seqlen = 2, 3
    x = torch.randn(bsz, seqlen, config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
    hc_fn = torch.randn(config.hc_mult, config.hc_mult * config.hidden_size)
    hc_scale = torch.tensor([0.6])
    hc_base = torch.randn(config.hc_mult)
    norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
    head = DeepseekV4Head(config)
    head.weight.data.copy_(torch.randn_like(head.weight) / 10)

    actual = head(x, hc_fn, hc_scale, hc_base, norm)

    flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    mixes = F.linear(flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + config.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * flat.view_as(x.float()), dim=2).to(x.dtype)
    y = norm(y)
    expected = F.linear(y.float(), head.weight)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (bsz, seqlen, config.vocab_size)
