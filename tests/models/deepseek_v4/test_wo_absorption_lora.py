"""LoRA correctness of DeepSeek V4's weight-consuming o-projection absorption.

The absorption reads ``wo_a.weight`` for a grouped einsum instead of calling
the module, so a plain ``.weight`` read drops the LoRA delta.  These tests pin
the ``get_effective_weight`` fix on both call sites: the absorption output must
match a merged weight, and the adapter must stay in the autograd graph.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from torchturbo.models.deepseek_v4 import model as turbo_model
from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Config,
    PostAttentionOps,
)

from .test_mtp import _make_mtp_model


def _config() -> DeepseekV4Config:
    """Only the fields the o-projection absorption reads.

    o_groups=2 keeps the grouped matmul a real grouped matmul (o_groups=1
    would degenerate it into a plain linear call).
    """
    return DeepseekV4Config(
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        o_groups=2,
        o_lora_rank=4,
        qk_rope_head_dim=2,
    )


def _randomize_adapters(layer: nn.Module) -> None:
    """Make both LoRA factors non-zero (PEFT zeroes lora_B at init)."""
    with torch.no_grad():
        for adapter in layer.lora_A:
            layer.lora_A[adapter].weight.normal_()
            layer.lora_B[adapter].weight.normal_()


def _merged_linear(layer: nn.Module) -> nn.Linear:
    """A plain Linear holding base + delta of a single-adapter LoRA layer."""
    with torch.no_grad():
        merged = (
            layer.base_layer.weight
            + layer.lora_B["default"].weight
            @ layer.lora_A["default"].weight
            * layer.scaling["default"]
        )
    linear = nn.Linear(merged.shape[1], merged.shape[0], bias=False)
    linear.weight = nn.Parameter(merged)
    return linear


def _assert_adapter_in_graph(layer: nn.Module) -> None:
    for name in ("lora_A", "lora_B"):
        grad = getattr(layer, name)["default"].weight.grad
        assert grad is not None and grad.abs().sum() > 0


class _OProjection(nn.Module):
    """Holds wo_a/wo_b under their real attribute names for PEFT matching."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.wo_a = nn.Linear(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank, config.hidden_size, bias=False
        )


def test_post_attention_absorption_matches_merged_weight():
    config = _config()
    holder = get_peft_model(
        _OProjection(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["wo_a", "wo_b"]),
    )
    wrapped = holder.base_model.model
    _randomize_adapters(wrapped.wo_a)
    _randomize_adapters(wrapped.wo_b)

    # Pin the production forward (model.py) so the test does not depend on
    # another test having patched the class first.
    turbo_model.TurboDeepseekV4ForCausalLM.patch_deepseek_v4_attention_for_sp(
        sp_group=None, sp_size=1, sp_rank=0
    )
    ops = PostAttentionOps(config)

    torch.manual_seed(11)
    bsz, seqlen = 2, 3
    o = torch.randn(bsz, seqlen, config.num_attention_heads, config.head_dim)
    # Identity rotation keeps the comparison focused on the wo_a absorption.
    freqs_cis = torch.ones(
        config.max_position_embeddings,
        config.qk_rope_head_dim // 2,
        dtype=torch.complex64,
    )
    position_ids = torch.arange(seqlen).unsqueeze(0).expand(bsz, -1)

    def run(wo_a, wo_b):
        return ops(
            o, freqs_cis, bsz, seqlen, config.o_groups, wo_a, wo_b, position_ids
        )

    reference = run(
        _merged_linear(wrapped.wo_a), _merged_linear(wrapped.wo_b)
    ).detach()
    torch.testing.assert_close(run(wrapped.wo_a, wrapped.wo_b).detach(), reference)

    # Pre-fix the absorption read wo_a.weight (base only), dropping the adapter
    # from the autograd graph.
    run(wrapped.wo_a, wrapped.wo_b).sum().backward()
    _assert_adapter_in_graph(wrapped.wo_a)

    # After merge the delta folds into the base weight; output stays identical.
    holder.merge_adapter()
    torch.testing.assert_close(run(wrapped.wo_a, wrapped.wo_b).detach(), reference)


def test_mtp_absorption_keeps_adapter_in_graph():
    model = _make_mtp_model(num_stages=1)
    lora_model = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["wo_a", "wo_b"])
    )
    wrapped = lora_model.base_model.model
    _randomize_adapters(wrapped.mtp[0].attn.wo_a)
    _randomize_adapters(wrapped.mtp[0].attn.wo_b)

    lora_model(
        input_ids=torch.arange(1, 11).unsqueeze(0)
    ).mtp_logits[0].square().sum().backward()

    _assert_adapter_in_graph(wrapped.mtp[0].attn.wo_a)
