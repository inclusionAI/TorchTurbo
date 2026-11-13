from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from torchturbo.models.bailing_hybrid.model import TurboBailingMoeV3ForCausalLM


class _FakeRotaryEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.device("meta"):
            self.register_buffer(
                "inv_freq",
                torch.empty(2),
                persistent=False,
            )
        self.attention_scaling = 1.0
        self.config = object()
        self.rope_init_fn = lambda config, device: (
            torch.ones(2, device=device),
            1.0,
        )


class _MaterializedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(rotary_emb=_FakeRotaryEmbedding())
        self.param = nn.Parameter(torch.empty(1))


def test_on_materialized_rebuilds_meta_rotary_buffer():
    model = _MaterializedModel()

    TurboBailingMoeV3ForCausalLM.on_materialized(model)

    inv_freq = model.model.rotary_emb.inv_freq
    assert inv_freq.is_meta is False
    assert inv_freq.device == torch.device("cpu")


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "turbo_ulysses"])
def test_mla_forward_pads_v_and_slices_output_under_fa_kernels(
    monkeypatch, attn_implementation
):
    """The SP registry name must keep MLA's V-pad/slice behavior of FA2.

    MLA pads V to the QK head dim before the attention interface and slices
    the output back to the V head dim after it, gated on the config name.
    The pair must stay active for every FA-style implementation,
    ``turbo_ulysses`` included.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from torchturbo.models.bailing_hybrid.configuration_bailing_moe_v3 import (
        BailingMoeV3Config,
    )
    from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
        BailingMoeV3MultiLatentAttention,
    )

    config = BailingMoeV3Config(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=None,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=2,
        v_head_dim=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = attn_implementation
    attention = BailingMoeV3MultiLatentAttention(config, layer_idx=0)
    assert attention.qk_head_dim != attention.v_head_dim

    batch, seq, heads = 1, 3, 2
    hidden_states = torch.randn(batch, seq, config.hidden_size)
    rope = torch.ones(batch, seq, config.qk_rope_head_dim)
    interface = Mock(
        return_value=(
            torch.zeros(batch, heads, seq, attention.qk_head_dim),
            None,
        )
    )
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, attn_implementation, interface)

    attn_output, _, _ = attention.forward(hidden_states, (rope, rope), None)

    interface.assert_called_once()
    _, query, key, value, mask = interface.call_args.args[:5]
    # V padded to the QK head dim on the way in.
    assert value.shape[-1] == attention.qk_head_dim
    assert query.shape[-1] == attention.qk_head_dim
    assert key.shape[-1] == attention.qk_head_dim
    assert mask is None
    # Output sliced back to the V head dim, then reshaped to hidden states.
    assert attn_output.shape == (batch, seq, heads * attention.v_head_dim)
