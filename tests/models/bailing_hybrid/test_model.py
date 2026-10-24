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


@pytest.mark.parametrize(
    ("cu_seqlens", "expected"),
    [
        (None, [0, 8]),
        (torch.tensor([0, 8], dtype=torch.int32), [0, 8]),
        (torch.tensor([0, 3, 8], dtype=torch.int32), [0, 3, 8]),
        (torch.tensor([0, 3, 6], dtype=torch.int32), [0, 3, 6, 8]),
    ],
)
def test_kcp_builds_global_boundaries(monkeypatch, cu_seqlens, expected):
    cp_module = pytest.importorskip("fla.ops.cp")

    from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
        BailingMoeV3KimiDeltaAttention,
    )

    class ContextCaptured(Exception):
        pass

    group = object()
    captured = {}

    def build_cp_context(cu_seqlens, *, group, conv1d_kernel_size):
        captured["cu_seqlens"] = cu_seqlens
        captured["group"] = group
        captured["conv_size"] = conv1d_kernel_size
        raise ContextCaptured

    monkeypatch.setattr(cp_module, "build_cp_context", build_cp_context)
    # Restore the class-level patch after the test.
    monkeypatch.setattr(
        BailingMoeV3KimiDeltaAttention,
        "forward",
        BailingMoeV3KimiDeltaAttention.forward,
    )
    TurboBailingMoeV3ForCausalLM.patch_kda_forward(group, 2, 0, True)

    layer = SimpleNamespace(mode="chunk", conv_size=4)
    hidden_states = torch.zeros(1, 4, 8)
    with pytest.raises(ContextCaptured):
        BailingMoeV3KimiDeltaAttention.forward(
            layer, hidden_states, cu_seqlens=cu_seqlens
        )

    assert captured["cu_seqlens"].tolist() == expected
    expected_dtype = torch.int32 if cu_seqlens is None else torch.int64
    assert captured["cu_seqlens"].dtype == expected_dtype
    assert captured["cu_seqlens"].device == hidden_states.device
    assert captured["group"] is group
    assert captured["conv_size"] == 4


@pytest.mark.parametrize("sp_size", [1, 2])
def test_kcp_rejects_multiple_samples_before_building_context(monkeypatch, sp_size):
    cp_module = pytest.importorskip("fla.ops.cp")

    from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
        BailingMoeV3KimiDeltaAttention,
    )

    def build_cp_context(*args, **kwargs):
        pytest.fail("Multi-sample inputs must be rejected before building CP context.")

    monkeypatch.setattr(cp_module, "build_cp_context", build_cp_context)
    monkeypatch.setattr(
        BailingMoeV3KimiDeltaAttention,
        "forward",
        BailingMoeV3KimiDeltaAttention.forward,
    )
    TurboBailingMoeV3ForCausalLM.patch_kda_forward(None, sp_size, 0, True)

    layer = SimpleNamespace(mode="chunk", conv_size=4)
    hidden_states = torch.zeros(2, 4, 8)
    attention_mask = torch.ones(2, 4 * sp_size, dtype=torch.long)
    with pytest.raises(ValueError, match="FLA KCP requires batch_size=1, got 2"):
        BailingMoeV3KimiDeltaAttention.forward(layer, hidden_states, attention_mask)


@pytest.mark.parametrize(
    "attention_mask",
    [
        None,
        torch.tensor([[1, 1, 1, 0]]),
        torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        torch.ones(2, 4, dtype=torch.long),
    ],
)
def test_ulysses_ordinary_batches_use_fixed_length_convolution(monkeypatch, attention_mask):
    from torchturbo.models.bailing_hybrid.modeling_bailing_moe_v3 import (
        BailingMoeV3KimiDeltaAttention,
    )
    pytest.importorskip("fla.ops.kda")
    import torchturbo.ops as ops_module

    class ConvolutionCaptured(Exception):
        pass

    captured = {}

    def causal_conv1d(*, x, packed_params, **kwargs):
        captured["shape"] = x.shape
        captured["packed_params"] = packed_params
        raise ConvolutionCaptured

    monkeypatch.setattr(ops_module, "ops", SimpleNamespace(causal_conv1d=causal_conv1d))
    monkeypatch.setattr(
        BailingMoeV3KimiDeltaAttention,
        "forward",
        BailingMoeV3KimiDeltaAttention.forward,
    )
    TurboBailingMoeV3ForCausalLM.patch_kda_forward(None, 1, 0, False)

    layer = SimpleNamespace(
        mode="chunk",
        q_proj=nn.Identity(),
        k_proj=nn.Identity(),
        v_proj=nn.Identity(),
        q_conv1d=SimpleNamespace(
            weight=torch.ones(8, 1, 4), bias=None, activation="silu"
        ),
    )
    batch_size = attention_mask.size(0) if attention_mask is not None else 2
    hidden_states = torch.zeros(batch_size, 4, 8)
    with pytest.raises(ConvolutionCaptured):
        BailingMoeV3KimiDeltaAttention.forward(layer, hidden_states, attention_mask)

    assert captured["shape"] == hidden_states.shape
    assert captured["packed_params"] is None
