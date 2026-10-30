"""Contracts for Bailing attention adapters shared across model classes."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchturbo.ops as turbo_ops
from torchturbo.models.bailing_hybrid import attention
from torchturbo.models.bailing_hybrid.naive_kda import RMSNormGated


def test_rms_norm_patch_only_changes_supplied_class(monkeypatch):
    calls = []

    def fake_rms_norm(hidden_states, weight, eps):
        calls.append((hidden_states.shape, eps))
        return hidden_states

    monkeypatch.setattr(
        turbo_ops, "ops", SimpleNamespace(rms_norm=fake_rms_norm)
    )

    class TargetNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.variance_epsilon = 1e-5

        def forward(self, hidden_states):
            return hidden_states

    class OtherNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.variance_epsilon = 1e-5

        def forward(self, hidden_states):
            return hidden_states

    attention.patch_rms_norm(TargetNorm)
    hidden = torch.randn(2, 4)
    TargetNorm()(hidden)
    OtherNorm()(hidden)

    assert len(calls) == 1
    assert calls[0] == (hidden.shape, 1e-5)


@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("no_kda_lora", [True, False])
@pytest.mark.parametrize("cu,expected", [([0, 2, 4], [0, 2, 4]), ([0, 2, 3], [0, 2, 3, 4])])
def test_kda_preserves_packed_boundaries_and_local_parameters(
    monkeypatch, sp_size, no_kda_lora, cu, expected
):
    sp_rank = sp_size - 1
    conv_calls = []
    kda_calls = []

    def conv(**kwargs):
        conv_calls.append(kwargs)
        return kwargs["x"], None

    def chunk_kda(**kwargs):
        kda_calls.append(kwargs)
        return kwargs["v"], None

    def gated_norm(x, g, weight, bias, activation, **kwargs):
        assert bias is None
        assert activation == "sigmoid"
        return x

    exchanged_streams = []
    waited_streams = []
    waited_events = []
    recorded_events = []
    created_streams = []
    created_events = []

    def gather_sequence(x, seq_dim, head_dim, sp_group, sp_size, a2a_stream=None):
        exchanged_streams.append(a2a_stream)
        full_sequence = torch.cat([x] * sp_size, dim=seq_dim)
        return full_sequence.chunk(sp_size, dim=head_dim)[sp_rank]

    def gather_heads(x, seq_dim, head_dim, sp_group, sp_size, a2a_stream=None):
        exchanged_streams.append(a2a_stream)
        full_heads = torch.cat([x] * sp_size, dim=head_dim)
        return full_heads.chunk(sp_size, dim=head_dim)[sp_rank]

    monkeypatch.setattr(
        turbo_ops, "ops", SimpleNamespace(causal_conv1d=conv, kda=chunk_kda)
    )
    monkeypatch.setitem(
        sys.modules,
        "fla.modules.fused_norm_gate",
        SimpleNamespace(rms_norm_gated=gated_norm),
    )

    from torchturbo.models.modules.attention.ulysses import ulysses as ulysses_module

    monkeypatch.setattr(ulysses_module, "gather_seq_scatter_heads", gather_sequence)
    monkeypatch.setattr(ulysses_module, "gather_heads_scatter_seq", gather_heads)

    def new_stream(priority=0):
        stream = SimpleNamespace()
        created_streams.append(stream)
        return stream

    def new_event():
        event = SimpleNamespace()
        event.record = lambda stream: recorded_events.append((event, stream))
        created_events.append(event)
        return event

    fake_stream = SimpleNamespace(
        wait_stream=waited_streams.append,
        wait_event=waited_events.append,
    )
    fake_runtime = SimpleNamespace(
        current_stream=lambda: fake_stream,
        new_stream=new_stream,
        new_event=new_event,
    )
    monkeypatch.setattr(attention, "get_device_runtime", lambda: fake_runtime)

    class TargetKDA(nn.Module):
        def forward(self, x):
            return x

    class OtherKDA(nn.Module):
        def forward(self, x):
            return x

    module = TargetKDA()
    module.mode = "chunk"
    module.no_kda_lora = no_kda_lora
    module.num_heads = 2
    module.head_dim = module.head_k_dim = 2
    module.q_proj = module.k_proj = module.v_proj = nn.Identity()
    module.f_proj = module.g_proj = nn.Identity()
    module.f_a_proj = module.f_b_proj = nn.Identity()
    module.g_a_proj = module.g_b_proj = nn.Identity()
    module.b_proj = nn.Linear(4, 2, bias=False)
    module.q_conv1d = module.k_conv1d = module.v_conv1d = SimpleNamespace(
        weight=torch.arange(8).reshape(4, 1, 2), bias=None, activation="silu"
    )
    module.A_log = torch.arange(2)
    module.dt_bias = torch.arange(4)
    module.safe_gate = True
    module.lower_bound = -5.0
    module.o_norm = RMSNormGated(2, activation="sigmoid")
    module.o_proj = nn.Identity()

    attention.patch_kda_forward(TargetKDA, object(), sp_size, sp_rank)
    x = torch.ones(1, 4 // sp_size, 4)
    output, weights, cache = module(
        x, cu_seqlens=torch.tensor(cu, dtype=torch.int32)
    )
    assert weights is cache is None
    assert OtherKDA()(x) is x
    if sp_size > 1:
        (a2a_stream,) = created_streams
        q_event, k_event, v_event = created_events
        # All exchanges share one a2a stream; each convolution waits only
        # on its gather's event, while the chunk and the gated norm wait on
        # the stream covering the trailing gathers and the output scatter.
        assert exchanged_streams == [a2a_stream] * 6
        assert [event for event, _ in recorded_events] == [
            q_event, k_event, v_event,
        ]
        assert [stream for _, stream in recorded_events] == [a2a_stream] * 3
        assert waited_events == [q_event, k_event, v_event]
        assert waited_streams == [a2a_stream, a2a_stream]
    else:
        assert created_streams == []
        assert created_events == []
        assert exchanged_streams == []
        assert waited_streams == []
        assert waited_events == []

    assert len(conv_calls) == 3
    assert len(kda_calls) == 1
    # Convolutions receive full-length metadata (the padded tail becomes a
    # final sequence); recurrent KDA shares the same boundaries.
    for call in conv_calls:
        assert call["packed_params"].cu_seqlens_list == expected
    assert kda_calls[0]["packed_params"] is conv_calls[0]["packed_params"]
    for call in conv_calls:
        assert call["x"].shape == (1, 4, 4 // sp_size)
        expected_weight = module.q_conv1d.weight.chunk(sp_size, dim=0)[sp_rank]
        torch.testing.assert_close(call["weight"], expected_weight.squeeze(1))
    torch.testing.assert_close(
        kda_calls[0]["A_log"], module.A_log.chunk(sp_size)[sp_rank]
    )
    torch.testing.assert_close(
        kda_calls[0]["dt_bias"], module.dt_bias.chunk(sp_size)[sp_rank]
    )


def test_kda_gate_without_lower_bound_matches_softplus():
    from torchturbo.models.bailing_hybrid.naive_kda import naive_kda_lowerbound_gate

    gate = torch.tensor([[[[0.1, -0.2], [0.3, -0.4]]]])
    decay = torch.tensor([0.2, 0.4])
    bias = torch.tensor([0.1, 0.0, -0.1, 0.2])
    expected = -decay.exp()[:, None] * nn.functional.softplus(gate + bias.view(2, 2))
    torch.testing.assert_close(
        naive_kda_lowerbound_gate(gate, decay, bias, None), expected
    )


def _make_vl_kda():
    from torchturbo.models.bailing_moe_v3_vl.modeling_bailing_moe_v3_vl import (
        BailingMoeV3KimiDeltaAttention,
    )

    config = SimpleNamespace(
        hidden_size=32,
        short_conv_kernel_size=4,
        head_dim=16,
        num_attention_heads=2,
        no_kda_lora=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        rms_norm_eps=1e-6,
    )
    module = BailingMoeV3KimiDeltaAttention(config, layer_idx=0)
    module.eval()
    return module


def test_kda_unpad_resolves_through_patched_module(monkeypatch):
    """KDA unpadding must resolve the engine-patchable module symbol at call time."""
    import transformers.modeling_flash_attention_utils as flash_utils

    from torchturbo.utils.flash_attn_utils import get_unpad_data

    module = _make_vl_kda()
    calls = []

    def counting_unpad(mask):
        calls.append(mask)
        return get_unpad_data(mask)

    monkeypatch.setattr(flash_utils, "_get_unpad_data", counting_unpad)
    hidden = torch.randn(1, 6, 32)
    output, _, _ = module(hidden, attention_mask=torch.ones(1, 6, dtype=torch.long))

    assert output.shape == hidden.shape
    assert len(calls) == 1


def test_kda_unpad_segments_packed_labels_when_patched(monkeypatch):
    """Packed label masks isolate segments once the engine patch is in place."""
    import transformers.modeling_flash_attention_utils as flash_utils

    from torchturbo.utils.flash_attn_utils import get_unpad_data

    module = _make_vl_kda()
    monkeypatch.setattr(flash_utils, "_get_unpad_data", get_unpad_data)

    hidden = torch.randn(1, 5, 32)
    labels = torch.tensor([[1, 1, 2, 2, 0]])
    packed, _, _ = module(hidden, attention_mask=labels)

    segment_a, _, _ = module(hidden[:, :2], attention_mask=None)
    segment_b, _, _ = module(hidden[:, 2:4], attention_mask=None)
    torch.testing.assert_close(packed[:, :2], segment_a)
    torch.testing.assert_close(packed[:, 2:4], segment_b)
    assert torch.count_nonzero(packed[:, 4]) == 0
