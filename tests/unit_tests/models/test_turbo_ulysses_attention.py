"""Binding, masking, and layout tests for the ``turbo_ulysses`` interface."""

from __future__ import annotations

import pytest
import torch

from torchturbo.models.base import TurboModel, TurboVLMModel
from torchturbo.models.modules.attention.qk_clip import QKClipState
from torchturbo.models.modules.attention.ulysses import (
    TURBO_ULYSSES_ATTENTION,
    register_turbo_ulysses_attention,
)
from torchturbo.models.modules.attention.ulysses import ulysses as ulysses_module

from .flash_attention_fakes import lazy_flash_attention_kernels, process_flash_kwargs


class _AttnConfig:
    def __init__(self) -> None:
        self._attn_implementation = "flash_attention_2"


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.setattr(
        type(ALL_MASK_ATTENTION_FUNCTIONS),
        "_global_mapping",
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.copy(),
    )
    saved = ALL_ATTENTION_FUNCTIONS.get(TURBO_ULYSSES_ATTENTION)
    yield
    if saved is None:
        ALL_ATTENTION_FUNCTIONS.pop(TURBO_ULYSSES_ATTENTION, None)
    else:
        ALL_ATTENTION_FUNCTIONS[TURBO_ULYSSES_ATTENTION] = saved


class TestRegisterTurboUlyssesAttention:
    def test_binds_partial_and_switches_config(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        config = _AttnConfig()
        sp_group = object()

        register_turbo_ulysses_attention(config, sp_group, 2)

        entry = ALL_ATTENTION_FUNCTIONS[TURBO_ULYSSES_ATTENTION]
        assert entry.func is ulysses_module.ulysses_attention_forward
        assert entry.keywords["sp_group"] is sp_group
        assert entry.keywords["sp_size"] == 2
        assert config._attn_implementation == TURBO_ULYSSES_ATTENTION

    def test_base_method_registers_at_sp1(self):
        """SP=1 binds the same interface; it only skips the collectives.

        QK clipping records maximum logits through the interface, so the
        base method must not skip the switch when SP is off.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        class _Text(TurboModel):
            def __init__(self) -> None:
                self.config = _AttnConfig()

        model = _Text()
        assert model.get_sp_attention_config() is model.config

        model.register_sp_attention_interface({"sp_group": None, "sp_size": 1})

        entry = ALL_ATTENTION_FUNCTIONS[TURBO_ULYSSES_ATTENTION]
        assert entry.func is ulysses_module.ulysses_attention_forward
        assert entry.keywords["sp_size"] == 1
        assert entry.keywords["sp_group"] is None
        assert model.config._attn_implementation == TURBO_ULYSSES_ATTENTION

    def test_engine_fa_binding_untouched(self):
        """Registration must not overwrite HF's shared flash-attention helper.

        The engine binds ``integrations.flash_attention._flash_attention_forward``
        to turbo's ``attention_forward`` once at startup; the registry path
        must leave both module bindings intact so the native flash path
        (e.g. VLM vision towers) keeps working.
        """
        import transformers.integrations.flash_attention as fa
        import transformers.modeling_flash_attention_utils as mfu

        fa_before = fa._flash_attention_forward
        mfu_before = mfu._flash_attention_forward

        register_turbo_ulysses_attention(_AttnConfig(), object(), 2)

        assert fa._flash_attention_forward is fa_before
        assert mfu._flash_attention_forward is mfu_before


@pytest.mark.parametrize("sp_rank", [0, 1])
@pytest.mark.parametrize(
    "padding_mask",
    [[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]],
    ids=["left-padding", "right-padding", "all-valid"],
)
def test_sliding_window_mask_spans_the_global_sequence(
    monkeypatch, sp_rank, padding_mask
):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeModel,
    )

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        use_sliding_window=True,
        sliding_window=4,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3MoeModel(config)
    attention_masks = []

    def capture_attention(hidden_states, attention_mask, **kwargs):
        attention_masks.append(attention_mask)
        return torch.zeros_like(hidden_states), None

    monkeypatch.setattr(model.layers[0].self_attn, "forward", capture_attention)
    register_turbo_ulysses_attention(config, object(), 2)

    # Q/K/V are local until Ulysses gathers them, but the padding mask is global.
    local_seq_length = 2
    positions = torch.arange(local_seq_length) + sp_rank * local_seq_length
    full_mask = torch.tensor([padding_mask])
    model(
        inputs_embeds=torch.zeros(1, local_seq_length, config.hidden_size),
        attention_mask=full_mask,
        position_ids=positions.unsqueeze(0),
        use_cache=False,
    )

    assert len(attention_masks) == 1
    if full_mask.all():
        assert attention_masks[0] is None
    else:
        torch.testing.assert_close(attention_masks[0], full_mask.bool())


def _sdpa_flash_kernels():
    """CPU stand-ins for the dense and varlen flash-attention kernels."""
    import torch.nn.functional as F

    def dense(query, key, value, causal, softmax_scale, **kwargs):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=causal,
            scale=softmax_scale,
        ).transpose(1, 2)

    def varlen(query, key, value, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale, **kwargs):
        # Self-attention: query and key boundaries are identical.
        outputs = []
        for start, end in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
            segment = dense(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                causal,
                softmax_scale,
            )
            outputs.append(segment[0])
        return torch.cat(outputs)

    return dense, varlen


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked-rows"])
def test_sp1_interface_matches_flash_attention_2(monkeypatch, packed):
    """At SP=1 the registered interface reproduces the flash_attention_2 path."""
    import transformers.integrations.flash_attention as flash_integration
    import transformers.modeling_flash_attention_utils as flash_utils
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeModel,
    )

    from torchturbo.models.modules.attention.interface import attention_forward

    kernels = lazy_flash_attention_kernels(*_sdpa_flash_kernels())
    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        lambda *args, **kwargs: (kernels, process_flash_kwargs),
    )
    # TorchTurboEngine._patch_transformers binds this once at startup.
    monkeypatch.setattr(flash_integration, "_flash_attention_forward", attention_forward)

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3MoeModel(config).eval()
    if packed:
        # Two samples followed by a tail beyond the last boundary.
        cu_seq_lens = torch.tensor([0, 5, 11], dtype=torch.int32)
        inputs = dict(
            inputs_embeds=torch.randn(1, 13, config.hidden_size),
            position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 0]]),
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=6,
            max_length_k=6,
        )
    else:
        inputs = dict(inputs_embeds=torch.randn(2, 7, config.hidden_size))

    config._attn_implementation = "flash_attention_2"
    expected = model(**inputs).last_hidden_state
    register_turbo_ulysses_attention(config, None, 1)
    actual = model(**inputs).last_hidden_state

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _ModuleStub:
    is_causal = True
    training = False


class TestUlyssesAttentionInterface:
    def test_layout_and_module_fallback(self, monkeypatch):
        calls = {}

        def fake_core(query, key, value, attention_mask, query_length, **kwargs):
            calls["query_shape"] = tuple(query.shape)
            calls["query_length"] = query_length
            calls["kwargs"] = kwargs
            return query

        monkeypatch.setattr(
            ulysses_module, "ulysses_flash_attention_forward", fake_core
        )

        batch, heads, seq, head_dim = 1, 4, 8, 16
        q = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)

        out, weights = ulysses_module.ulysses_attention_forward(
            _ModuleStub(),
            q,
            k,
            v,
            None,
            dropout=0.1,
            scaling=0.5,
            sp_group=None,
            sp_size=1,
        )

        assert calls["query_shape"] == (batch, seq, heads, head_dim)
        assert calls["query_length"] == seq
        assert out.shape == (batch, seq, heads, head_dim)
        assert out.dtype == torch.bfloat16
        assert weights is None
        assert calls["kwargs"]["is_causal"] is True
        assert calls["kwargs"]["dropout"] == 0.1
        assert calls["kwargs"]["softmax_scale"] == 0.5

    def test_q_head_repeat_and_dedup(self, monkeypatch):
        calls = {}

        def fake_core(query, key, value, attention_mask, query_length, **kwargs):
            calls["query_shape"] = tuple(query.shape)
            return query

        monkeypatch.setattr(
            ulysses_module, "ulysses_flash_attention_forward", fake_core
        )

        batch, heads, seq, head_dim = 1, 2, 8, 16
        q = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seq, head_dim, dtype=torch.bfloat16)

        out, _ = ulysses_module.ulysses_attention_forward(
            _ModuleStub(),
            q,
            k,
            v,
            None,
            sp_group=object(),
            sp_size=4,
        )

        # lcm(4, 2) // 2 == 2: heads repeated for the all-to-all, deduped after.
        assert calls["query_shape"] == (batch, seq, heads * 2, head_dim)
        assert out.shape == (batch, seq, heads, head_dim)

    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("training", [True, False])
    def test_qk_clip_state_is_forwarded_only_while_collecting(
        self, monkeypatch, training, enabled
    ):
        calls = {}

        def fake_core(query, key, value, attention_mask, query_length, **kwargs):
            calls["qk_clip_state"] = kwargs["qk_clip_state"]
            return query

        monkeypatch.setattr(
            ulysses_module, "ulysses_flash_attention_forward", fake_core
        )

        module = _ModuleStub()
        module.training = training
        state = QKClipState(torch.full((2,), -torch.inf))
        # An offloaded optimizer suspends collection without removing the state.
        state.enabled = enabled
        module.qk_clip_state = state
        q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)

        ulysses_module.ulysses_attention_forward(
            module, q, q, q, None, sp_group=None, sp_size=1
        )

        assert calls["qk_clip_state"] is (state if training and enabled else None)


class TestSpAttentionConfig:
    def test_vlm_targets_text_config(self):
        class _Vlm(TurboVLMModel):
            def __init__(self) -> None:
                class _Config:
                    text_config = _AttnConfig()

                self.config = _Config()

        assert isinstance(_Vlm().get_sp_attention_config(), _AttnConfig)

    def test_text_model_targets_own_config(self):
        class _Text(TurboModel):
            def __init__(self) -> None:
                self.config = _AttnConfig()

        config = _Text().get_sp_attention_config()
        assert isinstance(config, _AttnConfig)
