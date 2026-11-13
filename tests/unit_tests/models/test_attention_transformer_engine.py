from collections.abc import Iterator
import gc
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import weakref

import pytest
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchturbo.ops.providers.cuda import attention as cuda_attention


@pytest.fixture(autouse=True)
def reset_availability_cache() -> Iterator[None]:
    cuda_attention.is_transformer_engine_available.cache_clear()
    yield
    cuda_attention.is_transformer_engine_available.cache_clear()


@pytest.fixture
def reset_block_mask_caches() -> Iterator[None]:
    torch.compiler.reset()
    cuda_attention._dense_block_mask.cache_clear()
    cuda_attention._masked_block_mask.cache_clear()
    cuda_attention._compiled_create_block_mask.cache_clear()
    yield
    cuda_attention._dense_block_mask.cache_clear()
    cuda_attention._masked_block_mask.cache_clear()
    cuda_attention._compiled_create_block_mask.cache_clear()
    torch.compiler.reset()


def _reference_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    scaling: float,
    is_causal: bool,
) -> tuple[Tensor, Tensor]:
    batch_size, seq_length = query.shape[:2]
    allowed = torch.ones(batch_size, 1, seq_length, seq_length, dtype=torch.bool)
    if mask is not None:
        allowed &= (mask[:, None, :, None] == mask[:, None, None, :])
        allowed &= (mask[:, None, :, None] != 0) & (mask[:, None, None, :] != 0)
    if is_causal:
        allowed &= torch.ones(seq_length, seq_length, dtype=torch.bool).tril()
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed, scale=scaling
    ).transpose(1, 2)
    scores = (query @ key.transpose(-1, -2) * scaling).masked_fill(~allowed, -torch.inf)
    return output, scores.amax(dim=(0, 2, 3))


class _MockDotProductAttention(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.config = kwargs
        self.calls: list[tuple[Tensor, Tensor, Tensor, dict[str, Any]]] = []

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, **kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        self.calls.append((query, key, value, kwargs))
        causal = "causal" in self.config["attn_mask_type"]
        scaling = self.config["softmax_scale"]
        if self.config["qkv_format"] == "bshd":
            output, maxima = _reference_attention(query, key, value, None, scaling, causal)
            return output.flatten(-2), maxima
        boundaries = kwargs["cu_seqlens_q"].tolist()
        outputs, maxima = [], []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start == end:
                continue
            sequence = [tensor[None, start:end] for tensor in (query, key, value)]
            output, head_max = _reference_attention(*sequence, None, scaling, causal)
            outputs.append(output.squeeze(0).flatten(-2))
            maxima.append(head_max)
        return torch.cat(outputs), torch.stack(maxima).amax(dim=0)


@pytest.fixture
def mock_dpa(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_MockDotProductAttention]]:
    instances = []

    def construct(**kwargs: Any) -> _MockDotProductAttention:
        attention = _MockDotProductAttention(**kwargs)
        instances.append(attention)
        return attention

    pytorch_module = ModuleType("transformer_engine.pytorch")
    pytorch_module.DotProductAttention = construct
    quantization_module = ModuleType("transformer_engine.pytorch.quantization")
    quantization_module.FP8GlobalStateManager = SimpleNamespace(is_fp8_enabled=lambda: False)
    monkeypatch.setitem(sys.modules, "transformer_engine", ModuleType("transformer_engine"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch_module)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.quantization", quantization_module)
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    monkeypatch.delenv("NVTE_DPA_FP8_RECIPE", raising=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    cuda_attention._get_attention.cache_clear()
    yield instances
    cuda_attention._get_attention.cache_clear()


@pytest.mark.parametrize(
    ("version", "available"),
    [
        ("2.8.0", False),
        ("2.9.0", False),
        ("2.10.0", False),
        ("2.11.0rc1", False),
        ("2.11.0", True),
        ("2.12.0", True),
    ],
)
def test_availability_requires_thd_max_logit_release(
    monkeypatch: pytest.MonkeyPatch, version: str, available: bool
) -> None:
    monkeypatch.setattr(cuda_attention, "version", lambda name: version)
    assert cuda_attention.TransformerEngineAttentionWithMaxLogits().available() is available


def test_missing_distribution_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_version(name: str) -> str:
        raise cuda_attention.PackageNotFoundError(name)

    monkeypatch.setattr(cuda_attention, "version", missing_version)
    assert not cuda_attention.TransformerEngineAttentionWithMaxLogits().available()


def test_availability_queries_package_metadata_once(monkeypatch: pytest.MonkeyPatch) -> None:
    queried_packages = []

    def installed_version(name: str) -> str:
        queried_packages.append(name)
        return "2.11.0"

    monkeypatch.setattr(cuda_attention, "version", installed_version)
    provider = cuda_attention.TransformerEngineAttentionWithMaxLogits()
    assert provider.available()
    assert provider.available()
    assert queried_packages == ["transformer-engine"]


@pytest.mark.parametrize("is_causal", [False, True])
def test_bshd_constructor_output_and_gradients(
    mock_dpa: list[_MockDotProductAttention], is_causal: bool
) -> None:
    torch.manual_seed(8)
    query = torch.randn(2, 5, 3, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 5, 3, 2, dtype=torch.float64, requires_grad=True)
    output, maxima = cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, key, value, None, 0.37, 0.0, is_causal
    )
    expected, expected_maxima = _reference_attention(query, key, value, None, 0.37, is_causal)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(maxima, expected_maxima.float())
    assert maxima.dtype == torch.float32
    assert not maxima.requires_grad
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)
    assert mock_dpa[0].config == {
        "num_attention_heads": 3,
        "kv_channels": (4, 2),
        "num_gqa_groups": 3,
        "attention_dropout": 0.0,
        "qkv_format": "bshd",
        "attn_mask_type": "causal" if is_causal else "no_mask",
        "softmax_scale": 0.37,
        "return_max_logit": True,
        "sequence_parallel": False,
        "tp_size": 1,
        "tp_group": None,
        "cp_group": None,
    }
    assert mock_dpa[0].calls[0][0] is query
    assert mock_dpa[0].calls[0][-1] == {}


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_thd_packing_padding_and_gradients(
    mock_dpa: list[_MockDotProductAttention], packed: bool, is_causal: bool
) -> None:
    torch.manual_seed(9)
    query = torch.randn(2, 6, 3, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 6, 3, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 1, 1, 2, 2, 2]])
    expected_lengths = [0, 2, 4, 7, 10]
    if not packed:
        mask = mask != 0
        expected_lengths = [0, 4, 10]
    output, maxima = cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, key, value, mask, 0.37, 0.0, is_causal
    )
    expected, expected_maxima = _reference_attention(query, key, value, mask, 0.37, is_causal)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(maxima, expected_maxima.float())
    assert not output[mask == 0].any()
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)
        assert not gradient[mask == 0].any()
    attention = mock_dpa[0]
    assert attention.config["qkv_format"] == "thd"
    assert attention.config["attn_mask_type"] == ("padding_causal" if is_causal else "padding")
    packed_query, packed_key, packed_value, call_kwargs = attention.calls[0]
    for actual, original in zip((packed_query, packed_key, packed_value), (query, key, value)):
        torch.testing.assert_close(actual, original[mask != 0])
    assert call_kwargs["cu_seqlens_q"].dtype == torch.int32
    assert call_kwargs["cu_seqlens_q"].tolist() == expected_lengths
    assert call_kwargs["cu_seqlens_kv"] is call_kwargs["cu_seqlens_q"]
    assert call_kwargs["max_seqlen_q"] == (3 if packed else 6)
    assert call_kwargs["max_seqlen_kv"] == call_kwargs["max_seqlen_q"]
    assert call_kwargs["pad_between_seqs"] is False
    assert "return_max_logit" not in call_kwargs


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("length_source", ["inferred", "integer", "tensor"])
@pytest.mark.parametrize("boundary_dtype", [torch.int32, torch.int64])
def test_thd_uses_cumulative_lengths_without_unpadding(
    monkeypatch: pytest.MonkeyPatch,
    mock_dpa: list[_MockDotProductAttention],
    is_causal: bool,
    length_source: str,
    boundary_dtype: torch.dtype,
) -> None:
    from torchturbo.utils import flash_attn_utils

    def unexpected_unpadding(*args, **kwargs):
        pytest.fail("Explicit cumulative lengths must bypass mask unpadding.")

    monkeypatch.setattr(flash_attn_utils, "get_unpad_data", unexpected_unpadding)
    torch.manual_seed(29)
    query = torch.randn(2, 6, 3, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 6, 3, 2, dtype=torch.float64, requires_grad=True)
    boundaries = torch.tensor([0, 2, 7, 12], dtype=boundary_dtype)
    max_length = {"inferred": None, "integer": 5, "tensor": torch.tensor(5)}[length_source]
    output, maxima = cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query,
        key,
        value,
        None,
        0.37,
        0.0,
        is_causal,
        cu_seq_lens_q=boundaries,
        cu_seq_lens_k=boundaries,
        max_length_q=max_length,
        max_length_k=max_length,
    )
    # The middle sequence crosses the original batch boundary at token six.
    mask = torch.tensor([[1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]])
    flattened = [tensor.flatten(0, 1).unsqueeze(0) for tensor in (query, key, value)]
    expected, expected_maxima = _reference_attention(*flattened, mask, 0.37, is_causal)
    torch.testing.assert_close(output, expected.reshape(value.shape))
    torch.testing.assert_close(maxima, expected_maxima.float())
    assert not maxima.requires_grad
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)
    attention = mock_dpa[0]
    assert attention.config["qkv_format"] == "thd"
    packed_query, packed_key, packed_value, call_kwargs = attention.calls[0]
    for actual, original in zip((packed_query, packed_key, packed_value), (query, key, value)):
        torch.testing.assert_close(actual, original.flatten(0, 1))
        assert actual.data_ptr() == original.data_ptr()
    torch.testing.assert_close(call_kwargs["cu_seqlens_q"], boundaries.int())
    torch.testing.assert_close(call_kwargs["cu_seqlens_kv"], boundaries.int())
    if length_source == "inferred":
        assert call_kwargs["max_seqlen_q"] is None
        assert call_kwargs["max_seqlen_kv"] is None
    else:
        assert type(call_kwargs["max_seqlen_q"]) is int
        assert type(call_kwargs["max_seqlen_kv"]) is int
        assert call_kwargs["max_seqlen_q"] == call_kwargs["max_seqlen_kv"] == 5
    assert call_kwargs["pad_between_seqs"] is False


def test_thd_mask_takes_precedence_over_cumulative_lengths(
    mock_dpa: list[_MockDotProductAttention],
) -> None:
    query = torch.randn(1, 5, 2, 4)
    mask = torch.tensor([[1, 1, 2, 2, 0]])
    boundaries = torch.tensor([0, 5], dtype=torch.int32)
    output, maxima = cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, query, query, mask, 0.5, 0.0, True,
        cu_seq_lens_q=boundaries, cu_seq_lens_k=boundaries,
        max_length_q=5, max_length_k=5,
    )
    expected, expected_maxima = _reference_attention(query, query, query, mask, 0.5, True)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(maxima, expected_maxima)
    call_kwargs = mock_dpa[0].calls[0][-1]
    assert call_kwargs["cu_seqlens_q"].tolist() == [0, 2, 4]
    assert call_kwargs["max_seqlen_q"] == 2


def test_empty_packed_batch_preserves_zero_gradients(
    mock_dpa: list[_MockDotProductAttention],
) -> None:
    query = torch.randn(2, 3, 2, 4, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 3, 2, 5, requires_grad=True)
    output, maxima = cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, key, value, torch.zeros(2, 3, dtype=torch.long), 0.5, 0.0, True
    )
    assert not mock_dpa
    assert output.shape == value.shape
    assert not output.any()
    assert torch.isneginf(maxima).all()
    assert maxima.dtype == torch.float32
    for gradient in torch.autograd.grad(output.sum(), (query, key, value)):
        assert not gradient.any()


def test_reuses_dpa_across_sequence_and_batch_sizes(
    mock_dpa: list[_MockDotProductAttention],
) -> None:
    for batch_size, seq_length in ((1, 2), (3, 5), (1, 2)):
        query = torch.randn(batch_size, seq_length, 2, 4)
        cuda_attention.TransformerEngineAttentionWithMaxLogits()(
            query, query, query, None, 0.5, 0.1, True
        )
    assert len(mock_dpa) == 1
    assert len(mock_dpa[0].calls) == 3
    assert mock_dpa[0].config["attention_dropout"] == 0.1


@pytest.mark.parametrize("change", ["scale", "dropout", "dtype", "causal", "packing", "heads", "value_dim"])
def test_different_execution_signatures_do_not_share_dpa(
    mock_dpa: list[_MockDotProductAttention], change: str
) -> None:
    query = torch.randn(1, 3, 2, 4)
    cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, query, query, None, 0.5, 0.0, True
    )
    if change == "dtype":
        query = query.double()
    if change == "heads":
        query = query[:, :, :1]
    value = query[:, :, :, :2] if change == "value_dim" else query
    cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query,
        query,
        value,
        torch.ones(1, 3, dtype=torch.long) if change == "packing" else None,
        0.25 if change == "scale" else 0.5,
        0.1 if change == "dropout" else 0.0,
        change != "causal",
    )
    assert len(mock_dpa) == 2


@pytest.mark.parametrize("policy", ["torch", "environment"])
def test_deterministic_policy_partitions_cache(
    monkeypatch: pytest.MonkeyPatch, mock_dpa: list[_MockDotProductAttention], policy: str
) -> None:
    query = torch.randn(1, 3, 2, 4)
    cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, query, query, None, 0.5, 0.0, True
    )
    if policy == "torch":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    else:
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, query, query, None, 0.5, 0.0, True
    )
    assert len(mock_dpa) == 2


def test_negative_maxima_are_preserved(mock_dpa: list[_MockDotProductAttention]) -> None:
    query = torch.ones(1, 3, 2, 4)
    _, maxima = cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, -query, query, None, 0.5, 0.0, True
    )
    torch.testing.assert_close(maxima, torch.full((2,), -2.0))


@pytest.mark.parametrize(("fp8_dpa", "fp8_mha"), [(True, False), (False, True), (True, True)])
def test_fp8_attention_is_rejected_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    mock_dpa: list[_MockDotProductAttention],
    fp8_dpa: bool,
    fp8_mha: bool,
) -> None:
    from transformer_engine.pytorch import quantization

    manager = SimpleNamespace(
        is_fp8_enabled=lambda: True,
        get_fp8_recipe=lambda: SimpleNamespace(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha),
    )
    monkeypatch.setattr(quantization, "FP8GlobalStateManager", manager)
    query = torch.randn(1, 3, 2, 4)
    with pytest.raises(ValueError, match="do not support Transformer Engine FP8 attention"):
        cuda_attention.TransformerEngineAttentionWithMaxLogits()(
            query, query, query, None, 0.5, 0.0, True
        )
    assert not mock_dpa


@pytest.mark.parametrize("f16_override", [False, True])
def test_fp8_gemms_with_non_fp8_attention_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
    mock_dpa: list[_MockDotProductAttention],
    f16_override: bool,
) -> None:
    from transformer_engine.pytorch import quantization

    manager = SimpleNamespace(
        is_fp8_enabled=lambda: True,
        get_fp8_recipe=lambda: SimpleNamespace(fp8_dpa=f16_override, fp8_mha=f16_override),
    )
    monkeypatch.setattr(quantization, "FP8GlobalStateManager", manager)
    if f16_override:
        monkeypatch.setenv("NVTE_DPA_FP8_RECIPE", "F16")
    query = torch.randn(1, 3, 2, 4)
    cuda_attention.TransformerEngineAttentionWithMaxLogits()(
        query, query, query, None, 0.5, 0.0, True
    )
    assert len(mock_dpa) == 1


def test_te_execution_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, mock_dpa: list[_MockDotProductAttention]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        raise RuntimeError("TE kernel rejected inputs")

    monkeypatch.setattr(_MockDotProductAttention, "forward", fail)
    query = torch.randn(1, 3, 2, 4)
    with pytest.raises(RuntimeError, match="TE kernel rejected inputs"):
        cuda_attention.TransformerEngineAttentionWithMaxLogits()(
            query, query, query, None, 0.5, 0.0, True
        )


def test_maxima_must_follow_execution_head_order(
    monkeypatch: pytest.MonkeyPatch, mock_dpa: list[_MockDotProductAttention]
) -> None:
    def wrong_shape(
        self: _MockDotProductAttention, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor]:
        return value.flatten(-2), torch.zeros(1, query.shape[2])

    monkeypatch.setattr(_MockDotProductAttention, "forward", wrong_shape)
    query = torch.randn(1, 3, 2, 4)
    with pytest.raises(RuntimeError, match="one maximum per attention head"):
        cuda_attention.TransformerEngineAttentionWithMaxLogits()(
            query, query, query, None, 0.5, 0.0, True
        )


@pytest.mark.parametrize(
    "provider_type",
    [
        cuda_attention.TransformerEngineAttentionWithMaxLogits,
        cuda_attention.FlexAttentionWithMaxLogits,
    ],
)
def test_cuda_providers_reject_four_dimensional_masks(provider_type) -> None:
    query = torch.ones(1, 3, 2, 4)
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="4D masks only in the CPU reference"):
        provider_type()(query, query, query, mask, 0.5, 0.0, True)


@pytest.mark.parametrize("with_auxiliary_api", [False, True])
def test_flex_availability_requires_auxiliary_api_without_loading_kernel(
    monkeypatch: pytest.MonkeyPatch, with_auxiliary_api: bool
) -> None:
    flex_module = ModuleType("torch.nn.attention.flex_attention")

    def unexpected_execution(*args, **kwargs):
        pytest.fail("Availability must not construct or execute attention kernels.")

    flex_module.create_block_mask = unexpected_execution
    flex_module.flex_attention = unexpected_execution
    if with_auxiliary_api:
        flex_module.AuxRequest = unexpected_execution
    monkeypatch.setitem(sys.modules, flex_module.__name__, flex_module)
    monkeypatch.setattr(torch, "compile", unexpected_execution)

    assert cuda_attention.FlexAttentionWithMaxLogits().available() is with_auxiliary_api


def test_flex_rejects_dropout_before_loading_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    query = torch.ones(1, 3, 2, 4)
    flex_module = ModuleType("torch.nn.attention.flex_attention")
    monkeypatch.setitem(sys.modules, flex_module.__name__, flex_module)

    with pytest.raises(ValueError, match="FlexAttention.*attention_dropout=0"):
        cuda_attention.FlexAttentionWithMaxLogits()(
            query, query, query, None, 0.5, 0.1, True
        )


@pytest.mark.usefixtures("reset_block_mask_caches")
@pytest.mark.parametrize("is_causal", [False, True])
def test_flex_cumulative_lengths_preserve_cross_row_sequences_and_cache(
    monkeypatch: pytest.MonkeyPatch, is_causal: bool
) -> None:
    from torch.nn.attention.flex_attention import create_block_mask

    block_masks = []

    def attention(query, key, value, *, block_mask, scale, return_aux):
        assert return_aux.max_scores
        block_masks.append(block_mask)
        batch = torch.arange(query.shape[0])[:, None, None]
        indices = torch.arange(query.shape[2])
        allowed = block_mask.mask_mod(batch, 0, indices[None, :, None], indices[None, None, :])
        allowed = allowed.unsqueeze(1)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, scale=scale)
        scores = (query @ key.transpose(-1, -2) * scale).masked_fill(~allowed, -torch.inf)
        return output, SimpleNamespace(max_scores=scores.amax(dim=-1))

    monkeypatch.setattr(cuda_attention, "_compiled_flex_attention", lambda: attention)
    monkeypatch.setattr(cuda_attention, "_compiled_create_block_mask", lambda: create_block_mask)
    torch.manual_seed(30)
    query = torch.randn(2, 6, 2, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 6, 2, 3, dtype=torch.float64, requires_grad=True)
    boundaries = torch.tensor([0, 2, 7, 12], dtype=torch.int32)
    provider = cuda_attention.FlexAttentionWithMaxLogits()

    def execute():
        return provider(
            query, key, value, None, 0.37, 0.0, is_causal,
            cu_seq_lens_q=boundaries, cu_seq_lens_k=boundaries,
            max_length_q=5, max_length_k=5,
        )

    output, maxima = execute()
    repeated_output, repeated_maxima = execute()
    assert block_masks[0] is block_masks[1]
    mask = torch.tensor([[1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]])
    flattened = [tensor.flatten(0, 1).unsqueeze(0) for tensor in (query, key, value)]
    expected, expected_maxima = _reference_attention(*flattened, mask, 0.37, is_causal)
    torch.testing.assert_close(output, expected.reshape(value.shape))
    torch.testing.assert_close(maxima, expected_maxima)
    torch.testing.assert_close(repeated_output, output)
    torch.testing.assert_close(repeated_maxima, maxima)
    assert not maxima.requires_grad
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)

    boundaries[1] = 3
    updated_output, updated_maxima = execute()
    assert block_masks[2] is not block_masks[1]
    mask[0, 2] = 1
    expected, expected_maxima = _reference_attention(*flattened, mask, 0.37, is_causal)
    torch.testing.assert_close(updated_output, expected.reshape(value.shape))
    torch.testing.assert_close(updated_maxima, expected_maxima)


@pytest.mark.usefixtures("reset_block_mask_caches")
@pytest.mark.parametrize("is_causal", [False, True])
def test_flex_dense_inference_before_training_preserves_gradients_and_cache(
    monkeypatch: pytest.MonkeyPatch, is_causal: bool
) -> None:
    from torch.nn.attention.flex_attention import flex_attention

    block_masks = []

    def attention(query, key, value, *, block_mask, scale, return_aux):
        assert return_aux.max_scores
        block_masks.append(block_mask)
        # CPU FlexAttention saves real block metadata but cannot return max_scores.
        output = flex_attention(query, key, value, block_mask=block_mask, scale=scale)
        scores = query @ key.transpose(-1, -2) * scale
        indices = torch.arange(query.shape[2])
        allowed = block_mask.mask_mod(0, 0, indices[:, None], indices[None, :])
        maxima = scores.masked_fill(~allowed, -torch.inf).amax(dim=-1)
        return output, SimpleNamespace(max_scores=maxima)

    monkeypatch.setattr(cuda_attention, "_compiled_flex_attention", lambda: attention)
    torch.manual_seed(31)
    query, key, value = [torch.randn(1, 4, 2, 16, requires_grad=True) for _ in range(3)]
    provider = cuda_attention.FlexAttentionWithMaxLogits()
    with torch.inference_mode():
        provider(query, key, value, None, 0.37, 0.0, is_causal)

    output, maxima = provider(query, key, value, None, 0.37, 0.0, is_causal)
    repeated_output, repeated_maxima = provider(query, key, value, None, 0.37, 0.0, is_causal)
    assert block_masks[0] is not block_masks[1]
    assert block_masks[1] is block_masks[2]
    assert block_masks[0].kv_indices.is_inference()
    assert not block_masks[1].kv_indices.is_inference()
    assert cuda_attention._dense_block_mask.cache_info().currsize == 1
    assert cuda_attention._dense_block_mask.cache_info().hits == 1

    expected, expected_maxima = _reference_attention(query, key, value, None, 0.37, is_causal)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(maxima, expected_maxima)
    torch.testing.assert_close(repeated_output, output)
    torch.testing.assert_close(repeated_maxima, maxima)
    assert not maxima.requires_grad
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.usefixtures("reset_block_mask_caches")
@pytest.mark.parametrize("inference", [False, True])
def test_flex_rebuilds_block_metadata_after_mask_mutation(
    monkeypatch: pytest.MonkeyPatch, inference: bool
) -> None:
    from torch.nn.attention.flex_attention import create_block_mask

    block_masks = []

    def attention(query, key, value, *, block_mask, scale, return_aux):
        block_masks.append(block_mask)
        return value, SimpleNamespace(max_scores=query.new_zeros(query.shape[:3]))

    monkeypatch.setattr(cuda_attention, "_compiled_flex_attention", lambda: attention)
    monkeypatch.setattr(cuda_attention, "_compiled_create_block_mask", lambda: create_block_mask)
    with torch.inference_mode(inference):
        query = torch.randn(1, 257, 1, 4)
        mask = torch.ones(1, 257, dtype=torch.long)
        provider = cuda_attention.FlexAttentionWithMaxLogits()
        provider(query, query, query, mask, 0.5, 0.0, False)
        provider(query, query, query, mask, 0.5, 0.0, False)
        assert (block_masks[0] is block_masks[1]) is not inference

        mask[:, 128:] = 2
        provider(query, query, query, mask, 0.5, 0.0, False)
        assert block_masks[2] is not block_masks[1]
        expected = create_block_mask(
            block_masks[2].mask_mod, B=1, H=None, Q_LEN=257, KV_LEN=257, device=query.device
        )
        torch.testing.assert_close(block_masks[2].as_tuple()[:-1], expected.as_tuple()[:-1])
        assert block_masks[0].full_kv_num_blocks.sum() > block_masks[2].full_kv_num_blocks.sum()
        if inference:
            assert cuda_attention._masked_block_mask.cache_info().currsize == 0


@pytest.mark.usefixtures("reset_block_mask_caches")
def test_flex_block_masks_are_cached_per_mask_and_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_builds = 0
    original_create = cuda_attention._compiled_create_block_mask()

    def counting_create(mask_mod, *args, **kwargs):
        nonlocal num_builds
        num_builds += 1
        return original_create(mask_mod, *args, **kwargs)

    monkeypatch.setattr(cuda_attention, "_compiled_create_block_mask", lambda: counting_create)
    device = torch.device("cpu")
    mask = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)

    cached = cuda_attention._masked_block_mask(mask, True, 4, device)
    assert cuda_attention._masked_block_mask(mask, True, 4, device) is cached
    other_mask = mask.clone()
    assert cuda_attention._masked_block_mask(other_mask, True, 4, device) is not cached
    non_causal = cuda_attention._masked_block_mask(mask, False, 4, device)
    assert non_causal is not cached
    dense = cuda_attention._dense_block_mask(True, 4, device)
    assert cuda_attention._dense_block_mask(True, 4, device) is dense

    assert num_builds == 4
    del other_mask
    assert cuda_attention._masked_block_mask(mask.clone(), True, 4, device) is not cached


@pytest.mark.usefixtures("reset_block_mask_caches")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
def test_compiled_block_mask_preserves_partial_blocks(packed, is_causal) -> None:
    from torch.nn.attention.flex_attention import create_block_mask

    seq_length = 257
    device = torch.device("cpu")
    if packed:
        mask = torch.ones(2, seq_length, dtype=torch.long)
        mask[0, 129:] = 2
        mask[1, 127:] = 0
        actual = cuda_attention._masked_block_mask(mask, is_causal, seq_length, device)
    else:
        actual = cuda_attention._dense_block_mask(is_causal, seq_length, device)

    expected = create_block_mask(
        actual.mask_mod,
        B=2 if packed else None,
        H=None,
        Q_LEN=seq_length,
        KV_LEN=seq_length,
        device=device,
    )
    torch.testing.assert_close(actual.as_tuple()[:-1], expected.as_tuple()[:-1])


@pytest.mark.usefixtures("reset_block_mask_caches")
@pytest.mark.parametrize("packed", [False, True])
def test_block_mask_construction_avoids_token_square_allocations(packed) -> None:
    seq_length = 2048
    device = torch.device("cpu")
    mask = torch.ones(2, seq_length, dtype=torch.long)
    mask[:, seq_length // 2:] = 2

    def build():
        if packed:
            return cuda_attention._masked_block_mask(mask, True, seq_length, device)
        return cuda_attention._dense_block_mask(True, seq_length, device)

    build()
    cuda_attention._dense_block_mask.cache_clear()
    cuda_attention._masked_block_mask.cache_clear()
    # Exclude compilation and force construction instead of measuring a cache hit.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        result = build()

    assert result.seq_lengths == (seq_length, seq_length)
    largest_allocation = max(event.self_cpu_memory_usage for event in profile.events())
    assert largest_allocation < seq_length * seq_length


@pytest.mark.usefixtures("reset_block_mask_caches")
def test_flex_mask_cache_releases_evicted_entries_and_cleared_entries() -> None:
    mask_refs, block_mask_refs = [], []
    for _ in range(17):
        mask = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
        block_mask = cuda_attention._masked_block_mask(mask, True, 4, mask.device)
        mask_refs.append(weakref.ref(mask))
        block_mask_refs.append(weakref.ref(block_mask))
        del mask, block_mask

    gc.collect()
    assert cuda_attention._masked_block_mask.cache_info().currsize == 16
    assert mask_refs[0]() is None
    assert block_mask_refs[0]() is None
    assert all(reference() is not None for reference in mask_refs[1:])
    assert all(reference() is not None for reference in block_mask_refs[1:])

    cuda_attention._masked_block_mask.cache_clear()
    gc.collect()
    assert all(reference() is None for reference in mask_refs)
    assert all(reference() is None for reference in block_mask_refs)


@pytest.mark.usefixtures("reset_block_mask_caches")
def test_flex_mask_cache_hit_preserves_recently_used_entry() -> None:
    mask_refs, block_mask_refs = [], []
    for _ in range(16):
        mask = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
        block_mask = cuda_attention._masked_block_mask(mask, True, 4, mask.device)
        mask_refs.append(weakref.ref(mask))
        block_mask_refs.append(weakref.ref(block_mask))
        del mask, block_mask

    recent_mask = mask_refs[0]()
    assert cuda_attention._masked_block_mask(
        recent_mask, True, 4, recent_mask.device
    ) is block_mask_refs[0]()
    del recent_mask

    new_mask = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    cuda_attention._masked_block_mask(new_mask, True, 4, new_mask.device)
    del new_mask
    gc.collect()

    assert cuda_attention._masked_block_mask.cache_info().currsize == 16
    assert mask_refs[0]() is not None
    assert block_mask_refs[0]() is not None
    assert mask_refs[1]() is None
    assert block_mask_refs[1]() is None


@pytest.mark.usefixtures("reset_block_mask_caches")
def test_flex_evicted_block_mask_remains_usable_by_attention() -> None:
    from torch.nn.attention.flex_attention import flex_attention

    mask = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    mask_ref = weakref.ref(mask)
    block_mask = cuda_attention._masked_block_mask(mask, True, 4, mask.device)
    del mask
    for _ in range(16):
        other_mask = torch.ones(1, 4, dtype=torch.long)
        cuda_attention._masked_block_mask(other_mask, True, 4, other_mask.device)
        del other_mask
    gc.collect()

    assert cuda_attention._masked_block_mask.cache_info().currsize == 16
    assert mask_ref() is not None
    torch.manual_seed(17)
    query, key, value = [torch.randn(1, 2, 4, 16, requires_grad=True) for _ in range(3)]
    output = flex_attention(query, key, value, block_mask=block_mask)
    allowed = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ])
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
    torch.testing.assert_close(output, expected)
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected_gradient)
