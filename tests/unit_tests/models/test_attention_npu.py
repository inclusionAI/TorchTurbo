import sys
from types import SimpleNamespace

import pytest
import torch
import transformers.modeling_flash_attention_utils as hf_attention
from transformers.integrations import flash_attention as flash_integration
from transformers.integrations import npu_flash_attention

from torchturbo.models.modules.attention import interface
from torchturbo.ops import OpDispatcher, registry
from torchturbo.ops.providers.npu import attention
from torchturbo.runtime.npu import NpuRuntime

from .flash_attention_fakes import lazy_flash_attention_kernels


@pytest.mark.parametrize("module", [None, SimpleNamespace()])
def test_npu_attention_availability_requires_kernel_api(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    assert not attention.NpuAttentionWithMaxLogits().available()


def test_npu_attention_availability_does_not_execute_kernel(monkeypatch):
    def unexpected_kernel(*args, **kwargs):
        pytest.fail("Availability checks must not execute the attention kernel.")

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=unexpected_kernel)
    )
    assert attention.NpuAttentionWithMaxLogits().available()


@pytest.fixture
def kernels(monkeypatch):
    calls = {"native": [], "flash": [], "varlen": []}

    def output(query, key, value):
        return query + (key.sum() + value.sum()) * 0

    def native(query, key, value, head_num, input_layout, **kwargs):
        calls["native"].append({
            "query": query, "key": key, "value": value,
            "head_num": head_num, "input_layout": input_layout, **kwargs,
        })
        head_values = torch.tensor([2.0, 7.0], requires_grad=True)
        if input_layout == "BSND":
            maxima = head_values.view(1, 2, 1, 1).expand(query.shape[0], 2, query.shape[1], 8)
        else:
            sequences = []
            start = 0
            for index, end in enumerate(kwargs["actual_seq_qlen"]):
                sequence = (head_values + index).view(2, 1, 1).expand(2, end - start, 8)
                sequences.append(sequence.reshape(-1, 8))
                start = end
            maxima = torch.cat(sequences).view(query.shape[0], 2, 8)
        return output(query, key, value), maxima, None

    def flash(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        calls["flash"].append({
            "query": query, "key": key, "value": value,
            "dropout_p": dropout_p, "softmax_scale": softmax_scale, "causal": causal, **kwargs,
        })
        return output(query, key, value)

    def varlen(
        query, key, value, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=None, max_seqlen_k=None,
        dropout_p=0.0, softmax_scale=None, causal=False, **kwargs,
    ):
        calls["varlen"].append({
            "query": query, "key": key, "value": value,
            "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_k": cu_seqlens_k,
            "dropout_p": dropout_p, "softmax_scale": softmax_scale, "causal": causal, **kwargs,
        })
        return output(query, key, value)

    process_kwargs = hf_attention._lazy_define_process_function(varlen)
    monkeypatch.setattr(
        hf_attention, "lazy_import_flash_attention",
        lambda *args, **kwargs: (lazy_flash_attention_kernels(flash, varlen), process_kwargs),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=native))
    monkeypatch.setattr(
        npu_flash_attention, "get_attn_mask_npu",
        lambda device: torch.ones(4, 4, dtype=torch.bool).triu(1),
    )
    monkeypatch.setattr(
        interface, "ops", OpDispatcher(device_runtime=NpuRuntime(), registry=registry)
    )
    return calls


def test_default_uses_hf_utils_helper_after_integration_sp_patch(monkeypatch, kernels):
    query, key, value = torch.randn(3, 1, 4, 2, 3)
    original = hf_attention._flash_attention_forward

    def sp_wrapper(*args, **kwargs):
        pytest.fail("The HF utils helper must not re-enter the integration SP wrapper.")

    monkeypatch.setattr(flash_integration, "_flash_attention_forward", sp_wrapper)
    result = interface.attention_forward(
        query, key, value, None, 4, True,
        dropout=0.2, softmax_scale=0.37, attn_implementation="flash_attention_2",
    )
    assert hf_attention._flash_attention_forward is original
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, query)
    assert not kernels["native"]
    call = kernels["flash"][0]
    assert call["dropout_p"] == 0.2
    assert call["softmax_scale"] == 0.37
    assert call["causal"] is True


@pytest.mark.parametrize("scenario", ["ordinary", "padding", "positions", "lengths", "cross_row_lengths"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_npu_auxiliary_path_reuses_hf_preprocessing(kernels, scenario, is_causal):
    batch_size, seq_length = (2, 4) if scenario == "padding" else (1, 6)
    if scenario == "cross_row_lengths":
        batch_size, seq_length = 2, 3
    query, key, value = torch.randn(3, batch_size, seq_length, 2, 3)
    mask = None
    kwargs = {"target_dtype": torch.bfloat16, "softmax_scale": 0.23, "dropout": 0.1}
    if scenario == "padding":
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    elif scenario == "positions":
        kwargs["position_ids"] = torch.tensor([[0, 1, 0, 1, 2, 0]])
    elif scenario in ("lengths", "cross_row_lengths"):
        kwargs.update(
            cu_seq_lens_q=torch.tensor([0, 2, 6], dtype=torch.int32),
            cu_seq_lens_k=torch.tensor([0, 2, 6], dtype=torch.int32),
            max_length_q=4, max_length_k=4,
        )
    expected = interface.attention_forward(
        query, key, value, mask, seq_length, is_causal, **kwargs
    )
    output, maxima = interface.attention_forward(
        query, key, value, mask, seq_length, is_causal,
        return_max_logits=True, **kwargs,
    )
    torch.testing.assert_close(output, expected)
    native = kernels["native"][0]
    baseline = (kernels["flash"] if scenario == "ordinary" else kernels["varlen"])[0]
    for name in ("query", "key", "value"):
        torch.testing.assert_close(native[name], baseline[name])
        assert native[name].dtype == torch.bfloat16
    assert native["scale"] == 0.23
    assert native["keep_prob"] == 0.9
    assert ("atten_mask" in native) == is_causal
    if is_causal:
        assert native["sparse_mode"] == npu_flash_attention.SPARSE_MODE
    if scenario == "ordinary":
        assert native["input_layout"] == "BSND"
        assert maxima.tolist() == [2, 7]
    else:
        assert native["input_layout"] == "TND"
        assert native["actual_seq_qlen"] == tuple(baseline["cu_seqlens_q"][1:].tolist())
        assert native["actual_seq_kvlen"] == tuple(baseline["cu_seqlens_k"][1:].tolist())
        num_sequences = len(native["actual_seq_qlen"])
        assert maxima.tolist() == [1 + num_sequences, 6 + num_sequences]
    assert maxima.dtype == torch.float32
    assert not maxima.requires_grad


@pytest.mark.parametrize("position_ids", [None, torch.tensor([[0, 1, 2, 3, 4, 0]])])
def test_integer_packing_mask_works_without_engine_patch(kernels, position_ids):
    query, key, value = torch.randn(3, 1, 6, 2, 3)
    mask = torch.tensor([[1, 1, 2, 2, 2, 0]])
    output, maxima = interface.attention_forward(
        query, key, value, mask, 6, True,
        position_ids=position_ids, return_max_logits=True,
    )
    native = kernels["native"][0]
    assert native["input_layout"] == "TND"
    assert native["actual_seq_qlen"] == (2, 5)
    assert native["actual_seq_kvlen"] == (2, 5)
    torch.testing.assert_close(native["query"], query[0, :5])
    torch.testing.assert_close(output[:, :5], query[:, :5])
    assert not output[:, 5:].any()
    assert maxima.tolist() == [3, 8]


def test_in_place_packing_mask_updates_sequence_boundaries(kernels):
    query, key, value = torch.randn(3, 1, 6, 2, 3)
    mask = torch.tensor([[1, 1, 2, 2, 2, 0]])
    positions = torch.tensor([[0, 1, 2, 3, 4, 0]])
    interface.attention_forward(
        query, key, value, mask, 6, True,
        position_ids=positions, return_max_logits=True,
    )
    mask[0, 2] = 1
    interface.attention_forward(
        query, key, value, mask, 6, True,
        position_ids=positions, return_max_logits=True,
    )
    first, second = kernels["native"]
    assert first["actual_seq_qlen"] == first["actual_seq_kvlen"] == (2, 5)
    assert second["actual_seq_qlen"] == second["actual_seq_kvlen"] == (3, 5)


def test_explicit_lengths_reach_kernel_without_unpadding(monkeypatch, kernels):
    query = torch.randn(2, 2, 2, 3, requires_grad=True)
    key, value = [torch.randn(2, 3, 2, 3, requires_grad=True) for _ in range(2)]
    query_lengths = torch.tensor([0, 1, 4], dtype=torch.int32)
    key_lengths = torch.tensor([0, 2, 6], dtype=torch.int32)
    original = attention._npu_fusion_attention

    def kernel(*args, **kwargs):
        assert kwargs["cu_seqlens_q"] is query_lengths
        assert kwargs["cu_seqlens_k"] is key_lengths
        return original(*args, **kwargs)

    def unexpected_unpadding(*args, **kwargs):
        pytest.fail("Explicit sequence boundaries must bypass unpadding and repadding.")

    monkeypatch.setattr(attention, "_npu_fusion_attention", kernel)
    for helper in ("_upad_input", "_unpad_input", "_index_first_axis", "_pad_input"):
        monkeypatch.setattr(hf_attention, helper, unexpected_unpadding)
    monkeypatch.setattr(
        "torchturbo.utils.flash_attn_utils.get_unpad_data", unexpected_unpadding
    )
    output, maxima = attention.NpuAttentionWithMaxLogits()(
        query, key, value, None, 0.5, 0.0, True,
        cu_seq_lens_q=query_lengths, cu_seq_lens_k=key_lengths,
    )
    native = kernels["native"][0]
    assert native["input_layout"] == "TND"
    assert native["actual_seq_qlen"] == (1, 4)
    assert native["actual_seq_kvlen"] == (2, 6)
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        torch.testing.assert_close(native[name], tensor.flatten(0, 1))
    torch.testing.assert_close(output, query)
    assert maxima.tolist() == [3, 8]
    output.sum().backward()
    torch.testing.assert_close(query.grad, torch.ones_like(query))
    for tensor in (key, value):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


def test_padding_mask_takes_priority_over_explicit_lengths(kernels):
    query, key, value = torch.randn(3, 1, 4, 2, 3)
    mask = torch.tensor([[0, 1, 1, 0]])
    lengths = torch.tensor([0, 4], dtype=torch.int32)
    output, _ = attention.NpuAttentionWithMaxLogits()(
        query, key, value, mask, 0.5, 0.0, True,
        cu_seq_lens_q=lengths, cu_seq_lens_k=lengths,
    )
    native = kernels["native"][0]
    assert native["actual_seq_qlen"] == (2,)
    assert native["actual_seq_kvlen"] == (2,)
    torch.testing.assert_close(output[:, 1:3], query[:, 1:3])
    assert not output[:, (0, 3)].any()


def test_partial_queries_preserve_hf_padding_and_static_cache_slicing(kernels):
    query = torch.randn(2, 2, 2, 3)
    key, value = torch.randn(2, 2, 6, 2, 3)
    mask = torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1]])
    scaling = query.shape[-1] ** -0.5
    expected = hf_attention._flash_attention_forward(
        query, key, value, mask, 2, True, softmax_scale=scaling
    )
    output, _ = attention.npu_attention_forward(
        query, key, value, mask, scaling, 0.0, True
    )
    torch.testing.assert_close(output, expected)
    native, baseline = kernels["native"][0], kernels["varlen"][0]
    for name in ("query", "key", "value"):
        torch.testing.assert_close(native[name], baseline[name])
    assert native["actual_seq_qlen"] == (2, 3)
    assert native["actual_seq_kvlen"] == (3, 4)
    assert native["key"].shape[0] == 4


def test_all_padding_skips_native_kernel_and_preserves_zero_gradients(kernels):
    query, key, value = [torch.randn(2, 4, 2, 3, requires_grad=True) for _ in range(3)]
    mask = torch.zeros(2, 4, dtype=torch.long)
    output, maxima = interface.attention_forward(
        query, key, value, mask, 4, True, return_max_logits=True
    )
    assert not kernels["native"]
    assert output.shape == query.shape
    assert not output.any()
    assert torch.isneginf(maxima).all()
    output.sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert not tensor.grad.any()


def test_top_left_single_query_uses_hf_causal_compatibility(kernels):
    query = torch.randn(1, 1, 2, 3)
    key, value = torch.randn(2, 1, 1, 2, 3)
    interface.attention_forward(
        query, key, value, None, 1, True,
        use_top_left_mask=True, return_max_logits=True,
    )
    assert "atten_mask" not in kernels["native"][0]


def test_explicit_lengths_do_not_require_unused_npu_max_lengths(kernels):
    query, key, value = torch.randn(3, 1, 5, 2, 3)
    lengths = torch.tensor([0, 2, 5], dtype=torch.int32)
    interface.attention_forward(
        query, key, value, None, 5, True,
        cu_seq_lens_q=lengths, cu_seq_lens_k=lengths, return_max_logits=True,
    )
    assert kernels["native"][0]["actual_seq_qlen"] == (2, 5)


def test_npu_maximum_logits_rejects_4d_masks_before_kernel(kernels):
    query, key, value = torch.randn(3, 1, 4, 2, 3)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="does not support 4D masks"):
        interface.attention_forward(
            query, key, value, mask, 4, True, return_max_logits=True
        )
    assert not kernels["native"]
