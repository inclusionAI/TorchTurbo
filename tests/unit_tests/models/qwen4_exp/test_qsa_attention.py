"""QSA varlen packing, gradients, and bounded saved activations."""

from __future__ import annotations

import copy
import itertools

import pytest
import torch
import torch.nn.functional as F
import transformers.modeling_flash_attention_utils as flash_utils
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from torchturbo.models.qwen4_exp import qsa
from torchturbo.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

from ..flash_attention_fakes import lazy_flash_attention_kernels, process_flash_kwargs
from .test_model import _small_config, _token_ids_to_mask


def _cpu_varlen_kernel(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    causal,
    softmax_scale=None,
    dropout_p=0.0,
    **kwargs,
):
    assert not causal
    assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
    assert int(cu_seqlens_q[-1]) == query.shape[0]
    assert int(cu_seqlens_k[-1]) == key.shape[0] == value.shape[0]
    outputs = []
    repeats = query.shape[1] // key.shape[1]
    for i, (start, stop) in enumerate(itertools.pairwise(cu_seqlens_k)):
        q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].transpose(0, 1)
        k = key[start:stop].repeat_interleave(repeats, dim=1).transpose(0, 1)
        v = value[start:stop].repeat_interleave(repeats, dim=1).transpose(0, 1)
        outputs.append(
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=softmax_scale,
                dropout_p=dropout_p,
            ).transpose(0, 1)
        )
    return torch.cat(outputs)


def _flash_kwargs(**kwargs):
    result = process_flash_kwargs(**kwargs)
    result["dropout_p"] = kwargs["dropout"]
    return result


@pytest.fixture
def flash_calls(monkeypatch):
    calls = []

    def varlen(query, key, value, **kwargs):
        calls.append((kwargs["cu_seqlens_q"].numel() - 1, key.shape[0]))
        return _cpu_varlen_kernel(query, key, value, **kwargs)

    kernels = lazy_flash_attention_kernels(None, varlen)
    monkeypatch.setattr(
        flash_utils,
        "lazy_import_flash_attention",
        lambda implementation: (kernels, _flash_kwargs),
    )
    return calls


@pytest.mark.parametrize("input_kind", ["fp32_leaf", "fp32_nonleaf", "bf16_leaf"])
def test_qsa_autocast_preserves_input_and_parameter_gradients(
    monkeypatch, flash_calls, input_kind
) -> None:
    torch.manual_seed(803)
    config = _small_config().text_config
    attention = Qwen4ExpTextAttention(config, layer_idx=3).float()
    reference = copy.deepcopy(attention)
    dtype = torch.bfloat16 if input_kind == "bf16_leaf" else torch.float32
    source = torch.randn(1, 8, config.hidden_size, dtype=dtype, requires_grad=True)
    hidden = source + 0 if input_kind == "fp32_nonleaf" else source
    reference_source = source.detach().clone().requires_grad_()
    reference_hidden = (
        reference_source + 0 if input_kind == "fp32_nonleaf" else reference_source
    )
    phases = torch.randn(1, 8, config.head_dim // 2).repeat(1, 1, 2)
    positions = (phases.cos(), phases.sin())
    lengths = torch.tensor([0, 3, 8], dtype=torch.int32)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, _ = qsa.qsa_forward(
            attention, hidden, positions, None, cu_seq_lens_q=lengths
        )
        token_ids = qsa.qsa_indexer_forward(
            attention.indexer, hidden, positions, lengths
        )
    indexer_forward = qsa.qsa_indexer_forward

    def select_without_cache(*args, **kwargs):
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            return indexer_forward(*args, **kwargs)

    # Keep normal Q/K/V cast accumulation, disabling caching only for selection.
    with monkeypatch.context() as patch:
        patch.setattr(qsa, "qsa_indexer_forward", select_without_cache)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reference_ids = qsa.qsa_indexer_forward(
                reference.indexer, reference_hidden, positions, lengths
            )
            expected, _ = qsa.qsa_forward(
                reference, reference_hidden, positions, None, cu_seq_lens_q=lengths
            )
    torch.testing.assert_close(token_ids, reference_ids, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    probe = torch.randn_like(output)
    output.backward(probe)
    expected.backward(probe)
    assert source.grad is not None
    assert torch.count_nonzero(source.grad) > 0
    torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)
    for (name, parameter), (_, expected_parameter) in zip(
        attention.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.requires_grad
        if name.startswith("indexer."):
            assert parameter.grad is None and expected_parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, rtol=0, atol=0
            )


@pytest.mark.parametrize("outer_checkpoint", [False, True])
def test_chunked_qsa_preserves_gradients_without_saving_expanded_kv(
    monkeypatch,
    flash_calls,
    outer_checkpoint,
) -> None:
    torch.manual_seed(12)
    query = torch.randn(1, 9, 4, 4, dtype=torch.double, requires_grad=True)
    key = torch.randn(1, 9, 2, 4, dtype=torch.double, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    token_ids = torch.tensor(
        [
            [
                [0, -1, -1, -1],
                [0, 1, -1, -1],
                [0, 2, 3, -1],
                [1, 2, 3, 4],
                [4, -1, -1, -1],
                [0, 4, 5, 6],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1],
            ]
        ],
        dtype=torch.int32,
    )
    # Two queries' worst-case KV expansion fits the budget.
    kv_bytes_per_query = 4 * (2 * 4 + 2 * 4) * query.element_size()
    monkeypatch.setattr(qsa, "_QSA_ATTENTION_MAX_KV_BYTES", 2 * kv_bytes_per_query)
    input_storages = {
        tensor.untyped_storage().data_ptr() for tensor in (query, key, value)
    }
    saved_float_storages = []

    def save(tensor):
        if tensor.is_floating_point() and tensor.numel():
            saved_float_storages.append(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        if outer_checkpoint:
            output = checkpoint(
                qsa.qsa_attention_forward,
                query,
                key,
                value,
                token_ids,
                use_reentrant=False,
                softmax_scale=0.4,
            )
        else:
            output = qsa.qsa_attention_forward(
                query,
                key,
                value,
                token_ids,
                softmax_scale=0.4,
            )
    assert saved_float_storages
    assert set(saved_float_storages) <= input_storages
    assert len(flash_calls) == 3
    assert all(
        num_queries <= 2 and num_keys <= 8 for num_queries, num_keys in flash_calls
    )

    reference_inputs = [
        tensor.detach().clone().requires_grad_() for tensor in (query, key, value)
    ]
    ref_query, ref_key, ref_value = reference_inputs
    expected = F.scaled_dot_product_attention(
        ref_query.transpose(1, 2),
        ref_key.repeat_interleave(2, dim=2).transpose(1, 2),
        ref_value.repeat_interleave(2, dim=2).transpose(1, 2),
        attn_mask=_token_ids_to_mask(token_ids, key.shape[1]),
        scale=0.4,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)
    probe = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (query, key, value), probe)
    expected_grads = torch.autograd.grad(expected, reference_inputs, probe)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)
    assert len(flash_calls) > 3


def test_padding_only_qsa_retains_zero_gradients(flash_calls) -> None:
    inputs = [torch.randn(1, 3, 2, 4, requires_grad=True) for _ in range(3)]
    token_ids = torch.full((1, 3, 4), -1, dtype=torch.int32)
    output = qsa.qsa_attention_forward(*inputs, token_ids)
    torch.testing.assert_close(output, torch.zeros_like(output))
    gradients = torch.autograd.grad(output.sum(), inputs)
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    assert flash_calls == []


@pytest.mark.parametrize("target_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padding_only", [False, True])
@pytest.mark.parametrize("needs_grad", [False, True])
def test_qsa_preserves_flash_dtype_correction(
    monkeypatch, flash_calls, target_dtype, input_dtype, padding_only, needs_grad
) -> None:
    torch.manual_seed(27)
    inputs = [
        torch.randn(1, 5, heads, 4, dtype=input_dtype, requires_grad=needs_grad)
        for heads in (4, 2, 2)
    ]
    token_ids = torch.tensor(
        [[[0, -1], [0, 1], [1, 2], [-1, -1], [-1, -1]]], dtype=torch.int32
    )
    if padding_only:
        token_ids.fill_(-1)
    # Include a separate padding-only chunk as well as padding within a chunk.
    monkeypatch.setattr(qsa, "_QSA_ATTENTION_MAX_KV_BYTES", 256)
    actual = qsa.qsa_attention_forward(
        *inputs, token_ids, target_dtype=target_dtype
    )
    reference_inputs = [
        tensor.detach().clone().requires_grad_(needs_grad) for tensor in inputs
    ]
    output_dtype = target_dtype if input_dtype == torch.float32 else input_dtype
    query, key, value = [tensor.to(output_dtype) for tensor in reference_inputs]
    # The 3-D varlen reference uses math SDPA; use the same rounding for dense SDPA.
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.repeat_interleave(2, dim=2).transpose(1, 2),
            value.repeat_interleave(2, dim=2).transpose(1, 2),
            attn_mask=_token_ids_to_mask(token_ids, key.shape[1]),
        ).transpose(1, 2)
    assert actual.dtype == output_dtype
    torch.testing.assert_close(actual, expected)
    if needs_grad:
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, inputs, probe)
        expected_grads = torch.autograd.grad(expected, reference_inputs, probe)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0.02, rtol=0.02)
    if padding_only:
        assert flash_calls == []


def test_qsa_checkpoint_preserves_dropout_rng(monkeypatch, flash_calls) -> None:
    torch.manual_seed(8)
    query = torch.randn(1, 4, 4, 4, dtype=torch.double, requires_grad=True)
    key = torch.randn(1, 4, 2, 4, dtype=torch.double, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    token_ids = torch.tensor([[[0, -1], [0, 1], [0, 2], [1, 3]]], dtype=torch.int32)
    monkeypatch.setattr(qsa, "_QSA_ATTENTION_MAX_KV_BYTES", 1)
    torch.manual_seed(21)
    output = qsa.qsa_attention_forward(query, key, value, token_ids, dropout=0.3)

    reference_inputs = [
        tensor.detach().clone().requires_grad_() for tensor in (query, key, value)
    ]
    ref_query, ref_key, ref_value = reference_inputs
    torch.manual_seed(21)
    outputs = []
    for i in range(4):
        indices = token_ids[0, i]
        indices = indices[indices >= 0].long()
        outputs.append(
            F.scaled_dot_product_attention(
                ref_query[0, i : i + 1].transpose(0, 1),
                ref_key[0, indices].repeat_interleave(2, dim=1).transpose(0, 1),
                ref_value[0, indices].repeat_interleave(2, dim=1).transpose(0, 1),
                dropout_p=0.3,
            ).transpose(0, 1)
        )
    expected = torch.cat(outputs).unsqueeze(0)
    torch.testing.assert_close(output, expected)
    probe = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (query, key, value), probe)
    expected_grads = torch.autograd.grad(expected, reference_inputs, probe)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_qsa_device_kernel_matches_dense_reference(monkeypatch, device_type) -> None:
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        pytest.importorskip("flash_attn")
    else:
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("requires an NPU")

    torch.manual_seed(19)
    query = torch.randn(1, 17, 4, 32).to(torch.bfloat16).float().requires_grad_()
    key = torch.randn(1, 17, 2, 32).to(torch.bfloat16).float().requires_grad_()
    value = torch.randn_like(key).to(torch.bfloat16).float().requires_grad_()
    token_ids = torch.full((1, 17, 5), -1, dtype=torch.int32)
    for i in range(15):
        count = min(i + 1, 5)
        token_ids[0, i, :count] = torch.randperm(i + 1)[:count].int()
    device_inputs = [
        tensor.detach().to(device=device_type, dtype=torch.bfloat16).requires_grad_()
        for tensor in (query, key, value)
    ]
    monkeypatch.setattr(qsa, "_QSA_ATTENTION_MAX_KV_BYTES", 4096)
    actual = qsa.qsa_attention_forward(*device_inputs, token_ids.to(device_type))
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.repeat_interleave(2, dim=2).transpose(1, 2),
        value.repeat_interleave(2, dim=2).transpose(1, 2),
        attn_mask=_token_ids_to_mask(token_ids, key.shape[1]),
    ).transpose(1, 2)
    probe = torch.randn_like(expected).to(torch.bfloat16).float()
    actual.backward(probe.to(device=device_type, dtype=torch.bfloat16))
    expected.backward(probe)

    torch.testing.assert_close(actual.float().cpu(), expected, rtol=0.02, atol=0.02)
    for device_input, reference in zip(device_inputs, (query, key, value), strict=True):
        torch.testing.assert_close(
            device_input.grad.float().cpu(),
            reference.grad,
            rtol=0.03,
            atol=0.03,
        )
