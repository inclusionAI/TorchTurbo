from __future__ import annotations

import importlib
from inspect import Parameter, signature
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
import transformers.integrations.flash_attention as flash_integration
import transformers.modeling_flash_attention_utils as flash_utils
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.utils._python_dispatch import TorchDispatchMode

from torchturbo.engine.core import TorchTurboEngine
from torchturbo.models.modules.attention import interface
from torchturbo.models.modules.attention.ulysses import register_turbo_ulysses_attention
from torchturbo.models.modules.attention.ulysses import TURBO_ULYSSES_ATTENTION, ulysses
from torchturbo.ops import OpDispatcher, OpPolicy, ops, registry
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.utils import flash_attn_utils

from .flash_attention_fakes import lazy_flash_attention_kernels


def _configure_ops(monkeypatch, runtime):
    dispatcher = OpDispatcher(device_runtime=runtime, registry=registry)
    monkeypatch.setattr(interface, "ops", dispatcher)
    return dispatcher


@pytest.fixture(autouse=True)
def _cpu_ops(monkeypatch):
    interface._position_sequence_lengths.cache_clear()
    interface._matching_cumulative_lengths.cache_clear()
    flash_attn_utils._packed_token_counts.cache_clear()
    return _configure_ops(monkeypatch, CpuRuntime())


def test_public_signature_preserves_hf_parameters_and_adds_only_maximum_logits() -> None:
    hf_parameters = signature(flash_utils._flash_attention_forward).parameters
    public_parameters = signature(interface.attention_forward).parameters

    assert [name for name in public_parameters if name != "return_max_logits"] == list(hf_parameters)
    for name, expected in hf_parameters.items():
        actual = public_parameters[name]
        assert actual.kind is expected.kind
        assert actual.default == expected.default
    maximum_logits = public_parameters["return_max_logits"]
    assert maximum_logits.kind is Parameter.KEYWORD_ONLY
    assert maximum_logits.default is False


def test_interface_preserves_metadata_for_ordinary_attention(monkeypatch):
    query = torch.randn(1, 4, 2, 3)
    mask = torch.ones(1, 4, dtype=torch.long)
    positions = torch.arange(4).reshape(1, 4)
    lengths = torch.tensor([0, 2, 4], dtype=torch.int32)
    output = torch.empty_like(query)
    ordinary = Mock(return_value=output)
    maximum = Mock()
    monkeypatch.setattr(flash_utils, "_flash_attention_forward", ordinary)
    monkeypatch.setattr(
        interface, "ops", SimpleNamespace(attention_with_max_logits=maximum)
    )
    metadata = dict(
        dropout=0.15,
        position_ids=positions,
        softmax_scale=0.7,
        sliding_window=8,
        use_top_left_mask=True,
        softcap=0.5,
        deterministic=True,
        cu_seq_lens_q=lengths,
        cu_seq_lens_k=lengths,
        max_length_q=2,
        max_length_k=2,
        target_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        s_aux=torch.ones(2),
        backend_option="retained",
    )

    actual = interface.attention_forward(
        query, query, query, mask, 4, False, **metadata,
    )

    assert actual is output
    ordinary.assert_called_once_with(query, query, query, mask, 4, False, **metadata)
    maximum.assert_not_called()


def test_hf_helper_preserves_output_identity(monkeypatch):
    query = torch.ones(1, 3, 2, 4)
    original = Mock(return_value=query)
    monkeypatch.setattr(flash_utils, "_flash_attention_forward", original)

    output = interface.attention_forward(query, query, query, None, 3, True)

    assert output is query
    assert original.call_args.args == (query, query, query, None, 3, True)
    assert "return_max_logits" not in original.call_args.kwargs


def test_ordinary_attention_does_not_require_ops_initialization(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(interface, "ops", ops)
    query = torch.ones(1, 2, 1, 3)
    original = Mock(return_value=query)
    monkeypatch.setattr(flash_utils, "_flash_attention_forward", original)

    assert interface.attention_forward(query, query, query, None, 2, True) is query
    original.assert_called_once()


def test_ordinary_attention_ignores_custom_ops_policy(monkeypatch):
    dispatcher = OpDispatcher(
        device_runtime=CpuRuntime(),
        registry=registry,
        default_policy=OpPolicy.CUSTOM,
    )
    monkeypatch.setattr(interface, "ops", dispatcher)
    query = torch.ones(1, 2, 1, 3)
    original = Mock(return_value=query)
    monkeypatch.setattr(flash_utils, "_flash_attention_forward", original)

    assert interface.attention_forward(query, query, query, None, 2, True) is query
    original.assert_called_once()


def test_maximum_logits_requires_explicit_ops_initialization(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(interface, "ops", ops)
    query = torch.ones(1, 2, 1, 3)

    with pytest.raises(RuntimeError, match="Ops dispatcher is not configured"):
        interface.attention_forward(
            query, query, query, None, 2, True, return_max_logits=True,
        )


@pytest.mark.parametrize(
    ("invalid_arguments", "message"),
    [
        ({"query_length": 3}, "query_length"),
        ({"softmax_scale": 0.0}, "positive finite scaling"),
        ({"softcap": 1.0}, "without logit softcapping"),
    ],
    ids=["query-length", "scale", "softcap"],
)
def test_invalid_common_arguments_fail_before_stats_op_lookup(
    monkeypatch, invalid_arguments, message,
):
    lookup = Mock(side_effect=AssertionError("Invalid input reached stats op resolution."))

    class Ops:
        @property
        def attention_with_max_logits(self):
            return lookup()

    monkeypatch.setattr(interface, "ops", Ops())
    query = torch.ones(1, 4, 2, 3)
    arguments = dict(attention_mask=None, query_length=4, is_causal=True)
    arguments.update(invalid_arguments)

    with pytest.raises(ValueError, match=message):
        interface.attention_forward(query, query, query, return_max_logits=True, **arguments)

    lookup.assert_not_called()


@pytest.mark.parametrize(
    "patch_sequence", [("ordinary",), ("sp",), ("ordinary", "sp")],
    ids=["ordinary", "sp", "ordinary-then-sp"],
)
def test_hf_integration_preserves_the_original_helper(
    monkeypatch, patch_sequence,
):
    original = flash_utils._flash_attention_forward
    monkeypatch.setattr(
        flash_integration, "_flash_attention_forward", flash_integration._flash_attention_forward,
    )
    monkeypatch.setattr(flash_utils, "_get_unpad_data", flash_utils._get_unpad_data)
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(interface, "ops", None)

    query = torch.ones(1, 2, 3, 4, dtype=torch.bfloat16)
    kernel = Mock(side_effect=lambda query, key, value, **kwargs: query)
    unused_kernel = Mock(side_effect=AssertionError("Dense input used the varlen kernel."))
    process_kwargs = Mock(return_value={})
    monkeypatch.setattr(
        flash_utils, "lazy_import_flash_attention",
        Mock(return_value=(
            lazy_flash_attention_kernels(kernel, unused_kernel),
            process_kwargs,
        )),
    )
    gather = Mock(side_effect=lambda tensor, **kwargs: tensor.repeat_interleave(2, dim=1)[:, :, ::2])
    scatter = Mock(side_effect=lambda tensor, **kwargs: tensor[:, ::2].repeat_interleave(2, dim=2))
    monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", gather)
    monkeypatch.setattr(ulysses, "gather_heads_scatter_seq", scatter)

    engine = object.__new__(TorchTurboEngine)
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        is_causal=True,
        layer_idx=0,
        training=False,
    )
    try:
        for _ in range(2):
            for patch_kind in patch_sequence:
                if patch_kind == "ordinary":
                    engine._patch_transformers()
                else:
                    register_turbo_ulysses_attention(module.config, object(), 2)
                assert flash_utils._flash_attention_forward is original

        # Dispatch the way attention modules do: by config name.
        attention_fn = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
        output, weights = attention_fn(module, query, query, query, None, scaling=0.5)
    finally:
        # The attention entry is SP-bound and instance-local; drop it. The
        # mask-table entry is a class-level global declaration with no
        # instance state, so it intentionally outlives the test.
        ALL_ATTENTION_FUNCTIONS.pop(TURBO_ULYSSES_ATTENTION, None)

    torch.testing.assert_close(output, query.transpose(1, 2))
    assert weights is None
    assert flash_utils._flash_attention_forward is original
    kernel.assert_called_once()
    unused_kernel.assert_not_called()
    process_kwargs.assert_called_once()
    assert process_kwargs.call_args.kwargs["softmax_scale"] == 0.5
    if "sp" in patch_sequence:
        assert gather.call_count == 3
        scatter.assert_called_once()
        assert process_kwargs.call_args.kwargs["query_length"] == 6
    else:
        gather.assert_not_called()
        scatter.assert_not_called()
        assert flash_integration._flash_attention_forward is interface.attention_forward


def test_maximum_logits_reject_attention_sinks():
    query = torch.ones(1, 4, 2, 3)
    with pytest.raises(ValueError, match="attention sinks"):
        interface.attention_forward(
            query, query, query, None, 4, True,
            s_aux=torch.ones(2), return_max_logits=True,
        )


def test_maximum_logit_op_receives_normalized_arguments_and_corrected_dtype(monkeypatch):
    query = torch.ones(1, 4, 2, 3)
    mask = torch.ones(1, 4, dtype=torch.long)
    expected = (torch.zeros(1, 4, 2, 3, dtype=torch.bfloat16), torch.ones(2))
    compute = Mock(return_value=expected)
    monkeypatch.setattr(
        interface, "ops", SimpleNamespace(attention_with_max_logits=compute)
    )

    actual = interface.attention_forward(
        query, query, query, mask, 4, False,
        dropout=0.15, softmax_scale=0.7, target_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", return_max_logits=True,
    )

    torch.testing.assert_close(actual, expected)
    assert actual[1] is expected[1]
    compute.assert_called_once()
    arguments = compute.call_args.args
    assert len(arguments) == 7
    for tensor in arguments[:3]:
        torch.testing.assert_close(tensor, query.to(torch.bfloat16))
    assert query.dtype is torch.float32
    assert arguments[3] is mask
    assert arguments[4:] == (0.7, 0.15, False)
    assert compute.call_args.kwargs == {}


@pytest.mark.parametrize("packing", ["positions", "lengths", "lengths-without-maxima"])
def test_maximum_logit_op_receives_varlen_metadata_and_original_layout(
    monkeypatch, packing,
):
    query = torch.arange(36.0).reshape(2, 3, 2, 3)
    value = torch.arange(24.0).reshape(2, 3, 2, 2)
    maxima = torch.ones(2)
    compute = Mock(side_effect=lambda query, key, value, *args, **kwargs: (value + 1, maxima))
    monkeypatch.setattr(
        interface, "ops", SimpleNamespace(attention_with_max_logits=compute)
    )
    if packing != "positions":
        lengths = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        max_length = 3 if packing == "lengths" else None
        metadata = dict(
            cu_seq_lens_q=lengths, cu_seq_lens_k=lengths,
            max_length_q=max_length, max_length_k=max_length,
            position_ids=torch.zeros(2, 3, dtype=torch.long),
        )
        expected_lengths = lengths
    else:
        metadata = dict(position_ids=torch.tensor([[0, 1, 0], [0, 0, 1]]))
        expected_lengths = torch.tensor([0, 2, 3, 4, 6], dtype=torch.int32)
        max_length = 2

    output, actual_maxima = interface.attention_forward(
        query, query, value, None, 3, True, return_max_logits=True, **metadata,
    )

    torch.testing.assert_close(output, value + 1)
    assert actual_maxima is maxima
    compute.assert_called_once()
    arguments = compute.call_args.args
    assert len(arguments) == 7
    assert arguments[0] is query
    assert arguments[1] is query
    assert arguments[2] is value
    assert arguments[3] is None
    assert arguments[4:] == (3 ** -0.5, 0.0, True)
    received_metadata = compute.call_args.kwargs
    assert set(received_metadata) == {
        "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k",
    }
    for name in ("cu_seq_lens_q", "cu_seq_lens_k"):
        torch.testing.assert_close(received_metadata[name], expected_lengths)
        if packing != "positions":
            assert received_metadata[name] is lengths
    assert received_metadata["max_length_q"] == max_length
    assert received_metadata["max_length_k"] == max_length


@pytest.mark.parametrize(
    ("mask_values", "positions", "mask_dtype"),
    [
        ([1, 1, 2, 2], [0, 1, 2, 3], torch.int64),
        ([1, 1, 2, 2], [0, 1, 2, 3], torch.float32),
        ([1, 1, 1, 0], [0, 1, 2, 3], torch.int64),
        ([1, 1, 1, 0], [0, 1, 0, 1], torch.bool),
        ([1, 1, 1, 0], [0, 1, 0, 1], torch.int64),
        ([1, 1, 1, 0], [0, 1, 0, 1], torch.float32),
    ],
    ids=["packing-int", "packing-float", "padding", "resets-bool", "resets-int", "resets-float"],
)
def test_existing_masks_are_preserved_without_reading_device_scalars(
    monkeypatch, mask_values, positions, mask_dtype,
):
    class RejectScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                raise AssertionError("Mask normalization read a device scalar into Python.")
            return func(*args, **(kwargs or {}))

    query = torch.ones(2, 4, 1, 1, dtype=torch.bfloat16)
    mask = torch.tensor([mask_values], dtype=mask_dtype).expand(2, -1)
    position_ids = torch.tensor([positions])
    compute = Mock(return_value=(query, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))

    with RejectScalarReads():
        interface.attention_forward(
            query, query, query, mask, 4, True,
            position_ids=position_ids, return_max_logits=True,
        )

    assert compute.call_args.args[3] is mask
    assert compute.call_args.args[3].dtype is mask_dtype
    assert compute.call_args.kwargs == {}


def test_existing_mask_takes_precedence_over_position_and_varlen_metadata(monkeypatch):
    query = torch.ones(1, 4, 1, 1)
    mask = torch.tensor([[1, 1, 2, 2]])
    compute = Mock(return_value=(query, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))

    interface.attention_forward(
        query, query, query, mask, 4, True,
        position_ids=torch.zeros(1, 4, dtype=torch.long),
        cu_seq_lens_q=torch.tensor([0, 3, 4], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_length_q=3, max_length_k=2, return_max_logits=True,
    )

    assert compute.call_args.args[3] is mask
    assert compute.call_args.kwargs == {}


@pytest.mark.parametrize("position_batches", [1, 2])
def test_dense_positions_keep_the_unmasked_path(monkeypatch, position_batches):
    query = torch.ones(2, 4, 1, 1, dtype=torch.bfloat16)
    position_ids = torch.arange(4).repeat(position_batches, 1)
    compute = Mock(return_value=(query, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))

    interface.attention_forward(
        query, query, query, None, 4, True,
        position_ids=position_ids, return_max_logits=True,
    )

    assert compute.call_args.args[3] is None
    assert compute.call_args.kwargs == {}


@pytest.mark.parametrize(
    ("query_length", "use_top_left_mask", "is_causal", "expected_causal"),
    [(1, True, True, False), (2, True, True, True), (1, False, True, True), (1, True, False, False)],
)
def test_maximum_logits_normalize_hf_causal_alignment(
    monkeypatch, query_length, use_top_left_mask, is_causal, expected_causal
):
    query = torch.ones(1, query_length, 2, 3)
    compute = Mock(return_value=(query, torch.ones(2)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))

    interface.attention_forward(
        query, query, query, None, query_length, is_causal,
        use_top_left_mask=use_top_left_mask, return_max_logits=True,
    )

    assert compute.call_args.args[6] is expected_causal


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("empty_segment", [False, True])
def test_cumulative_lengths_span_batch_rows_with_correct_gradients_and_maxima(
    is_causal: bool, empty_segment: bool
) -> None:
    torch.manual_seed(16)
    query = torch.randn(2, 3, 2, 3, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 3, 2, 2, requires_grad=True)
    boundaries = [0, 2, 2, 5, 6] if empty_segment else [0, 2, 5, 6]
    lengths = torch.tensor(boundaries, dtype=torch.int32)
    scaling = 0.37

    output, maxima = interface.attention_forward(
        query, key, value, None, 3, is_causal,
        softmax_scale=scaling, cu_seq_lens_q=lengths, cu_seq_lens_k=lengths,
        return_max_logits=True,
    )

    flat_query, flat_key, flat_value = (
        tensor.flatten(0, 1) for tensor in (query, key, value)
    )
    outputs, expected_maxima = [], []
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            continue
        q, k, v = (
            tensor[start:end].transpose(0, 1).unsqueeze(0)
            for tensor in (flat_query, flat_key, flat_value)
        )
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scaling
        )
        outputs.append(attention.squeeze(0).transpose(0, 1))
        scores = q @ k.transpose(-1, -2) * scaling
        if is_causal:
            future = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, -torch.inf)
        expected_maxima.append(scores.detach().amax(dim=(0, 2, 3)))
    expected_output = torch.cat(outputs).reshape_as(value)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(maxima, torch.stack(expected_maxima).amax(dim=0))
    assert maxima.dtype == torch.float32
    assert not maxima.requires_grad
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected_output.square().sum(), (query, key, value))
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)


def test_explicit_cumulative_lengths_take_precedence_over_position_resets() -> None:
    query = torch.ones(2, 3, 1, 1)
    key = torch.arange(1.0, 7.0).reshape_as(query)
    lengths = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    positions = torch.zeros(2, 3, dtype=torch.long)
    kwargs = dict(cu_seq_lens_q=lengths, cu_seq_lens_k=lengths, return_max_logits=True)

    expected = interface.attention_forward(query, key, key, None, 3, True, **kwargs)
    actual = interface.attention_forward(
        query, key, key, None, 3, True, position_ids=positions, **kwargs
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("is_causal", [False, True])
def test_prefix_lengths_exclude_padding_from_maxima_and_gradients(
    batch_size: int, is_causal: bool,
) -> None:
    query = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0, 100.0]).reshape(batch_size, -1, 1, 1)
    query.requires_grad_()
    key = query.detach().clone().requires_grad_()
    value = torch.arange(12.0).reshape(batch_size, -1, 1, 2).requires_grad_()
    lengths = torch.tensor([0, 2, 4], dtype=torch.int32)
    mask = torch.tensor([[1, 1, 2, 2, 0, 0]])
    expected_output, expected_maxima = interface.attention_forward(
        query.reshape(1, 6, 1, 1), key.reshape(1, 6, 1, 1), value.reshape(1, 6, 1, 2),
        mask, 6, is_causal, softmax_scale=1.0, return_max_logits=True,
    )

    output, maxima = interface.attention_forward(
        query, key, value, None, query.shape[1], is_causal,
        cu_seq_lens_q=lengths, cu_seq_lens_k=lengths,
        max_length_q=2, max_length_k=2, softmax_scale=1.0, return_max_logits=True,
    )

    torch.testing.assert_close(output, expected_output.reshape_as(value))
    torch.testing.assert_close(maxima, expected_maxima)
    torch.testing.assert_close(maxima, torch.tensor([16.0]))
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected_output.square().sum(), (query, key, value))
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
        assert not actual.flatten(0, 1)[4:].any()


def test_ordinary_prefix_lengths_preserve_unequal_query_key_lengths(monkeypatch) -> None:
    query = torch.randn(2, 3, 1, 2, requires_grad=True)
    key = torch.randn(2, 4, 1, 2, requires_grad=True)
    value = torch.randn(2, 4, 1, 3, requires_grad=True)
    query_lengths = torch.tensor([0, 2, 4], dtype=torch.int32)
    key_lengths = torch.tensor([0, 3, 6], dtype=torch.int32)
    observed = []

    def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
        observed.append((q.shape, k.shape, v.shape, cu_seqlens_q, cu_seqlens_k))
        outputs = []
        for qs, qe, ks, ke in zip(
            cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_k[:-1], cu_seqlens_k[1:],
        ):
            outputs.append(F.scaled_dot_product_attention(
                q[qs:qe].transpose(0, 1), k[ks:ke].transpose(0, 1), v[ks:ke].transpose(0, 1),
            ).transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setattr(
        flash_utils, "lazy_import_flash_attention",
        lambda *args, **kwargs: (
            lazy_flash_attention_kernels(None, varlen_attention),
            lambda **kwargs: {},
        ),
    )
    output = interface.attention_forward(
        query, key, value, None, 3, False,
        cu_seq_lens_q=query_lengths, cu_seq_lens_k=key_lengths,
        max_length_q=2, max_length_k=3,
    )

    flat_query, flat_key, flat_value = (tensor.flatten(0, 1) for tensor in (query, key, value))
    expected = torch.cat([
        F.scaled_dot_product_attention(
            flat_query[qs:qs + 2].transpose(0, 1),
            flat_key[ks:ks + 3].transpose(0, 1),
            flat_value[ks:ks + 3].transpose(0, 1),
        ).transpose(0, 1)
        for qs, ks in ((0, 0), (2, 3))
    ])
    expected = F.pad(expected, (0, 0, 0, 0, 0, 2)).reshape(2, 3, 1, 3)
    torch.testing.assert_close(output, expected)
    assert observed[0][:3] == ((4, 1, 2), (6, 1, 2), (6, 1, 3))
    assert observed[0][3] is query_lengths
    assert observed[0][4] is key_lengths
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected_gradient)


@pytest.mark.parametrize("return_max_logits", [False, True])
@pytest.mark.parametrize("num_valid_tokens", [3, 4])
@pytest.mark.parametrize("separate_key_lengths", [False, True])
def test_prefix_counts_are_reused_across_layers_and_invalidated_on_mutation(
    monkeypatch, return_max_logits: bool, num_valid_tokens: int, separate_key_lengths: bool,
) -> None:
    class RejectScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.equal.default):
                raise AssertionError("Cached prefix counts must not read device scalars.")
            return func(*args, **(kwargs or {}))

    query = torch.ones(1, 4, 1, 1)
    lengths = torch.tensor([0, num_valid_tokens], dtype=torch.int32)
    key_lengths = lengths.clone() if separate_key_lengths else lengths
    compute = Mock(side_effect=lambda q, k, v, *args, **kwargs: (v, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))
    ordinary = Mock(side_effect=lambda q, k, v, *args, **kwargs: v)
    monkeypatch.setattr(flash_utils, "_flash_attention_forward", ordinary)

    def forward(cu, cu_key):
        return interface.attention_forward(
            query, query, query, None, 4, True,
            cu_seq_lens_q=cu, cu_seq_lens_k=cu_key, max_length_q=4, max_length_k=4,
            return_max_logits=return_max_logits,
        )

    forward(lengths, key_lengths)
    with RejectScalarReads():
        forward(lengths.view_as(lengths), key_lengths.view_as(key_lengths))
        forward(lengths, key_lengths)

    lengths[-1] = 2
    key_lengths[-1] = 2
    output = forward(lengths, key_lengths)
    if return_max_logits:
        output = output[0]
    torch.testing.assert_close(output.flatten(), torch.tensor([1.0, 1.0, 0.0, 0.0]))


@pytest.mark.parametrize("inference_scope", ["metadata", "execution"])
def test_prefix_count_cache_bypasses_inference_tensors_and_mode(
    monkeypatch, inference_scope: str,
) -> None:
    query = torch.ones(1, 4, 1, 1)
    ordinary = Mock(side_effect=lambda q, k, v, *args, **kwargs: v)
    monkeypatch.setattr(flash_utils, "_flash_attention_forward", ordinary)
    with torch.inference_mode(inference_scope == "metadata"):
        lengths = torch.tensor([0, 3], dtype=torch.int32)

    with torch.inference_mode():
        for num_valid_tokens in (3, 2):
            lengths[-1] = num_valid_tokens
            output = interface.attention_forward(
                query, query, query, None, 4, True,
                cu_seq_lens_q=lengths, cu_seq_lens_k=lengths,
                max_length_q=3, max_length_k=3,
            )
            assert output.shape == query.shape
            assert output.sum() == num_valid_tokens


@pytest.mark.parametrize(
    "lengths",
    [
        torch.tensor([0.0, 2.0, 6.0]),
        torch.tensor([False, True]),
        torch.tensor([[0, 2, 6]], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    ],
    ids=["float", "bool", "matrix", "empty", "single"],
)
def test_invalid_cumulative_lengths_are_rejected(lengths: torch.Tensor) -> None:
    query = torch.ones(2, 3, 1, 1)

    with pytest.raises(ValueError):
        interface.attention_forward(
            query, query, query, None, 3, True,
            cu_seq_lens_q=lengths, cu_seq_lens_k=lengths, return_max_logits=True,
        )


@pytest.mark.parametrize(
    "key_lengths",
    [
        torch.tensor([0, 3, 6], dtype=torch.int32),
        torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        torch.tensor([[0, 2, 6]], dtype=torch.int32),
    ],
    ids=["different-boundaries", "different-segments", "matrix"],
)
def test_mismatched_query_and_key_cumulative_lengths_are_rejected(key_lengths):
    query = torch.ones(2, 3, 1, 1)
    query_lengths = torch.tensor([0, 2, 6], dtype=torch.int32)

    with pytest.raises(ValueError):
        interface.attention_forward(
            query, query, query, None, 3, True,
            cu_seq_lens_q=query_lengths, cu_seq_lens_k=key_lengths,
            return_max_logits=True,
        )


def test_position_lengths_are_reused_across_layers_and_metadata_views(monkeypatch):
    query = torch.ones(2, 3, 1, 1)
    compute = Mock(side_effect=lambda query, key, value, *args, **kwargs: (value, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))
    positions = torch.tensor([[0, 1, 0], [0, 1, 2]])
    expected_lengths = torch.tensor([0, 2, 3, 6], dtype=torch.int32)

    interface.attention_forward(
        query, query, query, None, 3, True,
        position_ids=positions, return_max_logits=True,
    )
    first_lengths = compute.call_args.kwargs["cu_seq_lens_q"]
    next_query = query + 1
    interface.attention_forward(
        next_query, next_query, next_query, None, 3, True,
        position_ids=positions.view_as(positions), return_max_logits=True,
    )
    assert compute.call_args.args[3] is None
    assert compute.call_args.kwargs["cu_seq_lens_q"] is first_lengths
    assert compute.call_args.kwargs["cu_seq_lens_k"] is first_lengths
    torch.testing.assert_close(first_lengths, expected_lengths)

    interface.attention_forward(
        query, query, query, None, 3, True,
        position_ids=positions.clone(), return_max_logits=True,
    )
    assert compute.call_args.kwargs["cu_seq_lens_q"] is not first_lengths
    torch.testing.assert_close(compute.call_args.kwargs["cu_seq_lens_q"], expected_lengths)


def test_position_length_cache_tracks_in_place_changes(monkeypatch):
    query = torch.ones(1, 4, 1, 1)
    compute = Mock(return_value=(query, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))
    positions = torch.tensor([[0, 1, 0, 1]])

    interface.attention_forward(
        query, query, query, None, 4, True,
        position_ids=positions, return_max_logits=True,
    )
    first_lengths = compute.call_args.kwargs["cu_seq_lens_q"]
    positions.copy_(torch.tensor([[0, 1, 2, 0]]))
    interface.attention_forward(
        query, query, query, None, 4, True,
        position_ids=positions, return_max_logits=True,
    )
    next_lengths = compute.call_args.kwargs["cu_seq_lens_q"]
    assert next_lengths is not first_lengths
    torch.testing.assert_close(first_lengths, torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(next_lengths, torch.tensor([0, 3, 4], dtype=torch.int32))
    assert compute.call_args.kwargs["max_length_q"] == 3

    positions.copy_(torch.arange(4).unsqueeze(0))
    interface.attention_forward(
        query, query, query, None, 4, True,
        position_ids=positions, return_max_logits=True,
    )
    assert compute.call_args.args[3] is None
    assert compute.call_args.kwargs == {}


def test_explicit_cumulative_lengths_are_forwarded_without_conversion(monkeypatch):
    query = torch.ones(2, 3, 1, 1)
    lengths = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    compute = Mock(return_value=(query, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))

    for supplied_lengths in (lengths, lengths.view_as(lengths), lengths.clone()):
        interface.attention_forward(
            query, query, query, None, 3, True,
            cu_seq_lens_q=supplied_lengths, cu_seq_lens_k=supplied_lengths,
            return_max_logits=True,
        )
        assert compute.call_args.args[:3] == (query, query, query)
        assert compute.call_args.args[3] is None
        assert compute.call_args.kwargs["cu_seq_lens_q"] is supplied_lengths
        assert compute.call_args.kwargs["cu_seq_lens_k"] is supplied_lengths
        assert compute.call_args.kwargs["max_length_q"] is None
        assert compute.call_args.kwargs["max_length_k"] is None

    lengths[1] = 3
    interface.attention_forward(
        query, query, query, None, 3, True,
        cu_seq_lens_q=lengths, cu_seq_lens_k=lengths, return_max_logits=True,
    )
    assert compute.call_args.kwargs["cu_seq_lens_q"] is lengths
    torch.testing.assert_close(lengths, torch.tensor([0, 3, 5, 6], dtype=torch.int32))


@pytest.mark.parametrize("inference_scope", ["metadata", "execution"])
def test_position_length_cache_bypasses_inference_tensors_and_mode(
    monkeypatch, inference_scope,
):
    query = torch.ones(1, 4, 1, 1)
    compute = Mock(return_value=(query, torch.ones(1)))
    monkeypatch.setattr(interface, "ops", SimpleNamespace(attention_with_max_logits=compute))
    with torch.inference_mode(inference_scope == "metadata"):
        positions = torch.tensor([[0, 1, 0, 1]])

    with torch.inference_mode(inference_scope == "execution"):
        interface.attention_forward(
            query, query, query, None, 4, True,
            position_ids=positions, return_max_logits=True,
        )
    first_lengths = compute.call_args.kwargs["cu_seq_lens_q"]
    interface.attention_forward(
        query, query, query, None, 4, True,
        position_ids=positions, return_max_logits=True,
    )
    second_lengths = compute.call_args.kwargs["cu_seq_lens_q"]
    assert second_lengths is not first_lengths
    assert not second_lengths.is_inference()
    torch.testing.assert_close(second_lengths, torch.tensor([0, 2, 4], dtype=torch.int32))
