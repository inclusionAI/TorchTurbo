"""CPU coverage of KDA sequence batching and its autograd wiring."""

import sys
from itertools import accumulate, pairwise
from types import ModuleType

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from torchturbo.ops import OpDispatcher, OpRegistry
from torchturbo.ops import dispatcher as dispatcher_module
from torchturbo.ops.packed_params import build_packed_params
from torchturbo.ops.providers.npu import kda
from torchturbo.runtime.npu import NpuRuntime


def _inputs(lengths, batch=1):
    generator = torch.Generator().manual_seed(42)
    num_tokens = sum(lengths)
    shapes = [(batch, num_tokens, 2, 4)] * 4 + [(batch, num_tokens, 2), (2,), (8,)]
    tensors = [torch.randn(shape, generator=generator, requires_grad=True) for shape in shapes]
    boundaries = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)
    return tensors, build_packed_params(boundaries, chunk_sizes=(64,))


def _call(tensors, packed, output_final_state=True, normalize=False):
    return kda.ascendc_chunk_kda(
        *tensors[:5], A_log=tensors[5], dt_bias=tensors[6],
        packed_params=packed, output_final_state=output_final_state,
        safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=normalize,
    )


def _reference_apply(calls):
    def apply(query, key, value, gate, beta, a_log, dt_bias, scale,
              output_final_state, lower_bound, use_gate_in_kernel, metadata):
        assert all(tensor.is_contiguous() for tensor in (query, key, value, gate, beta))
        assert use_gate_in_kernel and scale == 0.5 and lower_bound == -5.0
        if metadata is None:
            token_axis = 1
        else:
            boundaries, indices = metadata
            assert len(boundaries) - 1 <= kda._MAX_KDA_SEQUENCES
            assert boundaries[0] == 0 and boundaries[-1] == query.size(1)
            expected_indices = []
            for sequence_index, (left, right) in enumerate(pairwise(boundaries)):
                for chunk_index in range((right - left + 63) // 64):
                    expected_indices.extend((sequence_index, chunk_index))
            assert indices == expected_indices
            token_axis = 0
        calls.append(metadata)
        combined = query + 2 * key + 3 * value + 4 * gate + 5 * beta.unsqueeze(-1)
        output = combined.transpose(token_axis, token_axis + 1)
        output = output + a_log.reshape(2, 1) + dt_bias.reshape(2, 4)
        final_state = None
        if output_final_state:
            assert metadata is not None
            final_state = torch.stack([
                output[left:right].sum(dim=0).unsqueeze(-2).expand(2, 4, 4)
                for left, right in pairwise(boundaries)
            ])
        return output, final_state
    return apply


@pytest.mark.parametrize("num_sequences", [1023, 1024, 1025, 2048, 2049])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_sequence_count_boundaries(monkeypatch, num_sequences, output_final_state):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    tensors, packed = _inputs([1] * num_sequences)
    output, final_state = _call(tensors, packed, output_final_state)

    assert [len(metadata[0]) - 1 for metadata in calls] == [
        min(kda._MAX_KDA_SEQUENCES, num_sequences - start)
        for start in range(0, num_sequences, kda._MAX_KDA_SEQUENCES)
    ]
    assert output.shape == (1, num_sequences, 2, 4)
    assert (final_state is not None) == output_final_state
    if output_final_state:
        assert final_state.shape == (num_sequences, 2, 4, 4)


@pytest.mark.parametrize("use_raw_offsets", [False, True])
def test_explicit_dispatch_reaches_batched_provider(monkeypatch, use_raw_offsets):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    monkeypatch.setattr(kda.AscendCKDA, "available", lambda self: True)
    monkeypatch.setattr(dispatcher_module, "iter_provider_modules", lambda device: (kda.__name__,))
    registry = OpRegistry()
    registry.register(kda.AscendCKDA.op_spec())
    dispatcher = OpDispatcher(registry=registry, device_runtime=NpuRuntime())
    tensors, packed = _inputs([1] * 1025)
    metadata = {"cu_seqlens": packed.cu_seqlens} if use_raw_offsets else {"packed_params": packed}
    output, final_state = dispatcher.kda(
        *tensors[:5], A_log=tensors[5], dt_bias=tensors[6],
        use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
        output_final_state=True, impl="ascendc_kda", **metadata,
    )
    assert [len(group[0]) - 1 for group in calls] == [1023, 2]
    assert output.shape == (1, 1025, 2, 4)
    assert final_state.shape == (1025, 2, 4, 4)


@pytest.mark.parametrize("use_checkpoint", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_batched_outputs_and_gradients_match_single_call(
    monkeypatch, use_checkpoint, output_final_state, normalize,
):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    triton = ModuleType("fla_npu.ops.triton")
    triton.l2norm = lambda tensor: torch.nn.functional.normalize(tensor, dim=-1)
    monkeypatch.setitem(sys.modules, "fla_npu.ops.triton", triton)
    lengths = [1, 63, 64, 65, 0, 2, 0, 127, 1]
    baseline_inputs, packed = _inputs(lengths)
    split_inputs = [tensor.detach().clone().requires_grad_() for tensor in baseline_inputs]
    baseline, baseline_state = _call(baseline_inputs, packed, output_final_state, normalize)
    upstream = torch.randn_like(baseline)
    (baseline * upstream).sum().backward()

    monkeypatch.setattr(kda, "_MAX_KDA_SEQUENCES", 4)
    if use_checkpoint:
        actual, actual_state = checkpoint(
            lambda *inputs: _call(inputs, packed, output_final_state, normalize),
            *split_inputs, use_reentrant=True,
        )
    else:
        actual, actual_state = _call(split_inputs, packed, output_final_state, normalize)
    (actual * upstream).sum().backward()

    torch.testing.assert_close(actual, baseline)
    if output_final_state:
        torch.testing.assert_close(actual_state, baseline_state)
    for actual_input, baseline_input in zip(split_inputs, baseline_inputs):
        torch.testing.assert_close(actual_input.grad, baseline_input.grad)
    assert [metadata[0] for metadata in calls[1:4]] == [
        [0, 1, 64, 128, 193], [0, 0, 2, 2, 129], [0, 1],
    ]


@pytest.mark.parametrize("max_sequences", [2, 1024])
def test_packed_metadata_stays_reusable_without_device_transfers(monkeypatch, max_sequences):
    tensors, packed = _inputs([0, 65, 63, 1])
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    monkeypatch.setattr(kda, "_MAX_KDA_SEQUENCES", max_sequences)

    def unexpected_transfer(*args, **kwargs):
        pytest.fail("KDA must consume the precomputed host metadata")

    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_transfer)
    monkeypatch.setattr(torch.Tensor, "tolist", unexpected_transfer)
    output, _ = _call(tensors, packed)
    assert output.shape == (1, 129, 2, 4)
    assert packed.cu_seqlens_list == [0, 0, 65, 128, 129]
    assert packed.chunk_indices_list["64"] == [1, 0, 1, 1, 2, 0, 3, 0]
    if max_sequences == 1024:
        assert calls[0][0] is packed.cu_seqlens_list
        assert calls[0][1] is packed.chunk_indices_list["64"]


def test_padding_tail_counts_as_an_isolated_sequence(monkeypatch):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    tensors, _ = _inputs([1027])
    packed = build_packed_params(torch.arange(1025), chunk_sizes=(64,), total_tokens=1027)
    output, final_state = _call(tensors, packed)
    assert [len(metadata[0]) - 1 for metadata in calls] == [1023, 2]
    assert calls[-1] == ([0, 1, 4], [0, 0, 1, 0])
    assert output.shape == (1, 1027, 2, 4)
    assert final_state.shape == (1025, 2, 4, 4)


def test_long_packed_sequence_is_not_split_by_token_count(monkeypatch):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    tensors, packed = _inputs([2049])
    output, final_state = _call(tensors, packed)
    assert len(calls) == 1 and calls[0][0] == [0, 2049]
    assert output.shape == (1, 2049, 2, 4)
    assert final_state.shape == (1, 2, 4, 4)


@pytest.mark.parametrize("batch", [1, 2])
def test_nonpacked_keeps_single_call(monkeypatch, batch):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    tensors, _ = _inputs([2049], batch=batch)
    output, final_state = _call(tensors, None, output_final_state=False)
    assert calls == [None]
    assert output.shape == (batch, 2049, 2, 4)
    assert final_state is None


def test_noncontiguous_inputs_keep_values_and_gradients(monkeypatch):
    calls = []
    monkeypatch.setattr(kda.ChunkKDAFunction, "apply", _reference_apply(calls))
    original, _ = _inputs([12])
    views = [tensor[:, ::2] for tensor in original[:5]] + original[5:]
    contiguous = [tensor.detach().clone().requires_grad_() for tensor in views]
    packed = build_packed_params(torch.tensor([0, 1, 3, 4, 6]), chunk_sizes=(64,))
    expected, _ = _call(contiguous, packed)
    expected.sum().backward()
    monkeypatch.setattr(kda, "_MAX_KDA_SEQUENCES", 2)
    actual, _ = _call(views, packed)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    for source, reference in zip(original[:5], contiguous[:5]):
        torch.testing.assert_close(source.grad[:, ::2], reference.grad)
        assert torch.count_nonzero(source.grad[:, 1::2]) == 0
    for source, reference in zip(original[5:], contiguous[5:]):
        torch.testing.assert_close(source.grad, reference.grad)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_autograd_function_receives_group_metadata_and_empty_states(monkeypatch, use_checkpoint):
    forward_calls, backward_calls = [], []
    ascendc = ModuleType("fla_npu.ops.ascendc")

    def forward(query, key, value, gate, beta, scale, chunk_size, **kwargs):
        boundaries, indices = kwargs["cu_seqlens"], kwargs["chunk_indices"]
        forward_calls.append((boundaries, indices))
        assert 0 < query.size(1) and len(boundaries) - 1 <= 2
        assert kwargs["layout"] == "NTD" and kwargs["disable_recompute"]
        assert all(tensor.is_contiguous() for tensor in (query, key, value, gate, beta))
        state = torch.ones(len(boundaries) - 1, query.size(0), query.size(-1), value.size(-1))
        return (value.transpose(0, 1), state, gate, query, key, value,
                value, query, key, value, state, None)

    def backward(query, key, value, beta, gate_cumsum, aqk, akk, weights,
                 query_gated, key_gated, value_new, state, grad_output, scale, **kwargs):
        backward_calls.append((kwargs["cu_seqlens"], kwargs["chunk_indices"]))
        assert kwargs["implementation"] == "legacy"
        assert kwargs["disable_recompute"] and grad_output.shape == value.shape
        assert grad_output.is_contiguous()
        torch.testing.assert_close(grad_output, torch.ones_like(value))
        return (
            torch.zeros_like(query), torch.zeros_like(key), grad_output,
            torch.zeros_like(beta), torch.zeros_like(kwargs["raw_g"]), None,
            torch.ones_like(kwargs["A_log"]), torch.ones_like(kwargs["dt_bias"]),
        )

    ascendc.chunk_kda_fwd = forward
    ascendc.chunk_kda_bwd = backward
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)
    monkeypatch.setattr(kda, "_MAX_KDA_SEQUENCES", 2)
    tensors, packed = _inputs([0, 0, 65, 1, 0, 0, 2])
    states = []

    def run(*inputs):
        output, final_state = _call(inputs, packed)
        states.append(final_state.detach())
        return output

    output = checkpoint(run, *tensors, use_reentrant=True) if use_checkpoint else run(*tensors)
    assert output.shape == (1, 68, 2, 4)
    torch.testing.assert_close(states[0][:, 0, 0, 0], torch.tensor([0., 0., 1., 1., 0., 0., 1.]))
    output.sum().backward()
    expected_calls = [([0, 65, 66], [0, 0, 0, 1, 1, 0]), ([0, 2], [0, 0])]
    assert forward_calls == expected_calls * (2 if use_checkpoint else 1)
    assert sorted(backward_calls) == sorted(expected_calls)
    for tensor_index, tensor in enumerate(tensors):
        expected_grad = 2 if tensor_index >= 5 else int(tensor_index == 2)
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, expected_grad))


@pytest.mark.parametrize("output_final_state", [False, True])
@pytest.mark.parametrize("use_gate_in_kernel", [False, True])
def test_empty_groups_return_zero_states_and_gradients(output_final_state, use_gate_in_kernel):
    tensors, packed = _inputs([0] * 2049)
    output, final_state = kda.ascendc_chunk_kda(
        *tensors[:5], A_log=tensors[5], dt_bias=tensors[6], packed_params=packed,
        output_final_state=output_final_state, safe_gate=True,
        use_gate_in_kernel=use_gate_in_kernel,
    )
    assert output.shape == (1, 0, 2, 4)
    if output_final_state:
        assert final_state.shape == (2049, 2, 4, 4)
        assert final_state.dtype == torch.float32
        assert torch.count_nonzero(final_state) == 0
    else:
        assert final_state is None
    output.sum().backward()
    for tensor in tensors[:5] + (tensors[5:] if use_gate_in_kernel else []):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))
    if not use_gate_in_kernel:
        assert tensors[5].grad is None and tensors[6].grad is None


def test_empty_groups_preserve_final_state_gradient_rejection():
    tensors, packed = _inputs([0] * 1025)
    _, final_state = _call(tensors, packed)
    with pytest.raises(NotImplementedError, match="gradients through final_state"):
        final_state.sum().backward()
