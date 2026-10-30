from __future__ import annotations

import copy
from functools import partial
from itertools import pairwise
from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torchturbo.models.qwen4_exp import ple as impl
from torchturbo.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from torchturbo.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids


def new_reference() -> Qwen4ExpTextPLELayer:
    config = Qwen4ExpTextConfig(
        hidden_size=8,
        hc_count=2,
        ple_embed_dim=16,
        ngram_vocab_size_base=11,
        heads_per_ngram=2,
        eos_token_id=2,
        ple_layer_ids=[1],
        layer_types=["linear_attention"],
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
    )
    torch.manual_seed(17)
    layer = Qwen4ExpTextPLELayer(config, 0, 0)
    layer.ple_embedding.ngram_embedding.weight.requires_grad_(False)
    return layer


def row_ids(ngram, ids: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    return impl.NGramHasher.from_module(ngram).row_ids(ids, segments)


def adapted(reference: Qwen4ExpTextPLELayer) -> Qwen4ExpTextPLELayer:
    layer = copy.deepcopy(reference)
    rows = layer.ple_embedding.ngram_embedding.weight.detach().clone()
    layer.ple_embedding.ngram_embedding = impl.NGramEmbedding(rows)
    layer.forward = MethodType(impl.ple_forward, layer)
    return layer


@pytest.mark.parametrize("sp_size", [1, 2, 4, 8, 24])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ngram_matches_upstream_eos_and_packed_samples(sp_size, dtype):
    ngram = new_reference().ple_embedding
    ids = torch.tensor(
        [5, 2, 7, 8, 9, 6, 7, 2, 2, 8, 3, 2, 5, 6, 7, 8, 9, 3, 4, 5, 0, 0, 0, 0],
        dtype=dtype,
    )
    boundaries = [0, 5, 7, 20]
    segments = cu_seqlens_to_segment_ids(torch.tensor(boundaries), ids.numel())[0]
    indices = row_ids(ngram, ids, segments)
    actual = ngram.ngram_embedding(indices).flatten(-2)
    expected = torch.cat(
        [ngram(ids[None, a:b], None)[0] for a, b in pairwise(boundaries)]
    )
    torch.testing.assert_close(actual[:20], expected)
    length = ids.numel() // sp_size
    for rank in range(sp_size):
        start = rank * length
        window = slice(max(0, start - ngram.context_len), start + length)
        torch.testing.assert_close(
            row_ids(ngram, ids[window], segments[window])[-length:],
            indices[start : start + length],
        )


def test_shared_table_and_upstream_fallback(tmp_path, node_group):
    reference = new_reference()
    dense = reference.ple_embedding.ngram_embedding.weight
    layer = adapted(reference)
    embedding = layer.ple_embedding.ngram_embedding
    embedding.weight.directory = tmp_path
    embedding.weight.materialize_()
    ids = torch.tensor([[3, 2, 4, 5, 6]])
    torch.testing.assert_close(
        layer.ple_embedding(ids, None), reference.ple_embedding(ids, None)
    )
    assert dict(layer.ple_embedding.ngram_embedding.named_parameters())
    assert not dict(embedding.named_buffers())
    assert "_rows" not in embedding.state_dict()
    assert list(tmp_path.iterdir()) == []
    assert embedding.weight.is_shared()
    torch.testing.assert_close(
        embedding.gather(torch.tensor([1, 1, 3])), dense[[1, 1, 3]]
    )


@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
def test_nonpacked_ple_casts_only_lookup_results(table_dtype, compute_dtype):
    reference = new_reference()
    reference.ple_embedding.ngram_embedding.to(table_dtype)
    layer = adapted(reference).to(compute_dtype)
    layer.ple_embedding.ngram_embedding.to(table_dtype)
    reference.to(compute_dtype)
    ids = torch.tensor([[3, 2, 4, 5, 6]])
    hidden = torch.randn(1, 5, 16, dtype=compute_dtype, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    mask = torch.tensor([[0, 1, 1, 1, 1]])
    actual = layer(hidden, ids, None, mask)
    expected = reference(expected_hidden, ids, None, mask)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    assert layer.ple_embedding.ngram_embedding.weight.dtype == table_dtype


@pytest.mark.parametrize("window_tokens", [8192, 5])
@pytest.mark.parametrize("length", [1, 4, 8, 12])
@pytest.mark.parametrize("kernel_size", [1, 4])
def test_segment_conv_matches_independent_conv_and_gradients(
    monkeypatch, window_tokens, length, kernel_size
):
    monkeypatch.setattr(impl, "_PLE_CONV_WINDOW_TOKENS", window_tokens)
    torch.manual_seed(31)
    x = torch.randn(1, length, 4, requires_grad=True)
    weight = torch.randn(4, 1, kernel_size, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_w = weight.detach().clone().requires_grad_()
    split = max(1, length // 2)
    segments = torch.tensor([[1] * split + [2] * (length - split)])
    actual = impl.segment_causal_conv(x, segments, weight, 3)
    pieces = [ref_x[:, :split]]
    if split < length:
        pieces.append(ref_x[:, split:])
    expected = torch.cat(
        [
            F.silu(
                F.conv1d(
                    F.pad(part.transpose(1, 2), ((kernel_size - 1) * 3, 0)),
                    ref_w,
                    dilation=3,
                    groups=4,
                )
            ).transpose(1, 2)
            for part in pieces
        ],
        1,
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(weight.grad, ref_w.grad)


@pytest.mark.parametrize("length", [4, 12], ids=["short_shard", "long_shard"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
def test_local_conv_with_halos_matches_owned_gradients(length, dtype, needs_grad):
    torch.manual_seed(37)
    history, dilation, left_halo = 9, 3, 9
    start, stop = 12, 12 + length
    total = stop + history
    boundaries = (0, start + length // 2, total)
    segments = torch.tensor(
        [[1] * boundaries[1] + [2] * (total - boundaries[1])]
    )
    x = torch.randn(1, total, 3, dtype=dtype, requires_grad=True)
    weight = torch.randn(3, 1, 4, dtype=dtype, requires_grad=True)
    expected = torch.cat(
        [
            F.conv1d(
                F.pad(x[:, a:b].transpose(1, 2), (history, 0)),
                weight,
                dilation=dilation,
                groups=3,
            ).transpose(1, 2)
            for a, b in pairwise(boundaries)
        ],
        dim=1,
    )
    gradient = torch.randn_like(expected)
    expected_dx = torch.autograd.grad(expected, x, gradient, retain_graph=True)[0]
    # Future outputs contribute to local dX, but only owned outputs define dW.
    expected_dw = torch.autograd.grad(
        expected[:, start:stop], weight, gradient[:, start:stop]
    )[0]
    x_with_left = x.detach()[:, start - left_halo : stop]
    local_segments = segments[:, start - left_halo : stop + history]
    local_gradient = torch.cat(
        (gradient.new_zeros(1, left_halo, 3), gradient[:, start : stop + history]),
        dim=1,
    )

    actual = impl._local_conv_forward(
        x_with_left, local_segments, weight.detach(), dilation, 5, left_halo, dtype
    )
    actual_dx, actual_dw = impl._local_conv_backward(
        x_with_left,
        local_segments,
        weight.detach(),
        local_gradient,
        dilation,
        5,
        left_halo,
        dtype,
        needs_input_grad=needs_grad[0],
        needs_weight_grad=needs_grad[1],
    )

    torch.testing.assert_close(actual, expected[:, start:stop])
    if needs_grad[0]:
        torch.testing.assert_close(actual_dx, expected_dx[:, start:stop])
    else:
        assert actual_dx is None
    if needs_grad[1]:
        torch.testing.assert_close(actual_dw, expected_dw)
    else:
        assert actual_dw is None


@pytest.mark.parametrize("window_tokens", [8192, 16])
@pytest.mark.parametrize("autocast", [False, True])
def test_segment_conv_preserves_bfloat16_accumulation(
    monkeypatch, window_tokens, autocast
):
    monkeypatch.setattr(impl, "_PLE_CONV_WINDOW_TOKENS", window_tokens)
    torch.manual_seed(31)
    x = torch.randn(1, 40, 128).bfloat16().requires_grad_()
    weight = torch.randn(128, 1, 4)
    if not autocast:
        weight = weight.bfloat16()
    weight.requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    ref_w = weight.detach().clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = impl.segment_causal_conv(
            x, torch.ones(1, 40, dtype=torch.long), weight, 3
        )
        expected = F.silu(
            F.conv1d(
                F.pad(ref_x.transpose(1, 2), (9, 0)), ref_w, dilation=3, groups=128
            )
        ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    # Some CPU BF16 kernels round partial input gradients. Use an FP32
    # convolution oracle, with the same BF16 rounding before the activation.
    expected = F.silu(
        F.conv1d(
            F.pad(ref_x.float().transpose(1, 2), (9, 0)),
            ref_w.bfloat16().float(),
            dilation=3,
            groups=128,
        ).bfloat16()
    ).transpose(1, 2)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(weight.grad, ref_w.grad)


@pytest.mark.parametrize(
    "dtype,current_weight,left_grad,right_grad,expected",
    [
        (torch.bfloat16, 1.0078125, 510, -512, 0.9921875),
        (torch.float16, 1.0009765625, 4094, -4096, 0.9990234375),
    ],
)
def test_segment_conv_accumulates_boundary_gradients_before_rounding(
    dtype, current_weight, left_grad, right_grad, expected
):
    boundary = impl._PLE_CONV_WINDOW_TOKENS
    x = torch.zeros(1, boundary + 1, 1, dtype=dtype, requires_grad=True)
    weight = torch.tensor([[[1.0, current_weight]]], dtype=dtype, requires_grad=True)
    segments = torch.ones(1, boundary + 1, dtype=torch.long)
    output = impl.segment_causal_conv(x, segments, weight, 1)
    gradient = torch.zeros_like(output)
    gradient[:, boundary - 1] = left_grad
    gradient[:, boundary] = right_grad
    output.backward(gradient)

    # SiLU'(0) = 0.5. Round only after summing both output contributions.
    assert x.grad[0, boundary - 1, 0].item() == expected


@pytest.mark.parametrize("length", [8192, 32768])
def test_segment_conv_does_not_retain_fp32_activation_windows(length):
    x = torch.randn(1, length, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(16, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    segments = torch.ones(1, length, dtype=torch.long)
    fp32_storage = {}

    def record(tensor):
        if tensor.dtype == torch.float32:
            storage = tensor.untyped_storage()
            fp32_storage[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        output = impl.segment_causal_conv(x, segments, weight, 3)

    # A small FP32 weight copy is allowed; token activations stay low precision.
    assert sum(fp32_storage.values()) <= weight.numel() * 4
    output.sum().backward()


@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
def test_segment_conv_gradcheck_with_strided_inputs(monkeypatch, needs_grad):
    monkeypatch.setattr(impl, "_PLE_CONV_WINDOW_TOKENS", 3)
    x = torch.randn(2, 14, 2, dtype=torch.float64)[:, ::2]
    weight = torch.randn(2, 1, 6, dtype=torch.float64)[..., ::2]
    x.requires_grad_(needs_grad[0])
    weight.requires_grad_(needs_grad[1])
    segments = torch.tensor([[1, 1, 1, 2, 2, 2, 2], [1, 1, 2, 2, 2, 3, 3]])

    assert torch.autograd.gradcheck(
        lambda x, weight: impl.segment_causal_conv(x, segments, weight, 3),
        (x, weight),
    )


@pytest.mark.parametrize("reentrant", [None, False, True])
def test_packed_ple_matches_upstream_with_checkpoint(reentrant):
    reference = new_reference()
    layer = adapted(reference)
    ids = torch.tensor([[3, 2, 5, 6, 4, 7, 2, 8, 3, 6, 0, 0]])
    boundaries = torch.tensor([0, 4, 10])
    segments = cu_seqlens_to_segment_ids(boundaries, ids.shape[1])
    indices = row_ids(reference.ple_embedding, ids[0], segments[0])
    embeddings = layer.ple_embedding.ngram_embedding(indices).flatten(-2).unsqueeze(0)
    batch = impl.PLEBatch((embeddings,), segments)
    hidden = torch.randn(1, 12, 16, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    fn = partial(layer, ple_inputs=batch, past_key_values=None)
    actual = (
        fn(hidden)
        if reentrant is None
        else checkpoint(fn, hidden, use_reentrant=reentrant)
    )
    expected = torch.cat(
        [
            reference(ref_hidden[:, :4], ids[:, :4], None),
            reference(ref_hidden[:, 4:10], ids[:, 4:10], None),
            ref_hidden[:, 10:] * 0,
        ],
        1,
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    reference_params = dict(reference.named_parameters())
    for name, parameter in layer.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            continue
        torch.testing.assert_close(parameter.grad, reference_params[name].grad)


@pytest.mark.parametrize("reentrant", [False, True])
def test_prefetch_reuses_rows_during_decoder_recomputation(reentrant):
    layer = adapted(new_reference())

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.ple = layer

        def forward(self, hidden, ple_input_ids):
            return self.ple(hidden, ple_input_ids, None)

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Decoder()])

        def forward(self, inputs_embeds, ple_input_ids, **kwargs):
            fn = partial(self.layers[0], ple_input_ids=ple_input_ids)
            return checkpoint(fn, inputs_embeds, use_reentrant=reentrant)

    text = TextModel()
    impl.install_ple_prefetch(text, sp_size=1, sp_rank=0)
    embedding = layer.ple_embedding.ngram_embedding
    calls = []
    gather = embedding.gather

    def counted_gather(indices):
        calls.append(indices.clone())
        return gather(indices)

    embedding.gather = counted_gather
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    actual = text(
        inputs_embeds=hidden,
        ple_input_ids=torch.tensor([[3, 4, 5, 6]]),
        cu_seq_lens_q=torch.tensor([0, 4]),
    )
    actual.sum().backward()
    assert len(calls) == 1
    assert hidden.grad is not None
    assert layer.key_proj.weight.grad is not None


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize(
    "hash_device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="requires MPS"
            ),
        ),
    ],
)
def test_prefetch_uses_hash_buffers_reloaded_after_forward(assign, hash_device):
    reference = new_reference()
    layer = adapted(reference)
    ngram = layer.ple_embedding
    ngram.layer_multipliers = ngram.layer_multipliers.to(hash_device)
    ngram.ngram_heads_vocab_sizes = ngram.ngram_heads_vocab_sizes.to(hash_device)
    ngram.ngram_heads_offsets = ngram.ngram_heads_offsets.to(hash_device)

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            decoder = nn.Module()
            decoder.ple = layer
            self.layers = nn.ModuleList([decoder])

        def forward(self, input_ids, ple_input_ids=None, **kwargs):
            return ple_input_ids.rows[0]

    text = TextModel()
    impl.install_ple_prefetch(text, sp_size=1, sp_rank=0)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    boundaries = torch.tensor([0, 6], dtype=torch.int32)
    before = text(input_ids=ids, cu_seq_lens_q=boundaries)
    torch.testing.assert_close(before, reference.ple_embedding(ids, None))

    multipliers = reference.ple_embedding.layer_multipliers + 2
    reference.ple_embedding.load_state_dict(
        {"layer_multipliers": multipliers}, strict=False
    )
    ngram.load_state_dict(
        {"layer_multipliers": multipliers.to(hash_device)},
        strict=False,
        assign=assign,
    )
    expected = reference.ple_embedding(ids, None)
    assert not torch.equal(before, expected)
    actual = text(input_ids=ids, cu_seq_lens_q=boundaries)
    torch.testing.assert_close(actual, expected)


def test_deepcopy_prefetch_reads_the_copied_models_table():
    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            decoder = nn.Module()
            decoder.ple = adapted(new_reference())
            self.layers = nn.ModuleList([decoder])

        def forward(self, input_ids, ple_input_ids=None, **kwargs):
            return ple_input_ids.rows[0]

    source = TextModel()
    impl.install_ple_prefetch(source, sp_size=1, sp_rank=0)
    target = copy.deepcopy(source)
    target.layers[0].ple.ple_embedding.ngram_embedding.weight.add_(10)
    inputs = {
        "input_ids": torch.tensor([[3, 4, 5, 6]]),
        "cu_seq_lens_q": torch.tensor([0, 4], dtype=torch.int32),
    }
    torch.testing.assert_close(target(**inputs), source(**inputs) + 10)


def test_table_state_dict_restores_current_rows_without_source(tmp_path):
    reference = new_reference()
    source = adapted(reference).ple_embedding.ngram_embedding
    state = source.state_dict()
    assert set(state) == {"weight"}
    snapshot = {name: value.clone() + 3 for name, value in state.items()}
    original_parameters = dict(source.named_parameters())
    original_storage = source.weight.untyped_storage().data_ptr()
    source.load_state_dict(snapshot)
    assert source.weight.untyped_storage().data_ptr() == original_storage
    assert all(
        dict(source.named_parameters())[name] is param
        for name, param in original_parameters.items()
    )
    torch.testing.assert_close(
        source.weight, reference.ple_embedding.ngram_embedding.weight + 3
    )
    assert all(not param.requires_grad for param in source.parameters())
