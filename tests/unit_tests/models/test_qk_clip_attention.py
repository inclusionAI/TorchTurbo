import pytest
import torch
import torch.nn.functional as F
import transformers.modeling_flash_attention_utils as hf_attention
from torch.utils.checkpoint import checkpoint

from torchturbo.models.modules.attention import interface
from torchturbo.ops import OpDispatcher, registry
from torchturbo.ops.providers.cuda.attention import TransformerEngineAttentionWithMaxLogits
from torchturbo.models.modules.attention.interface import attention_forward
from torchturbo.models.modules.attention.qk_clip import QKClipState
from torchturbo.models.modules.attention.ulysses import ulysses
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime

from .flash_attention_fakes import lazy_flash_attention_kernels, process_flash_kwargs


@pytest.fixture(autouse=True)
def _cpu_ops(monkeypatch):
    monkeypatch.setattr(interface, "ops", OpDispatcher(device_runtime=CpuRuntime(), registry=registry))


@pytest.mark.parametrize("is_causal", [True, False])
def test_output_gradients_and_scaled_maxima(is_causal):
    torch.manual_seed(12)
    query = torch.randn(2, 5, 3, 4, requires_grad=True)
    key = torch.randn(2, 5, 3, 4, requires_grad=True)
    value = torch.randn(2, 5, 3, 2, requires_grad=True)
    scaling = 0.37
    output, maxima = attention_forward(
        query, key, value, None, query.size(1), is_causal=is_causal,
        softmax_scale=scaling, return_max_logits=True,
    )
    reference = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        is_causal=is_causal, scale=scaling,
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference)
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(reference.square().sum(), (query, key, value))
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * scaling
    if is_causal:
        scores = scores.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), -torch.inf)
    torch.testing.assert_close(maxima, scores.amax((0, 2, 3)))
    assert maxima.dtype == torch.float32
    assert not maxima.requires_grad


@pytest.mark.parametrize("use_position_ids", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
def test_packed_sequences_and_padding_match_independent_attention(use_position_ids, is_causal):
    torch.manual_seed(25)
    seq_length = 5 if use_position_ids else 6
    query = torch.randn(1, seq_length, 2, 4, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    mask = torch.tensor([[1, 1, 2, 2, 2, 0]])
    positions = None
    if use_position_ids:
        mask = None
        positions = torch.tensor([[0, 1, 0, 1, 2]])
    output, observed_maxima = attention_forward(
        query, key, value, mask, query.size(1), is_causal=is_causal,
        softmax_scale=0.5, position_ids=positions, return_max_logits=True,
    )
    references, maxima = [], []
    for start, end in ((0, 2), (2, 5)):
        q, k, v = (tensor[:, start:end].transpose(1, 2) for tensor in (query, key, value))
        references.append(F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=0.5
        ).transpose(1, 2))
        scores = q @ k.transpose(-1, -2) * 0.5
        if is_causal:
            future = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, -torch.inf)
        maxima.append(scores.amax((0, 2, 3)))
    if not use_position_ids:
        references.append(torch.zeros_like(output[:, :1]))
    expected = torch.cat(references, dim=1)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(observed_maxima, torch.stack(maxima).amax(0))
    gradients = torch.autograd.grad(output.sum(), (query, key, value), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), (query, key, value))
    for actual, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected_gradient)
        if not use_position_ids:
            assert not actual[:, -1].any()


def test_padding_position_reset_does_not_merge_masked_sequences():
    query = torch.tensor([1.0, 1.0, 100.0, 100.0, 0.0]).reshape(1, 5, 1, 1)
    key = torch.tensor([100.0, 100.0, 1.0, 1.0, 0.0]).reshape_as(query)
    value = torch.tensor([3.0, 4.0, 5.0, 6.0, 999.0]).reshape_as(query)
    mask = torch.tensor([[1, 1, 2, 2, 0]])
    positions = torch.tensor([[0, 1, 2, 3, 0]])

    output, maxima = attention_forward(
        query, key, value, mask, 5, True, softmax_scale=1.0,
        position_ids=positions, return_max_logits=True,
    )

    torch.testing.assert_close(maxima, torch.tensor([100.0]))
    expected = torch.tensor([3.0, 3.5, 5.0, 5.5, 0.0]).reshape_as(output)
    torch.testing.assert_close(output, expected)


def test_other_batch_position_resets_do_not_change_packing_mask():
    query = torch.tensor([
        [1.0, 1.0, 100.0, 100.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]).reshape(2, 5, 1, 1)
    key = torch.tensor([
        [100.0, 100.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]).reshape_as(query)
    value = torch.tensor([3.0, 4.0, 5.0, 6.0, 999.0]).reshape(1, 5, 1, 1).expand_as(query)
    mask = torch.tensor([[1, 1, 2, 2, 0], [1, 1, 2, 2, 0]])
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 0, 1, 0]])

    output, maxima = attention_forward(
        query, key, value, mask, 5, True, softmax_scale=1.0,
        position_ids=positions, return_max_logits=True,
    )

    torch.testing.assert_close(maxima, torch.tensor([100.0]))
    expected = torch.tensor([3.0, 3.5, 5.0, 5.5, 0.0]).reshape(1, 5, 1, 1).expand_as(output)
    torch.testing.assert_close(output, expected)


def test_binary_mask_with_position_resets_matches_hf_mask_first_dispatch(monkeypatch):
    query = torch.tensor([1.0, 1.0, 100.0, 100.0, 0.0]).reshape(1, 5, 1, 1)
    key = torch.tensor([100.0, 100.0, 1.0, 1.0, 0.0]).reshape_as(query)
    value = torch.tensor([3.0, 4.0, 5.0, 6.0, 999.0]).reshape_as(query)
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    positions = torch.tensor([[0, 1, 0, 1, 0]])
    observed_lengths = []

    def varlen_attention(
        query, key, value, cu_seqlens_q, cu_seqlens_k,
        softmax_scale=None, causal=False, **kwargs,
    ):
        observed_lengths.append(cu_seqlens_q.clone())
        outputs = []
        boundaries = cu_seqlens_q.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q, k, v = (
                tensor[start:end].transpose(0, 1).unsqueeze(0)
                for tensor in (query, key, value)
            )
            output = F.scaled_dot_product_attention(
                q, k, v, scale=softmax_scale, is_causal=causal,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setattr(
        hf_attention, "lazy_import_flash_attention",
        lambda *args, **kwargs: (
            lazy_flash_attention_kernels(None, varlen_attention),
            process_flash_kwargs,
        ),
    )
    expected = hf_attention._flash_attention_forward(
        query, key, value, mask, 5, True, softmax_scale=1.0, position_ids=positions,
    )
    output, maxima = attention_forward(
        query, key, value, mask, 5, True, softmax_scale=1.0,
        position_ids=positions, return_max_logits=True,
    )

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(maxima, torch.tensor([10000.0]))
    torch.testing.assert_close(observed_lengths[0], torch.tensor([0, 4], dtype=torch.int32))


def test_explicit_cumulative_lengths_match_packed_mask():
    torch.manual_seed(24)
    query, key, value = torch.randn(3, 1, 6, 2, 4)
    mask = torch.tensor([[1, 1, 2, 2, 2, 2]])
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    expected = attention_forward(
        query, key, value, mask, query.size(1), is_causal=True, return_max_logits=True
    )
    actual = attention_forward(
        query, key, value, None, query.size(1), is_causal=True, return_max_logits=True,
        cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("is_causal", [False, True])
def test_packed_position_offsets_preserve_independent_batch_rows(is_causal):
    torch.manual_seed(51)
    query = torch.randn(2, 4, 2, 3, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(2, 4, 2, 2, requires_grad=True)
    positions = torch.tensor([[0, 1, 0, 1], [4, 5, 6, 7]])
    lengths = torch.tensor([0, 2, 4, 8], dtype=torch.int32)
    kwargs = dict(softmax_scale=0.37, return_max_logits=True)

    output, maxima = attention_forward(
        query, key, value, None, 4, is_causal, position_ids=positions, **kwargs,
    )
    expected_output, expected_maxima = attention_forward(
        query, key, value, None, 4, is_causal,
        cu_seq_lens_q=lengths, cu_seq_lens_k=lengths, **kwargs,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(maxima, expected_maxima)
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(expected_output.square().sum(), (query, key, value))
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(positions, torch.tensor([[0, 1, 0, 1], [4, 5, 6, 7]]))


def test_future_masked_logits_do_not_trigger_clipping():
    query = torch.tensor([10.0, 0.0]).reshape(1, 2, 1, 1)
    key = torch.tensor([0.0, 100.0]).reshape(1, 2, 1, 1)
    _, maxima = attention_forward(
        query, key, key, None, query.size(1), is_causal=True,
        softmax_scale=1.0, return_max_logits=True,
    )
    assert maxima.item() == 0


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_cpu_reference_accepts_four_dimensional_masks(mask_dtype):
    query = torch.ones(1, 3, 2, 1)
    key = torch.tensor([1.0, 4.0, 100.0]).reshape(1, 3, 1, 1).expand_as(query)
    mask = torch.tensor([[[[True, False, False], [True, True, False], [False, False, False]]]])
    if mask_dtype == torch.float32:
        mask = torch.zeros_like(mask, dtype=mask_dtype).masked_fill(~mask, torch.finfo(mask_dtype).min)
    output, maxima = attention_forward(
        query, key, key, mask, query.size(1), is_causal=True,
        softmax_scale=0.5, return_max_logits=True,
    )
    assert maxima.tolist() == [2, 2]
    assert not output[:, -1].any()


def test_state_borrows_packed_storage_and_preserves_negative_maxima():
    packed = torch.full((6,), -torch.inf)
    state = QKClipState(packed[1:5])
    assert not state.observed
    state.record(torch.tensor([-3.0, -7.0], requires_grad=True), head_offset=1)
    state.record(torch.tensor([-5.0, -2.0]), head_offset=1)
    assert state.observed
    assert packed.tolist() == [-torch.inf, -torch.inf, -3, -2, -torch.inf, -torch.inf]
    assert not state.max_logits.requires_grad
    state.reset()
    assert not state.observed
    assert torch.isneginf(packed).all()


def test_state_ignores_nonfinite_heads_and_accepts_later_finite_observations():
    packed = torch.full((6,), -torch.inf)
    state = QKClipState(packed[1:5])
    state.record(torch.tensor([torch.nan, torch.inf, -torch.inf]), head_offset=1)
    assert state.observed
    assert torch.isneginf(packed).all()

    state.record(torch.tensor([-4.0, 200.0, -7.0]), head_offset=1)
    state.record(torch.tensor([torch.nan, torch.inf, 9.0]), head_offset=1)

    assert packed.tolist() == [-torch.inf, -torch.inf, -4, 200, 9, -torch.inf]


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_recomputation_and_microbatch_accumulation(use_reentrant):
    state = QKClipState(torch.full((1,), -torch.inf))

    def forward(query):
        return ulysses.ulysses_flash_attention_forward(
            query, query, query, None, query.size(1), is_causal=True,
            softmax_scale=1.0, sp_group=None, sp_size=1, qk_clip_state=state,
        )

    first = torch.full((1, 2, 1, 1), 3.0, requires_grad=True)
    checkpoint(forward, first, use_reentrant=use_reentrant).sum().backward()
    assert state.max_logits.item() == 9
    second = torch.full_like(first, 2.0, requires_grad=True)
    checkpoint(forward, second, use_reentrant=use_reentrant).sum().backward()
    assert state.max_logits.item() == 9
    state.reset()
    with torch.no_grad():
        forward(second)
    assert state.max_logits.item() == 4


@pytest.mark.parametrize("local_mask", [False, True])
@pytest.mark.parametrize("local_positions", [False, True])
def test_sp_collects_once_at_global_head_offsets(monkeypatch, local_mask, local_positions):
    query = torch.arange(1.0, 5.0).reshape(1, 1, 4, 1).expand(1, 4, 4, 1)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    head_shards = iter(tensor[:, :, 2:] for tensor in (query, key, value))
    forward_collectives, output_collectives, gathered = [], [], []

    def gather_sequence(tensor, **kwargs):
        forward_collectives.append(tensor)
        return next(head_shards)

    def scatter_sequence(output, **kwargs):
        output_collectives.append(output)
        return torch.cat((output, output), dim=2)[:, :2]

    def all_gather(outputs, tensor, group):
        gathered.append(tensor.clone())
        for output in outputs:
            output.copy_(tensor)

    monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", gather_sequence)
    monkeypatch.setattr(ulysses, "gather_heads_scatter_seq", scatter_sequence)
    monkeypatch.setattr(ulysses.dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(ulysses.dist, "all_gather", all_gather)
    state = QKClipState(torch.full((4,), -torch.inf))
    mask = torch.ones(1, 2 if local_mask else 4, dtype=torch.long)
    positions = torch.tensor([[0, 1]]) if local_positions else torch.tensor([[0, 1, 0, 1]])
    output = ulysses.ulysses_flash_attention_forward(
        query[:, :2], key[:, :2], value[:, :2], mask, 2, True,
        softmax_scale=1.0, position_ids=positions,
        sp_group=object(), sp_size=2, qk_clip_state=state,
    )
    assert output.shape == (1, 2, 4, 1)
    assert state.max_logits.tolist() == [-torch.inf, -torch.inf, 3, 4]
    assert len(gathered) == int(local_mask) + int(local_positions)
    assert len(forward_collectives) == 3
    assert len(output_collectives) == 1


def test_sp_explicit_prefix_lengths_exclude_padding_without_gathering_positions(monkeypatch):
    query = torch.tensor([1.0, 1.0, 1.0, 100.0]).reshape(1, 4, 1, 1).expand(1, 4, 2, 1)
    head_shards = iter(tensor[:, :, :1] for tensor in (query, query, query))
    lengths = torch.tensor([0, 2, 3], dtype=torch.int32)
    state = QKClipState(torch.full((2,), -torch.inf))

    monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", lambda tensor, **kwargs: next(head_shards))
    monkeypatch.setattr(
        ulysses, "gather_heads_scatter_seq",
        lambda tensor, **kwargs: tensor[:, :2].expand(1, 2, 2, 1),
    )
    monkeypatch.setattr(ulysses.dist, "get_rank", lambda group: 0)

    def unexpected_gather(*args, **kwargs):
        pytest.fail("Explicit global boundaries must not gather local position IDs.")

    monkeypatch.setattr(ulysses.dist, "all_gather", unexpected_gather)
    output = ulysses.ulysses_flash_attention_forward(
        query[:, :2], query[:, :2], query[:, :2], None, 2, True,
        position_ids=torch.tensor([[0, 1]]), cu_seq_lens_q=lengths, cu_seq_lens_k=lengths,
        max_length_q=2, max_length_k=2, softmax_scale=1.0,
        sp_group=object(), sp_size=2, qk_clip_state=state,
    )

    torch.testing.assert_close(output, torch.ones_like(output))
    torch.testing.assert_close(state.max_logits, torch.tensor([1.0, -torch.inf]))
    torch.testing.assert_close(lengths, torch.tensor([0, 2, 3], dtype=torch.int32))


def test_disabled_path_preserves_flash_attention_arguments(monkeypatch):
    query = torch.randn(1, 4, 2, 3)
    mask = torch.ones(1, 4, dtype=torch.long)
    positions = torch.arange(4).unsqueeze(0)
    captured = {}

    def flash_attention(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return query

    def unexpected_statistics(*args, **kwargs):
        pytest.fail("Disabled QK clipping must not call a statistics kernel.")

    monkeypatch.setattr(hf_attention, "_flash_attention_forward", flash_attention)
    monkeypatch.setattr(interface.ops, "attention_with_max_logits", unexpected_statistics)
    output = ulysses.ulysses_flash_attention_forward(
        query, query, query, mask, 4, False,
        softmax_scale=0.7, dropout=0.2, position_ids=positions,
        sp_group=None, sp_size=1, deterministic=True,
    )
    assert output is query
    assert captured["args"] == (query, query, query, mask, 4, False)
    assert captured["kwargs"] == {
        "position_ids": positions, "softmax_scale": 0.7,
        "dropout": 0.2, "deterministic": True,
        "sliding_window": None, "use_top_left_mask": False, "softcap": None,
        "cu_seq_lens_q": None, "cu_seq_lens_k": None,
        "max_length_q": None, "max_length_k": None,
        "target_dtype": None, "attn_implementation": None,
    }


def test_missing_cuda_max_score_api_is_rejected_on_forward(monkeypatch):
    import torch.nn.attention.flex_attention as flex_module

    query = torch.ones(1, 4, 2, 3)
    monkeypatch.setattr(TransformerEngineAttentionWithMaxLogits, "available", lambda self: False)
    monkeypatch.delattr(flex_module, "AuxRequest", raising=False)
    monkeypatch.setattr(interface, "ops", OpDispatcher(device_runtime=CudaRuntime(), registry=registry))
    with pytest.raises(RuntimeError, match="No implementation.*attention_with_max_logits"):
        attention_forward(query, query, query, None, 4, True, return_max_logits=True)
