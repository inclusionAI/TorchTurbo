"""GBSA sparse-pattern, autograd, packing, and optional-backend contracts."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from torchturbo.models.modules.attention.ulysses import all_gather_sp_with_grad_reduce
from torchturbo.models.qwen4_exp import gbsa, qsa
from torchturbo.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

from .test_model import _small_config, _token_ids_to_mask


def _reference(query, key, value, mask, scale):
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)
    lse = scores.logsumexp(dim=-1).transpose(0, 1).contiguous()
    probabilities = torch.nan_to_num(scores.softmax(dim=-1))
    output = torch.einsum("hqk,khd->qhd", probabilities, value.float()).to(query.dtype)
    return output, lse


def _block_mask(indices, counts, q_length, k_length, block_size):
    assert torch.equal(indices, indices[:1].expand_as(indices))
    mask = torch.zeros(q_length, k_length, dtype=torch.bool)
    for row in range(q_length):
        end = k_length - q_length + row + 1
        complete = end // block_size
        count = int(counts[0, row])
        selected = indices[0, row, :count].tolist()
        assert selected == sorted(set(selected))
        assert all(0 <= block < complete for block in selected)
        for block in selected:
            mask[row, block * block_size:(block + 1) * block_size] = True
        mask[row, complete * block_size:end] = True
    return mask


def _install_fake_gbsa(monkeypatch):
    calls = Counter()

    def metadata(indices, counts, n_query, n_key, dim, block_shape, **options):
        calls["metadata"] += 1
        assert indices.ndim == 3 and counts.shape == indices.shape[:-1]
        assert indices.dtype == counts.dtype == torch.int32
        assert options["residual_block_mode"] == 1
        assert options["softmax_precision"] == 0
        assert options["layout_q"] == options["layout_kv"] == "TND"
        assert options["mask_mode"] == 1
        assert options["cu_seqlens_q"].dtype == torch.int64
        assert options["cu_seqlens_kv"].dtype == torch.int64
        assert block_shape[0] == 1
        cu_query = options["cu_seqlens_q"].tolist()
        cu_key = options["cu_seqlens_kv"].tolist()
        q_lengths = [stop - start for start, stop in zip(cu_query[:-1], cu_query[1:])]
        kv_lengths = [stop - start for start, stop in zip(cu_key[:-1], cu_key[1:])]
        assert options["max_seqlen_q"] == max(q_lengths)
        assert options["max_seqlen_kv"] == max(kv_lengths)
        return torch.zeros(1024, dtype=torch.int32)

    def forward(query, key, value, indices, counts, block_shape, **options):
        calls["forward"] += 1
        assert options["return_softmax_lse"]
        assert all(tensor.is_contiguous() for tensor in (query, key, value, indices, counts))
        cu_query = options["cu_seqlens_q"].tolist()
        cu_key = options["cu_seqlens_kv"].tolist()
        outputs, lses = [], []
        for q_start, q_stop, k_start, k_stop in zip(
            cu_query[:-1], cu_query[1:], cu_key[:-1], cu_key[1:]
        ):
            mask = _block_mask(
                indices[:, q_start:q_stop], counts[:, q_start:q_stop],
                q_stop - q_start, k_stop - k_start, block_shape[1],
            )
            output, lse = _reference(
                query[q_start:q_stop], key[k_start:k_stop], value[k_start:k_stop],
                mask, options["softmax_scale"],
            )
            outputs.append(output)
            lses.append(lse)
        return torch.cat(outputs, dim=0), torch.cat(lses, dim=0).unsqueeze(-1)

    def backward(query, key, value, dout, output, lse, indices, counts,
                 block_shape, **options):
        calls["backward"] += 1
        assert indices.ndim == 4 and counts.shape == indices.shape[:-1]
        assert options["layout_q"] == options["layout_kv"] == "TND"
        assert options["layout_sparse_pattern"] == 0
        assert options["residual_block_mode"] == 1
        assert options["metadata"] is None
        assert options["softmax_precision"] == 0
        assert int(options["cu_seqlens_q"][-1]) == query.shape[0]
        assert int(options["cu_seqlens_kv"][-1]) == key.shape[0]
        mask = _block_mask(indices[0], counts[0], query.shape[0], key.shape[0], block_shape[1])
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_() for tensor in (query, key, value)]
            expected, expected_lse = _reference(*inputs, mask, options["softmax_scale"])
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(lse, expected_lse.unsqueeze(-1), rtol=0, atol=0)
            return torch.autograd.grad(expected, inputs, dout)

    monkeypatch.setattr(gbsa, "_load_gbsa_ops", lambda: (metadata, forward, backward))
    return calls


@pytest.fixture
def fake_gbsa(monkeypatch):
    return _install_fake_gbsa(monkeypatch)


def _selection(boundaries, storage_length, block_size=4, topk=2):
    ids = torch.full((1, storage_length, topk * block_size + block_size - 1),
                     -1, dtype=torch.int32)
    for start, stop in zip(boundaries, boundaries[1:]):
        for row in range(start, stop):
            complete = (row - start + 1) // block_size
            blocks = list(range(complete))[-topk:][::-1]
            tokens = [start + block * block_size + offset
                      for block in blocks for offset in range(block_size)]
            tokens += list(range(start + complete * block_size, row + 1))
            ids[0, row, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    return ids


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_complete_blocks_exclude_causal_tail(block_size):
    boundaries = [0, 3, 21]
    ids = _selection(boundaries, 24, block_size)
    segment = ids[0, 3:21]
    indices, counts = gbsa._selected_blocks(segment, 3, block_size, 2)
    reconstructed = _block_mask(indices, counts, 18, 18, block_size)
    expected = _token_ids_to_mask(ids, 24)[0, 0, 3:21, 3:21]
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)


def test_full_512_block_budget_still_keeps_three_tail_tokens():
    ids = torch.arange(2051, dtype=torch.int32).unsqueeze(0)
    indices, counts = gbsa._selected_blocks(ids, 0, 4, 2)
    assert indices.shape == (2, 1, 512)
    torch.testing.assert_close(counts, torch.full((2, 1), 512, dtype=torch.int32))
    mask = _block_mask(indices, counts, 1, 2051, 4)
    assert mask.all()


def test_complete_samples_share_forward_but_keep_sample_backwards(fake_gbsa):
    torch.manual_seed(421)
    boundaries = [0, 3, 13, 29]
    ids = _selection(boundaries, 29)
    inputs = [torch.randn(1, 29, heads, 16, dtype=torch.bfloat16, requires_grad=True)
              for heads in (4, 2, 2)]
    output = gbsa.qsa_gbsa_attention(
        *inputs, ids, torch.tensor(boundaries, dtype=torch.int32),
        block_size=4, softmax_scale=0.23,
    )
    assert output.shape == inputs[0].shape
    assert fake_gbsa["metadata"] == fake_gbsa["forward"] == 1
    torch.autograd.grad(output, inputs, torch.randn_like(output))
    assert fake_gbsa["backward"] == len(boundaries) - 1


@pytest.mark.parametrize("offset,length", [(0, 32), (0, 16), (16, 16), (29, 3), (0, 0)])
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_sp_slices_keep_outputs_and_gradients(fake_gbsa, offset, length, outer_checkpoint, dtype):
    torch.manual_seed(420)
    boundaries = [0, 3, 13, 29]
    ids = _selection(boundaries, 32)
    inputs = [torch.randn(1, size, heads, 16, dtype=dtype, requires_grad=True)
              for size, heads in ((length, 4), (32, 2), (32, 2))]
    reference = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    local_ids = ids[:, offset:offset + length]

    def attend(query, key, value):
        return gbsa.qsa_gbsa_attention(
            query, key, value, local_ids, torch.tensor(boundaries, dtype=torch.int32),
            block_size=4, query_offset=offset, softmax_scale=0.23,
        )

    output = checkpoint(attend, *inputs, use_reentrant=False) if outer_checkpoint else attend(*inputs)
    mask = _token_ids_to_mask(local_ids, 32)[0, 0]
    expected, _ = _reference(*(tensor[0] for tensor in reference), mask, 0.23)
    expected = expected.unsqueeze(0)
    torch.testing.assert_close(output, expected, atol=0.008, rtol=0.008)
    probe = torch.randn_like(output)
    grads = torch.autograd.grad(output, inputs, probe)
    expected_grads = torch.autograd.grad(expected, reference, probe)
    for actual, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(actual, expected_grad, atol=0.02, rtol=0.02)
    if offset >= 29 or length == 0:
        assert fake_gbsa["forward"] == fake_gbsa["backward"] == 0
    else:
        assert fake_gbsa["metadata"] == fake_gbsa["forward"] > 0
        assert fake_gbsa["backward"] > 0


def test_qsa_forward_dispatches_to_gbsa_and_preserves_projection_grads(monkeypatch, fake_gbsa):
    monkeypatch.setenv("TORCHTURBO_QSA_BACKEND", "gbsa")
    config = _small_config().text_config
    attention = Qwen4ExpTextAttention(config, layer_idx=3).bfloat16()
    hidden = torch.randn(1, 9, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    phases = torch.randn(1, 9, config.head_dim // 2).repeat(1, 1, 2)
    output, _ = qsa.qsa_forward(
        attention, hidden, (phases.cos().bfloat16(), phases.sin().bfloat16()),
        None, cu_seq_lens_q=torch.tensor([0, 3, 9], dtype=torch.int32),
    )
    output.float().square().sum().backward()
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad)
    assert fake_gbsa["metadata"] == fake_gbsa["forward"] == 1
    assert fake_gbsa["backward"] == 2
    for name, parameter in attention.named_parameters():
        if name.startswith("indexer."):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None, name


@pytest.mark.parametrize("options,match", [
    ({"dropout": 0.1}, "dropout"),
    ({"block_size": 3}, "power of two"),
    ({"query_offset": -1}, "query slice"),
    ({"softmax_scale": 0.0}, "softmax_scale"),
])
def test_unsupported_contracts_fail_before_loading_dependencies(monkeypatch, options, match):
    def unexpected_load():
        raise AssertionError("Dependencies must not load for invalid input")

    monkeypatch.setattr(gbsa, "_load_gbsa_ops", unexpected_load)
    inputs = [torch.ones(1, 8, 2, 16, dtype=torch.bfloat16) for _ in range(3)]
    kwargs = {"block_size": 4, **options}
    with pytest.raises(ValueError, match=match):
        gbsa.qsa_gbsa_attention(*inputs, _selection([0, 8], 8), torch.tensor([0, 8]), **kwargs)


def test_enabled_backend_does_not_fall_back_when_dependencies_are_missing(monkeypatch):
    def missing_dependency():
        raise ImportError("missing GBSA dependency")

    monkeypatch.setattr(gbsa, "_load_gbsa_ops", missing_dependency)
    inputs = [torch.ones(1, 8, 2, 16, dtype=torch.bfloat16) for _ in range(3)]
    with pytest.raises(ImportError, match="missing GBSA dependency"):
        gbsa.qsa_gbsa_attention(*inputs, _selection([0, 8], 8), torch.tensor([0, 8]), block_size=4)


def test_fp32_inputs_cast_to_target_dtype_without_losing_gradients(fake_gbsa):
    inputs = [torch.randn(1, 8, 2, 16, requires_grad=True) for _ in range(3)]
    output = gbsa.qsa_gbsa_attention(
        *inputs, _selection([0, 8], 8), torch.tensor([0, 8]),
        block_size=4, target_dtype=torch.bfloat16,
    )
    assert output.dtype == torch.bfloat16
    gradients = torch.autograd.grad(output.sum(), inputs)
    assert all(gradient.dtype == torch.float32 for gradient in gradients)
    assert fake_gbsa["backward"] == 1


def _run_gbsa_sp(rank, init_file):
    torch.set_num_threads(1)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        group = dist.group.WORLD
        with pytest.MonkeyPatch.context() as patch:
            _install_fake_gbsa(patch)
            for boundaries in ([0, 3, 13, 29], [0, 3, 11], [0, 5, 32]):
                for use_checkpoint in (False, True):
                    torch.manual_seed(756)
                    inputs = [torch.randn(1, 32, heads, 16, dtype=torch.bfloat16,
                                          requires_grad=True) for heads in (4, 2, 2)]
                    probe = torch.randn_like(inputs[0])
                    ids = _selection(boundaries, 32)
                    local = [tensor[:, rank * 16:(rank + 1) * 16].detach().clone().requires_grad_()
                             for tensor in inputs]
                    mask = _token_ids_to_mask(ids, 32)[0, 0]
                    expected, _ = _reference(*(tensor[0] for tensor in inputs), mask, 0.23)
                    expected = expected.unsqueeze(0) * torch.sigmoid(inputs[0])
                    expected.backward(probe)

                    def attend(query, key, value):
                        kv = all_gather_sp_with_grad_reduce(
                            torch.cat((key, value), dim=-1), 1, group, 2,
                        )
                        global_key, global_value = kv.chunk(2, dim=-1)
                        attended = gbsa.qsa_gbsa_attention(
                            query, global_key, global_value, ids[:, rank * 16:(rank + 1) * 16],
                            torch.tensor(boundaries, dtype=torch.int32), block_size=4,
                            query_offset=rank * 16, softmax_scale=0.23,
                        )
                        # Match qsa_forward's post-attention gate: even a padded
                        # rank must replay the KV collective under checkpoint.
                        return attended * torch.sigmoid(query)

                    actual = (checkpoint(attend, *local, use_reentrant=False)
                              if use_checkpoint else attend(*local))
                    torch.testing.assert_close(actual, expected[:, rank * 16:(rank + 1) * 16],
                                               rtol=0.008, atol=0.008)
                    actual.backward(probe[:, rank * 16:(rank + 1) * 16])
                    for tensor, reference in zip(local, inputs):
                        torch.testing.assert_close(tensor.grad, reference.grad[:, rank * 16:(rank + 1) * 16],
                                                   rtol=0.025, atol=0.025)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not (dist.is_available() and dist.is_gloo_available()), reason="requires Gloo")
def test_gbsa_sp_collectives_keep_remote_kv_gradients_and_padding_ranks(tmp_path):
    mp.spawn(_run_gbsa_sp, args=(str(tmp_path / "gbsa-sp"),), nprocs=2, join=True)
