"""Verify compressed-attention teacher targets and indexer auxiliary gradients."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.deepseek_v4 import model as turbo_model
from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler

from .test_packed_inputs import (
    _make_model,
    _packed_inputs,
    _parallelize,
    restore_attention_forwards,
)


@pytest.fixture(autouse=True)
def restore_auxiliary_loss_scale(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "loss_scale", 1.0)


def _loss_inputs():
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(2, 4, 2, 3, generator=generator, requires_grad=True)
    kv = torch.randn(2, 4, 3, generator=generator, requires_grad=True)
    q_indexer = torch.randn(2, 4, 2, 3, generator=generator, requires_grad=True)
    k_indexer = torch.randn(2, 4, 3, generator=generator, requires_grad=True)
    weights = torch.randn(2, 4, 2, generator=generator, requires_grad=True)
    selected = torch.tensor([
        [[-1, -1, -1], [2, 0, -1], [1, 2, -1], [-1, -1, -1]],
        [[0, -1, -1], [0, 2, 1], [3, 1, -1], [-1, -1, -1]],
    ])
    mask = torch.tensor([
        [[1, 1, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
        [[0, 1, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]],
    ], dtype=torch.bool)
    scores = torch.randn(2, 4, 3, generator=generator, requires_grad=True)
    return q, kv, q_indexer, k_indexer, weights, selected, scores, mask


def _reference_kl(q, kv, selected, scores, mask, scale):
    losses = []
    for batch_idx in range(q.size(0)):
        for query_idx in range(q.size(1)):
            valid = selected[batch_idx, query_idx] >= 0
            if not valid.any():
                continue
            teacher_logits = q[batch_idx, query_idx].detach() @ kv[batch_idx].detach().T
            teacher_logits = (teacher_logits * scale).masked_fill(
                mask[batch_idx, query_idx], float("-inf")
            )
            teacher = teacher_logits.softmax(dim=-1).sum(dim=0)
            teacher = teacher[selected[batch_idx, query_idx, valid]]
            teacher = teacher / teacher.sum()
            log_student = scores[batch_idx, query_idx, valid].log_softmax(dim=-1)
            losses.append((teacher * (teacher.log() - log_student)).sum())
    return torch.stack(losses).mean()


def test_selected_teacher_kl_matches_manual_reference_and_detaches_teacher():
    q, kv, qi, ki, weights, selected, scores, mask = _loss_inputs()
    offset = 17
    indices = torch.where(selected >= 0, selected + offset, -1)
    loss = dsv4.LiLoss(2, 0.7, 4)(
        q, kv, qi, ki, weights, indices, scores, mask, offset
    )
    expected = _reference_kl(q, kv, selected, scores, mask, 0.7)
    torch.testing.assert_close(loss, expected)
    grads = torch.autograd.grad(loss, (q, kv, scores), allow_unused=True)
    assert grads[0] is grads[1] is None
    expected_grad = torch.autograd.grad(expected, scores)[0]
    torch.testing.assert_close(grads[2], expected_grad)
    assert not grads[2][selected < 0].any()


def test_invalid_selections_and_appended_padding_do_not_change_loss_or_gradients():
    q, kv, qi, ki, weights, selected, scores, mask = _loss_inputs()
    loss_fn = dsv4.LiLoss(2, 0.7, 4)
    loss = loss_fn(q, kv, qi, ki, weights, selected, scores, mask, 0)
    expected_grad = torch.autograd.grad(loss, scores)[0]

    changed_scores = scores.detach().masked_fill(selected < 0, 1e4).requires_grad_()
    padded_scores = torch.cat([changed_scores, torch.full((2, 3, 3), -1e4)], dim=1)
    padded_q = torch.cat([q.detach(), torch.full((2, 3, 2, 3), 1e4)], dim=1)
    padded_qi = torch.cat([qi.detach(), torch.full((2, 3, 2, 3), 1e4)], dim=1)
    padded_weights = torch.cat([weights.detach(), torch.ones(2, 3, 2)], dim=1)
    padded_selected = torch.cat([selected, torch.full((2, 3, 3), -1)], dim=1)
    padded_mask = torch.cat([mask, torch.ones(2, 3, 4, dtype=torch.bool)], dim=1)
    padded_loss = loss_fn(
        padded_q, kv, padded_qi, ki, padded_weights,
        padded_selected, padded_scores, padded_mask, 0,
    )
    torch.testing.assert_close(padded_loss, loss)
    torch.testing.assert_close(
        torch.autograd.grad(padded_loss, changed_scores)[0], expected_grad
    )


@pytest.mark.parametrize("reconstruct_scores", [False, True])
def test_no_valid_selections_produce_finite_differentiable_zero(reconstruct_scores):
    q, kv, qi, ki, weights, selected, scores, mask = _loss_inputs()
    selected = torch.full_like(selected, -1)
    loss = dsv4.LiLoss(2, 0.7, 4)(
        q, kv, qi, ki, weights, selected,
        None if reconstruct_scores else scores, torch.ones_like(mask), 0,
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0
    parameters = (qi, ki, weights) if reconstruct_scores else (scores,)
    for gradient in torch.autograd.grad(loss, parameters):
        assert torch.isfinite(gradient).all()
        assert not gradient.any()


def test_reconstructed_indexer_logits_match_explicit_values_and_gradients():
    q, kv, qi, ki, weights, selected, _, mask = _loss_inputs()
    full_scores = (torch.einsum("bshd,bnd->bshn", qi, ki).relu() * weights.unsqueeze(-1)).sum(2)
    scores = full_scores.gather(-1, selected.clamp_min(0))
    loss_fn = dsv4.LiLoss(2, 0.7, 4)
    expected = loss_fn(q, kv, qi, ki, weights, selected, scores, mask, 0)
    actual = loss_fn(q, kv, qi, ki, weights, selected, None, mask, 0)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (qi, ki, weights)),
        torch.autograd.grad(expected, (qi, ki, weights)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_missing_mask_uses_causal_compression_visibility():
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(1, 12, 2, 3, generator=generator)
    kv = torch.randn(1, 3, 3, generator=generator)
    scores = torch.randn(1, 12, 3, generator=generator, requires_grad=True)
    mask = torch.arange(3) >= (torch.arange(12).unsqueeze(-1) + 1) // 4
    selected = torch.arange(3).expand(1, 12, 3).masked_fill(mask, -1)
    loss_fn = dsv4.LiLoss(2, 0.7, 4)
    actual = loss_fn(q, kv, None, None, None, selected, scores, None, 0)
    expected = _reference_kl(q, kv, selected, scores, mask.unsqueeze(0), 0.7)
    torch.testing.assert_close(actual, expected)


def _indexer_gradients(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.layers[0].attn.indexer.named_parameters()
        if parameter.grad is not None
    }


def _auxiliary_backward(model, embeddings, positions=None, cu=None):
    logits = model(
        inputs_embeds=embeddings, position_ids=positions, cu_seq_lens_q=cu,
    ).logits
    assert torch.isfinite(logits).all()
    # The attached auxiliary loss must train the indexer without a main loss.
    (logits.sum() * 0).backward()
    gradients = _indexer_gradients(model)
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients.values())
    assert sum(grad.abs().sum() for grad in gradients.values()) > 0
    return gradients


def test_packed_training_indexer_gradients_scale_with_coefficient():
    model = _make_model(4)
    assert model.config.dsa_indexer_loss_coeff == 0.0
    model.train()
    _parallelize(model)
    embeddings, positions, cu = _packed_inputs([9, 13], padding=2)
    model.config.dsa_indexer_loss_coeff = 0.2
    baseline = _auxiliary_backward(model, embeddings, positions, cu)
    model.zero_grad(set_to_none=True)
    model.config.dsa_indexer_loss_coeff = 0.8
    scaled = _auxiliary_backward(model, embeddings.detach(), positions, cu)
    assert baseline.keys() == scaled.keys()
    for name in baseline:
        torch.testing.assert_close(scaled[name], baseline[name] * 4)
    model.zero_grad(set_to_none=True)
    model.config.dsa_indexer_loss_coeff = 0.2
    MoEAuxLossAutoScaler.set_loss_scale(0.25)
    quarter_scale = _auxiliary_backward(model, embeddings.detach(), positions, cu)
    for name in baseline:
        torch.testing.assert_close(quarter_scale[name], baseline[name] * 0.25)


def test_packed_indexer_gradients_match_valid_query_weighted_independent_samples():
    model = _make_model(4)
    model.train()
    _parallelize(model)
    lengths = [9, 13]
    embeddings, positions, cu = _packed_inputs(lengths, padding=2)
    model.config.dsa_indexer_loss_coeff = 0.7
    packed = _auxiliary_backward(model, embeddings, positions, cu)
    model.zero_grad(set_to_none=True)
    valid_queries = [length - 4 + 1 for length in lengths]
    for sample, count in zip(embeddings[:, :sum(lengths)].detach().split(lengths, dim=1), valid_queries):
        model.config.dsa_indexer_loss_coeff = 0.7 * count / sum(valid_queries)
        _auxiliary_backward(model, sample)
    independent = _indexer_gradients(model)
    assert packed.keys() == independent.keys()
    for name in packed:
        torch.testing.assert_close(packed[name], independent[name], atol=1e-6, rtol=2e-4)


def _sp_auxiliary_worker(rank, init_file, lengths, padding):
    torch.set_num_threads(1)
    turbo_model.is_npu_available = lambda: False
    MoEAuxLossAutoScaler.set_loss_scale(1.0)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = _make_model(4)
        model.train()
        _parallelize(model)
        embeddings, positions, cu = _packed_inputs(lengths, padding=padding)
        model.config.dsa_indexer_loss_coeff = 0.7
        reference = _auxiliary_backward(model, embeddings, positions, cu)
        model.zero_grad(set_to_none=True)
        _parallelize(model, sp_size=2, sp_rank=rank, sp_group=dist.group.WORLD)
        local_length = embeddings.size(1) // 2
        local_slice = slice(rank * local_length, (rank + 1) * local_length)
        actual = _auxiliary_backward(
            model, embeddings[:, local_slice].detach(), positions[:, local_slice], cu
        )
        assert actual.keys() == reference.keys()
        for name, gradient in actual.items():
            dist.all_reduce(gradient)
            gradient /= 2
            torch.testing.assert_close(gradient, reference[name], atol=1e-6, rtol=2e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("lengths,padding", [([9, 13], 2), ([3, 9], 0)])
def test_sp_indexer_gradients_match_unsharded_packed_training(tmp_path, lengths, padding):
    mp.spawn(
        _sp_auxiliary_worker,
        args=(str(tmp_path / "indexer_gloo"), lengths, padding),
        nprocs=2,
        join=True,
    )
