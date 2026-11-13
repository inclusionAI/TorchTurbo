"""Verify model-owned packed boundaries reach window and compressed attention."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.deepseek_v4 import model as turbo_model
from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from torchturbo.models.deepseek_v4.model import TurboDeepseekV4ForCausalLM


@pytest.fixture(autouse=True)
def restore_attention_forwards(monkeypatch):
    monkeypatch.setattr(dsv4, "DSAIndexerLossAutoScaler", dsv4.DSAIndexerLossAutoScaler)
    for cls in (
        dsv4.DeepseekV4Compressor,
        dsv4.DeepseekV4Indexer,
        dsv4.GetWindowTopkIdxs,
        dsv4.GetCompressTopkIdxs,
        dsv4.LiCompute,
        dsv4.InnerAttentionOps,
        dsv4.PreAttentionOps,
        dsv4.PostAttentionOps,
        dsv4.SparseAttention,
        dsv4.DeepseekV4Attention,
    ):
        monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(turbo_model, "is_npu_available", lambda: False)


def _make_model(ratio):
    torch.manual_seed(7)
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        qk_nope_head_dim=2,
        v_head_dim=4,
        head_dim=4,
        o_groups=1,
        o_lora_rank=4,
        sliding_window=8,
        compress_ratios=[ratio],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        use_cache=False,
    )
    # Retain PyTorch reference norms while exercising the Turbo parallelism hook.
    model = object.__new__(TurboDeepseekV4ForCausalLM)
    dsv4.DeepseekV4ForCausalLM.__init__(model, config)
    model.eval()
    return model


def _parallelize(model, sp_size=1, sp_rank=0, sp_group=None):
    model.on_parallelized(
        {"sp": {"sp_size": sp_size, "sp_rank": sp_rank, "sp_group": sp_group}}
    )


def _packed_inputs(lengths, padding):
    valid_length = sum(lengths)
    embeddings = torch.randn(1, valid_length + padding, 8, requires_grad=True)
    positions = torch.cat(
        [torch.arange(length) for length in lengths] + [torch.zeros(padding)]
    ).long().unsqueeze(0)
    cu = torch.tensor([0, lengths[0], valid_length], dtype=torch.int32)
    return embeddings, positions, cu


def _independent_logits(model, embeddings, lengths):
    return torch.cat(
        [
            model(inputs_embeds=row).logits
            for row in embeddings[:, : sum(lengths)].split(lengths, dim=1)
        ],
        dim=1,
    )


@pytest.mark.parametrize("ratio", [1, 4, 128])
@pytest.mark.parametrize("padding", [0, 3])
def test_packed_forward_and_input_gradients_match_independent_sequences(ratio, padding):
    model = _make_model(ratio)
    _parallelize(model)
    lengths = [ratio + 1, 2 * ratio + 1]
    embeddings, positions, cu = _packed_inputs(lengths, padding)
    valid_length = sum(lengths)

    packed = model(
        inputs_embeds=embeddings,
        position_ids=positions,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
    ).logits[:, :valid_length]
    independent = _independent_logits(model, embeddings, lengths)
    torch.testing.assert_close(packed, independent, atol=1e-6, rtol=1e-5)

    packed_grad = torch.autograd.grad(packed.square().sum(), embeddings)[0]
    independent_grad = torch.autograd.grad(independent.square().sum(), embeddings)[0]
    torch.testing.assert_close(packed_grad, independent_grad, atol=1e-6, rtol=1e-5)
    assert not packed_grad[:, valid_length:].any()
    assert cu.tolist() == [0, lengths[0], valid_length]


@pytest.mark.parametrize("ratio", [4, 128])
def test_rope_resets_within_segment_keep_compressed_attention_causal(ratio):
    model = _make_model(ratio)
    model.train()
    _parallelize(model)
    embeddings = torch.randn(1, 4 * ratio, 8, requires_grad=True)
    positions = torch.arange(2 * ratio).repeat(2).unsqueeze(0)
    cu = torch.tensor([0, 4 * ratio], dtype=torch.int32)

    logits = model(
        inputs_embeds=embeddings,
        position_ids=positions,
        cu_seq_lens_q=cu,
    ).logits
    prefix_logits = model(
        inputs_embeds=embeddings[:, :ratio],
        position_ids=positions[:, :ratio],
        cu_seq_lens_q=torch.tensor([0, ratio], dtype=torch.int32),
    ).logits
    torch.testing.assert_close(logits[:, :ratio], prefix_logits, atol=1e-6, rtol=1e-5)

    # The supervised target at this position is still in the first document;
    # the later document with reset RoPE coordinates must not influence it.
    grad = torch.autograd.grad(logits[0, ratio - 1].square().sum(), embeddings)[0]
    assert grad[:, :ratio].abs().sum() > 0
    assert not grad[:, ratio:].any()


@pytest.mark.parametrize("ratio", [1, 4])
def test_unpacked_rows_without_cu_match_independent_sequences(ratio):
    """DPO-style batches carry neither cu nor mask: each row is causal."""
    model = _make_model(ratio)
    _parallelize(model)
    lengths = [5, 9]
    # Right padding: row 0 holds 5 valid tokens followed by a padded tail.
    embeddings = torch.randn(2, max(lengths), 8)
    logits = model(inputs_embeds=embeddings).logits
    for row, length in enumerate(lengths):
        expected = model(inputs_embeds=embeddings[row : row + 1, :length]).logits
        torch.testing.assert_close(
            logits[row : row + 1, :length], expected, atol=1e-6, rtol=1e-5
        )


def _sp_worker(rank, init_file):
    turbo_model.is_npu_available = lambda: False
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        model = _make_model(4)
        _parallelize(model)
        lengths = [5, 9]
        embeddings, positions, cu = _packed_inputs(lengths, padding=2)
        independent = _independent_logits(model, embeddings, lengths)
        reference_grad = torch.autograd.grad(
            independent.square().sum(), embeddings
        )[0]

        _parallelize(model, sp_size=2, sp_rank=rank, sp_group=dist.group.WORLD)
        start = rank * 8
        local = embeddings[:, start : start + 8].detach().clone().requires_grad_()
        captured = []
        hook = model.layers[0].attn.register_forward_pre_hook(
            lambda _module, _args, kwargs: captured.append(kwargs),
            with_kwargs=True,
        )
        packed = model(
            inputs_embeds=local,
            position_ids=positions[:, start : start + 8],
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=9,
            max_length_k=9,
        ).logits
        hook.remove()

        # No mask is rebuilt: the global cu boundaries reach attention as-is.
        assert captured[0]["attention_mask"] is None
        assert captured[0]["cu_seq_lens"] is cu
        valid_length = min(8, sum(lengths) - start)
        torch.testing.assert_close(
            packed[:, :valid_length],
            independent[:, start : start + valid_length],
            atol=1e-6,
            rtol=1e-5,
        )
        packed[:, :valid_length].square().sum().backward()
        torch.testing.assert_close(
            local.grad, reference_grad[:, start : start + 8], atol=1e-6, rtol=1e-5
        )

        for ratio in (4, 128):
            model = _make_model(ratio)
            model.train()
            _parallelize(model)
            embeddings = torch.randn(1, 4 * ratio, 8, requires_grad=True)
            positions = torch.arange(2 * ratio).repeat(2).unsqueeze(0)
            cu = torch.tensor([0, 4 * ratio], dtype=torch.int32)
            reference = model(
                inputs_embeds=embeddings,
                position_ids=positions,
                cu_seq_lens_q=cu,
            ).logits
            reference_grad = torch.autograd.grad(
                reference[0, ratio - 1].square().sum(), embeddings
            )[0]

            _parallelize(model, sp_size=2, sp_rank=rank, sp_group=dist.group.WORLD)
            local_slice = slice(rank * 2 * ratio, (rank + 1) * 2 * ratio)
            local = embeddings[:, local_slice].detach().clone().requires_grad_()
            packed = model(
                inputs_embeds=local,
                position_ids=positions[:, local_slice],
                cu_seq_lens_q=cu,
            ).logits
            torch.testing.assert_close(
                packed, reference[:, local_slice], atol=1e-6, rtol=1e-5
            )
            # Every rank joins backward; only a query on rank zero contributes
            # loss, so future tokens on both shards must receive zero gradient.
            loss = packed[0, ratio - 1].square().sum() if rank == 0 else packed.sum() * 0
            loss.backward()
            torch.testing.assert_close(
                local.grad, reference_grad[:, local_slice], atol=1e-6, rtol=1e-5
            )
            future_grad = local.grad[:, ratio:] if rank == 0 else local.grad
            assert not future_grad.any()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_sp_packed_forward_and_input_gradients_match_independent_sequences(tmp_path):
    mp.spawn(_sp_worker, args=(str(tmp_path / "gloo_init"),), nprocs=2, join=True)
