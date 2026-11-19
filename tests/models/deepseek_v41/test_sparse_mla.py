"""Runtime patch and numerical tests for the A3 MLA pair."""

import os

import pytest
import torch

from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import (
    DeepseekV41TextConfig,
)
from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41InnerAttentionOps,
    get_window_topk_idxs,
)
from torchturbo.models.deepseek_v41.npu import (
    packed_window_matches,
    patch_deepseek_v41_npu_sfa,
)
from torchturbo.models.deepseek_v41.packed_metadata import (
    checked_block_info as _checked_block_info,
)


def window_indices_match(indices: torch.Tensor, batch: int, seq: int) -> bool:
    """Check the unpadded single-sequence special case."""
    mask = torch.ones((batch, seq), dtype=torch.long, device=indices.device)
    return packed_window_matches(indices, mask)


def make_module(backend: str) -> DeepseekV41InnerAttentionOps:
    module = DeepseekV41InnerAttentionOps(
        DeepseekV41TextConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=512,
        ),
        0,
    )

    if backend == "cann":
        patch_deepseek_v41_npu_sfa(module)
    return module


@pytest.mark.parametrize("seq", [1, 16, 128, 137])
def test_window_contract(seq: int) -> None:
    mask = torch.ones(2, seq, dtype=torch.long)
    indices = get_window_topk_idxs(128, 2, seq, mask.device, mask)
    assert window_indices_match(indices, 2, seq)
    indices = indices.clone()
    indices[0, -1, -1] = -1
    assert not window_indices_match(indices, 2, seq)


def test_packed_and_padding_are_not_dense_windows() -> None:
    for mask in [torch.tensor([[1, 1, 2, 2]]), torch.tensor([[1, 1, 0, 0]])]:
        indices = get_window_topk_idxs(128, 1, 4, mask.device, mask)
        assert not window_indices_match(indices, 1, 4)


def test_strict_backend_rejects_cpu() -> None:
    q, kv, sinks = torch.zeros(1, 4, 2, 512), torch.zeros(1, 4, 512), torch.zeros(2)
    indices = get_window_topk_idxs(128, 1, 4, q.device, torch.ones(1, 4))
    with pytest.raises(ValueError, match="NPU"):
        make_module("cann")(q, kv, indices, None, None, sinks)


@pytest.mark.skipif(
    os.environ.get("RUN_CANN_MLA_TESTS") != "1",
    reason="set RUN_CANN_MLA_TESTS=1 on an A3 with the custom package",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch, seq, checkpointed", [(2, 16, False), (1, 137, True)])
def test_npu_forward_backward(
    dtype: torch.dtype, batch: int, seq: int, checkpointed: bool
) -> None:
    import torch_npu  # noqa: F401 -- Register the NPU backend.

    torch.npu.set_device(0)
    torch.manual_seed(123)
    q = (torch.randn(batch, seq, 2, 512) * 0.1).to("npu", dtype).requires_grad_()
    kv = (torch.randn(batch, seq, 512) * 0.1).to("npu", dtype).requires_grad_()
    sinks = (torch.randn(2) * 0.2).to("npu").requires_grad_()
    mask = torch.ones(batch, seq, device="npu", dtype=torch.long)
    win = get_window_topk_idxs(128, batch, seq, q.device, mask)
    inputs = (q, kv, win, None, None, sinks)
    if checkpointed:
        from torch.utils.checkpoint import checkpoint

        actual = checkpoint(make_module("cann"), *inputs, use_reentrant=False)
    else:
        actual = make_module("cann")(*inputs)
    expected = make_module("eager")(*inputs)
    dy = torch.randn_like(actual) * 0.1
    ga = torch.autograd.grad(actual, (q, kv, sinks), dy, retain_graph=True)
    ge = torch.autograd.grad(expected, (q, kv, sinks), dy)
    torch.npu.synchronize()
    tol = 0.025 if dtype == torch.bfloat16 else 0.005
    for a, e in zip((actual, *ga), (expected, *ge)):
        assert torch.isfinite(a).all().item()
        torch.testing.assert_close(a.float(), e.float(), rtol=tol, atol=tol * 0.1)


def test_packed_compression_offsets() -> None:
    mask = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2, 0]])
    spans = _checked_block_info(mask[0], 2)
    assert [
        (start, end, offset, offset + count)
        for start, end, offset, count in spans.values()
    ] == [
        (0, 3, 0, 1),
        (3, 8, 1, 3),
    ]
    win = get_window_topk_idxs(128, 1, 9, mask.device, mask)
    assert packed_window_matches(win, mask)


def test_repeated_segment_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _checked_block_info(torch.tensor([1, 2, 1]), 1)


@pytest.mark.skipif(
    os.environ.get("RUN_CANN_MLA_TESTS") != "1",
    reason="requires A3 and the corrected custom package",
)
@pytest.mark.parametrize("ratio", [0, 1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_csa_shared_gradients(ratio: int, dtype: torch.dtype, reference_npu_ops) -> None:
    import torch_npu  # noqa: F401 -- Register the NPU backend.
    from torch.utils.checkpoint import checkpoint

    from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41Compressor,
    )

    torch.npu.set_device(0)
    torch.manual_seed(77)
    config = DeepseekV41TextConfig(
        hidden_size=32,
        head_dim=512,
        num_attention_heads=2,
        num_hidden_layers=1,
        compress_ratios=[ratio],
    )
    fused = DeepseekV41InnerAttentionOps(config, 0)
    patch_deepseek_v41_npu_sfa(fused)
    eager = DeepseekV41InnerAttentionOps(config, 0)
    mask = torch.tensor(
        [[1] * 9 + [2] * 7 + [0] * 3, [1] * 3 + [2] * 13 + [0] * 3],
        device="npu",
        dtype=torch.long,
    )
    positions = torch.tensor(
        [
            list(range(9)) + list(range(7)) + [0] * 3,
            list(range(3)) + list(range(13)) + [0] * 3,
        ],
        device="npu",
        dtype=torch.long,
    )
    q = (torch.randn(2, 19, 2, 512) * 0.1).to("npu", dtype).requires_grad_()
    kv = (torch.randn(2, 19, 512) * 0.1).to("npu", dtype).requires_grad_()
    sinks = torch.randn(2, device="npu", requires_grad=True)
    win = get_window_topk_idxs(128, 2, 19, mask.device, mask)
    compressed = selected = None
    variables = [q, kv, sinks]
    if ratio:
        compressor = DeepseekV41Compressor(config, ratio).to("npu")
        if ratio == 1:
            compressor = compressor.to(dtype)
        x = (torch.randn(2, 19, 32) * 0.1).to("npu", dtype).requires_grad_()
        compressed = compressor(x, mask, positions)
        selected = torch.full((2, 19, 5), -1, device="npu", dtype=torch.long)
        for batch, row in enumerate(mask):
            for start, end, offset, _ in _checked_block_info(row, ratio).values():
                for t in range(end - start):
                    # Sparse choices, not merely all visible compressed tokens.
                    candidates = list(range(offset, offset + (t + 1) // ratio, 2))[:5]
                    if candidates:
                        selected[batch, start + t, : len(candidates)] = torch.tensor(
                            candidates, device="npu"
                        )
        variables += [compressed, x, compressor.wkv.weight]
        if ratio > 1:
            variables += [compressor.wgate.weight]
    args = (q, kv, win, compressed, selected, sinks, mask, positions)
    actual = checkpoint(fused, *args, use_reentrant=False)
    expected = eager(*args)
    if ratio:
        # A second consumer reuses the same compressor graph and new indices.
        reindexed = selected.flip(-1).contiguous()
        actual = actual + fused(
            q * 0.7, kv * 0.8, win, compressed, reindexed, sinks, mask, positions
        )
        expected = expected + eager(
            q * 0.7, kv * 0.8, win, compressed, reindexed, sinks, mask, positions
        )
    dy = torch.randn_like(actual) * 0.1
    actual_grads = torch.autograd.grad(actual, variables, dy, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, variables, dy)
    torch.npu.synchronize()
    tol = 0.03 if dtype == torch.bfloat16 else 0.006
    for name, a, e in zip(
        ["out", "q", "kv", "sinks", "compressed", "compressor_input", "wkv", "wgate"],
        [actual, *actual_grads],
        [expected, *expected_grads],
    ):
        assert torch.isfinite(a).all().item(), name
        if name in ("compressor_input", "wkv", "wgate"):
            # RMSNorm amplifies low-precision differences at near-zero entries.
            # Bound whole-gradient error and peak error without dividing by
            # individual reference entries close to zero.
            delta = a.float() - e.float()
            rel_l2 = delta.norm() / e.float().norm().clamp_min(1e-8)
            rel_peak = delta.abs().max() / e.float().abs().max().clamp_min(1e-8)
            limit = 0.04 if dtype == torch.bfloat16 else 0.008
            print(name, float(rel_l2), float(rel_peak))
            assert rel_l2 < limit and rel_peak < 2 * limit, name
        else:
            torch.testing.assert_close(
                a.float(),
                e.float(),
                rtol=tol,
                atol=tol * 0.1,
                msg=lambda msg, name=name: name + ": " + msg,
            )


@pytest.mark.skipif(
    os.environ.get("RUN_CANN_MLA_TESTS") != "1",
    reason="requires A3 and the corrected custom package",
)
def test_four_layer_model_full_reindex_reuse() -> None:
    import copy

    import torch_npu  # noqa: F401 -- Register the NPU backend.

    from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import (
        DeepseekV41Config,
    )
    from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41ForCausalLM,
    )

    torch.npu.set_device(0)
    torch.manual_seed(345)
    text = DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        compress_ratios=[0, 1, 1, 1],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1, 2],
        candidate_source_layer_id=1,
        index_n_heads=2,
        index_head_dim=64,
        index_topk=512,
        candidate_topk_blocks=32,
        candidate_block_size=2,
        hc_mult=2,
        engram_layer_ids=[],
        use_cache=False,
        rope_scaling={"original_max_position_embeddings": 4096, "factor": 1.0},
    )
    reference = DeepseekV41ForCausalLM(
        DeepseekV41Config(
            text_config=text.to_dict(), vision_config={"num_hidden_layers": 0}
        )
    ).to(device="npu", dtype=torch.bfloat16)
    for name, param in reference.named_parameters():
        if "attn_sink" in name or "hc_" in name:
            param.data = param.data.float()
    actual = copy.deepcopy(reference)
    patch_deepseek_v41_npu_sfa(actual)
    actual.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    reference.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    reference.train()
    actual.train()
    ids = torch.randint(3, 32, (1, 12), device="npu")
    mask = torch.tensor([[1] * 7 + [2] * 5], device="npu")
    pos = torch.tensor([list(range(7)) + list(range(5))], device="npu")
    expected = reference(input_ids=ids, attention_mask=mask, position_ids=pos).logits
    output = actual(input_ids=ids, attention_mask=mask, position_ids=pos).logits
    torch.testing.assert_close(output.float(), expected.float(), rtol=0.03, atol=0.003)
    loss_weights = torch.randn_like(output)
    (output * loss_weights).sum().backward()
    (expected * loss_weights).sum().backward()
    for name in [
        "layers.0.attn.wkv.weight",
        "layers.1.attn.compressor.wkv.weight",
        "layers.1.attn.attn_sink",
        "layers.2.attn.wq_b.weight",
        "layers.3.attn.wq_b.weight",
        "layers.3.attn.attn_sink",
    ]:
        a = actual.get_parameter(name).grad.float()
        e = reference.get_parameter(name).grad.float()
        assert torch.isfinite(a).all().item(), name
        assert e.norm() > 0, name
        error = (a - e).norm() / e.norm()
        assert error < 0.05, (name, error.item())


@pytest.mark.skipif(
    os.environ.get("RUN_CANN_MLA_TESTS") != "1",
    reason="requires A3 and the corrected custom package",
)
@pytest.mark.parametrize("padding", [False, True])
def test_csa_64_heads_and_empty_rows(padding: bool) -> None:
    import torch_npu  # noqa: F401 -- Register the NPU backend.

    torch.npu.set_device(0)
    torch.manual_seed(78)
    seq, heads, topk = 137, 64, 16
    config = DeepseekV41TextConfig(
        num_hidden_layers=1,
        num_attention_heads=heads,
        head_dim=512,
        compress_ratios=[1],
    )
    fused = DeepseekV41InnerAttentionOps(config, 0)
    patch_deepseek_v41_npu_sfa(fused)
    eager = DeepseekV41InnerAttentionOps(config, 0)
    q = (
        (torch.randn(1, seq, heads, 512) * 0.1)
        .to("npu", torch.bfloat16)
        .requires_grad_()
    )
    kv = (torch.randn(1, seq, 512) * 0.1).to("npu", torch.bfloat16).requires_grad_()
    sinks = torch.randn(heads, device="npu", requires_grad=True)
    mask = torch.full((1, seq), 0 if padding else 1, device="npu", dtype=torch.long)
    pos = torch.arange(seq, device="npu")[None]
    win = get_window_topk_idxs(128, 1, seq, q.device, mask)
    if padding:
        cmp = None
        selected = None
        tensors = (q, kv, sinks)
    else:
        cmp = (
            (torch.randn(1, seq, 512) * 0.1).to("npu", torch.bfloat16).requires_grad_()
        )
        selected = torch.full((1, seq, topk), -1, device="npu", dtype=torch.long)
        for t in range(seq):
            count = min(t + 1, topk)
            selected[0, t, :count] = torch.arange(t + 1 - count, t + 1, device="npu")
        tensors = (q, kv, sinks, cmp)
    args = (q, kv, win, cmp, selected, sinks, mask, pos)
    actual, expected = fused(*args), eager(*args)
    dy = torch.randn_like(actual) * 0.1
    ga = torch.autograd.grad(actual, tensors, dy, retain_graph=True)
    ge = torch.autograd.grad(expected, tensors, dy)
    for a, e in zip((actual, *ga), (expected, *ge)):
        torch.testing.assert_close(a.float(), e.float(), rtol=0.03, atol=0.003)


@pytest.mark.parametrize("ratio", [0, 2])
def test_packed_samples_share_one_tnd_call(
    monkeypatch: pytest.MonkeyPatch, ratio: int
) -> None:
    from types import SimpleNamespace

    from sparse_mla_test_utils import make_compressed_indices

    from torchturbo.models.deepseek_v41 import npu

    q = torch.randn(1, 12, 2, 512)
    kv = torch.randn(1, 12, 512)
    mask = torch.tensor([[1] * 4 + [2] * 8])
    compressed = torch.randn(1, 6, 512) if ratio else None
    selected = None
    if ratio:
        parts = []
        offset = 0
        for length in (4, 8):
            values = make_compressed_indices(length, ratio, 2, torch.device("cpu"))[
                :, 0
            ]
            parts.append(torch.where(values >= 0, values + offset, values))
            offset += length // ratio
        selected = torch.cat(parts).unsqueeze(0)
    calls = []

    def fake_fused(query, original, compressed_kv, indices, residual, *args, **kwargs):
        calls.append(kwargs)
        return query

    # npu.ops is the lazy dispatcher proxy (raises while unconfigured), so the
    # fake replaces the module-level name instead of a proxy attribute.
    monkeypatch.setattr(npu, "ops", SimpleNamespace(sparse_flash_mla=fake_fused))
    result = npu.run_cann_attention(
        q,
        kv,
        compressed,
        selected,
        torch.zeros(2),
        512**-0.5,
        max(ratio, 1),
        mask,
    )
    torch.testing.assert_close(result, q, rtol=0, atol=0)
    assert len(calls) == 1
    assert calls[0]["cu_seq_lens_q"].tolist() == [0, 4, 12]
    assert calls[0]["cu_seq_lens_ori_kv"].tolist() == [0, 4, 12]
    assert calls[0]["max_seqlens"] == (8, 8, 4 if ratio else 0)
    if ratio:
        assert calls[0]["cu_seq_lens_cmp_kv"].tolist() == [0, 2, 6]


def test_runtime_patch_is_instance_local_and_idempotent() -> None:
    fused = make_module("cann")
    reference = make_module("eager")
    original = DeepseekV41InnerAttentionOps.forward
    assert fused.forward.__func__ is not original
    assert reference.forward.__func__ is original
    assert DeepseekV41InnerAttentionOps.forward is original
    patch_deepseek_v41_npu_sfa(fused)
    installed = fused.forward.__func__
    assert patch_deepseek_v41_npu_sfa(fused) == 1
    assert fused.forward.__func__ is installed


@pytest.mark.parametrize("legacy_backend", [None, "eager", "cann"])
def test_parallelized_hook_always_installs_fused(monkeypatch, legacy_backend) -> None:

    from torchturbo import runtime
    from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM
    from torchturbo.models.deepseek_v41.npu import _npu_sfa

    model = torch.nn.Module()
    model.attention = make_module("eager")
    registered = []
    model._register_publish_shared_for_fsdp = lambda: registered.append(True)
    from torchturbo.models.deepseek_v41 import npu
    monkeypatch.setattr(npu, "patch_deepseek_v41_npu_sfa_li", lambda model: None)
    model._freeze_indexers = lambda: None
    monkeypatch.setattr(runtime, "is_npu_available", lambda: True)
    if legacy_backend is None:
        monkeypatch.delenv("SPARSE_MLA_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SPARSE_MLA_BACKEND", legacy_backend)
    TurboDeepseekV41ForCausalLM.on_parallelized(model, {"sp": {"sp_size": 1, "sp_rank": 0, "sp_group": None}})
    assert registered == [True]
    assert model.attention.forward.__func__ is _npu_sfa


@pytest.mark.parametrize("missing", [None, "li", "sli", "kl"])
def test_indexer_bundle_preserves_mla_and_reference(monkeypatch, missing) -> None:
    import copy
    import importlib

    available = missing is None

    from torchturbo import runtime
    from torchturbo.models.deepseek_v41 import npu
    from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM
    from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41Attention,
    )

    config = DeepseekV41TextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=512,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        compress_ratios=[1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        index_n_heads=2,
        index_head_dim=64,
        indexer_kl_loss_coeff=0.1,
    )
    model = torch.nn.Module()
    model.config = config
    model.attention = DeepseekV41Attention(config, 0)
    reference = copy.deepcopy(model)
    model._register_publish_shared_for_fsdp = lambda: None
    freeze_calls = []
    model._freeze_indexers = lambda: freeze_calls.append(True)
    monkeypatch.setattr(runtime, "is_npu_available", lambda: True)
    for key, module_name, probe in (
        ("li", "lightning_indexer_v2", "has_lightning_indexer_v2"),
        ("sli", "sparse_lightning_indexer", "has_sparse_lightning_indexer"),
        ("kl", "sparse_lightning_indexer_kl_loss_grad", "has_sli_kl_loss_grad"),
    ):
        provider = importlib.import_module("torchturbo.ops.providers.npu." + module_name)
        monkeypatch.setattr(provider, probe, lambda key=key: key != missing)
    # A missing sparse kernel degrades the block selection to the reference
    # path instead of failing the bundle; the source op and KL still raise.
    raises = missing in ("li", "kl")
    if raises:
        with pytest.raises(RuntimeError, match="NPU training requires"):
            TurboDeepseekV41ForCausalLM.on_parallelized(model, {"sp": {"sp_size": 1, "sp_rank": 0, "sp_group": None}})
    else:
        TurboDeepseekV41ForCausalLM.on_parallelized(model, {"sp": {"sp_size": 1, "sp_rank": 0, "sp_group": None}})
    assert model.attention.inner_attention_ops.forward.__func__ is npu._npu_sfa
    if not raises:
        assert model._npu_sfa_li_patched is True
    assert freeze_calls == ([] if raises else [True])
    if not raises:
        assert model.attention._prepare_indexer_loss.__func__ is npu._prepare_indexer_loss
    if available:
        assert model.attention.indexer._select_blocks.__func__ is npu._npu_indexer_select
    else:
        assert (
            model.attention.indexer._select_blocks.__func__
            is type(model.attention.indexer)._select_blocks
        )
    assert reference.attention.forward.__func__ is DeepseekV41Attention.forward
    assert reference.attention.inner_attention_ops.forward.__func__ is (
        DeepseekV41InnerAttentionOps.forward
    )
    assert (
        reference.attention.indexer.score.__func__
        is type(reference.attention.indexer).score
    )


def test_kl_target_keeps_joint_normalization_and_detaches() -> None:
    from torchturbo.models.deepseek_v41.npu import _IndexerLossContext

    module = make_module("eager")
    q = torch.zeros(1, 2, 2, 512, requires_grad=True)
    kv = torch.zeros(1, 2, 512, requires_grad=True)
    compressed = torch.zeros(1, 2, 512, requires_grad=True)
    sink = torch.zeros(2, requires_grad=True)
    window = torch.tensor([[[0, -1], [0, 1]]])
    selected = torch.tensor([[[0, -1], [0, 1]]])
    target = _IndexerLossContext.target_probs(module, q, kv, window, compressed, selected, sink)
    # Zero scores give equal mass to valid window keys, compressed keys and sink.
    torch.testing.assert_close(target, torch.tensor([[[1 / 3, 0], [1 / 5, 1 / 5]]]))
    assert not target.requires_grad


@pytest.mark.skipif(
    os.environ.get("RUN_CANN_MLA_TESTS") != "1",
    reason="requires A3 and the custom MLA package",
)
@pytest.mark.parametrize("coefficient", [0.0, 0.1])
def test_fused_attention_with_optional_kl(monkeypatch, coefficient, unit_kl_scale) -> None:
    """Exercise real MLA forward/backward with a controlled auxiliary KL loss."""
    import copy

    import torch_npu  # noqa: F401 -- Register the NPU backend.

    from torchturbo.models.deepseek_v41 import npu
    from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41Attention,
    )

    torch.npu.set_device(0)
    torch.manual_seed(91)
    config = DeepseekV41TextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        compress_ratios=[1],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        index_n_heads=2,
        index_head_dim=64,
        indexer_kl_loss_coeff=coefficient,
        rope_scaling={"original_max_position_embeddings": 4096, "factor": 1.0},
    )
    plain = DeepseekV41Attention(config, 0).to("npu", torch.bfloat16)
    plain.attn_sink.data = torch.zeros(2, device="npu", dtype=torch.float32)
    patched = copy.deepcopy(plain)
    npu.patch_deepseek_v41_npu_sfa(plain)
    npu.patch_deepseek_v41_npu_sfa(patched)
    npu.patch_deepseek_v41_npu(enable_li=True, enable_indexer_kl=True, model=patched)
    x = torch.randn(1, 8, 16, device="npu", dtype=torch.bfloat16).requires_grad_()
    compressed = torch.randn(
        1, 8, 512, device="npu", dtype=torch.bfloat16
    ).requires_grad_()
    index_k = torch.randn(1, 8, 64, device="npu").requires_grad_()
    mask = torch.tensor([[1] * 3 + [2] * 5], device="npu")
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]], device="npu")
    selected = torch.tensor([[[0], [1], [2], [3], [4], [5], [6], [7]]], device="npu")
    calls = []

    def controlled_kl(attn, hidden, keys, indices, target, inv, actual_mask, pos, *, packed=None):
        assert not target.requires_grad
        assert actual_mask is mask and pos is positions
        calls.append(True)
        return keys.square().sum()

    monkeypatch.setattr(npu, "_indexer_kl_loss", controlled_kl)
    if coefficient == 0:

        def unexpected_target(*args):
            pytest.fail("KL-disabled attention must not recompute target probabilities")

        monkeypatch.setattr(npu._IndexerLossContext, "target_probs", staticmethod(unexpected_target))
    args = (x, compressed, index_k, selected, None, mask, positions)
    expected, _ = plain(*args)
    actual, _ = patched(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(
        expected, (x, compressed, plain.attn_sink), weights, retain_graph=True
    )
    actual_grads = torch.autograd.grad(
        actual,
        (x, compressed, patched.attn_sink, index_k),
        weights,
        allow_unused=True,
    )
    for actual_grad, expected_grad in zip(actual_grads[:3], expected_grads):
        # The FP32 sink reduction can differ by roundoff between kernel launches.
        tolerance = 1e-6 if actual_grad.dtype == torch.float32 else 0
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=tolerance, atol=tolerance
        )
    assert calls == ([True] if coefficient else [])
    if coefficient:
        torch.testing.assert_close(actual_grads[3], 2 * coefficient * index_k)
    else:
        assert actual_grads[3] is None
