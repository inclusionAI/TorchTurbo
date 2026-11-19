"""Initialization, adapter ordering and eager/fused interface contracts."""

import copy
import inspect

import pytest
import torch

from torchturbo.models.deepseek_v41 import npu
from torchturbo.models.deepseek_v41 import modeling_deepseek_v41 as modeling
from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig


def config():
    return DeepseekV41TextConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        head_dim=8, qk_rope_head_dim=4, q_lora_rank=8, o_lora_rank=8,
        o_groups=1, compress_ratios=[1], kv_source_layer_ids=[0],
        index_source_layer_ids=[0], index_n_heads=2, index_head_dim=8,
        indexer_kl_loss_coeff=0.1,
        rope_scaling={"original_max_position_embeddings": 4096, "factor": 1.0},
    )


@pytest.mark.parametrize("patcher", [npu.patch_deepseek_v41_npu_sfa, npu.patch_deepseek_v41_npu_engram])
@pytest.mark.parametrize("size,rank", [(0, 0), (1, -1), (1, 1), (2, 0)])
def test_invalid_sp_context_fails_before_mutation(patcher, size, rank):
    model = modeling.DeepseekV41Attention(config(), 0)
    original = model.inner_attention_ops.forward.__func__
    with pytest.raises(ValueError, match="SP"):
        patcher(model, sp_size=size, sp_rank=rank)
    assert model.inner_attention_ops.forward.__func__ is original
    assert model.indexer._sp_size == 1


@pytest.mark.parametrize("li_first", [False, True])
def test_adapter_order_preserves_parallel_context_and_state_dict(li_first):
    model = modeling.DeepseekV41Attention(config(), 0)
    keys = tuple(model.state_dict())
    reference = copy.deepcopy(model)
    group = object()
    def install_sfa():
        npu.patch_deepseek_v41_npu_sfa(model, sp_group=group, sp_size=2, sp_rank=1)

    def install_li():
        npu.patch_deepseek_v41_npu(enable_li=True, enable_indexer_kl=True, model=model)

    for install in ([install_li, install_sfa] if li_first else [install_sfa, install_li]):
        install()
    for module in [model.pre_attention_ops, model.inner_attention_ops, model.compressor, model.indexer]:
        assert module._sp_group is group and module._sp_size == 2 and module._sp_rank == 1
    assert model._prepare_indexer_loss.__func__ is npu._prepare_indexer_loss
    assert model.inner_attention_ops.forward.__func__ is npu._npu_sfa
    assert model.indexer._select_blocks.__func__ is npu._npu_indexer_select
    assert reference.indexer._sp_size == 1 and reference.indexer._sp_group is None
    assert tuple(model.state_dict()) == keys


def test_inner_attention_accepts_the_same_arguments():
    eager = inspect.signature(modeling.DeepseekV41InnerAttentionOps.forward).parameters
    fused = inspect.signature(npu._npu_sfa).parameters
    assert list(eager) == list(fused)
    assert [(p.kind, p.default) for p in eager.values()] == [(p.kind, p.default) for p in fused.values()]


@pytest.mark.parametrize("source,expected", [(-1, None), (0, 2)])
def test_indexer_derives_candidate_ratio_without_a_patch(source, expected):
    cfg = config()
    cfg.candidate_source_layer_id = source
    cfg.compress_ratios = [2]
    indexer = modeling.DeepseekV41Indexer(cfg, 2, owns_k=True, is_candidate_source=False)
    assert indexer.candidate_source_ratio == expected
    assert indexer._sp_size == 1 and indexer._sp_rank == 0 and indexer._sp_group is None


@pytest.mark.parametrize("coefficient", [0.0, 0.1])
def test_kl_adapter_can_use_eager_inner_attention(monkeypatch, unit_kl_scale, coefficient):
    cfg = config()
    cfg.indexer_kl_loss_coeff = coefficient
    plain = modeling.DeepseekV41Attention(cfg, 0)
    plain.attn_sink.data.zero_()
    patched = copy.deepcopy(plain)
    npu.patch_deepseek_v41_npu(enable_li=False, enable_indexer_kl=True, model=patched)
    x = torch.randn(1, 8, 16, requires_grad=True)
    compressed = torch.randn(1, 8, 8, requires_grad=True)
    index_k = torch.randn(1, 8, 8, requires_grad=True)
    mask = torch.tensor([[1] * 3 + [2] * 5])
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
    selected = torch.arange(8).view(1, 8, 1)
    calls = []

    def controlled_kl(attn, hidden, keys, indices, target, inv, actual_mask, pos, *, packed):
        assert packed.local_start == 0 and not target.requires_grad
        calls.append(True)
        return keys.square().sum()

    monkeypatch.setattr(npu, "_indexer_kl_loss", controlled_kl)
    args = (x, compressed, index_k, selected, None, mask, positions)
    expected, _ = plain(*args)
    actual, _ = patched(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.sum(), (x, compressed, index_k), allow_unused=True)
    reference = torch.autograd.grad(expected.sum(), (x, compressed))
    for a, e in zip(gradients[:2], reference):
        torch.testing.assert_close(a, e)
    assert calls == ([True] if coefficient else [])
    if coefficient:
        torch.testing.assert_close(gradients[2], 2 * coefficient * index_k)
    else:
        assert gradients[2] is None


@pytest.mark.parametrize("fused", [False, True])
def test_indexer_shares_projection_before_backend_selection(monkeypatch, fused):
    cfg = config()
    indexer = modeling.DeepseekV41Indexer(cfg, 1, owns_k=True, is_candidate_source=False)
    if fused:
        npu.patch_deepseek_v41_npu(enable_li=True, enable_indexer_kl=False, model=indexer)
    assert indexer.score.__func__ is modeling.DeepseekV41Indexer.score
    expected_backend = npu._npu_indexer_select if fused else modeling.DeepseekV41Indexer._select_blocks
    assert indexer._select_blocks.__func__ is expected_backend
    x = torch.randn(2, 6, 16, requires_grad=True)
    qr = torch.randn(2, 6, 8, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2]]).expand(2, -1)
    inv_freq = torch.tensor([1.0, 0.01])
    mask = torch.tensor([[1, 1, 1, 2, 2, 2]]).expand(2, -1)
    keys = torch.randn(2, 6, 8)
    selected = torch.zeros(2, 6, 1, dtype=torch.long)
    calls = []

    def capture(q, weights, index_k, candidates, attention_mask, position_ids):
        assert index_k is keys and candidates is None
        assert attention_mask is mask and position_ids is positions
        calls.append((q, weights))
        return selected, None, None

    monkeypatch.setattr(indexer, "_select_blocks", capture)
    actual, _, _ = indexer.score(x, qr, keys, inv_freq, None, mask, positions)
    assert actual is selected and len(calls) == 1
    q, weights = calls[0]
    reference_q = modeling._rope_tail(
        indexer.wq_b(qr).view(2, 6, 2, 8), inv_freq, positions, 4
    )
    reference_w = indexer.weights_proj(x) * (8 ** -0.5 * 2 ** -0.5)
    torch.testing.assert_close(q, reference_q, rtol=0, atol=0)
    torch.testing.assert_close(weights, reference_w, rtol=0, atol=0)
    variables = (x, qr, indexer.wq_b.weight, indexer.weights_proj.weight)
    actual_grads = torch.autograd.grad(q.square().sum() + weights.square().sum(), variables)
    expected_grads = torch.autograd.grad(reference_q.square().sum() + reference_w.square().sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("window", [1, 4, 128])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_window_shards_match_explicit_causal_positions(window, rank):
    mask = torch.tensor([[1] * 5 + [2] * 132 + [0] * 7, [1] * 144])
    start, end = rank * 48, (rank + 1) * 48
    actual = modeling.get_window_topk_idxs(
        window, 2, 144, mask.device, mask, query_start=start, query_end=end
    )
    for batch in range(2):
        for local, position in enumerate(range(start, end)):
            expected = [key for key in range(max(0, position - window + 1), position + 1)
                        if mask[batch, position] > 0 and mask[batch, key] == mask[batch, position]]
            assert actual[batch, local][actual[batch, local] >= 0].tolist() == expected
    if window == 128:
        assert npu.packed_window_matches(actual, mask, sp_rank=rank)
        actual = actual.clone()
        actual[0, 0, 0] = 999
        assert not npu.packed_window_matches(actual, mask, sp_rank=rank)


def test_unsupported_window_fails_before_installation():
    cfg = config()
    cfg.sliding_window = 64
    model = modeling.DeepseekV41Attention(cfg, 0)
    original = model.inner_attention_ops.forward.__func__
    with pytest.raises(ValueError, match="sliding_window=128"):
        npu.patch_deepseek_v41_npu_sfa(model)
    assert model.inner_attention_ops.forward.__func__ is original
    assert model.pre_attention_ops.forward.__func__ is modeling.DeepseekV41PreAttentionOps.forward


@pytest.mark.parametrize("ratio", [0, 2])
@pytest.mark.parametrize("size,rank", [(1, 0), (2, 0), (2, 1)])
def test_shared_pre_attention_matches_full_sequence_gradients(monkeypatch, ratio, size, rank):
    cfg = config()
    cfg.compress_ratios = [ratio]
    reference = modeling.DeepseekV41Attention(cfg, 0)
    patched = copy.deepcopy(reference)
    npu.patch_deepseek_v41_npu_sfa(patched, sp_size=size, sp_rank=rank,
                                   sp_group=object() if size > 1 else None)
    assert patched.pre_attention_ops.forward.__func__ is modeling.DeepseekV41PreAttentionOps.forward
    calls = []

    def gather(value, *, dim, sp_group, sp_size):
        calls.append(value)
        assert dim == 1 and sp_size == size
        return torch.cat([value] * size, dim=1)

    monkeypatch.setattr(modeling, "all_gather_sp_with_grad_reduce", gather)
    x = torch.randn(2, 6, 16, requires_grad=True)
    positions = torch.arange(6).expand(2, -1)
    mask = torch.arange(1, size + 1).repeat_interleave(6).expand(2, -1)
    plain, yarn = torch.tensor([1., .01]), torch.tensor([.5, .005])
    topk = torch.zeros(2, 6, 1, dtype=torch.long)

    def run(model, hidden, pos):
        return model.pre_attention_ops(
            hidden, plain, yarn, model.wq_a, model.q_norm, model.wq_b,
            model.wkv, model.kv_norm, None, None, None, topk, None, mask, pos
        )

    actual = run(patched, x, positions)
    expected = run(reference, x.repeat(1, size, 1), positions.repeat(1, size))
    start, end = rank * 6, (rank + 1) * 6
    torch.testing.assert_close(actual[0], expected[0][:, start:end], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2][:, start:end], rtol=0, atol=0)
    assert actual[4] is (topk if ratio else None)
    assert len(calls) == (1 if size > 1 else 0)
    parameter_names = ['wq_a.weight', 'q_norm.weight', 'wq_b.weight', 'wkv.weight', 'kv_norm.weight']
    aparams, eparams = dict(patched.named_parameters()), dict(reference.named_parameters())
    ga = torch.autograd.grad(actual[0].square().sum() + actual[1].square().sum(),
                             [x] + [aparams[name] for name in parameter_names])
    ge = torch.autograd.grad(expected[0][:, start:end].square().sum() + expected[1].square().sum(),
                             [x] + [eparams[name] for name in parameter_names])
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("size", [1, 2])
def test_compressor_shared_forward_preserves_communication_and_provider(monkeypatch, ratio, size):
    from types import SimpleNamespace
    import torchturbo.ops as op_package

    cfg = config()
    compressor = modeling.DeepseekV41Compressor(cfg, ratio)
    keys = tuple(compressor.state_dict())
    npu.patch_deepseek_v41_npu_sfa(compressor, sp_size=size, sp_rank=size - 1,
                                   sp_group=object() if size > 1 else None)
    assert compressor.forward.__func__ is modeling.DeepseekV41Compressor.forward
    assert tuple(compressor.state_dict()) == keys
    x = torch.randn(2, 8, 16, requires_grad=True)
    positions = torch.arange(8).expand(2, -1)
    mask = torch.ones(2, 8 * size, dtype=torch.long)
    gathered = []
    provider_calls = []

    def gather(value, *, dim, sp_group, sp_size):
        assert dim == 1 and sp_size == size
        gathered.append(value)
        return torch.cat([value] * size, dim=1)

    def provider(values, wkv, wgate, norm, actual_mask, actual_positions,
                 actual_ratio, head_dim, eps, *, impl):
        provider_calls.append(values)
        assert wkv is compressor.wkv.weight and wgate is compressor.wgate.weight
        assert norm is compressor.norm.weight and actual_mask is mask
        assert actual_ratio == ratio and head_dim == cfg.head_dim
        assert impl == (None if ratio == 2 else "torch_compressor")
        torch.testing.assert_close(values, torch.cat([x] * size, dim=1))
        torch.testing.assert_close(actual_positions, torch.cat([positions] * size, dim=1))
        return values

    monkeypatch.setattr(modeling, "all_gather_sp_with_grad_reduce", gather)
    monkeypatch.setattr(op_package, "ops", SimpleNamespace(compressor=provider))
    result = compressor(x, mask, positions)
    assert len(gathered) == (2 if size > 1 else 0)
    assert len(provider_calls) == int(ratio > 1)
    if size > 1:
        expected = x if ratio > 1 else compressor.wkv(x)
        torch.testing.assert_close(gathered[0], expected, rtol=0, atol=0)
        assert gathered[1] is positions
    assert result.shape[:2] == (2, 8 * size)



def test_turbo_logits_does_not_mutate_head():
    from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM
    holder = TurboDeepseekV41ForCausalLM.__new__(TurboDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(holder)
    cfg = config()
    cfg.vocab_size = 23
    holder.head = modeling.DeepseekV41Head(cfg)
    # DeepseekV41Head leaves the weight as torch.empty; init it so the
    # assertions below never read allocator garbage (NaNs) left behind by
    # earlier tests in the same process.
    holder.head.init_weights(0.02)
    holder.norm = modeling.DeepseekV41RMSNorm(16)
    x = torch.randn(1, 3, 16, requires_grad=True)
    keys = tuple(holder.state_dict())
    original = holder.head
    torch.testing.assert_close(holder._compute_logits(x), original(x, holder.norm), rtol=0, atol=0)
    calls = []

    def external_head(normalized):
        assert holder.head is external_head
        calls.append(True)
        return normalized.square()

    object.__setattr__(holder, "head", external_head)
    actual = holder._compute_logits(x)
    expected = holder.norm(x).float().square()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ga = torch.autograd.grad(actual.sum(), (x, holder.norm.weight))
    ge = torch.autograd.grad(expected.sum(), (x, holder.norm.weight))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert calls == [True] and holder.head is external_head
    assert tuple(holder.state_dict()) == keys and holder._modules["head"] is original


def test_text_config_forms_and_moe_replacement_scope():
    from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config, text_config_of
    from torchturbo.models.deepseek_v41.parallel_plan import _replace_moe_only
    text = config()
    nested = DeepseekV41Config(text_config=text.to_dict(), vision_config={"num_hidden_layers": 0})
    assert text_config_of(text) is text
    assert text_config_of(nested) is nested.text_config
    unrelated = torch.nn.Linear(2, 2)
    assert _replace_moe_only(unrelated, text, {}, {}, None) is unrelated


@pytest.mark.parametrize("nested", [False, True])
def test_mhc_installation_preserves_decoder_and_parameter_names(monkeypatch, nested):
    import importlib
    from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41Config
    provider = importlib.import_module("torchturbo.ops.providers.npu.mhc_pre_sinkhorn_premix")
    monkeypatch.setattr(provider, "has_mhc_pre_sinkhorn_premix", lambda: True)
    cfg = config()
    cfg.hidden_size, cfg.hc_mult, cfg.hc_sinkhorn_iters = 128, 4, 20
    holder = torch.nn.Module()
    holder.config = DeepseekV41Config(text_config=cfg.to_dict()) if nested else cfg
    layer = modeling.DeepseekV41DecoderLayer.__new__(modeling.DeepseekV41DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.register_parameter("sentinel", torch.nn.Parameter(torch.ones(1)))
    holder.layer = layer
    keys = tuple(holder.state_dict())
    reference = copy.deepcopy(holder)
    npu.patch_deepseek_v41_npu_mhc(holder)
    npu.patch_deepseek_v41_npu_mhc(holder)
    assert layer._hc_pre.__func__ is npu._npu_hc_pre
    assert layer.forward.__func__ is modeling.DeepseekV41DecoderLayer.forward
    assert reference.layer._hc_pre.__func__ is modeling.DeepseekV41DecoderLayer._hc_pre
    assert tuple(holder.state_dict()) == keys


@pytest.mark.parametrize("empty_selection", [False, True])
def test_window_only_execution_accepts_unused_compression_state(monkeypatch, empty_selection):
    from types import SimpleNamespace

    q = torch.randn(1, 4, 2, 512, requires_grad=True)
    kv = torch.randn(1, 4, 512, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.long)
    compressed = torch.empty(1, 0, 512, requires_grad=True) if empty_selection else None
    selected = torch.empty(1, 4, 0, dtype=torch.long) if empty_selection else torch.zeros(1, 4, 1, dtype=torch.long)

    def window_kernel(query, original, cmp, indices, residual, sinks, *args, **kwargs):
        assert cmp is None and indices is None and residual is None
        return query + original.sum() * 0.01 + sinks.sum() * 0.1

    # npu.ops is the lazy dispatcher proxy (raises while unconfigured), so the
    # fake replaces the module-level name instead of a proxy attribute.
    monkeypatch.setattr(npu, "ops", SimpleNamespace(sparse_flash_mla=window_kernel))
    actual = npu.run_cann_attention(q, kv, compressed, selected, sink, 512**-0.5, 2, mask)
    expected = npu.run_cann_attention(q, kv, None, None, sink, 512**-0.5, 2, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    variables = (q, kv, sink)
    ga = torch.autograd.grad(actual.sum(), variables, retain_graph=True)
    ge = torch.autograd.grad(expected.sum(), variables)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    if empty_selection:
        assert torch.autograd.grad(actual.sum(), compressed)[0].shape == compressed.shape


def test_unknown_dtype_is_not_silently_replaced_with_fp32():
    from types import SimpleNamespace
    from torchturbo.models.deepseek_v41.parallel_plan import _dtype_from_config
    assert _dtype_from_config(SimpleNamespace(torch_dtype="bf16")) == torch.bfloat16
    assert _dtype_from_config(SimpleNamespace(torch_dtype=None)) == torch.float32
    with pytest.raises(KeyError):
        _dtype_from_config(SimpleNamespace(torch_dtype="bfloatl6"))
