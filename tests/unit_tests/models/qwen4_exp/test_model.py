from __future__ import annotations

import pytest
import torch

from torchturbo.models.modules.attention.ulysses import (
    SequencePadding,
)
from torchturbo.models.modules.cpu_resident import CPUResidentParameter
from torchturbo.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from torchturbo.models.qwen4_exp.model import (
    Qwen4ExpWeightConverter,
    TurboQwen4ExpForConditionalGeneration,
)
from torchturbo.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpModel,
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextModel,
    Qwen4ExpTextPLELayer,
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRMSNorm,
    Qwen4ExpTextRMSNormGated,
    Qwen4ExpVisionModel,
)
from torchturbo.models.qwen4_exp.qsa import (
    qsa_attention_forward,
    qsa_indexer_forward,
)
from torchturbo.models.registry import get_model
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids


def _small_config() -> Qwen4ExpConfig:
    text_config = {
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "hc_count": 4,
        "hc_lowrank": 16,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "ple_layer_ids": [2],
        "ple_embed_dim": 64,
        "ngram_vocab_size_base": 256,
        "split_ngram_parts": 4,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "max_position_embeddings": 128,
        "eos_token_id": 2,
        "use_cache": False,
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "out_hidden_size": 64,
        "num_heads": 4,
        "num_position_embeddings": 64,
    }
    return Qwen4ExpConfig(text_config=text_config, vision_config=vision_config)


def _new_qsa_indexer(config, layer_idx: int) -> Qwen4ExpTextQSAIndexer:
    return Qwen4ExpTextQSAIndexer(config, layer_idx)


def _token_ids_to_mask(token_ids: torch.Tensor, key_length: int) -> torch.Tensor:
    scatter_indices = torch.where(token_ids >= 0, token_ids, key_length)
    selected = torch.zeros(
        *token_ids.shape[:-1],
        key_length + 1,
        dtype=torch.bool,
        device=token_ids.device,
    )
    selected.scatter_(-1, scatter_indices, True)
    return selected[..., :key_length].unsqueeze(1)


def _cpu_qsa_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    _attention_mask: torch.Tensor | None,
    _query_length: int,
    _is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: float | None = None,
    *,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    outputs = []
    for query_idx in range(cu_seq_lens_q.numel() - 1):
        q = query[:, cu_seq_lens_q[query_idx] : cu_seq_lens_q[query_idx + 1]].transpose(1, 2)
        key_start = int(cu_seq_lens_k[query_idx])
        key_stop = int(cu_seq_lens_k[query_idx + 1])
        k = key[:, key_start:key_stop].transpose(1, 2)
        v = value[:, key_start:key_stop].transpose(1, 2)
        repeats = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout,
                scale=softmax_scale,
            ).transpose(1, 2)
        )
    return torch.cat(outputs, dim=1)


def _patch_cpu_qsa(monkeypatch) -> None:
    monkeypatch.setattr(
        "torchturbo.models.qwen4_exp.qsa.attention_forward",
        _cpu_qsa_varlen_attention,
    )


def test_parallel_forwards_are_bound_per_model_instance() -> None:
    first_owner = TurboQwen4ExpForConditionalGeneration(_small_config())
    second_owner = TurboQwen4ExpForConditionalGeneration(_small_config())
    first = first_owner.model.language_model
    second = second_owner.model.language_model
    original_class_forward = Qwen4ExpTextGatedDeltaNet.forward
    original_ple_forward = Qwen4ExpTextPLELayer.forward
    first_group = object()
    second_group = object()

    TurboQwen4ExpForConditionalGeneration.on_parallelized(
        first_owner,
        {"sp": {"sp_group": first_group, "sp_size": 2, "sp_rank": 0}},
    )
    first_linear = first.layers[0].linear_attn
    first_sp_forward = first_linear.forward
    first_qsa_forward = first.layers[-1].self_attn.forward
    first_ple_forward = first.layers[1].ple.forward

    TurboQwen4ExpForConditionalGeneration.on_parallelized(
        second_owner,
        {"sp": {"sp_group": second_group, "sp_size": 4, "sp_rank": 3}},
    )
    second_linear = second.layers[0].linear_attn
    first_qsa = first.layers[-1].self_attn
    first_ple = first.layers[1].ple
    second_ple_forward = second.layers[1].ple.forward
    second_qsa_forward = second.layers[-1].self_attn.forward

    assert Qwen4ExpTextGatedDeltaNet.forward is original_class_forward
    assert first_linear.forward == first_sp_forward
    assert first_qsa.forward == first_qsa_forward
    assert first_ple.forward == first_ple_forward
    assert Qwen4ExpTextPLELayer.forward is original_ple_forward
    assert not hasattr(first_linear, "_sp_group")
    assert not hasattr(first_linear, "_sp_size")
    assert not hasattr(first_linear, "_sp_rank")
    assert not hasattr(second_linear, "_sp_group")
    assert not hasattr(second_linear, "_sp_size")
    assert not hasattr(second_linear, "_sp_rank")
    for module in (first_qsa, first_qsa.indexer, first_ple):
        assert not hasattr(module, "_sp_group")
        assert not hasattr(module, "_sp_size")
        assert not hasattr(module, "_sp_rank")
    assert first_linear.forward.__func__ is not second_linear.forward.__func__

    TurboQwen4ExpForConditionalGeneration.on_parallelized(
        first_owner,
        {"sp": {"sp_group": None, "sp_size": 1, "sp_rank": 0}},
    )
    assert first_linear.forward.__func__ is not original_class_forward
    assert second_linear.forward.__func__ is not original_class_forward
    assert second.layers[1].ple.forward == second_ple_forward
    assert second.layers[-1].self_attn.forward == second_qsa_forward


def test_packed_text_forward_uses_modeling_implementation() -> None:
    owner = TurboQwen4ExpForConditionalGeneration(_small_config())
    language_model = owner.model.language_model

    assert type(language_model) is Qwen4ExpTextModel
    assert language_model.forward.__func__ is Qwen4ExpTextModel.forward
    assert not hasattr(language_model, "_sp_group")
    assert not hasattr(language_model, "_sp_size")
    assert not hasattr(language_model, "_sp_rank")

    spec = (
        TurboQwen4ExpForConditionalGeneration.get_sequence_parallel_input_slicing_spec(
            owner
        )
    )
    assert spec.original_forward is Qwen4ExpTextModel.forward


@pytest.mark.parametrize("input_mode", ["keyword", "positional"])
def test_packed_text_accepts_ids_without_redundant_ple_ids(monkeypatch, input_mode):
    config = _small_config()
    config.text_config.layer_types = ["full_attention"] * 4
    owner = TurboQwen4ExpForConditionalGeneration(config).eval()
    text = owner.model.language_model
    _patch_cpu_qsa(monkeypatch)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    kwargs = {
        "cu_seq_lens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 3, 8], dtype=torch.int32),
        "max_length_q": 5, "max_length_k": 5, "use_cache": False,
    }
    expected = text(input_ids=ids, ple_input_ids=ids, **kwargs).last_hidden_state
    actual = text(ids, **kwargs) if input_mode == "positional" else text(input_ids=ids, **kwargs)
    torch.testing.assert_close(actual.last_hidden_state, expected)


def test_packed_text_rejects_embeddings_without_token_ids(monkeypatch):
    config = _small_config()
    config.text_config.layer_types = ["full_attention"] * 4
    owner = TurboQwen4ExpForConditionalGeneration(config).eval()
    text = owner.model.language_model
    _patch_cpu_qsa(monkeypatch)
    embeddings = text.embed_tokens(torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]]))
    kwargs = {
        "cu_seq_lens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 3, 8], dtype=torch.int32),
        "max_length_q": 5, "max_length_k": 5, "use_cache": False,
    }
    # Recovering IDs from embeddings compares every token with the vocabulary.
    with pytest.raises(ValueError, match="ple_input_ids or input_ids"):
        text(inputs_embeds=embeddings, **kwargs)


def test_packed_text_rejects_attention_mask_with_cu_seq_lens():
    owner = TurboQwen4ExpForConditionalGeneration(_small_config()).eval()
    text = owner.model.language_model
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    with pytest.raises(ValueError, match="attention_mask must be None"):
        text(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=5,
            max_length_k=5,
            use_cache=False,
        )


def test_qwen4_sp1_gdn_uses_packed_ops(monkeypatch) -> None:
    import torchturbo.ops as turbo_ops
    from torchturbo.ops.dispatcher import OpDispatcher

    captured = []

    def dispatch(op_name, *args, **kwargs):
        packed = kwargs["packed_params"]
        captured.append(packed)
        if op_name == "causal_conv1d":
            return args[0], None
        assert op_name == "gated_delta_rule"
        return args[2], None

    dispatcher = OpDispatcher.__new__(OpDispatcher)
    monkeypatch.setattr(dispatcher, "_dispatch", dispatch)
    monkeypatch.setattr(turbo_ops, "ops", dispatcher)

    owner = TurboQwen4ExpForConditionalGeneration(_small_config())
    language_model = owner.model.language_model
    TurboQwen4ExpForConditionalGeneration.on_parallelized(
        owner,
        {"sp": {"sp_group": None, "sp_size": 1, "sp_rank": 0}},
    )
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    output = language_model.layers[0].linear_attn(
        torch.randn(1, 8, language_model.config.hidden_size),
        cu_seq_lens=cu_seqlens,
    )

    assert output.shape == (1, 8, language_model.config.hidden_size)
    assert captured[0] is captured[1]
    assert captured[0].cu_seqlens_list == [0, 3, 8]


def test_text_rms_norm_chunking_preserves_values_and_gradients() -> None:
    torch.manual_seed(0)
    reference = Qwen4ExpTextRMSNorm(16, group_size=4)
    chunked = Qwen4ExpTextRMSNorm(16, group_size=4)
    chunked.load_state_dict(reference.state_dict())
    chunked._temporary_bytes = 2 * 3 * 16 * torch.float32.itemsize
    reference_input = torch.randn(2, 11, 16, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    probe = torch.randn_like(reference_input)

    reference_output = reference(reference_input)
    chunked_output = chunked(chunked_input)
    (reference_output * probe).sum().backward()
    (chunked_output * probe).sum().backward()

    torch.testing.assert_close(chunked_output, reference_output)
    torch.testing.assert_close(chunked_input.grad, reference_input.grad)
    torch.testing.assert_close(chunked.weight.grad, reference.weight.grad)


def test_text_rms_norm_chunking_preserves_npu_bf16_gradients() -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is unavailable")
    torch.manual_seed(1)
    reference = Qwen4ExpTextRMSNorm(64, group_size=16).npu()
    chunked = Qwen4ExpTextRMSNorm(64, group_size=16).npu()
    chunked.load_state_dict(reference.state_dict())
    chunked._temporary_bytes = 2 * 5 * 64 * torch.float32.itemsize
    reference_input = torch.randn(2, 17, 64, dtype=torch.bfloat16).npu()
    reference_input.requires_grad_(True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    probe = torch.randn_like(reference_input)

    reference_output = reference(reference_input)
    chunked_output = chunked(chunked_input)
    (reference_output.float() * probe).sum().backward()
    (chunked_output.float() * probe).sum().backward()
    torch.npu.synchronize()

    torch.testing.assert_close(chunked_output.cpu(), reference_output.cpu())
    torch.testing.assert_close(chunked_input.grad.cpu(), reference_input.grad.cpu())
    torch.testing.assert_close(chunked.weight.grad.cpu(), reference.weight.grad.cpu())


def test_text_rms_norm_gated_chunking_preserves_values_and_gradients() -> None:
    torch.manual_seed(2)
    reference = Qwen4ExpTextRMSNormGated(16)
    chunked = Qwen4ExpTextRMSNormGated(16)
    chunked.load_state_dict(reference.state_dict())
    chunked._temporary_bytes = 4 * 3 * 16 * torch.float32.itemsize
    reference_input = torch.randn(2, 11, 4, 16, requires_grad=True)
    reference_gate = torch.randn_like(reference_input, requires_grad=True)
    chunked_input = reference_input.detach().clone().requires_grad_(True)
    chunked_gate = reference_gate.detach().clone().requires_grad_(True)
    probe = torch.randn_like(reference_input)

    reference_output = reference(reference_input, reference_gate)
    chunked_output = chunked(chunked_input, chunked_gate)
    (reference_output * probe).sum().backward()
    (chunked_output * probe).sum().backward()

    torch.testing.assert_close(chunked_output, reference_output)
    torch.testing.assert_close(chunked_input.grad, reference_input.grad)
    torch.testing.assert_close(chunked_gate.grad, reference_gate.grad)
    torch.testing.assert_close(chunked.weight.grad, reference.weight.grad)


def test_gated_norm_commutes_with_ulysses_vector_permutation() -> None:
    torch.manual_seed(3)
    norm_before = Qwen4ExpTextRMSNormGated(16)
    norm_after = Qwen4ExpTextRMSNormGated(16)
    norm_after.load_state_dict(norm_before.state_dict())
    permutation = torch.randperm(24)

    value_before = torch.randn(24, 16, requires_grad=True)
    gate_before = torch.randn_like(value_before, requires_grad=True)
    value_after = value_before.detach().clone().requires_grad_(True)
    gate_after = gate_before.detach().clone().requires_grad_(True)
    probe = torch.randn_like(value_before)

    output_before = norm_before(value_before, gate_before)[permutation]
    output_after = norm_after(value_after[permutation], gate_after[permutation])
    (output_before * probe).sum().backward()
    (output_after * probe).sum().backward()

    torch.testing.assert_close(output_before, output_after)
    torch.testing.assert_close(value_before.grad, value_after.grad)
    torch.testing.assert_close(gate_before.grad, gate_after.grad)
    torch.testing.assert_close(norm_before.weight.grad, norm_after.weight.grad)


@pytest.mark.parametrize("sp_size", [2, 4, 8])
def test_qsa_indexer_selects_only_local_queries(monkeypatch, sp_size) -> None:
    torch.manual_seed(7)
    config = _small_config().text_config
    config.indexer_n_heads = 5
    indexer = _new_qsa_indexer(config, layer_idx=3)
    hidden_states = torch.randn(1, 32, config.hidden_size)
    phases = torch.randn(1, 32, config.indexer_head_dim // 2).repeat(1, 1, 2)
    cos, sin = phases.cos(), phases.sin()
    cu_seqlens = torch.tensor([0, 3, 14, 25], dtype=torch.int32)
    expected = qsa_indexer_forward(indexer, hidden_states, (cos, sin), cu_seqlens)
    raw_keys = indexer.index_qk_proj(hidden_states)[..., -config.indexer_head_dim :]
    full_state = torch.cat((raw_keys, cos, sin), dim=-1).movedim(1, 0)
    local_length = hidden_states.shape[1] // sp_size
    calls = []

    def gather(tensor, tensor_parallel_output_grad, group):
        assert tensor_parallel_output_grad is False
        assert group == "qsa-sp"
        calls.append(tensor)
        return full_state

    monkeypatch.setattr(
        "torchturbo.models.qwen4_exp.qsa.gather_from_sequence_parallel_region", gather
    )
    for rank in range(sp_size):
        shard = slice(rank * local_length, (rank + 1) * local_length)
        actual = qsa_indexer_forward(
            indexer,
            hidden_states[:, shard],
            (cos[:, shard], sin[:, shard]),
            cu_seqlens,
            sp_group="qsa-sp",
            sp_size=sp_size,
            sp_rank=rank,
        )
        torch.testing.assert_close(actual, expected[:, shard])
        torch.testing.assert_close(calls[-1], full_state[shard])
    assert len(calls) == sp_size


def test_qsa_indexer_preserves_pooled_key_dtype_under_autocast(monkeypatch) -> None:
    torch.manual_seed(23)
    config = _small_config().text_config
    indexer = _new_qsa_indexer(config, layer_idx=3)
    hidden = torch.randn(1, 24, config.hidden_size)
    cos = torch.ones(1, 24, config.indexer_head_dim)
    sin = torch.zeros_like(cos)
    lengths = torch.tensor([0, 24], dtype=torch.int32)
    pooled_dtypes = []
    handle = indexer.k_layernorm.register_forward_pre_hook(
        lambda module, args: pooled_dtypes.append(args[0].dtype)
    )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = qsa_indexer_forward(indexer, hidden, (cos, sin), lengths)
            raw_keys = indexer.index_qk_proj(hidden)[..., -config.indexer_head_dim :]
            full_state = torch.cat((raw_keys, cos, sin), dim=-1).movedim(1, 0)
            assert full_state.dtype == torch.float32
            monkeypatch.setattr(
                "torchturbo.models.qwen4_exp.qsa.gather_from_sequence_parallel_region",
                lambda *args, **kwargs: full_state,
            )
            actual = qsa_indexer_forward(
                indexer,
                hidden[:, 12:],
                (cos[:, 12:], sin[:, 12:]),
                lengths,
                sp_size=2,
                sp_rank=1,
            )
    finally:
        handle.remove()
    assert pooled_dtypes == [torch.bfloat16, torch.bfloat16]
    torch.testing.assert_close(actual, expected[:, 12:])


def test_qwen4_exp_registers_with_torchturbo() -> None:
    assert get_model("qwen4_exp") is TurboQwen4ExpForConditionalGeneration


def test_qwen4_exp_advertises_flash_attention_support() -> None:
    with torch.device("meta"):
        TurboQwen4ExpForConditionalGeneration(_small_config())
    assert TurboQwen4ExpForConditionalGeneration._supports_flash_attn is True
    assert Qwen4ExpTextModel._supports_flash_attn is True
    assert Qwen4ExpVisionModel._supports_flash_attn is True


def test_flash_init_keeps_original_container_and_submodel_backends(monkeypatch) -> None:
    import transformers.modeling_utils as hf_modeling

    # Emulate backend availability while retaining HF's model support checks.
    monkeypatch.setitem(
        hf_modeling.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2],
        "general_availability_check",
        lambda: True,
    )
    monkeypatch.setattr(
        hf_modeling.PreTrainedModel,
        "_flash_attn_import_error",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hf_modeling, "lazy_import_flash_attention", lambda *args, **kwargs: None
    )
    config = _small_config()
    config._attn_implementation = "flash_attention_2"
    config.dtype = torch.bfloat16
    with torch.device("meta"):
        model = TurboQwen4ExpForConditionalGeneration(config)

    assert type(model.model) is Qwen4ExpModel
    assert type(model.model.language_model) is Qwen4ExpTextModel
    assert type(model.model.visual) is Qwen4ExpVisionModel
    assert Qwen4ExpModel._supports_flash_attn is True
    assert Qwen4ExpTextModel._supports_flash_attn is True
    assert Qwen4ExpVisionModel._supports_flash_attn is True
    assert config._attn_implementation == "flash_attention_2"
    assert config.text_config._attn_implementation == "flash_attention_2"
    assert config.vision_config._attn_implementation == "flash_attention_2"
    model.on_parallelized({"sp": {"sp_group": None, "sp_size": 1, "sp_rank": 0}})
    assert config.text_config._attn_implementation == "flash_attention_2"
    assert config.vision_config._attn_implementation == "flash_attention_2"


def test_qsa_keeps_queries_local_and_gathers_kv_per_instance(monkeypatch) -> None:
    from torchturbo.models.qwen4_exp import qsa

    models = [TurboQwen4ExpForConditionalGeneration(_small_config()) for _ in range(2)]
    groups = [object(), object()]
    for model, group, size in zip(models, groups, (2, 4)):
        model.on_parallelized(
            {"sp": {"sp_group": group, "sp_size": size, "sp_rank": size - 1}}
        )
    calls = []
    current = {}

    def select(indexer, hidden, positions, lengths, *, sp_group, sp_size, sp_rank):
        assert sp_group is current["group"]
        assert sp_size == current["size"]
        assert sp_rank == sp_size - 1
        calls.append("indexer")
        return torch.zeros(1, 4, 1, dtype=torch.int32)

    def gather(tensor, dim, sp_group, sp_size):
        assert sp_group is current["group"]
        assert sp_size == current["size"]
        assert dim == 1
        assert tensor.shape == (1, 4, 1, 32)
        calls.append("gather")
        return tensor.repeat(1, sp_size, 1, 1)

    def attend(query, key, value, selected, **kwargs):
        assert query.shape == (1, 4, 4, 16)
        assert key.shape == value.shape == (1, 4 * current["size"], 1, 16)
        assert selected.shape == (1, 4, 1)
        calls.append("attention")
        return torch.zeros_like(query)

    monkeypatch.setattr(qsa, "qsa_indexer_forward", select)
    monkeypatch.setattr(qsa, "all_gather_sp_with_grad_reduce", gather)
    monkeypatch.setattr(qsa, "qsa_attention_forward", attend)
    for model, group, size in zip(models, groups, (2, 4)):
        current.update(group=group, size=size)
        attention = model.model.language_model.layers[-1].self_attn
        hidden = torch.randn(1, 4, 64)
        positions = (torch.ones(1, 4, 16), torch.zeros(1, 4, 16))
        lengths = torch.tensor([0, 4 * size], dtype=torch.int32)
        output, _ = attention(hidden, positions, None, cu_seq_lens_q=lengths)
        assert output.shape == hidden.shape
    assert calls == ["indexer", "gather", "attention"] * 2


def test_qwen4_uses_upstream_multimodal_forward_with_ids_and_embeddings() -> None:
    model = TurboQwen4ExpForConditionalGeneration(_small_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    inputs_embeds = model.get_input_embeddings()(input_ids)
    positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, -1, -1)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=positions,
        )

    assert model.model.forward.__func__ is Qwen4ExpModel.forward
    assert output.logits.shape == (1, input_ids.shape[1], 256)


def test_qwen4_sp_input_contract_contains_only_real_qwen4_fields() -> None:
    model = TurboQwen4ExpForConditionalGeneration(_small_config())
    spec = model.get_sequence_parallel_input_slicing_spec()

    assert spec.sliced_inputs == (("inputs_embeds", 1), ("position_ids", -1))
    assert spec.padded_inputs == (
        ("input_ids", SequencePadding.ZERO),
        ("mm_token_type_ids", SequencePadding.ZERO),
        ("labels", SequencePadding.IGNORE_INDEX),
        ("position_ids", SequencePadding.POSITION),
    )
    assert all(name != "text_position_ids" for name, _ in spec.padded_inputs)


def test_qwen4_exp_meta_construction_registers_frozen_table_parameters() -> None:
    config = _small_config()
    with torch.device("meta"):
        model = TurboQwen4ExpForConditionalGeneration(config)

    ngram_embedding = model.model.language_model.layers[
        1
    ].ple.ple_embedding.ngram_embedding
    with pytest.raises(RuntimeError, match="not been loaded"):
        ngram_embedding.gather(torch.tensor([0]))
    parameters = dict(ngram_embedding.named_parameters())
    assert set(parameters) == {"weight"}
    assert all(isinstance(parameter, CPUResidentParameter) for parameter in parameters.values())
    assert all(parameter.is_meta for parameter in parameters.values())
    assert all(not parameter.requires_grad for parameter in parameters.values())


def test_latest_modeling_accepts_three_axis_mrope_positions(monkeypatch) -> None:
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 0, 1]],
            [[0, 1, 2, 0, 1]],
            [[0, 1, 2, 0, 1]],
        ]
    )

    model = TurboQwen4ExpForConditionalGeneration(_small_config()).eval()
    _patch_cpu_qsa(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=3,
            max_length_k=3,
        )
    assert output.logits.shape == (1, 5, 256)


def test_qsa_attention_matches_masked_reference_and_gradients(monkeypatch) -> None:
    captured = {}

    def fake_varlen_attention(
        query, key, value, attention_mask, query_length, is_causal, **kwargs
    ):
        captured.update(
            query=query,
            key=key,
            cu_q=kwargs["cu_seq_lens_q"],
            cu_k=kwargs["cu_seq_lens_k"],
            is_causal=is_causal,
        )
        return _cpu_qsa_varlen_attention(
            query, key, value, attention_mask, query_length, is_causal, **kwargs
        )

    monkeypatch.setattr(
        "torchturbo.models.qwen4_exp.qsa.attention_forward",
        fake_varlen_attention,
    )
    query = torch.randn(1, 4, 4, 8, dtype=torch.double, requires_grad=True)
    key = torch.randn(1, 4, 2, 8, dtype=torch.double, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    token_ids = torch.tensor(
        [[[0, -1, -1], [0, 1, -1], [0, 2, 3], [-1, -1, -1]]],
        dtype=torch.int32,
    )
    output = qsa_attention_forward(
        query,
        key,
        value,
        token_ids,
        softmax_scale=8**-0.5,
    )

    assert captured["query"].shape[1] == int(captured["cu_q"][-1])
    assert captured["query"].shape[1] * captured["query"].shape[2] == 3 * 4
    assert captured["key"].shape == (1, 6, 2, 8)
    torch.testing.assert_close(
        captured["cu_q"],
        torch.arange(4, dtype=torch.int32) * (4 // captured["query"].shape[2]),
    )
    torch.testing.assert_close(
        captured["cu_k"], torch.tensor([0, 1, 3, 6], dtype=torch.int32)
    )
    assert captured["is_causal"] is False

    reference_inputs = [
        tensor.detach().clone().requires_grad_() for tensor in (query, key, value)
    ]
    ref_query, ref_key, ref_value = reference_inputs
    expected = torch.nn.functional.scaled_dot_product_attention(
        ref_query.transpose(1, 2),
        ref_key.repeat_interleave(2, dim=2).transpose(1, 2),
        ref_value.repeat_interleave(2, dim=2).transpose(1, 2),
        attn_mask=_token_ids_to_mask(token_ids, key.shape[1]),
        scale=8**-0.5,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    (expected * probe).sum().backward()
    for actual, reference in zip((query, key, value), reference_inputs):
        torch.testing.assert_close(actual.grad, reference.grad)


@pytest.mark.parametrize("query_chunk_size", [3, 256])
def test_qsa_indexer_matches_packed_reference(monkeypatch, query_chunk_size) -> None:
    torch.manual_seed(1)
    monkeypatch.setattr(
        "torchturbo.models.qwen4_exp.qsa._QSA_INDEXER_QUERY_CHUNK_SIZE",
        query_chunk_size,
    )
    config = _small_config().text_config
    indexer = _new_qsa_indexer(config, layer_idx=3)
    hidden_states = torch.randn(1, 26, config.hidden_size)
    phases = torch.randn(1, 26, config.indexer_head_dim // 2).repeat(1, 1, 2)
    cos, sin = phases.cos(), phases.sin()
    cu_seqlens = torch.tensor([0, 5, 12, 24], dtype=torch.int32)
    sequence_ids = cu_seqlens_to_segment_ids(cu_seqlens, num_tokens=26)
    positions = torch.arange(26)
    packed_causal = (
        (sequence_ids[:, :, None] == sequence_ids[:, None, :])
        & sequence_ids[:, :, None].ne(0)
        & (positions[None, None, :] <= positions[None, :, None])
    ).unsqueeze(1)

    expected = indexer(hidden_states, (cos, sin), packed_causal, past_key_values=None)
    token_ids = qsa_indexer_forward(indexer, hidden_states, (cos, sin), cu_seqlens)

    assert token_ids.dtype == torch.int32
    assert token_ids.shape == (1, 26, indexer.token_budget + indexer.compress_ratio - 1)
    valid = token_ids >= 0
    assert not (~valid[..., :-1] & valid[..., 1:]).any()
    actual = _token_ids_to_mask(token_ids, hidden_states.shape[1])
    torch.testing.assert_close(actual, expected)
    assert not (actual & ~packed_causal).any()


def test_qsa_indexer_keeps_upstream_forward() -> None:
    indexer = _new_qsa_indexer(_small_config().text_config, layer_idx=3)

    assert indexer.forward.__func__ is Qwen4ExpTextQSAIndexer.forward


def test_npu_qsa_selection_matches_cpu() -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an NPU")

    config = _small_config().text_config
    cpu_indexer = _new_qsa_indexer(config, layer_idx=3)
    npu_indexer = _new_qsa_indexer(config, layer_idx=3).npu()
    npu_indexer.load_state_dict(cpu_indexer.state_dict())
    generator = torch.Generator().manual_seed(20260913)
    hidden_states = torch.randn(1, 24, config.hidden_size, generator=generator)
    cos = torch.ones(1, 24, config.indexer_head_dim)
    sin = torch.zeros_like(cos)
    cu_seqlens = torch.tensor([0, 5, 12, 24], dtype=torch.int32)

    expected = qsa_indexer_forward(cpu_indexer, hidden_states, (cos, sin), cu_seqlens)
    actual = qsa_indexer_forward(
        npu_indexer,
        hidden_states.npu(),
        (cos.npu(), sin.npu()),
        cu_seqlens,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected)


def test_qwen4_exp_weight_converter_recognizes_ple_buffers() -> None:
    converter = object.__new__(Qwen4ExpWeightConverter)
    name = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
    assert converter.is_persist_buffer(name)
    assert converter.convert_persist_buffer(name) == name


def test_qwen4_exp_rebuilds_meta_rotary_buffers(monkeypatch) -> None:
    model = None
    config = _small_config()
    config.text_config.ple_layer_ids = []
    with torch.device("meta"):
        model = TurboQwen4ExpForConditionalGeneration(config)
    model.to_empty(device="cpu")
    monkeypatch.setattr(
        "torchturbo.models.qwen4_exp.model.get_device_runtime",
        lambda: type("Runtime", (), {"device": torch.device("cpu")})(),
    )

    model.on_materialized()

    assert model.model.language_model.rotary_emb.inv_freq.device.type == "cpu"
    assert model.model.visual.rotary_pos_emb.inv_freq.device.type == "cpu"


def test_qwen4_exp_text_sft_forward_backward(monkeypatch) -> None:
    torch.manual_seed(1)
    model = TurboQwen4ExpForConditionalGeneration(_small_config()).float().train()
    _patch_cpu_qsa(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
        cu_seq_lens_q=torch.tensor([0, 3, 8], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 3, 8], dtype=torch.int32),
        max_length_q=5,
        max_length_k=5,
    )

    assert outputs.logits.shape == (1, 8, 256)
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    ple = model.model.language_model.layers[1].ple
    assert ple.key_proj.weight.grad is not None
    assert ple.value_proj.weight.grad is not None
    table = ple.ple_embedding.ngram_embedding.weight
    assert not table.requires_grad
    assert table.grad is None


def test_packed_qsa_ple_matches_independent_sequences(monkeypatch) -> None:
    """Packed execution must preserve the semantics of independent samples."""
    config = _small_config()
    config.text_config.layer_types = ["full_attention"] * 4

    torch.manual_seed(19)
    packed_model = TurboQwen4ExpForConditionalGeneration(config).double().train()
    independent_model = TurboQwen4ExpForConditionalGeneration(config).double().train()
    independent_model.load_state_dict(packed_model.state_dict())
    for layer, reference_layer in zip(
        independent_model.model.language_model.layers,
        packed_model.model.language_model.layers,
    ):
        if layer.ple is not None:
            embedding = layer.ple.ple_embedding.ngram_embedding
            reference_embedding = reference_layer.ple.ple_embedding.ngram_embedding
            assert embedding.weight is not reference_embedding.weight
            torch.testing.assert_close(embedding.weight, reference_embedding.weight)
    _patch_cpu_qsa(monkeypatch)

    first = torch.tensor([[11, 12, 13]])
    second = torch.tensor([[21, 22, 23, 24, 25]])
    packed = torch.cat((first, second), dim=1)

    def run(model, input_ids, lengths, labels):
        offsets = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        text_positions = torch.cat([torch.arange(length) for length in lengths])
        position_ids = text_positions.view(1, 1, -1).expand(3, -1, -1)
        return model(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=offsets,
            cu_seq_lens_k=offsets,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
        )

    packed_labels = packed.clone()
    packed_labels[:, first.shape[1]] = -100
    packed_output = run(packed_model, packed, [3, 5], packed_labels)
    first_output = run(independent_model, first, [3], first)
    second_output = run(independent_model, second, [5], second)

    independent_logits = torch.cat((first_output.logits, second_output.logits), dim=1)
    torch.testing.assert_close(packed_output.logits, independent_logits)

    first_targets = first.shape[1] - 1
    second_targets = second.shape[1] - 1
    independent_loss = (
        first_output.loss * first_targets + second_output.loss * second_targets
    ) / (first_targets + second_targets)
    torch.testing.assert_close(packed_output.loss, independent_loss)

    packed_output.loss.backward()
    independent_loss.backward()
    parameter_names = (
        "lm_head.weight",
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
    )
    packed_parameters = dict(packed_model.named_parameters())
    independent_parameters = dict(independent_model.named_parameters())
    for name in parameter_names:
        torch.testing.assert_close(
            packed_parameters[name].grad,
            independent_parameters[name].grad,
        )


def test_npu_packed_model_matches_independent_sequences(monkeypatch) -> None:
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an NPU")

    import importlib

    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime import get_device_runtime

    ops_module = importlib.import_module("torchturbo.ops.ops")
    monkeypatch.setattr(ops_module, "_OPS", None)
    get_or_create_ops(device_runtime=get_device_runtime())

    config = _small_config()
    text_config = config.text_config
    text_config.hidden_size = 128
    text_config.num_attention_heads = 4
    text_config.head_dim = 32
    text_config.indexer_head_dim = 32
    text_config.linear_key_head_dim = 128
    text_config.linear_value_head_dim = 128
    text_config.hc_lowrank = 32
    text_config.ple_embed_dim = 128
    text_config.max_position_embeddings = 256
    config.vision_config.out_hidden_size = 128

    torch.manual_seed(17)
    model = TurboQwen4ExpForConditionalGeneration(config).bfloat16().eval()

    def place_compute_tensor(tensor: torch.Tensor) -> torch.Tensor:
        # Match backend placement while keeping the lookup table on CPU.
        if isinstance(tensor, CPUResidentParameter):
            return tensor
        return tensor.npu()

    model._apply(place_compute_tensor)
    model.on_parallelized({"sp": {"sp_group": None, "sp_size": 1, "sp_rank": 0}})

    def run(input_ids, lengths):
        offsets = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        text_positions = torch.cat([torch.arange(length) for length in lengths])
        position_ids = text_positions.view(1, 1, -1).expand(3, -1, -1).npu()
        return model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            cu_seq_lens_q=offsets,
            cu_seq_lens_k=offsets,
            max_length_q=max(lengths),
            max_length_k=max(lengths),
        ).logits

    first = torch.arange(11, 75).view(1, -1).npu()
    second = torch.arange(101, 165).view(1, -1).npu()
    with torch.no_grad():
        packed_logits = run(torch.cat((first, second), dim=1), [64, 64])
        independent_logits = torch.cat((run(first, [64]), run(second, [64])), dim=1)
    torch.npu.synchronize()

    torch.testing.assert_close(packed_logits.cpu(), independent_logits.cpu())


def test_qwen4_exp_accepts_ids_with_precomputed_embeddings() -> None:
    torch.manual_seed(2)
    model = TurboQwen4ExpForConditionalGeneration(_small_config()).float().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    inputs_embeds = model.get_input_embeddings()(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )

    assert outputs.logits.shape == (1, 4, 256)
    assert torch.isfinite(outputs.logits).all()


def test_qwen4_exp_packed_boundaries_are_cu_driven() -> None:
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    sequence_ids = cu_seqlens_to_segment_ids(cu_seqlens, num_tokens=8)

    torch.testing.assert_close(
        sequence_ids,
        torch.tensor([[1, 1, 1, 2, 2, 0, 0, 0]]),
    )
