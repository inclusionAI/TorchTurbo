"""CPU contracts for dense MTP using the established MoE recurrence."""

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import modeling_flash_attention_utils as flash_utils
from transformers.integrations import flash_attention as flash_integration

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import LossChunkConfig
from torchturbo.models.modules.attention.interface import attention_forward
from torchturbo.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5DecoderLayer,
    Qwen3_5MLP,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5MTPLayer,
    Qwen3_5MultiTokenPredictor,
    Qwen3_5TextRotaryEmbedding,
)
from torchturbo.models.qwen3_5.parallel_plan import get_parallel_plan
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMultiTokenPredictor
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

from ..flash_attention_fakes import lazy_flash_attention_kernels, process_flash_kwargs


def _config(layers=2):
    config = Qwen3_5TextConfig(
        vocab_size=17, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, layer_types=["full_attention"], mtp_num_hidden_layers=layers,
        pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [1, 1, 0]},
    )
    config._attn_implementation = "eager"
    return config


class _Backbone(nn.Module):
    """Replace only the expensive VLM backbone, not the MTP blocks."""

    def __init__(self, config):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.hidden = nn.Parameter(torch.randn(1, 5, config.hidden_size))
        self.input_ids = None

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.input_ids = input_ids
        return Qwen3_5ModelOutputWithPast(last_hidden_state=self.hidden)


def _model(layers=2):
    config = _config(layers)
    model = object.__new__(TurboQwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(return_dict=True, text_config=config)
    model.model = _Backbone(config)
    model.mtp = Qwen3_5MultiTokenPredictor(config) if layers else None
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    return model.train()


def _batch():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    return {"input_ids": ids, "labels": torch.tensor([[-100, 2, 3, -100, 5]]),
            "position_ids": torch.arange(5).view(1, 5)}


def _backend(chunked, scale):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.rank = 0
    backend.config = SimpleNamespace(
        loss_chunk=LossChunkConfig(enabled=chunked),
        mtp=SimpleNamespace(enabled=True, loss_scaling_factor=scale, log_loss=False),
    )
    backend.get_sp_info = lambda: {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    return backend


@pytest.fixture
def cpu_ops(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())


@pytest.mark.parametrize("layers", [0, 1, 2])
def test_constructor_and_checkpoint_roundtrip(layers):
    config = Qwen3_5Config(
        text_config=_config(layers).to_dict(),
        vision_config={"depth": 1, "hidden_size": 8, "intermediate_size": 16,
                       "num_heads": 2, "out_hidden_size": 8},
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config)
    if not layers:
        assert model.mtp is None
        assert not any(key.startswith("mtp.") for key in model.state_dict())
        return
    assert len(model.mtp.layers) == layers
    for index, layer in enumerate(model.mtp.layers):
        assert isinstance(layer.mlp, Qwen3_5MLP)
        assert layer.layer_type == "full_attention"
        assert layer.self_attn.layer_idx == config.text_config.num_hidden_layers + index
    keys = model.state_dict()
    for name in ("fc", "norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden",
                 "layers.0.mlp.gate_proj", "layers.0.self_attn.q_proj"):
        assert f"mtp.{name}.weight" in keys
    restored = Qwen3_5ForConditionalGeneration(config)
    restored.load_state_dict(keys, strict=True)
    for key in keys:
        torch.testing.assert_close(restored.state_dict()[key], keys[key])
    assert get_parallel_plan().plan["mtp.layers.*"].activation_checkpoint


def test_recurrence_shifts_ids_and_preserves_embedding_first_order():
    model = _model()
    seen_ids, seen_fc = [], []
    model.model.language_model.embed_tokens.register_forward_pre_hook(
        lambda module, args: seen_ids.append(args[0].clone())
    )
    model.mtp.fc.register_forward_pre_hook(
        lambda module, args: seen_fc.append(args[0].detach().clone())
    )
    batch = _batch()
    batch.pop("labels")
    outputs = model(**batch, inputs_embeds=torch.randn(1, 5, 8))
    assert model.model.input_ids is None
    assert [ids.tolist() for ids in seen_ids] == [[[2, 3, 4, 5, 0]], [[3, 4, 5, 0, 0]]]
    embeddings = model.model.language_model.embed_tokens(seen_ids[0])
    torch.testing.assert_close(seen_fc[0][..., :8], model.mtp.pre_fc_norm_embedding(embeddings))
    torch.testing.assert_close(seen_fc[0][..., 8:], model.mtp.pre_fc_norm_hidden(model.model.hidden))
    torch.testing.assert_close(seen_fc[1][..., 8:], model.mtp.pre_fc_norm_hidden(outputs.mtp_hidden_states[0]))
    assert [logits.shape for logits in outputs.mtp_logits] == [(1, 5, 17)] * 2


@pytest.mark.parametrize("scale", [0.0, 0.2])
def test_chunked_and_regular_loss_and_gradients_match(cpu_ops, scale):
    torch.manual_seed(7)
    regular = _model()
    chunked = copy.deepcopy(regular)
    reference_batch = _batch()
    labels = reference_batch.pop("labels")
    reference_outputs = regular(**reference_batch)
    reference_losses = []
    for shift, logits in enumerate([reference_outputs.logits, *reference_outputs.mtp_logits], 1):
        shifted = F.pad(labels, (0, shift), value=-100)[..., shift:]
        reference_losses.append(F.cross_entropy(logits.flatten(0, 1), shifted.flatten()))
    expected = reference_losses[0] + scale * sum(reference_losses[1:])
    loss, outputs = _backend(False, scale).compute_loss(regular, _batch(), return_outputs=True)
    chunk_loss, chunk_outputs = _backend(True, scale).compute_loss(chunked, _batch(), return_outputs=True)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(chunk_loss, expected)
    assert chunk_outputs.logits is None
    assert chunk_outputs.mtp_logits is None
    for actual, ref in zip(outputs.mtp_loss, reference_losses[1:]):
        torch.testing.assert_close(actual, ref)
    loss.backward()
    chunk_loss.backward()
    for (name, param), (other_name, other) in zip(regular.named_parameters(), chunked.named_parameters()):
        assert name == other_name
        if param.grad is None:
            assert other.grad is None
        elif other.grad is None:
            assert torch.count_nonzero(param.grad) == 0
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=2e-4, atol=2e-6)
    if scale:
        for name in ("model.hidden", "model.language_model.embed_tokens.weight", "lm_head.weight",
                     "mtp.fc.weight", "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.1.self_attn.q_proj.weight"):
            grad = dict(regular.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("training,layers", [(False, 2), (True, 0)])
def test_eval_and_disabled_mtp_leave_main_output_unchanged(training, layers):
    model = _model(layers)
    model.train(training)
    batch = _batch()
    batch.pop("labels")
    outputs = model(**batch)
    assert outputs.mtp_logits is None
    assert outputs.mtp_hidden_states is None
    torch.testing.assert_close(outputs.logits, model.lm_head(model.model.hidden))


@pytest.mark.parametrize("local_ids", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_sp_shifts_global_ids_before_local_embedding_slice(monkeypatch, local_ids, rank):
    ulysses = importlib.import_module("torchturbo.models.modules.attention.ulysses")
    original = Qwen3_5MultiTokenPredictor.forward
    moe_original = Qwen3_5MoeMultiTokenPredictor.forward
    monkeypatch.setattr(Qwen3_5MultiTokenPredictor, "forward", original)
    monkeypatch.setattr(ulysses, "slice_input_tensor",
                        lambda tensor, dim, group, rank: tensor.chunk(2, dim=dim)[rank])
    global_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def gather(outputs, inputs, group):
        for output, ids in zip(outputs, global_ids.chunk(2, dim=1)):
            output.copy_(ids)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    TurboQwen3_5MoeForConditionalGeneration.patch_mtp_for_sp(Qwen3_5MultiTokenPredictor, None, 2, rank)
    assert Qwen3_5MoeMultiTokenPredictor.forward is moe_original
    predictor = Qwen3_5MultiTokenPredictor(_config())
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    # Isolate the shift/slice contract from cross-rank attention communication.
    class IdentityLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            assert kwargs["position_ids"].shape[-1] == 3
            assert kwargs["position_embeddings"][0].shape[1] == 3
            assert kwargs["cu_seq_lens_q"] is cu
            assert kwargs["cu_seq_lens_k"] is cu
            assert kwargs["max_length_q"] == kwargs["max_length_k"] == 3
            return hidden_states

    predictor.layers = nn.ModuleList([IdentityLayer(), IdentityLayer()])
    seen = []
    predictor.pre_fc_norm_embedding.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].clone())
    )
    predictor(
        global_ids.chunk(2, dim=1)[rank] if local_ids else global_ids,
        torch.randn(1, 3, 8),
        lambda ids: ids.unsqueeze(-1).expand(-1, -1, 8).float(),
        position_ids=torch.arange(6).view(1, 6),
        position_embeddings=(torch.ones(1, 6, 4), torch.zeros(1, 6, 4)),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=3,
        max_length_k=3,
    )
    expected = ([[2, 3, 4], [3, 4, 5]] if rank == 0 else [[5, 6, 0], [6, 0, 0]])
    assert [tensor[0, :, 0].tolist() for tensor in seen] == expected


@pytest.mark.parametrize("axes", [2, 3, 4])
def test_position_rows_and_packed_metadata_reach_mtp(monkeypatch, axes):
    model = _model(1)
    text_positions = torch.tensor([[0, 1, 0, 1, 2]])
    rope_positions = torch.stack((text_positions + 2, text_positions + 4, text_positions + 6))
    positions = (text_positions if axes == 2 else rope_positions if axes == 3
                 else torch.cat((text_positions.unsqueeze(0), rope_positions)))
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    captured = {}

    def capture(input_ids, hidden_states, embed_tokens, **kwargs):
        captured.update(kwargs)
        return [hidden_states]

    monkeypatch.setattr(model.mtp, "forward", capture)
    model(input_ids=_batch()["input_ids"], position_ids=positions,
          cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=3, max_length_k=3)
    # Three-axis inputs use the temporal row for attention; four-axis inputs
    # carry a separate text row before the three rotary coordinates.
    expected_text_positions = rope_positions[0] if axes == 3 else text_positions
    torch.testing.assert_close(captured["position_ids"], expected_text_positions)
    expected_rope = model.model.language_model.rotary_emb(
        model.model.hidden, text_positions if axes == 2 else rope_positions
    )
    for actual, expected in zip(captured["position_embeddings"], expected_rope):
        torch.testing.assert_close(actual, expected)
    assert captured["cu_seq_lens_q"] is cu
    assert captured["cu_seq_lens_k"] is cu
    assert captured["max_length_q"] == captured["max_length_k"] == 3


def test_dense_mtp_matches_backbone_with_position_resets_inside_cu(monkeypatch):
    torch.manual_seed(13)
    config = _config(1)
    config._attn_implementation = "flash_attention_2"
    backbone_layer = Qwen3_5DecoderLayer(config, layer_idx=0)
    mtp_layer = Qwen3_5MTPLayer(config, layer_idx=1)
    mtp_layer.load_state_dict(backbone_layer.state_dict())
    hidden = torch.randn(1, 6, config.hidden_size)
    positions = torch.tensor([[0, 1, 0, 1, 0, 1]])
    cu = torch.tensor([0, 6], dtype=torch.int32)
    observed_metadata = []

    def varlen_attention(query, key, value, **kwargs):
        observed_metadata.append(kwargs)
        query_bounds = kwargs["cu_seqlens_q"].tolist()
        key_bounds = kwargs["cu_seqlens_k"].tolist()
        outputs = []
        for qs, qe, ks, ke in zip(
            query_bounds[:-1], query_bounds[1:], key_bounds[:-1], key_bounds[1:],
        ):
            output = F.scaled_dot_product_attention(
                query[qs:qe].transpose(0, 1).unsqueeze(0),
                key[ks:ke].transpose(0, 1).unsqueeze(0),
                value[ks:ke].transpose(0, 1).unsqueeze(0),
                is_causal=kwargs["causal"],
                scale=kwargs["softmax_scale"],
                enable_gqa=True,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setattr(flash_integration, "_flash_attention_forward", attention_forward)
    monkeypatch.setattr(
        flash_utils, "lazy_import_flash_attention",
        lambda *args, **kwargs: (
            lazy_flash_attention_kernels(None, varlen_attention),
            process_flash_kwargs,
        ),
    )
    inputs = {
        "position_embeddings": Qwen3_5TextRotaryEmbedding(config)(hidden, positions),
        "position_ids": positions,
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": 6,
        "max_length_k": 6,
    }

    expected = backbone_layer(hidden, **inputs)
    actual = mtp_layer(hidden, **inputs)

    torch.testing.assert_close(actual, expected)
    assert len(observed_metadata) == 2
    for metadata in observed_metadata:
        assert metadata["cu_seqlens_q"] is cu
        assert metadata["cu_seqlens_k"] is cu
        assert metadata["max_seqlen_q"] == metadata["max_seqlen_k"] == 6
