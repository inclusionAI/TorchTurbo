"""Public engine lifecycle adapter checks for the internal V4.1 implementation."""
import json
import sys
import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from torchturbo import EngineConfig, TorchTurboEngine
from torchturbo.engine.args import TorchTurboArgs, set_torchturbo_args
from torchturbo.models.deepseek_v41 import DeepseekV41Config, TurboDeepseekV41ForCausalLM
from torchturbo.models.deepseek_v41.host_offload_embedding import TorchHostOffloadEmbedding

@pytest.fixture(autouse=True)
def cpu_host_allocation(monkeypatch):
    # Pinned allocations select the unavailable MPS allocator on this CPU host.
    original = TorchHostOffloadEmbedding.__init__
    def initialize(self, *args, **kwargs):
        kwargs["pin_memory"] = False
        original(self, *args, **kwargs)
    monkeypatch.setattr(TorchHostOffloadEmbedding, "__init__", initialize)
    yield
    set_torchturbo_args(TorchTurboArgs())

@pytest.fixture
def checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "pretrained"
    checkpoint_dir.mkdir()
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
    vocab.update({f"word{i}": i for i in range(3, 32)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    config = DeepseekV41Config(
        vision_config={"num_hidden_layers": 0},
        text_config={
            "vocab_size": len(vocab),
            "hidden_size": 16,
            "moe_intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 8,
            "o_lora_rank": 4,
            "o_groups": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "gate_temp": 1.4,
            "sliding_window": 2,
            "compress_ratios": [2, 2, 2],
            "kv_source_layer_ids": [0],
            "index_source_layer_ids": [0, 1],
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_topk": 2,
            "candidate_source_layer_id": 0,
            "candidate_topk_blocks": 1,
            "candidate_block_size": 1,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 2,
            "rope_scaling": {"factor": 2, "original_max_position_embeddings": 8},
            "engram_layer_ids": [0],
            "engram_num_embeddings": [32],
            "engram_max_ngram_size": 2,
            "engram_vocab_size": 7,
            "engram_n_heads": 1,
            "engram_head_dim": 4,
            "engram_compressed_vocab_size": len(vocab),
        },
    )
    torch.manual_seed(41)
    model = AutoModelForCausalLM.from_config(config)
    assert isinstance(model, TurboDeepseekV41ForCausalLM)
    config.save_pretrained(checkpoint_dir)
    weights = model.state_dict()
    filename = "model-00001-of-00001.safetensors"
    save_file(weights, checkpoint_dir / filename, metadata={"format": "pt"})
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(weights, filename)})
    )
    return checkpoint_dir


@pytest.mark.parametrize("train_table", [False, True])
def test_internal_engine_checkpoint_lifecycle(checkpoint, tmp_path, monkeypatch, train_table):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    set_torchturbo_args(TorchTurboArgs(train_engram=train_table))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    try:
        engine = TorchTurboEngine(EngineConfig.from_dict({"mixed_precision":"fp32", "ops":{"policy":"auto"}}))
        model = engine.prepare_model(str(checkpoint), torch_dtype=torch.float32)
        table = model.layers[0].engram.embed
        assert table.weight.dtype == torch.bfloat16
        assert table.local_rows == 2048 and table.logical_num_embeddings == 32
        assert table.weight.device.type == "cpu" and table.weight.requires_grad == train_table
        assert model.layers[0].engram.q_weight.dtype == torch.float32
        optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=0.01, weight_decay=0.0)
        assert all(p is not table.weight for g in optimizer.param_groups for p in g["params"])
        ids = torch.tensor([[3,4,5,6,7]])
        batch = {"input_ids":ids, "attention_mask":torch.ones_like(ids), "labels":ids}
        before = table.weight.detach().clone()
        engine.compute_loss(model,batch).backward()
        assert bool(table.pending_gradients) == train_table
        engine.backend.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        assert (not torch.equal(before, table.weight)) == train_table
        saved = str(tmp_path / "saved")
        engine.save_model(model,saved)
        engine.save_optimizer_and_scheduler(model,optimizer,None,saved)
        if train_table:
            assert (tmp_path / "saved" / "engram_optimizer_ep0.pt").is_file()
        engine.compute_loss(model,batch).backward()
        optimizer.step()
        optimizer.zero_grad()
        expected = table.weight.detach().clone()
        engine.load_from_checkpoint(model,saved)
        engine.load_optimizer_and_scheduler(model,optimizer,None,saved)
        engine.compute_loss(model,batch).backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.testing.assert_close(table.weight,expected,rtol=0,atol=0)
        if train_table:
            table.weight.requires_grad_(False)
            before = table.weight.detach().clone()
            engine.compute_loss(model,batch).backward()
            assert not table.pending_gradients
            optimizer.step()
            optimizer.zero_grad()
            torch.testing.assert_close(table.weight,before,rtol=0,atol=0)
    finally:
        dist.destroy_process_group()


def test_recursive_engram_freeze_disables_source_table_training(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    set_torchturbo_args(TorchTurboArgs(train_engram=True))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    try:
        config = EngineConfig.from_dict({
            "mixed_precision": "fp32",
            "freeze_policy": {"patterns": ["layers.*.engram.**"]},
        })
        engine = TorchTurboEngine(config)
        model = engine.prepare_model(str(checkpoint), torch_dtype=torch.float32)
        optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=0.01, weight_decay=0.0)
        engram = model.layers[0].engram
        assert all(not p.requires_grad for p in engram.parameters())
        assert engine.backend.engram_optimizer is None
        before = engram.embed.weight.detach().clone()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        engine.compute_loss(model, {"input_ids": ids, "labels": ids}).backward()
        optimizer.step()
        assert all(not p.requires_grad for p in engram.parameters())
        assert not engram.embed.pending_gradients
        torch.testing.assert_close(engram.embed.weight, before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("freeze_phase", ["before_materialization", "after_policy"])
def test_engram_table_freeze_survives_materialization(checkpoint, tmp_path, monkeypatch, freeze_phase):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    set_torchturbo_args(TorchTurboArgs(train_engram=True))
    policy = {}
    if freeze_phase == "before_materialization":
        original = TurboDeepseekV41ForCausalLM.on_materialized

        def materialize(model):
            # Parameter freezing by PEFT happens before host storage exists.
            table = model.layers[0].engram.embed
            assert table.weight.is_meta
            table.requires_grad_(False)
            original(model)

        monkeypatch.setattr(TurboDeepseekV41ForCausalLM, "on_materialized", materialize)
    else:
        policy = {"patterns": ["layers.*.engram.**"]}
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    try:
        engine = TorchTurboEngine(EngineConfig.from_dict({
            "mixed_precision": "fp32", "freeze_policy": policy,
        }))
        model = engine.prepare_model(str(checkpoint), torch_dtype=torch.float32)
        if freeze_phase == "after_policy":
            # A lifecycle replay must not override the explicit engine policy.
            assert not model.layers[0].engram.embed.weight.requires_grad
            model.on_materialized()
        table = model.layers[0].engram.embed
        assert not table.weight.requires_grad
        optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=0.01, weight_decay=0.0)
        assert engine.backend.engram_optimizer is None
        before = table.weight.detach().clone()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        engine.compute_loss(model, {"input_ids": ids, "labels": ids}).backward()
        optimizer.step()
        assert not table.pending_gradients
        torch.testing.assert_close(table.weight, before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("train_table", [False, True])
def test_host_hf_export_and_reload(checkpoint, tmp_path, monkeypatch, train_table):
    from torchturbo.engine.checkpoint.format import CheckpointFormat

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    set_torchturbo_args(TorchTurboArgs(train_engram=train_table))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    try:
        engine = TorchTurboEngine(EngineConfig.from_dict({"mixed_precision": "fp32"}))
        model = engine.prepare_model(str(checkpoint), torch_dtype=torch.float32)
        table = model.layers[0].engram.embed
        expected_rows = table.weight[:table.logical_num_embeddings].detach().clone()
        expected_gate = model.layers[0].engram.q_weight.detach().clone()
        export = tmp_path / "hf_export"
        engine.save_model(model, str(export), format=CheckpointFormat.HF)
        hf_dir = export / "huggingface"
        index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
        key = "layers.0.engram.embed.weight"
        with safe_open(hf_dir / index["weight_map"][key], framework="pt") as handle:
            exported = handle.get_tensor(key)
        assert exported.shape == (table.logical_num_embeddings, table.embedding_dim)
        assert exported.dtype == torch.bfloat16
        torch.testing.assert_close(exported, expected_rows, rtol=0, atol=0)
        with torch.no_grad():
            table.weight.zero_()
            model.layers[0].engram.q_weight.zero_()
        engine.load_from_checkpoint(model, str(export))
        torch.testing.assert_close(table.weight[:table.logical_num_embeddings], expected_rows, rtol=0, atol=0)
        torch.testing.assert_close(model.layers[0].engram.q_weight, expected_gate, rtol=0, atol=0)
        assert table.weight.requires_grad == train_table
    finally:
        dist.destroy_process_group()


def test_lora_keeps_engram_host_table_frozen(checkpoint, tmp_path, monkeypatch):
    from peft import LoraConfig

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    set_torchturbo_args(TorchTurboArgs(train_engram=True))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/dist", rank=0, world_size=1)
    try:
        engine = TorchTurboEngine(EngineConfig.from_dict({"mixed_precision": "fp32"}))
        model = engine.prepare_model(
            str(checkpoint), torch_dtype=torch.float32,
            peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["wq_a"]),
        )
        table = model.layers[0].engram.embed
        assert not table.weight.requires_grad
        optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=0.01, weight_decay=0.0)
        assert engine.backend.engram_optimizer is None
        before = table.weight.detach().clone()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        engine.compute_loss(model, {"input_ids": ids, "labels": ids}).backward()
        assert any(p.grad is not None for name, p in model.named_parameters() if "lora_B" in name)
        optimizer.step()
        assert not table.pending_gradients
        torch.testing.assert_close(table.weight, before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
