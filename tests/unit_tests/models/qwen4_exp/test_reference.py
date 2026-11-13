from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import transformers.integrations.flash_attention as flash_integration
import transformers.modeling_flash_attention_utils as flash_utils
import transformers.modeling_utils as hf_modeling

from tests.engine.test_core import DummyBackend
from torchturbo.adapters.llamafactory_v0.patches import CreateRefModelPatch
from torchturbo.adapters.llamafactory_v0.trainer import TorchTurboDPOTrainer
from torchturbo.engine import EngineConfig, TorchTurboEngine
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.models.qwen4_exp import ple
from torchturbo.models.qwen4_exp.model import TurboQwen4ExpForConditionalGeneration
from torchturbo.runtime.cpu import CpuRuntime

from .test_model import _patch_cpu_qsa, _small_config


@pytest.fixture
def cpu_engine(monkeypatch, node_group):
    # Retain HF support checks and the real engine lifecycle on a CPU backend.
    monkeypatch.setitem(
        hf_modeling.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2],
        "general_availability_check",
        lambda: True,
    )
    monkeypatch.setattr(
        hf_modeling.PreTrainedModel, "_flash_attn_import_error", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        hf_modeling, "lazy_import_flash_attention", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        flash_integration,
        "_flash_attention_forward",
        flash_integration._flash_attention_forward,
    )
    monkeypatch.setattr(flash_utils, "_get_unpad_data", flash_utils._get_unpad_data)
    _patch_cpu_qsa(monkeypatch)
    engine = object.__new__(TorchTurboEngine)
    engine.config = EngineConfig()
    engine.backend = DummyBackend(engine.config, CpuRuntime())
    materializer = object.__new__(FSDP2Backend)
    materializer._device_runtime = CpuRuntime()
    materializer.fsdp_config = FSDPConfig()
    engine.backend.materialize = materializer.materialize
    engine.parallel_plan = None
    engine._checkpoint = None
    return engine


def _install_reference_factory(monkeypatch, config):
    class ModelArguments(SimpleNamespace):
        @classmethod
        def copyfrom(cls, args, **overrides):
            return cls(**(vars(args) | overrides))

    def noop(*args, **kwargs):
        return None

    modules = {
        "llamafactory.hparams": {"ModelArguments": ModelArguments},
        "llamafactory.model": {
            "load_config": lambda args: copy.deepcopy(config),
            "load_tokenizer": lambda args: {"tokenizer": object()},
        },
        "llamafactory.model.loader": {"_get_init_kwargs": lambda args: {}},
        "llamafactory.model.model_utils.liger_kernel": {"apply_liger_kernel": noop},
        "llamafactory.model.patcher": {"patch_config": noop, "patch_model": noop},
        "llamafactory.train.trainer_utils": {"create_ref_model": noop},
        "llamafactory.train.dpo.workflow": {"create_ref_model": noop},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return sys.modules["llamafactory.train.dpo.workflow"]


@pytest.mark.parametrize("table_dtype", [torch.float32, torch.bfloat16])
def test_default_dpo_reference_copies_registered_tables_without_source_loading(
    cpu_engine, monkeypatch, table_dtype
):
    config = _small_config()
    config.text_config.layer_types = ["full_attention"] * 4
    policy = TurboQwen4ExpForConditionalGeneration(config).eval()
    cpu_engine.model = policy
    source_ngram = policy.model.language_model.layers[1].ple.ple_embedding
    source_ngram.ngram_embedding = ple.NGramEmbedding(
        source_ngram.ngram_embedding.weight.to(table_dtype)
    )
    source_embedding = source_ngram.ngram_embedding
    sizes = (1, 0, 2, source_embedding.weight.shape[0] - 3)
    source_shards = source_embedding.weight.detach().clone().split(sizes)
    source_embedding.weight.load_(
        sizes, lambda index, target: target.copy_(source_shards[index])
    )
    source_ngram.layer_multipliers.add_(2)
    workflow = _install_reference_factory(monkeypatch, config)
    from torchturbo.engine.backends.fsdp2.checkpoint import weight_loader

    monkeypatch.setattr(
        weight_loader, "safe_open", lambda *a, **kw: pytest.fail("unexpected table I/O")
    )
    hashers = []
    from_module = ple.NGramHasher.from_module

    def record_hasher(cls, ngram):
        hasher = from_module(ngram)
        hashers.append(hasher)
        return hasher

    monkeypatch.setattr(ple.NGramHasher, "from_module", classmethod(record_hasher))
    patch = CreateRefModelPatch(SimpleNamespace(stage="dpo"))
    patch.patch(cpu_engine)
    finetuning_args = SimpleNamespace(ref_model=None)
    try:
        reference = workflow.create_ref_model(
            SimpleNamespace(
                model_name_or_path="/unused-source-checkpoint",
                compute_dtype=torch.float32,
            ),
            finetuning_args,
        )
    finally:
        patch.revert()

    target_ngram = reference.model.language_model.layers[1].ple.ple_embedding
    target_embedding = target_ngram.ngram_embedding
    assert target_embedding.weight is not source_embedding.weight
    assert target_embedding.weight.data_ptr() != source_embedding.weight.data_ptr()
    assert target_embedding.weight.device.type == "cpu"
    assert cpu_engine.model is policy
    assert cpu_engine.backend._checkpoint.calls == []
    assert hashers == []

    trainer = object.__new__(TorchTurboDPOTrainer)
    trainer.model = policy
    trainer.ref_model = reference
    trainer.finetuning_args = finetuning_args
    trainer._initialize_dpo_ref_model()
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    assert any(parameter.requires_grad for parameter in policy.parameters())
    assert target_embedding.weight.dtype == source_embedding.weight.dtype == table_dtype
    assert target_embedding.weight.shard_sizes == sizes
    target_state = reference.state_dict()
    for name, source in policy.state_dict().items():
        target = target_state[name]
        torch.testing.assert_close(target, source)
        assert (
            target.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        )
    assert hashers == []

    inputs = {
        "input_ids": torch.tensor([[3, 4, 2, 7, 8, 9]]),
        "cu_seq_lens_q": torch.tensor([0, 2, 6], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 6], dtype=torch.int32),
        "max_length_q": 4,
        "max_length_k": 4,
        "use_cache": False,
    }
    with torch.no_grad():
        expected = policy(**inputs).logits
        actual = reference(**inputs).logits
    torch.testing.assert_close(actual, expected)
    assert len(hashers) == 2
    assert hashers[0] is not hashers[1]
    source_rows = source_embedding.weight.clone()
    target_embedding.weight.add_(1)
    torch.testing.assert_close(source_embedding.weight, source_rows)
    assert not torch.equal(target_embedding.weight, source_embedding.weight)


@pytest.mark.parametrize(
    "wrap_source,wrap_target", [(True, False), (False, True), (True, True)]
)
def test_registered_table_copy_supports_peft_wrappers(
    wrap_source, wrap_target
):
    peft = pytest.importorskip("peft")
    source = TurboQwen4ExpForConditionalGeneration(_small_config())
    with torch.device("meta"):
        target = TurboQwen4ExpForConditionalGeneration(_small_config())
    source_embedding = source.model.language_model.layers[
        1
    ].ple.ple_embedding.ngram_embedding
    target_embedding = target.model.language_model.layers[
        1
    ].ple.ple_embedding.ngram_embedding
    lora_config = peft.LoraConfig(target_modules=["q_proj"], r=2)
    if wrap_source:
        source = peft.get_peft_model(source, lora_config)
    if wrap_target:
        target = peft.get_peft_model(target, lora_config)
    target.to_empty(device="cpu")
    target.on_materialized()
    target_embedding.load_state_dict(source_embedding.state_dict())
    assert target_embedding.weight is not source_embedding.weight
    assert target_embedding.weight.data_ptr() != source_embedding.weight.data_ptr()
    torch.testing.assert_close(target_embedding.weight, source_embedding.weight)
    assert all(not parameter.requires_grad for parameter in target_embedding.parameters())


def test_source_free_engine_build_materializes_registered_cpu_tables(cpu_engine):
    model = cpu_engine.build_model(
        "/unused-source-checkpoint",
        init_kwargs={"config": _small_config()},
        load_weights=False,
    )
    embedding = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding
    assert not embedding.weight.is_meta
    assert set(embedding.state_dict()) == {"weight"}
    assert all(parameter.device.type == "cpu" for parameter in embedding.parameters())
    assert all(not parameter.requires_grad for parameter in embedding.parameters())
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert cpu_engine.backend._checkpoint.calls == []
