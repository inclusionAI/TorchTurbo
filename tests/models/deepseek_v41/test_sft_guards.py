"""Guards for the first DeepSeek V4.1 small-operator SFT path."""

import json
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeek V4.1 SFT guard tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safetensors.torch import save_file  # noqa: E402

from torchturbo.models.deepseek_v41.configuration_deepseek_v41 import (  # noqa: E402
    DeepseekV41TextConfig,
)
from torchturbo.models.deepseek_v41.model import TurboDeepseekV41ForCausalLM  # noqa: E402
from torchturbo.models.deepseek_v41.modeling_deepseek_v41 import (  # noqa: E402
    DeepseekV41Indexer,
    DeepseekV41MoE,
    HostOffloadEngram,
)


def _text_config() -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        compress_ratios=[1],
        index_n_heads=2,
        index_head_dim=8,
        hc_mult=2,
        engram_layer_ids=[0],
        engram_num_embeddings=[64],
        engram_max_ngram_size=2,
        engram_n_heads=2,
        engram_head_dim=4,
    )


def test_moe_keeps_layer_idx_for_swiglu_limit():
    moe = DeepseekV41MoE(_text_config(), layer_idx=3, vision_enabled=False)
    assert moe.layer_idx == 3


def test_sp_greater_than_one_requires_npu(monkeypatch):
    from torchturbo import runtime

    monkeypatch.setattr(runtime, "is_npu_available", lambda: False)
    model = torch.nn.Module()
    from types import MethodType

    model.on_parallelized = MethodType(TurboDeepseekV41ForCausalLM.on_parallelized, model)
    try:
        model.on_parallelized({"sp": {"sp_size": 2, "sp_rank": 0, "sp_group": None}})
    except NotImplementedError as exc:
        assert "sp_size=2" in str(exc)
    else:
        raise AssertionError("sp_size=2 should be rejected")


def _indexer_holder(*, patched, coeff=0.0):
    """A minimal stand-in for the turbo model: indexer module plus the two
    attributes _freeze_indexers reads directly (loud-by-contract)."""
    from types import SimpleNamespace

    holder = torch.nn.Module()
    holder.indexer = DeepseekV41Indexer(
        _text_config(), compress_ratio=1, owns_k=True, is_candidate_source=True
    )
    holder._npu_sfa_li_patched = patched
    holder.config = SimpleNamespace(
        text_config=SimpleNamespace(indexer_kl_loss_coeff=coeff)
    )
    return holder


def test_indexer_parameters_are_frozen():
    holder = _indexer_holder(patched=False)
    assert any(param.requires_grad for param in holder.indexer.parameters())
    TurboDeepseekV41ForCausalLM._freeze_indexers(holder)
    assert all(not param.requires_grad for param in holder.indexer.parameters())


def test_indexer_parameters_unfrozen_with_kl_coeff():
    # Bundle installed AND a positive KL coefficient -> indexer trains.
    holder = _indexer_holder(patched=True, coeff=1e-3)
    TurboDeepseekV41ForCausalLM._freeze_indexers(holder)
    assert all(param.requires_grad for param in holder.indexer.parameters())


def test_indexer_parameters_frozen_without_kl_coeff():
    # Bundle installed but coeff 0 -> pure forward path, indexer stays frozen.
    holder = _indexer_holder(patched=True, coeff=0.0)
    TurboDeepseekV41ForCausalLM._freeze_indexers(holder)
    assert all(not param.requires_grad for param in holder.indexer.parameters())


def test_indexer_parameters_refrozen_when_kl_patch_missing():
    # Before NPU installation, the indexer stays frozen even with a positive coefficient.
    holder = _indexer_holder(patched=False, coeff=1e-3)
    TurboDeepseekV41ForCausalLM._freeze_indexers(holder)
    assert all(not param.requires_grad for param in holder.indexer.parameters())


def test_indexer_kl_coeff_cli_override(monkeypatch):
    # --tt_indexer_kl_loss_coeff (parsed args) beats the config field: it is
    # folded into text_config at model build time (_pre_init_patches), and
    # every later KL decision (freeze, patching) reads the merged field.
    from types import SimpleNamespace

    from torchturbo.engine.args import TorchTurboArgs

    monkeypatch.setattr(
        "torchturbo.engine.args._current_torchturbo_args",
        TorchTurboArgs(indexer_kl_loss_coeff=1e-3),
    )
    config = SimpleNamespace(text_config=SimpleNamespace(indexer_kl_loss_coeff=0.0))
    TurboDeepseekV41ForCausalLM._pre_init_patches(config)
    assert config.text_config.indexer_kl_loss_coeff == 1e-3
    holder = _indexer_holder(patched=True, coeff=config.text_config.indexer_kl_loss_coeff)
    TurboDeepseekV41ForCausalLM._freeze_indexers(holder)
    assert all(param.requires_grad for param in holder.indexer.parameters())

    # Without parsed args (or without the flag) the config field is untouched.
    monkeypatch.setattr("torchturbo.engine.args._current_torchturbo_args", None)
    config = SimpleNamespace(text_config=SimpleNamespace(indexer_kl_loss_coeff=0.25))
    TurboDeepseekV41ForCausalLM._pre_init_patches(config)
    assert config.text_config.indexer_kl_loss_coeff == 0.25


def test_indexer_kl_loss_coeff_default_and_override():
    assert DeepseekV41TextConfig().indexer_kl_loss_coeff == 0.0
    assert DeepseekV41TextConfig(indexer_kl_loss_coeff=0.5).indexer_kl_loss_coeff == 0.5


def test_mhc_installation_requires_kernel_and_supported_config(monkeypatch):
    import importlib
    import pytest
    from torchturbo.models.deepseek_v41.npu import patch_deepseek_v41_npu_mhc

    provider = importlib.import_module("torchturbo.ops.providers.npu.mhc_pre_sinkhorn_premix")
    monkeypatch.setattr(provider, "has_mhc_pre_sinkhorn_premix", lambda: False)
    with pytest.raises(RuntimeError, match="mhc_pre_sinkhorn_premix"):
        patch_deepseek_v41_npu_mhc(torch.nn.Module())
    monkeypatch.setattr(provider, "has_mhc_pre_sinkhorn_premix", lambda: True)
    for field, value in (("hidden_size", 4097), ("hc_mult", 8), ("hc_sinkhorn_iters", 19)):
        model = torch.nn.Module()
        model.config = DeepseekV41TextConfig(**{field: value})
        with pytest.raises(ValueError, match=field):
            patch_deepseek_v41_npu_mhc(model)


def test_sfa_li_degrades_to_reference_without_sparse_kernel(monkeypatch):
    import importlib
    import pytest

    from torchturbo.models.deepseek_v41.npu import patch_deepseek_v41_npu_sfa_li

    li = importlib.import_module("torchturbo.ops.providers.npu.lightning_indexer_v2")
    sparse = importlib.import_module("torchturbo.ops.providers.npu.sparse_lightning_indexer")
    kl = importlib.import_module(
        "torchturbo.ops.providers.npu.sparse_lightning_indexer_kl_loss_grad"
    )
    monkeypatch.setattr(li, "has_lightning_indexer_v2", lambda: True)
    monkeypatch.setattr(kl, "has_sli_kl_loss_grad", lambda: True)
    monkeypatch.setattr(sparse, "has_sparse_lightning_indexer", lambda: False)

    holder = _indexer_holder(patched=False)
    reference = holder.indexer._select_blocks
    # No raise: the fused block selection degrades to the reference path.
    patch_deepseek_v41_npu_sfa_li(holder)
    assert holder.indexer._select_blocks == reference

    monkeypatch.setattr(sparse, "has_sparse_lightning_indexer", lambda: True)
    patch_deepseek_v41_npu_sfa_li(holder)
    assert holder.indexer._select_blocks != reference

    monkeypatch.setattr(li, "has_lightning_indexer_v2", lambda: False)
    with pytest.raises(RuntimeError, match="lightning_indexer_v2"):
        patch_deepseek_v41_npu_sfa_li(holder)


def _write_engram_shard(directory: Path, engram: HostOffloadEngram) -> dict[str, torch.Tensor]:

    # tmp_path already exists when the fixture hands it over; mkdir must be
    # idempotent or these tests raise FileExistsError on every run.
    directory.mkdir(parents=True, exist_ok=True)
    # Build from shape/dtype so this also works while the module is still meta.
    payload = {
        f"layers.{engram.layer_idx}.engram.wkv.weight": torch.arange(
            engram.wkv.weight.numel(), dtype=engram.wkv.weight.dtype
        ).reshape(tuple(engram.wkv.weight.shape)),
        f"layers.{engram.layer_idx}.engram.q_weight": torch.full(
            tuple(engram.q_weight.shape), 0.25, dtype=engram.q_weight.dtype
        ),
        f"layers.{engram.layer_idx}.engram.k_weight": torch.full(
            tuple(engram.k_weight.shape), 0.5, dtype=engram.k_weight.dtype
        ),
    }
    filename = "model-00001-of-00001.safetensors"
    save_file(payload, directory / filename)
    index = {"weight_map": {key: filename for key in payload}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    return payload


def test_engram_load_params_replaces_meta_tensors(tmp_path):
    with torch.device("meta"):
        engram = HostOffloadEngram(_text_config(), layer_idx=0, layer_hash_index=0)
    assert engram.wkv.weight.is_meta
    expected = _write_engram_shard(tmp_path, engram)

    engram.load_params(str(tmp_path), device=torch.device("cpu"))

    assert not engram.wkv.weight.is_meta
    assert engram.wkv.weight.device.type == "cpu"
    torch.testing.assert_close(
        engram.wkv.weight, expected["layers.0.engram.wkv.weight"]
    )
    torch.testing.assert_close(engram.q_weight, expected["layers.0.engram.q_weight"])
    torch.testing.assert_close(engram.k_weight, expected["layers.0.engram.k_weight"])


def test_engram_load_params_copies_into_real_tensors(tmp_path):
    engram = HostOffloadEngram(_text_config(), layer_idx=0, layer_hash_index=0)
    assert not engram.wkv.weight.is_meta
    expected = _write_engram_shard(tmp_path, engram)

    engram.load_params(str(tmp_path), device=torch.device("cpu"))

    torch.testing.assert_close(
        engram.wkv.weight, expected["layers.0.engram.wkv.weight"]
    )
