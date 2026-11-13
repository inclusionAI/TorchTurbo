"""Tests for standalone LoRA adapter merging."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "checkpoint" / "merge_lora_adapter.py"
)
_SPEC = importlib.util.spec_from_file_location("merge_lora_adapter", _SCRIPT)
merge_lora_adapter = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = merge_lora_adapter
_SPEC.loader.exec_module(merge_lora_adapter)


def _write_adapter(
    root: Path,
    tensors: dict[str, torch.Tensor],
    base_model_name_or_path: str | None,
) -> None:
    root.mkdir(parents=True)
    save_file(tensors, root / "adapter_model.safetensors")
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "use_rslora": False,
                "use_dora": False,
                "base_model_name_or_path": base_model_name_or_path,
            }
        )
    )


def _merge_and_load(tmp_path: Path, base: dict[str, torch.Tensor], adapter: dict):
    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "output"
    base_dir.mkdir()
    save_file(base, base_dir / "model.safetensors")
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in base}})
    )
    (base_dir / "config.json").write_text('{"model_type": "test"}')
    _write_adapter(adapter_dir, adapter, str(base_dir))
    merge_lora_adapter.main(
        [
            "--adapter-dir",
            str(adapter_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    return load_file(output_dir / "model.safetensors"), output_dir


def test_merges_standard_linear_lora(tmp_path):
    base_weight = torch.randn(3, 4, dtype=torch.bfloat16)
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    merged, output_dir = _merge_and_load(
        tmp_path,
        {"model.q_proj.weight": base_weight},
        {
            "base_model.model.model.q_proj.lora_A.weight": a,
            "base_model.model.model.q_proj.lora_B.weight": b,
        },
    )
    expected = (base_weight.float() + 2.0 * (b @ a)).to(torch.bfloat16)
    assert torch.equal(merged["model.q_proj.weight"], expected)
    assert (output_dir / "config.json").is_file()


def test_merges_fused_grouped_moe_lora(tmp_path):
    base_gate_up = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    base_down = torch.randn(2, 3, 2, dtype=torch.bfloat16)
    a = torch.randn(2, 3, 2)
    b = torch.randn(2, 2, 4)
    a2 = torch.randn(2, 2, 2)
    b2 = torch.randn(2, 2, 3)
    prefix = "model.mlp.experts"
    merged, _ = _merge_and_load(
        tmp_path,
        {
            f"{prefix}.gate_up_proj.weight": base_gate_up,
            f"{prefix}.down_proj.weight": base_down,
        },
        {
            f"base_model.model.{prefix}.lora_A.weight": a,
            f"base_model.model.{prefix}.lora_B.weight": b,
            f"base_model.model.{prefix}.lora_A2.weight": a2,
            f"base_model.model.{prefix}.lora_B2.weight": b2,
        },
    )
    w1 = (2.0 * (a @ b)).transpose(1, 2)
    w2 = (2.0 * (a2 @ b2)).transpose(1, 2)
    assert torch.equal(
        merged[f"{prefix}.gate_up_proj.weight"],
        (base_gate_up.float() + w1).to(torch.bfloat16),
    )
    assert torch.equal(
        merged[f"{prefix}.down_proj.weight"],
        (base_down.float() + w2).to(torch.bfloat16),
    )


def test_merges_per_expert_grouped_moe_lora(tmp_path):
    base_gate = torch.randn(2, 3, dtype=torch.bfloat16)
    base_up = torch.randn(2, 3, dtype=torch.bfloat16)
    base_down = torch.randn(3, 2, dtype=torch.bfloat16)
    a = torch.randn(2, 3, 2)
    b = torch.randn(2, 2, 4)
    a2 = torch.randn(2, 2, 2)
    b2 = torch.randn(2, 2, 3)
    prefix = "model.mlp.experts"
    merged, _ = _merge_and_load(
        tmp_path,
        {
            f"{prefix}.0.gate_proj.weight": base_gate,
            f"{prefix}.1.up_proj.weight": base_up,
            f"{prefix}.1.down_proj.weight": base_down,
        },
        {
            f"base_model.model.{prefix}.lora_A.weight": a,
            f"base_model.model.{prefix}.lora_B.weight": b,
            f"base_model.model.{prefix}.lora_A2.weight": a2,
            f"base_model.model.{prefix}.lora_B2.weight": b2,
        },
    )
    w1 = (2.0 * (a @ b)).transpose(1, 2)
    w2 = (2.0 * (a2 @ b2)).transpose(1, 2)
    assert torch.equal(
        merged[f"{prefix}.0.gate_proj.weight"],
        (base_gate.float() + w1[0, :2]).to(torch.bfloat16),
    )
    assert torch.equal(
        merged[f"{prefix}.1.up_proj.weight"],
        (base_up.float() + w1[1, 2:]).to(torch.bfloat16),
    )
    assert torch.equal(
        merged[f"{prefix}.1.down_proj.weight"],
        (base_down.float() + w2[1]).to(torch.bfloat16),
    )


def test_rejects_dora_and_incomplete_pairs(tmp_path):
    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4,
        "use_dora": True,
    }
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="DoRA"):
        merge_lora_adapter.load_adapter_config(adapter_dir / "adapter_config.json")


def test_requires_base_dir_when_adapter_metadata_is_missing(tmp_path):
    adapter_dir = tmp_path / "adapter"
    _write_adapter(
        adapter_dir,
        {"base_model.model.model.q_proj.lora_A.weight": torch.randn(2, 2)},
        None,
    )

    with pytest.raises(ValueError, match="pass --base-dir"):
        merge_lora_adapter.main(
            [
                "--adapter-dir",
                str(adapter_dir),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
