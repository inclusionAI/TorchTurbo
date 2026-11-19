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


@pytest.mark.parametrize("weight_suffix", ["", ".weight"])
def test_merges_standard_per_expert_lora_into_fused_base(tmp_path, weight_suffix):
    base_gate_up = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    base_down = torch.randn(2, 3, 2, dtype=torch.bfloat16)
    gate_a = torch.randn(2, 2, 3)
    gate_b = torch.randn(2, 2, 2)
    up_a = torch.randn(2, 2, 3)
    up_b = torch.randn(2, 2, 2)
    down_a = torch.randn(2, 2, 2)
    down_b = torch.randn(2, 3, 2)
    prefix = "model.mlp.experts"
    adapter = {}
    for expert in range(2):
        for projection, a, b in (
            ("gate_proj", gate_a[expert], gate_b[expert]),
            ("up_proj", up_a[expert], up_b[expert]),
            ("down_proj", down_a[expert], down_b[expert]),
        ):
            adapter[f"base_model.model.{prefix}.{expert}.{projection}.lora_A.weight"] = a
            adapter[f"base_model.model.{prefix}.{expert}.{projection}.lora_B.weight"] = b

    merged, _ = _merge_and_load(
        tmp_path,
        {
            f"{prefix}.gate_up_proj{weight_suffix}": base_gate_up,
            f"{prefix}.down_proj{weight_suffix}": base_down,
        },
        adapter,
    )

    expected_gate_up = base_gate_up.float()
    expected_down = base_down.float()
    for expert in range(2):
        expected_gate_up[expert, :2] += 2.0 * (
            gate_b[expert] @ gate_a[expert]
        )
        expected_gate_up[expert, 2:] += 2.0 * (up_b[expert] @ up_a[expert])
        expected_down[expert] += 2.0 * (down_b[expert] @ down_a[expert])
    assert torch.equal(
        merged[f"{prefix}.gate_up_proj{weight_suffix}"],
        expected_gate_up.to(torch.bfloat16),
    )
    assert torch.equal(
        merged[f"{prefix}.down_proj{weight_suffix}"],
        expected_down.to(torch.bfloat16),
    )


def test_merges_standard_per_expert_lora_into_physical_base(tmp_path):
    prefix = "model.mlp.experts"
    base = {}
    adapter = {}
    expected = {}

    for expert in range(2):
        for projection, output_size, input_size in (
            ("gate_proj", 2, 3),
            ("up_proj", 2, 3),
            ("down_proj", 3, 2),
        ):
            base_weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)
            a = torch.randn(2, input_size)
            b = torch.randn(output_size, 2)
            base_key = f"{prefix}.{expert}.{projection}.weight"
            base[base_key] = base_weight
            adapter[f"base_model.model.{base_key}.lora_A.weight"] = a
            adapter[f"base_model.model.{base_key}.lora_B.weight"] = b
            expected[base_key] = (base_weight.float() + 2.0 * (b @ a)).to(
                torch.bfloat16
            )

    merged, _ = _merge_and_load(tmp_path, base, adapter)
    assert merged.keys() == expected.keys()
    for key, expected_weight in expected.items():
        assert torch.equal(merged[key], expected_weight)


def test_rejects_legacy_grouped_moe_lora(tmp_path):
    prefix = "model.mlp.experts"
    adapter = {
        f"base_model.model.{prefix}.lora_A.weight": torch.randn(2, 3, 2),
        f"base_model.model.{prefix}.lora_B.weight": torch.randn(2, 2, 4),
        f"base_model.model.{prefix}.lora_A2.weight": torch.randn(2, 2, 2),
        f"base_model.model.{prefix}.lora_B2.weight": torch.randn(2, 3, 2),
    }
    adapter_dir = tmp_path / "adapter"
    _write_adapter(adapter_dir, adapter, str(tmp_path / "base"))

    with pytest.raises(ValueError, match="Unsupported adapter tensors"):
        merge_lora_adapter.main(
            [
                "--base-dir",
                str(tmp_path / "base"),
                "--adapter-dir",
                str(adapter_dir),
                "--output-dir",
                str(tmp_path / "output"),
            ]
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
