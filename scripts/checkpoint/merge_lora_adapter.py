#!/usr/bin/env python
"""Merge a standalone TorchTurbo LoRA adapter into an HF base checkpoint.

The merge is performed directly on safetensors shards: no distributed process
group and no full model object are required.  The output preserves the base
checkpoint's shard boundaries and non-weight files.

Usage:
    python scripts/checkpoint/merge_lora_adapter.py \
        --adapter-dir output/checkpoint-10/huggingface \
        --output-dir output/checkpoint-10/merged-hf
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."


@dataclass(frozen=True)
class AdapterConfig:
    """The subset of PEFT LoRA settings needed for a numerical merge."""

    rank: int
    alpha: int
    use_rslora: bool
    use_dora: bool
    base_model_name_or_path: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the standalone merger command line."""
    parser = argparse.ArgumentParser(
        description="Merge a standalone LoRA adapter into an HF base checkpoint."
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Original HF base model directory. Defaults to the adapter's "
            "base_model_name_or_path; use this override after relocation or "
            "when the metadata names a Hub model ID."
        ),
    )
    parser.add_argument("--adapter-dir", required=True, help="Standalone LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Destination HF checkpoint directory.")
    return parser.parse_args(argv)


def load_adapter_config(path: Path) -> AdapterConfig:
    """Load and validate the adapter's PEFT configuration."""
    raw = json.loads(path.read_text())
    if raw.get("peft_type") != "LORA":
        raise ValueError(f"Only LoRA adapters are supported, got {raw.get('peft_type')!r}.")
    if raw.get("use_dora", False):
        raise ValueError("DoRA adapters cannot be merged by simple weight addition.")
    rank = raw.get("r")
    alpha = raw.get("lora_alpha")
    if not isinstance(rank, int) or not isinstance(alpha, int) or rank <= 0:
        raise ValueError("A positive integer LoRA r and lora_alpha are required.")
    if raw.get("rank_pattern") or raw.get("alpha_pattern"):
        raise ValueError("rank_pattern and alpha_pattern are not supported by this merger.")
    return AdapterConfig(
        rank=rank,
        alpha=alpha,
        use_rslora=bool(raw.get("use_rslora", False)),
        use_dora=False,
        base_model_name_or_path=raw.get("base_model_name_or_path"),
    )


def scaling(config: AdapterConfig) -> float:
    """Return the adapter's PEFT scaling factor."""
    denominator = config.rank**0.5 if config.use_rslora else config.rank
    return config.alpha / denominator


def read_adapter_keys(adapter_path: Path) -> set[str]:
    """Return all keys in the adapter safetensors file."""
    with safe_open(adapter_path, framework="pt") as adapter_file:
        return set(adapter_file.keys())


def group_adapter_keys(keys: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    """Group adapter keys by module and report modules with unusable pairs."""
    modules: dict[str, set[str]] = {}
    unsupported: set[str] = set()
    suffixes = ("lora_A.weight", "lora_B.weight", "lora_A2.weight", "lora_B2.weight")
    for key in keys:
        matched = next((suffix for suffix in suffixes if key.endswith(suffix)), None)
        if matched is None:
            unsupported.add(key)
            continue
        module = key[: -(len(matched) + 1)]
        modules.setdefault(module, set()).add(matched.removesuffix(".weight"))
    return modules, unsupported


def _base_shards(base_dir: Path) -> list[Path]:
    return sorted(base_dir.glob("*.safetensors"))


def _base_key_map(base_dir: Path, shards: list[Path]) -> dict[str, str]:
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return dict(weight_map)
    key_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as shard_file:
            for key in list(shard_file.keys()):
                key_map[key] = shard.name
    return key_map


def _delta_for_standard_module(
    adapter_path: Path, module: str, factor: float
) -> torch.Tensor:
    with safe_open(adapter_path, framework="pt") as adapter_file:
        a = adapter_file.get_tensor(f"{module}.lora_A.weight").to(torch.float32)
        b = adapter_file.get_tensor(f"{module}.lora_B.weight").to(torch.float32)
    return factor * torch.matmul(b, a)


def _grouped_deltas(
    adapter_path: Path, module: str, factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(adapter_path, framework="pt") as adapter_file:
        a = adapter_file.get_tensor(f"{module}.lora_A.weight").to(torch.float32)
        b = adapter_file.get_tensor(f"{module}.lora_B.weight").to(torch.float32)
        a2 = adapter_file.get_tensor(f"{module}.lora_A2.weight").to(torch.float32)
        b2 = adapter_file.get_tensor(f"{module}.lora_B2.weight").to(torch.float32)
    w1 = (factor * torch.matmul(a, b)).transpose(1, 2)
    w2 = (factor * torch.matmul(a2, b2)).transpose(1, 2)
    return w1.contiguous(), w2.contiguous()


def _target_keys_for_module(module: str, key_map: dict[str, str]) -> dict[str, str]:
    clean = module.removeprefix(ADAPTER_PREFIX)
    fused_w1 = f"{clean}.gate_up_proj.weight"
    fused_w2 = f"{clean}.down_proj.weight"
    if fused_w1 in key_map:
        return {fused_w1: "w1-fused", fused_w2: "w2-fused"}
    targets: dict[str, str] = {}
    kinds = {"gate_proj": "w1-gate", "up_proj": "w1-up", "down_proj": "w2-expert"}
    prefix = f"{clean}."
    for key in key_map:
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        parts = key[len(prefix) : -len(".weight")].split(".")
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in kinds:
            continue
        targets[key] = kinds[parts[1]]
    return targets


def _apply_module(
    tensors: dict[str, torch.Tensor],
    module: str,
    parts: set[str],
    adapter_path: Path,
    config: AdapterConfig,
    key_map: dict[str, str],
) -> None:
    clean = module.removeprefix(ADAPTER_PREFIX)
    factor = scaling(config)
    if parts == {"lora_A", "lora_B"}:
        target = f"{clean}.weight"
        if target not in tensors:
            return
        delta = _delta_for_standard_module(adapter_path, module, factor)
        tensors[target] = (tensors[target].to(torch.float32) + delta).to(tensors[target].dtype)
        return
    if parts != {"lora_A", "lora_B", "lora_A2", "lora_B2"}:
        raise ValueError(f"Incomplete LoRA tensor pair for {module}: {sorted(parts)}.")
    if not clean.endswith(".mlp.experts"):
        raise ValueError(f"Grouped LoRA tensors have an unexpected target module: {clean}.")

    target_kinds = _target_keys_for_module(module, key_map)
    needed = set(target_kinds).intersection(tensors)
    if not needed:
        return
    w1, w2 = _grouped_deltas(adapter_path, module, factor)
    intermediate = w1.shape[1] // 2
    for target in needed:
        kind = target_kinds[target]
        if kind == "w1-fused":
            delta = w1
        elif kind == "w2-fused":
            delta = w2
        elif kind == "w1-gate":
            expert = int(target.rsplit(".", 3)[-3])
            delta = w1[expert, :intermediate]
        elif kind == "w1-up":
            expert = int(target.rsplit(".", 3)[-3])
            delta = w1[expert, intermediate:]
        else:
            expert = int(target.rsplit(".", 3)[-3])
            delta = w2[expert]
        tensors[target] = (tensors[target].to(torch.float32) + delta).to(tensors[target].dtype)


def merge_adapter_to_base(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    """Merge ``adapter_dir`` into ``base_dir`` and write an HF checkpoint."""
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config = load_adapter_config(adapter_dir / "adapter_config.json")
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Missing {adapter_path}.")

    adapter_keys = read_adapter_keys(adapter_path)
    modules, unsupported = group_adapter_keys(adapter_keys)
    if unsupported:
        raise ValueError(f"Unsupported adapter tensors: {sorted(unsupported)}.")

    shards = _base_shards(base_dir)
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found under {base_dir}.")
    key_map = _base_key_map(base_dir, shards)
    output_dir.mkdir(parents=True, exist_ok=True)

    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard, framework="pt") as shard_file:
            for key in list(shard_file.keys()):
                tensors[key] = shard_file.get_tensor(key)
        for module, parts in sorted(modules.items()):
            _apply_module(tensors, module, parts, adapter_path, config, key_map)
        save_file(tensors, output_dir / shard.name, metadata={"format": "pt"})

    for source in base_dir.iterdir():
        if source.is_file() and source.suffix != ".safetensors":
            destination = output_dir / source.name
            shutil.copy2(source, destination)
    _validate_output(output_dir, key_map)


def _validate_output(output_dir: Path, key_map: dict[str, str]) -> None:
    output_keys: set[str] = set()
    for shard in _base_shards(output_dir):
        with safe_open(shard, framework="pt") as shard_file:
            output_keys.update(shard_file.keys())
    if output_keys != set(key_map):
        missing = sorted(set(key_map) - output_keys)
        extra = sorted(output_keys - set(key_map))
        raise RuntimeError(f"Output key validation failed; missing={missing}, extra={extra}.")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    adapter_dir = Path(args.adapter_dir)
    config = load_adapter_config(adapter_dir / "adapter_config.json")
    base_dir = args.base_dir or config.base_model_name_or_path
    if base_dir is None:
        raise ValueError(
            "The adapter metadata has no base_model_name_or_path; pass --base-dir."
        )
    merge_adapter_to_base(Path(base_dir), adapter_dir, Path(args.output_dir))


if __name__ == "__main__":
    main()
