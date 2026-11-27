#!/usr/bin/env python
"""Merge a standalone TorchTurbo LoRA adapter into an HF base checkpoint.

Usage:
    python scripts/checkpoint/merge_lora_adapter.py \
        [--base-dir /path/to/HF_base_model] \
        --adapter-dir /path/to/lora_adapter/huggingface \
        --output-dir /path/to/merged-hf
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."
PER_EXPERT_SUFFIX = re.compile(
    r"^(?P<parent>.+\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class AdapterConfig:
    """The subset of PEFT LoRA settings needed for a numerical merge."""

    rank: int
    alpha: int
    use_rslora: bool
    use_dora: bool
    base_model_name_or_path: str | None


@dataclass(frozen=True)
class FusedExpertTarget:
    """One standard PEFT expert target mapped into a fused base tensor."""

    module: str
    expert_index: int
    projection: str
    base_root: str


@dataclass(frozen=True)
class AdapterMergePlan:
    """Adapter modules grouped by the base layout they merge into.

    ``direct_modules`` contains ordinary Linear modules and standard per-expert
    modules whose base checkpoint also stores physical per-expert tensors. They
    are applied by logical module name.

    ``fused_expert_targets`` contains standard per-expert modules only when the
    base checkpoint stores a fused MoE tensor. They are grouped by ``base_root``
    so each adapter file is opened once per fused target and shard.
    """

    direct_modules: dict[str, set[str]]
    fused_expert_targets: dict[str, list[FusedExpertTarget]]


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
    suffixes = ("lora_A.weight", "lora_B.weight")
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


def _base_tensor_key(tensors: Mapping[str, Any], root: str) -> str | None:
    """Resolve either a bare parameter key or an ``nn.Module.weight`` key."""
    if root in tensors:
        return root
    weighted_name = f"{root}.weight"
    if weighted_name in tensors:
        return weighted_name
    return None


def _parse_fused_expert_target(module: str) -> FusedExpertTarget | None:
    """Parse a standard expert module and its fused base tensor root."""
    clean = module.removeprefix(ADAPTER_PREFIX)
    match = PER_EXPERT_SUFFIX.match(clean)
    if match is None:
        return None

    projection = match.group("projection")
    parent = match.group("parent")
    base_root = (
        f"{parent}.gate_up_proj"
        if projection in ("gate_proj", "up_proj")
        else f"{parent}.down_proj"
    )
    return FusedExpertTarget(
        module=module,
        expert_index=int(match.group("expert")),
        projection=projection,
        base_root=base_root,
    )


def _build_adapter_merge_plan(
    modules: dict[str, set[str]], key_map: dict[str, str]
) -> AdapterMergePlan:
    """Partition adapter modules by direct and fused base-target handling."""
    direct_modules: dict[str, set[str]] = {}
    fused_targets: dict[str, list[FusedExpertTarget]] = {}

    for module, parts in modules.items():
        target = _parse_fused_expert_target(module)
        is_standard_pair = parts == {"lora_A", "lora_B"}
        base_has_fused_tensor = (
            target is not None
            and _base_tensor_key(key_map, target.base_root) is not None
        )
        if is_standard_pair and base_has_fused_tensor and target is not None:
            fused_targets.setdefault(target.base_root, []).append(target)
        else:
            # Physical per-expert checkpoints take the same-name path. Malformed
            # pairs also remain here so the application path can report a
            # precise error.
            direct_modules[module] = parts

    return AdapterMergePlan(direct_modules, fused_targets)


def _apply_fused_expert_targets(
    tensors: dict[str, torch.Tensor],
    targets_by_root: dict[str, list[FusedExpertTarget]],
    adapter_path: Path,
    factor: float,
) -> None:
    """Merge standard per-expert PEFT tensors into fused MoE base tensors."""
    for base_root, targets in sorted(targets_by_root.items()):
        target_key = _base_tensor_key(tensors, base_root)
        if target_key is None:
            # The planner validated that this root exists in the full base
            # index. It simply lives in another safetensors shard.
            continue

        weight = tensors[target_key]
        intermediate_size = weight.shape[1] // 2
        with safe_open(adapter_path, framework="pt") as adapter_file:
            for target in targets:
                a = adapter_file.get_tensor(
                    f"{target.module}.lora_A.weight"
                ).to(torch.float32)
                b = adapter_file.get_tensor(
                    f"{target.module}.lora_B.weight"
                ).to(torch.float32)
                delta = factor * torch.matmul(b, a)

                if target.projection == "down_proj":
                    base_slice = weight[target.expert_index]
                elif target.projection == "gate_proj":
                    row_start, row_end = 0, intermediate_size
                    base_slice = weight[
                        target.expert_index, row_start:row_end
                    ]
                else:
                    row_start = intermediate_size
                    row_end = 2 * intermediate_size
                    base_slice = weight[
                        target.expert_index, row_start:row_end
                    ]

                if base_slice.shape != delta.shape:
                    raise ValueError(
                        f"Cannot merge {target.module}: adapter delta has shape "
                        f"{tuple(delta.shape)}, but the fused base slice has "
                        f"shape {tuple(base_slice.shape)}."
                    )

                merged_slice = (
                    base_slice.to(torch.float32) + delta
                ).to(weight.dtype)
                if target.projection == "down_proj":
                    weight[target.expert_index] = merged_slice
                else:
                    weight[
                        target.expert_index, row_start:row_end
                    ] = merged_slice


def _apply_direct_standard_module(
    tensors: dict[str, torch.Tensor],
    module: str,
    factor: float,
    adapter_path: Path,
    key_map: dict[str, str],
) -> None:
    """Merge an A/B pair into the same-named base tensor, if present."""
    clean = module.removeprefix(ADAPTER_PREFIX)
    target = _base_tensor_key(tensors, clean)
    if target is not None:
        delta = _delta_for_standard_module(adapter_path, module, factor)
        tensors[target] = (
            tensors[target].to(torch.float32) + delta
        ).to(tensors[target].dtype)
        return

    # ``tensors`` contains only the current shard. An absent target is valid
    # when the complete index places it in another shard; it is an error only
    # when no shard can ever provide that target.
    if _base_tensor_key(key_map, clean) is None:
        raise ValueError(f"No base weight found for LoRA target {clean}.")


def _apply_direct_modules(
    tensors: dict[str, torch.Tensor],
    modules: dict[str, set[str]],
    config: AdapterConfig,
    adapter_path: Path,
    key_map: dict[str, str],
) -> None:
    """Apply direct A/B modules and reject incomplete LoRA pairs."""
    factor = scaling(config)
    for module, parts in sorted(modules.items()):
        if parts != {"lora_A", "lora_B"}:
            raise ValueError(
                f"Incomplete LoRA tensor pair for {module}: {sorted(parts)}."
            )
        _apply_direct_standard_module(
            tensors, module, factor, adapter_path, key_map
        )


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
    merge_plan = _build_adapter_merge_plan(modules, key_map)
    output_dir.mkdir(parents=True, exist_ok=True)

    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard, framework="pt") as shard_file:
            for key in list(shard_file.keys()):
                tensors[key] = shard_file.get_tensor(key)
        _apply_fused_expert_targets(
            tensors, merge_plan.fused_expert_targets, adapter_path, scaling(config)
        )
        _apply_direct_modules(
            tensors, merge_plan.direct_modules, config, adapter_path, key_map
        )
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
