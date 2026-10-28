"""Offline conversion of sharded checkpoints to BF16."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from dequantize import DequantizationFormat, Dequantizer, TensorSource


@dataclass(frozen=True)
class ConversionSummary:
    converted_tensors: int
    copied_tensors: int
    output_shards: int
    total_size: int


class Converter:
    """Convert a sharded checkpoint to BF16."""

    INDEX_FILENAME = "model.safetensors.index.json"

    def __init__(self, tensor_format: DequantizationFormat) -> None:
        self._dequantizer = Dequantizer(tensor_format)

    def convert(
        self,
        model_dir: str | Path,
        output_dir: str | Path,
    ) -> ConversionSummary:
        model_dir = Path(model_dir).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self._validate_output_path(model_dir, output_dir)

        source_metadata, weight_map = self._load_index(model_dir)
        sources = self._dequantizer.resolve_sources(weight_map)
        self._validate_checkpoint_layout(
            model_dir,
            weight_map,
            sources.values(),
        )
        if not any(source.quantized for source in sources.values()):
            raise ValueError("Checkpoint contains no quantized tensor sources")

        output_dir.mkdir(parents=True, exist_ok=True)

        sources_by_output_file: dict[str, list[TensorSource]] = defaultdict(list)
        for source in sources.values():
            sources_by_output_file[weight_map[source.primary_key]].append(source)

        new_weight_map: dict[str, str] = {}
        total_size = 0
        converted = 0
        copied = 0

        for filename, shard_sources in sorted(sources_by_output_file.items()):
            required_keys = {
                key for source in shard_sources for key in source.physical_keys
            }
            physical_tensors = self._read_physical_tensors(
                model_dir,
                weight_map,
                required_keys,
            )
            output_tensors: dict[str, torch.Tensor] = {}

            for source in sorted(
                shard_sources,
                key=lambda item: item.logical_key,
            ):
                if source.quantized:
                    tensor = self._dequantizer.dequantize(
                        source,
                        physical_tensors,
                    )
                    converted += 1
                else:
                    tensor = physical_tensors[source.primary_key]
                    copied += 1
                output_tensors[source.logical_key] = tensor.contiguous()
                new_weight_map[source.logical_key] = filename
                total_size += tensor.numel() * tensor.element_size()

            output_path = output_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(output_tensors, output_path, metadata={"format": "pt"})
            converted_in_shard = sum(
                source.quantized for source in shard_sources
            )
            print(
                f"[done] wrote {filename} "
                f"(tensors={len(output_tensors)}, "
                f"converted={converted_in_shard})"
            )
            del output_tensors
            del physical_tensors

        self._copy_model_assets(model_dir, output_dir)
        self._mark_config_as_bf16(output_dir)

        metadata = dict(source_metadata)
        metadata["total_size"] = total_size
        index = {"metadata": metadata, "weight_map": new_weight_map}
        with (output_dir / self.INDEX_FILENAME).open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")

        return ConversionSummary(
            converted_tensors=converted,
            copied_tensors=copied,
            output_shards=len(sources_by_output_file),
            total_size=total_size,
        )

    @classmethod
    def _load_index(cls, model_dir: Path) -> tuple[dict, dict[str, str]]:
        index_path = model_dir / cls.INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"Checkpoint index not found: {index_path}")
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Checkpoint index has no weight_map: {index_path}")
        metadata = index.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Checkpoint metadata must be an object: {index_path}"
            )
        return metadata, weight_map

    @staticmethod
    def _validate_output_path(model_dir: Path, output_dir: Path) -> None:
        if output_dir == model_dir or output_dir.is_relative_to(model_dir):
            raise ValueError(
                "Output directory must be outside the input model directory"
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")

    @staticmethod
    def _read_physical_tensors(
        model_dir: Path,
        weight_map: Mapping[str, str],
        physical_keys: Collection[str],
    ) -> dict[str, torch.Tensor]:
        keys_by_file: dict[str, list[str]] = defaultdict(list)
        for key in physical_keys:
            keys_by_file[weight_map[key]].append(key)

        tensors: dict[str, torch.Tensor] = {}
        for filename, keys in sorted(keys_by_file.items()):
            shard_path = model_dir / filename
            with safe_open(shard_path, framework="pt", device="cpu") as reader:
                for key in keys:
                    tensors[key] = reader.get_tensor(key)
        return tensors

    @staticmethod
    def _validate_checkpoint_layout(
        model_dir: Path,
        weight_map: Mapping[str, str],
        sources: Collection[TensorSource],
    ) -> None:
        indexed_keys_by_file: dict[str, set[str]] = defaultdict(set)
        for key, filename in weight_map.items():
            indexed_keys_by_file[filename].add(key)

        direct_weight_keys = {
            source.primary_key
            for source in sources
            if not source.quantized and source.logical_key.endswith(".weight")
        }

        for filename, indexed_keys in sorted(indexed_keys_by_file.items()):
            shard_path = model_dir / filename
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint shard not found: {shard_path}"
                )
            with safe_open(shard_path, framework="pt", device="cpu") as reader:
                available = set(reader.keys())
                missing = indexed_keys - available
                if missing:
                    preview = ", ".join(sorted(missing)[:3])
                    raise KeyError(
                        f"Indexed tensors missing from {shard_path}: {preview}"
                    )
                for key in sorted(indexed_keys & direct_weight_keys):
                    if reader.get_tensor(key).element_size() == 1:
                        raise ValueError(
                            f"One-byte weight {key} has no recognized "
                            "quantization components"
                        )

    @classmethod
    def _copy_model_assets(cls, model_dir: Path, output_dir: Path) -> None:
        for item in model_dir.iterdir():
            if item.name == cls.INDEX_FILENAME or item.suffix == ".safetensors":
                continue
            destination = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)

    @staticmethod
    def _mark_config_as_bf16(output_dir: Path) -> None:
        config_path = output_dir / "config.json"
        if not config_path.is_file():
            return
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)

        config.pop("quantization_config", None)
        config["torch_dtype"] = "bfloat16"
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            text_config.pop("quantization_config", None)
            text_config["torch_dtype"] = "bfloat16"

        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
