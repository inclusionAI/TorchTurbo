#!/usr/bin/env python3
"""Kimi-K2.6 packed INT4 dequantization and offline CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import torch

from dequantize import DequantizationFormat, Dequantizer, TensorSource


class KimiK26Format(DequantizationFormat):
    """Describe and dequantize Kimi-K2.6 packed INT4 weights."""

    COMPONENT_SUFFIXES = {
        "packed": "_packed",
        "scale": "_scale",
        "shape": "_shape",
    }

    def __init__(
        self,
        group_size: int,
        *,
        unpack_fn: Callable | None = None,
    ) -> None:
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise ValueError(
                f"Kimi group_size must be an integer, got {group_size!r}"
            )
        if group_size == 0:
            raise ValueError("Kimi group_size must not be zero")
        self.group_size = group_size
        self._unpack_fn = unpack_fn

    @classmethod
    def from_config(
        cls,
        model_dir: str | Path,
        config_path: str | Path | None = None,
        *,
        unpack_fn: Callable | None = None,
    ) -> KimiK26Format:
        return cls(
            cls.load_group_size(model_dir, config_path),
            unpack_fn=unpack_fn,
        )

    @staticmethod
    def load_group_size(
        model_dir: str | Path,
        config_path: str | Path | None = None,
    ) -> int:
        path = (
            Path(config_path).expanduser()
            if config_path
            else Path(model_dir).expanduser() / "config.json"
        )
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)

        quantization_configs = []
        top_level = config.get("quantization_config")
        if isinstance(top_level, dict):
            quantization_configs.append(top_level)
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            nested = text_config.get("quantization_config")
            if isinstance(nested, dict):
                quantization_configs.append(nested)

        group_size = None
        for quantization_config in quantization_configs:
            try:
                group_size = quantization_config["config_groups"]["group_0"][
                    "weights"
                ]["group_size"]
                break
            except (KeyError, TypeError):
                continue
        if group_size is None:
            raise ValueError(f"Kimi group_size is missing from {path}")
        return group_size

    def resolve_sources(
        self,
        weight_map: Mapping[str, str],
    ) -> dict[str, TensorSource]:
        keys = set(weight_map)
        components_by_weight: dict[str, dict[str, str]] = {}

        for key in keys:
            for component, suffix in self.COMPONENT_SUFFIXES.items():
                if key.endswith(f".weight{suffix}"):
                    logical_key = key[: -len(suffix)]
                    components_by_weight.setdefault(logical_key, {})[
                        component
                    ] = key
                    break

        consumed: set[str] = set()
        sources: dict[str, TensorSource] = {}
        required = set(self.COMPONENT_SUFFIXES)
        for logical_key, components in sorted(components_by_weight.items()):
            missing = required - set(components)
            if missing:
                raise ValueError(
                    f"Incomplete Kimi INT4 source for {logical_key}: "
                    f"missing {', '.join(sorted(missing))}"
                )
            if logical_key in keys:
                raise ValueError(
                    "Checkpoint contains both direct and packed tensors for "
                    f"{logical_key}"
                )
            physical_keys = (
                components["packed"],
                components["scale"],
                components["shape"],
            )
            sources[logical_key] = TensorSource(
                logical_key=logical_key,
                physical_keys=physical_keys,
                quantized=True,
            )
            consumed.update(physical_keys)

        for key in sorted(keys - consumed):
            sources[key] = TensorSource(key, (key,))
        return sources

    def dequantize(
        self,
        source: TensorSource,
        tensors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.dequantize_int4(
            tensors[source.physical_keys[0]],
            tensors[source.physical_keys[1]],
            tensors[source.physical_keys[2]],
        )

    def dequantize_int4(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_shape: torch.Tensor | Iterable[int],
    ) -> torch.Tensor:
        """Unpack a Kimi INT4 weight, apply group scales, and return BF16."""

        shape = self._shape_tuple(weight_shape)
        unpacker = self._unpack_fn or self._load_unpacker()
        weight = unpacker(weight_packed.to(torch.int32), 4, shape)
        if tuple(weight.shape) != shape:
            raise ValueError(
                f"Unpacked Kimi weight shape {tuple(weight.shape)} "
                f"does not match {shape}"
            )

        scale = weight_scale.to(torch.float32)
        if self.group_size > 0:
            if scale.ndim == 1:
                scale = scale.unsqueeze(1)
            self._validate_matrix("Kimi weight_scale", scale)
            scales = torch.repeat_interleave(
                scale,
                repeats=self.group_size,
                dim=1,
            )
            if scales.shape[0] == shape[0] and scales.shape[1] >= shape[1]:
                scales = scales[:, : shape[1]]
        else:
            scales = scale

        if scales.shape != weight.shape and scales.numel() == weight.numel():
            scales = scales.reshape_as(weight)
        return Dequantizer.apply_scales_to_bf16(weight, scales)

    @staticmethod
    def _shape_tuple(
        weight_shape: torch.Tensor | Iterable[int],
    ) -> tuple[int, ...]:
        if isinstance(weight_shape, torch.Tensor):
            values = weight_shape.detach().cpu().reshape(-1).tolist()
        else:
            values = list(weight_shape)
        shape = tuple(int(value) for value in values)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(
                "Kimi weight_shape must contain two positive dimensions, "
                f"got {shape}"
            )
        return shape

    @staticmethod
    def _load_unpacker() -> Callable:
        try:
            from compressed_tensors.compressors import unpack_from_int32
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Kimi INT4 dequantization requires compressed-tensors; "
                "install it with `pip install compressed-tensors`"
            ) from exc
        return unpack_from_int32

    @staticmethod
    def _validate_matrix(name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}"
            )


def main(argv: list[str] | None = None) -> None:
    from converter import Converter

    parser = argparse.ArgumentParser(
        description="Convert a Kimi-K2.6 packed INT4 checkpoint to BF16."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Input Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output BF16 checkpoint directory (default: <model-dir>_bf16).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional config.json used to read the INT4 group size.",
    )
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(f"{model_dir}_bf16")
    )
    tensor_format = KimiK26Format.from_config(
        model_dir,
        args.config_path,
    )
    summary = Converter(tensor_format).convert(
        model_dir,
        output_dir,
    )
    print(
        "[complete] Kimi checkpoint converted to BF16: "
        f"converted={summary.converted_tensors}, copied={summary.copied_tensors}, "
        f"shards={summary.output_shards}, total_size={summary.total_size}"
    )


if __name__ == "__main__":
    main()
