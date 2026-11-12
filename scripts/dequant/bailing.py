#!/usr/bin/env python3
"""Ling FP8 block-wise dequantization and offline CLI."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from math import ceil

import torch

from torchturbo.scripts.dequant.dequantize import DequantizationFormat, Dequantizer, TensorSource


class BailingMoeV3Format(DequantizationFormat):
    """Convert Ling 128x128 FP8 blocks to BF16."""

    BLOCK_SIZE = 128
    SCALE_SUFFIX = ".weight_scale_inv"

    def resolve_sources(
        self,
        weight_map: Mapping[str, str],
    ) -> dict[str, TensorSource]:
        keys = set(weight_map)
        consumed: set[str] = set()
        sources: dict[str, TensorSource] = {}

        for key in sorted(keys):
            if not key.endswith(".weight"):
                continue
            scale_key = key[: -len(".weight")] + self.SCALE_SUFFIX
            if scale_key not in keys:
                continue
            sources[key] = TensorSource(
                logical_key=key,
                physical_keys=(key, scale_key),
                quantized=True,
            )
            consumed.update((key, scale_key))

        for key in sorted(keys - consumed):
            sources[key] = TensorSource(key, (key,))
        return sources

    def dequantize(
        self,
        source: TensorSource,
        tensors: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.dequantize_weight(
            tensors[source.physical_keys[0]],
            tensors[source.physical_keys[1]],
        )

    @classmethod
    def dequantize_weight(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Convert one Ling FP8 E4M3 weight to BF16."""

        cls._validate_matrix("Ling weight", weight)
        cls._validate_matrix("Ling scale", scale)
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Unsupported Ling weight dtype {weight.dtype}; "
                "expected FP8 E4M3"
            )
        if scale.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"Unsupported Ling scale dtype {scale.dtype}; "
                "expected BF16 or FP32"
            )

        scales = cls._expand_block_scales(scale, tuple(weight.shape))
        return Dequantizer.apply_scales_to_bf16(weight, scales)

    @classmethod
    def _expand_block_scales(
        cls,
        scale: torch.Tensor,
        output_shape: tuple[int, int],
    ) -> torch.Tensor:
        rows, columns = output_shape
        expected_shape = (
            ceil(rows / cls.BLOCK_SIZE),
            ceil(columns / cls.BLOCK_SIZE),
        )
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"Scale shape {tuple(scale.shape)} does not match expected "
                f"{expected_shape} for weight shape {output_shape} and "
                f"block size {cls.BLOCK_SIZE}"
            )

        expanded = scale.to(torch.float32).repeat_interleave(
            cls.BLOCK_SIZE,
            dim=1,
        ).repeat_interleave(cls.BLOCK_SIZE, dim=0)
        return expanded[:rows, :columns]

    @staticmethod
    def _validate_matrix(name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}"
            )


def main(argv: list[str] | None = None) -> None:
    from torchturbo.scripts.dequant.converter import Converter

    parser = argparse.ArgumentParser(
        description="Convert a Ling FP8 checkpoint to BF16."
    )
    parser.add_argument(
        "--input-fp8-hf-path",
        "--input_fp8_hf_path",
        dest="input_path",
        required=True,
        help="Input Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--output-hf-path",
        "--output_hf_path",
        dest="output_path",
        required=True,
        help="Output BF16 checkpoint directory.",
    )
    parser.add_argument(
        "--quant-type",
        "--quant_type",
        choices=["bfloat16"],
        default="bfloat16",
        help="Output dtype (only bfloat16 is supported).",
    )
    args = parser.parse_args(argv)

    summary = Converter(BailingMoeV3Format()).convert(
        args.input_path,
        args.output_path,
    )
    print(
        "[complete] Ling checkpoint converted to BF16: "
        f"converted={summary.converted_tensors}, "
        f"copied={summary.copied_tensors}, shards={summary.output_shards}, "
        f"total_size={summary.total_size}"
    )


if __name__ == "__main__":
    main()
