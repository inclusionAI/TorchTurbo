#!/usr/bin/env python3
"""DeepSeek V4 FP8/MXFP4 dequantization and offline CLI."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from math import ceil

import torch

from dequantize import DequantizationFormat, Dequantizer, TensorSource


class DeepSeekV4Format(DequantizationFormat):
    """Describe and dequantize DeepSeek V4 quantized weights."""

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
            scale_key = f"{key[:-len('.weight')]}.scale"
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
        """Convert a DeepSeek FP8 or packed MXFP4 weight to BF16."""

        cls._validate_matrix("DeepSeek weight", weight)
        if weight.dtype in (torch.int8, torch.uint8):
            unpacked = cls._unpack_mxfp4(weight)
            scales = cls._expand_block_scales(
                scale,
                tuple(unpacked.shape),
                32,
                rows_are_individual=True,
            )
            return Dequantizer.apply_scales_to_bf16(unpacked, scales)

        if weight.element_size() != 1 or not weight.is_floating_point():
            raise ValueError(
                f"Unsupported DeepSeek weight dtype {weight.dtype}; "
                "expected FP8 or packed INT8"
            )
        scales = cls._expand_block_scales(
            scale,
            tuple(weight.shape),
            128,
            rows_are_individual=False,
        )
        return Dequantizer.apply_scales_to_bf16(weight, scales)

    @classmethod
    def _unpack_mxfp4(cls, packed: torch.Tensor) -> torch.Tensor:
        cls._validate_matrix("packed MXFP4 weight", packed)
        values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=packed.device,
        )
        packed_uint8 = packed.view(torch.uint8)
        low = packed_uint8 & 0x0F
        high = (packed_uint8 >> 4) & 0x0F
        unpacked = torch.stack((low, high), dim=-1)
        output_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
        return values[unpacked.long()].reshape(output_shape)

    @classmethod
    def _expand_block_scales(
        cls,
        scale: torch.Tensor,
        output_shape: tuple[int, int],
        block_size: int,
        *,
        rows_are_individual: bool,
    ) -> torch.Tensor:
        scale = scale.to(torch.float32)
        rows, columns = output_shape
        if scale.ndim == 1 and rows_are_individual and rows == 1:
            scale = scale.unsqueeze(0)
        cls._validate_matrix("scale", scale)

        expected_rows = rows if rows_are_individual else ceil(rows / block_size)
        expected_columns = ceil(columns / block_size)
        expected_shape = (expected_rows, expected_columns)
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"Scale shape {tuple(scale.shape)} does not match expected "
                f"{expected_shape} for weight shape {output_shape} and "
                f"block size {block_size}"
            )

        expanded = scale.repeat_interleave(block_size, dim=1)
        if not rows_are_individual:
            expanded = expanded.repeat_interleave(block_size, dim=0)
        return expanded[:rows, :columns]

    @staticmethod
    def _validate_matrix(name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}"
            )


def main(argv: list[str] | None = None) -> None:
    from converter import Converter

    parser = argparse.ArgumentParser(
        description="Convert a DeepSeek V4 FP8/MXFP4 checkpoint to BF16."
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

    summary = Converter(DeepSeekV4Format()).convert(
        args.input_path,
        args.output_path,
    )
    print(
        "[complete] DeepSeek checkpoint converted to BF16: "
        f"converted={summary.converted_tensors}, copied={summary.copied_tensors}, "
        f"shards={summary.output_shards}, total_size={summary.total_size}"
    )


if __name__ == "__main__":
    main()
