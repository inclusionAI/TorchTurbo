#!/usr/bin/env python3
"""DeepSeek V4.1 FP8/FP4 dequantization and offline CLI."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from math import ceil

import torch

from dequantize import DequantizationFormat, Dequantizer, TensorSource


class DeepSeekV41Format(DequantizationFormat):
    """Describe and dequantize DeepSeek V4.1 quantized weights."""

    BLOCK_SIZE = 32
    SCALE_SUFFIXES = (".scale", ".weight_scale_inv")

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
            prefix = key[: -len(".weight")]
            scale_keys = [
                prefix + suffix
                for suffix in self.SCALE_SUFFIXES
                if prefix + suffix in keys
            ]
            if len(scale_keys) > 1:
                raise ValueError(
                    f"Multiple scale tensors found for {key}: "
                    + ", ".join(scale_keys)
                )
            if not scale_keys:
                continue

            scale_key = scale_keys[0]
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
        """Convert a DeepSeek V4.1 FP8 or packed FP4 weight to BF16."""

        cls._validate_matrix("DeepSeek V4.1 weight", weight)
        if weight.dtype in (torch.int8, torch.uint8):
            unpacked = cls._unpack_fp4(weight)
            scales = cls._expand_fp4_scales(scale, tuple(unpacked.shape))
            return Dequantizer.apply_scales_to_bf16(unpacked, scales)

        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Unsupported DeepSeek V4.1 weight dtype {weight.dtype}; "
                "expected FP8 E4M3 or packed INT8"
            )
        scales = cls._expand_fp8_scales(scale, tuple(weight.shape))
        return Dequantizer.apply_scales_to_bf16(weight, scales)

    @classmethod
    def _decode_e8m0_scale(cls, scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype == torch.uint8:
            exponent_bits = scale.to(torch.float32)
        elif scale.element_size() == 1 and scale.is_floating_point():
            # UE8M0 may be represented as a one-byte float dtype by torch.
            exponent_bits = scale.view(torch.uint8).to(torch.float32)
        else:
            raise ValueError(
                f"DeepSeek V4.1 scale dtype must be UE8M0, got {scale.dtype}"
            )
        return torch.exp2(exponent_bits - 127.0)

    @classmethod
    def _unpack_fp4(cls, packed: torch.Tensor) -> torch.Tensor:
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
        nibbles = torch.stack((low, high), dim=-1)
        output_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
        return values[nibbles.long()].reshape(output_shape)

    @classmethod
    def _expand_fp4_scales(
        cls,
        scale: torch.Tensor,
        output_shape: tuple[int, int],
    ) -> torch.Tensor:
        cls._validate_matrix("DeepSeek V4.1 FP4 scale", scale)
        rows, columns = output_shape
        expected_shape = (rows, ceil(columns / cls.BLOCK_SIZE))
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"FP4 scale shape {tuple(scale.shape)} does not match expected "
                f"{expected_shape} for weight shape {output_shape}"
            )
        decoded = cls._decode_e8m0_scale(scale)
        return decoded.repeat_interleave(cls.BLOCK_SIZE, dim=1)[:, :columns]

    @classmethod
    def _expand_fp8_scales(
        cls,
        scale: torch.Tensor,
        output_shape: tuple[int, int],
    ) -> torch.Tensor:
        cls._validate_matrix("DeepSeek V4.1 FP8 scale", scale)
        rows, columns = output_shape
        column_blocks = ceil(columns / cls.BLOCK_SIZE)
        row_shape = (rows, column_blocks)
        block_shapes = {
            block_size: (ceil(rows / block_size), ceil(columns / block_size))
            for block_size in (32, 128)
        }

        tensor_shape = tuple(scale.shape)
        if tensor_shape == row_shape:
            rows_are_individual = True
            block_size = cls.BLOCK_SIZE
        elif tensor_shape in block_shapes.values():
            rows_are_individual = False
            block_size = next(
                size for size, shape in block_shapes.items()
                if shape == tensor_shape
            )
        else:
            valid_shapes = [row_shape, *block_shapes.values()]
            raise ValueError(
                f"FP8 scale shape {tensor_shape} does not match any expected "
                f"shape {valid_shapes} for weight shape {output_shape}"
            )

        if scale.dtype == torch.float32:
            decoded = scale
        else:
            decoded = cls._decode_e8m0_scale(scale)
        expanded = decoded.repeat_interleave(block_size, dim=1)
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
        description="Convert a DeepSeek V4.1 FP8/FP4 checkpoint to BF16."
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

    summary = Converter(DeepSeekV41Format()).convert(
        args.input_path,
        args.output_path,
    )
    print(
        "[complete] DeepSeek V4.1 checkpoint converted to BF16: "
        f"converted={summary.converted_tensors}, copied={summary.copied_tensors}, "
        f"shards={summary.output_shards}, total_size={summary.total_size}"
    )


if __name__ == "__main__":
    main()
