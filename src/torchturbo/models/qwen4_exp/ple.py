"""Frozen CPU n-gram lookup and sequence-parallel PLE execution.

The n-gram tables are frozen and shared by the processes on each node through
one memory-mapped file per table in ``/dev/shm``. The file is unlinked once
every process has mapped it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributed import ProcessGroup
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from ...utils.flash_attn_utils import cu_seqlens_to_segment_ids
from ..modules.comm import all_gather_tensor
from ..modules.cpu_resident import NodeSharedCPUResidentParameter
from .modeling_qwen4_exp import (
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
    apply_mask_to_padding_states,
)

__all__ = [
    "NGramEmbedding",
    "NGramHasher",
    "PLEBatch",
    "install_ple_prefetch",
    "ple_forward",
    "segment_causal_conv",
]

# Bound temporary FP32 activations in both convolution passes.
_PLE_CONV_WINDOW_TOKENS = 8192


# ---------------------------------------------------------------------------
# Frozen n-gram tables: node-shared CPU rows and upstream row hashing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NGramHasher:
    """Upstream n-gram hashing with EOS history resets and isolated packed samples.

    The buffers are CPU copies taken from a materialized n-gram module, so the
    hash runs on CPU token IDs without touching the compute device.
    """

    eos_token_id: int
    ngram_size: int
    heads_per_ngram: int
    layer_multipliers: torch.Tensor
    head_vocab_sizes: torch.Tensor
    head_offsets: torch.Tensor

    @classmethod
    def from_module(cls, ngram: Qwen4ExpTextNGramEmbedding) -> NGramHasher:
        return cls(
            eos_token_id=ngram.eos_token_id,
            ngram_size=ngram.ngram_size,
            heads_per_ngram=ngram.heads_per_ngram,
            layer_multipliers=ngram.layer_multipliers.cpu(),
            head_vocab_sizes=ngram.ngram_heads_vocab_sizes.cpu(),
            head_offsets=ngram.ngram_heads_offsets.cpu(),
        )

    @property
    def context_len(self) -> int:
        """Number of preceding tokens that take part in the longest n-gram."""
        return self.ngram_size - 1

    def row_ids(
        self, token_ids: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        """Hash tokens into table rows.

        Args:
            token_ids: One-dimensional integer token IDs, optionally with left context.
            segment_ids: Matching segment IDs; zero denotes trailing padding.

        Returns:
            Table row IDs shaped [num_tokens, (ngram_size - 1) * heads_per_ngram].
        """
        token_ids = token_ids.long()
        positions = torch.arange(token_ids.numel(), device=token_ids.device)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        inclusive = eos_positions.cummax(0).values
        previous_eos = F.pad(inclusive[:-1], (1, 0), value=-1)
        # A history token is replaced by EOS when it lies before the sequence,
        # in another packed sample, or behind an EOS in the current sample.
        shifted_tokens = [token_ids]
        for shift in range(1, self.ngram_size):
            source_positions = positions - shift
            indices = source_positions.clamp_min(0)
            valid = (
                (source_positions >= 0)
                & segment_ids.ne(0)
                & segment_ids.eq(segment_ids[indices])
                & (positions - previous_eos - 1 >= shift)
            )
            shifted_tokens.append(
                torch.where(valid, token_ids[indices], self.eos_token_id)
            )
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            start = (ngram - 2) * self.heads_per_ngram
            stop = start + self.heads_per_ngram
            indices = torch.remainder(
                mixed_ids.unsqueeze(-1), self.head_vocab_sizes[start:stop]
            )
            blocks.append(indices + self.head_offsets[start:stop])
        return torch.cat(blocks, dim=-1)


class NGramEmbedding(nn.Module):
    """A frozen CPU embedding whose full weight is shared within each node."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.weight = NodeSharedCPUResidentParameter(
            weight.detach(),
            requires_grad=False,
            directory="/dev/shm",
        )

    def gather(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Look up flattened CPU int64 row IDs, returning new CPU storage."""
        if self.weight.is_meta:
            raise RuntimeError(
                "PLE table has not been loaded from its source checkpoint"
            )
        return self.weight.index_select(0, row_ids)

    def forward(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Look up on CPU and transfer only selected rows to the indices' device."""
        rows = self.gather(row_ids.flatten().cpu())
        return rows.view(*row_ids.shape, self.embedding_dim).to(row_ids.device)


# ---------------------------------------------------------------------------
# Packed PLE execution: prefetched rows, segment-aware convolution, SP halo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PLEBatch:
    """Constant local rows and global segment metadata for one micro-batch."""

    rows: tuple[torch.Tensor, ...]
    segment_ids: torch.Tensor


@dataclass(frozen=True)
class _ConvHalo:
    """Align convolution inputs, gradients, and segments across SP ranks.

    Only interval and process-group metadata live here. Autograd saves the
    aligned tensors separately so their lifetime follows save_for_backward.
    """

    history: int
    local_length: int
    group: ProcessGroup | None
    group_size: int
    group_rank: int

    @property
    def left_halo(self) -> int:
        return min(self.history, self.group_rank * self.local_length)

    def prepare_inputs(
        self, x: torch.Tensor, segment_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend left history and retain segments through the right dY halo."""
        if self.group_size > 1 and self.history:
            halo_len = min(self.history, self.local_length)
            tails = all_gather_tensor(
                x[:, -halo_len:].transpose(0, 1).contiguous(), self.group
            ).transpose(0, 1)
            stop = self.group_rank * halo_len
            x = torch.cat(
                (tails[:, max(0, stop - self.history) : stop], x), dim=1
            )
        start = self.group_rank * self.local_length
        stop = min(start + self.local_length + self.history, segment_ids.shape[1])
        segments = segment_ids[:, start - self.left_halo : stop]
        return x, segments

    def prepare_grad_output(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Align local and right-side dY with the saved input interval.

        Left-history positions have zero dY. Every backward rank participates,
        including padding-only ranks and ranks requiring only weight gradients.
        """
        if self.group_size <= 1 or self.history == 0:
            return grad_output
        halo_len = min(self.history, self.local_length)
        heads = all_gather_tensor(
            grad_output[:, :halo_len].transpose(0, 1).contiguous(), self.group
        ).transpose(0, 1)
        start = (self.group_rank + 1) * halo_len
        return torch.cat(
            (
                grad_output.new_zeros(
                    grad_output.shape[0], self.left_halo, grad_output.shape[2]
                ),
                grad_output,
                heads[:, start : start + self.history],
            ),
            dim=1,
        )


def _segment_causal_conv_window(
    x: torch.Tensor,
    segment_ids: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
    num_outputs: int,
) -> torch.Tensor:
    """Convolve one window and keep its last ``num_outputs`` positions."""
    length = x.shape[1]
    kernel_size = weight.shape[-1]
    output = torch.zeros_like(x)
    for kernel_index in range(kernel_size):
        lag = (kernel_size - 1 - kernel_index) * dilation
        if lag >= length:
            continue
        contribution = x[:, : length - lag] * weight[:, 0, kernel_index]
        if lag:
            crosses_boundary = segment_ids[:, lag:].ne(segment_ids[:, :-lag])
            contribution.masked_fill_(crosses_boundary.unsqueeze(-1), 0)
        output[:, lag:] += contribution
    return output[:, length - num_outputs :]


def _local_conv_forward(
    x: torch.Tensor,
    segment_ids: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
    window_tokens: int,
    left_halo: int,
    accumulation_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute local outputs from inputs prefixed with left history."""
    history = (weight.shape[-1] - 1) * dilation
    weight = weight.to(accumulation_dtype)
    output = torch.empty_like(x[:, left_halo:])
    for start in range(left_halo, x.shape[1], window_tokens):
        stop = min(start + window_tokens, x.shape[1])
        window_start = max(0, start - history)
        window_output = _segment_causal_conv_window(
            x[:, window_start:stop].to(accumulation_dtype),
            segment_ids[:, window_start:stop],
            weight,
            dilation,
            stop - start,
        )
        output[:, start - left_halo : stop - left_halo].copy_(window_output)
    return output


def _local_conv_backward(
    x: torch.Tensor,
    segment_ids: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    dilation: int,
    window_tokens: int,
    left_halo: int,
    accumulation_dtype: torch.dtype,
    *,
    needs_input_grad: bool,
    needs_weight_grad: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute local dX and owned dW from aligned inputs and output gradients.

    Inputs include left history. Output gradients have a zero left prefix,
    local dY, and right-side dY; segment IDs cover all three regions.
    """
    length = x.shape[1]
    grad_x = torch.empty_like(x[:, left_halo:]) if needs_input_grad else None
    grad_weight = (
        torch.zeros_like(weight, dtype=accumulation_dtype)
        if needs_weight_grad
        else None
    )
    weight_dtype = weight.dtype
    weight = weight.to(accumulation_dtype)
    kernel_size = weight.shape[-1]
    # Partition input positions, so all future-output contributions to one
    # input are accumulated together before the single low-precision cast.
    for start in range(0, length, window_tokens):
        stop = min(start + window_tokens, length)
        grad_window = (
            torch.zeros_like(x[:, start:stop], dtype=accumulation_dtype)
            if grad_x is not None
            else None
        )
        input_window = (
            x[:, start:stop].to(accumulation_dtype)
            if grad_weight is not None
            else None
        )
        for kernel_index in range(kernel_size):
            lag = (kernel_size - 1 - kernel_index) * dilation
            end = min(stop, grad_output.shape[1] - lag)
            if end <= start:
                continue
            count = end - start
            grad = grad_output[:, start + lag : end + lag].to(accumulation_dtype)
            if lag:
                crosses_boundary = segment_ids[:, start:end].ne(
                    segment_ids[:, start + lag : end + lag]
                )
                grad = grad.masked_fill(crosses_boundary.unsqueeze(-1), 0)
            if grad_window is not None:
                grad_window[:, :count] += grad * weight[:, 0, kernel_index]
            if grad_weight is not None:
                # Weight gradients belong to local outputs only; the future
                # dY halo contributes exclusively to locally owned dX.
                first = max(0, left_halo - start - lag)
                last = min(count, length - start - lag)
                if last > first:
                    grad_weight[:, 0, kernel_index] += (
                        input_window[:, first:last] * grad[:, first:last]
                    ).sum(dim=(0, 1))
        if grad_x is not None:
            first = max(start, left_halo)
            if stop > first:
                grad_x[:, first - left_halo : stop - left_halo].copy_(
                    grad_window[:, first - start :]
                )
    if grad_weight is not None:
        grad_weight = grad_weight.to(weight_dtype)
    return grad_x, grad_weight


class _SegmentCausalConv(torch.autograd.Function):
    """Coordinate SP history exchange and local convolution in one autograd node."""

    @staticmethod
    def forward(
        ctx, x, segment_ids, weight, dilation, window_tokens, sp_group, sp_size, sp_rank
    ):
        halo = _ConvHalo(
            history=(weight.shape[-1] - 1) * dilation,
            local_length=x.shape[1],
            group=sp_group,
            group_size=sp_size,
            group_rank=sp_rank,
        )
        x, segment_ids = halo.prepare_inputs(x, segment_ids)
        ctx.save_for_backward(x, segment_ids, weight)
        ctx.halo = halo
        ctx.dilation = dilation
        ctx.window_tokens = window_tokens
        ctx.accumulation_dtype = (
            torch.float32 if x.dtype in (torch.bfloat16, torch.float16) else x.dtype
        )
        return _local_conv_forward(
            x,
            segment_ids,
            weight,
            dilation,
            window_tokens,
            halo.left_halo,
            ctx.accumulation_dtype,
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, segment_ids, weight = ctx.saved_tensors
        # Exchange original dY so local and remote dX contributions are summed
        # in FP32 before the local helper casts the complete input gradients.
        grad_output = ctx.halo.prepare_grad_output(grad_output)
        grad_x, grad_weight = _local_conv_backward(
            x,
            segment_ids,
            weight,
            grad_output,
            ctx.dilation,
            ctx.window_tokens,
            ctx.halo.left_halo,
            ctx.accumulation_dtype,
            needs_input_grad=ctx.needs_input_grad[0],
            needs_weight_grad=ctx.needs_input_grad[2],
        )
        return grad_x, None, grad_weight, None, None, None, None, None


def segment_causal_conv(
    x: torch.Tensor,
    segment_ids: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
    *,
    sp_group: ProcessGroup | None = None,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Apply depthwise causal convolution and SiLU without crossing samples.

    Save inputs in their original dtype and recreate FP32 windows during
    backward. Both passes keep FP32 activation working storage window-sized.
    Under SP, ``x`` is local and ``segment_ids`` describes the global sequence.
    """
    if torch.is_autocast_enabled(x.device.type) and x.dtype != torch.float64:
        dtype = torch.get_autocast_dtype(x.device.type)
        x, weight = x.to(dtype), weight.to(dtype)
    # Native low-precision Conv1d accumulates in FP32 and rounds before SiLU.
    output = _SegmentCausalConv.apply(
        x,
        segment_ids,
        weight,
        dilation,
        _PLE_CONV_WINDOW_TOKENS,
        sp_group,
        sp_size,
        sp_rank,
    )
    return F.silu(output)


def ple_forward(
    self: Qwen4ExpTextPLELayer,
    hidden_states: torch.Tensor,
    ple_inputs: torch.Tensor | PLEBatch,
    past_key_values: Any,
    conv_mask: torch.Tensor | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    *,
    sp_group: ProcessGroup | None = None,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Run local PLE projections and convolution with sequence-parallel halos."""
    if isinstance(ple_inputs, PLEBatch):
        embeddings = ple_inputs.rows[self.ple_embedding.ple_layer_index]
    else:
        embeddings = self.ple_embedding(ple_inputs, past_key_values)
    embeddings = embeddings.to(hidden_states.dtype)
    key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(
        -1, (self.hc_count, self.hidden_size)
    )
    value = self.value_proj(embeddings)
    query_normed = self.norm_query(hidden_states).unflatten(
        -1, (self.hc_count, self.hidden_size)
    )
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
        self.hidden_size
    )
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated_value_normed = self.norm_conv(gated_value.flatten(-2))
    gated_value = gated_value.flatten(-2)
    if not isinstance(ple_inputs, PLEBatch):
        if conv_mask is not None:
            gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = apply_mask_to_padding_states(
                gated_value_normed, conv_mask
            )
        return gated_value + self._short_conv(gated_value_normed, past_key_values)

    length = hidden_states.shape[1]
    start = sp_rank * length
    segments = ple_inputs.segment_ids
    keep = segments[:, start : start + length].ne(0).unsqueeze(-1)
    gated_value = gated_value * keep
    gated_value_normed = gated_value_normed * keep

    conv_output = segment_causal_conv(
        gated_value_normed,
        segments,
        self.conv1d.weight,
        self.conv1d.dilation[0],
        sp_group=sp_group,
        sp_size=sp_size,
        sp_rank=sp_rank,
    )
    return (gated_value + conv_output).to(hidden_states.dtype)


def install_ple_prefetch(
    text_model: nn.Module, *, sp_size: int, sp_rank: int
) -> RemovableHandle:
    """Gather local rows before text forward, outside decoder recomputation."""
    def prefetch(module, args, kwargs):
        cu = kwargs.get("cu_seq_lens_q")
        if cu is None:
            return None
        ple_layers = sorted(
            (layer.ple for layer in module.layers if layer.ple is not None),
            key=lambda ple: ple.ple_embedding.ple_layer_index,
        )
        if not ple_layers:
            return None
        ids = kwargs.get("ple_input_ids")
        if ids is None:
            ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise ValueError(
                "Packed PLE requires ple_input_ids or input_ids; token IDs are "
                "not recovered from inputs_embeds"
            )
        if ids.shape[0] != 1 or ids.shape[1] % sp_size:
            raise ValueError(
                "Packed PLE requires batch size 1 and SP-aligned token IDs"
            )
        total = ids.shape[1]
        length = total // sp_size
        start = sp_rank * length
        ids_cpu = ids[0].cpu()
        segments = cu_seqlens_to_segment_ids(cu.cpu(), total)
        rows = []
        for ple in ple_layers:
            # Read current buffers after checkpoint reloads and model copies.
            hasher = NGramHasher.from_module(ple.ple_embedding)
            window = slice(max(0, start - hasher.context_len), start + length)
            indices = hasher.row_ids(ids_cpu[window], segments[0, window])[-length:]
            embeddings = ple.ple_embedding.ngram_embedding.gather(indices.flatten())
            rows.append(embeddings.view(1, length, -1).to(ids.device))
        kwargs["ple_input_ids"] = PLEBatch(tuple(rows), segments.to(ids.device))
        return args, kwargs

    return text_model.register_forward_pre_hook(prefetch, with_kwargs=True)
