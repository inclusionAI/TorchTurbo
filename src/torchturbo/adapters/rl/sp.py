"""Sequence-parallel helpers for RL training."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from torchturbo.models.modules.attention.ulysses import (
    all_gather_sp,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)

from .packed import shifted_packed_tensor


def _sp_group_and_size(sp_info: dict[str, Any]) -> tuple[dist.ProcessGroup | None, int]:
    return sp_info.get("sp_group"), int(sp_info.get("sp_size", 1))


def sp_gather_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    unpad_dim: int,
    padding_size: int,
    sp_group: dist.ProcessGroup,
    grad_scaler: bool = True,
) -> torch.Tensor:
    """Gather a tensor across the SP group and remove sequence padding."""

    sp_size = dist.get_world_size(group=sp_group)
    x = all_gather_sp(x, gather_dim, sp_group, sp_size, grad_scaler)
    if padding_size > 0:
        slc = [slice(None)] * x.dim()
        slc[unpad_dim] = slice(0, -padding_size)
        x = x[tuple(slc)]
    return x


def gather_logprobs_and_shift(
    logprobs_local: torch.Tensor,
    sp_pad_size: int,
    cu_seqlens: torch.Tensor | None,
    sp_info: dict[str, Any],
    grad_scaler: bool = True,
) -> torch.Tensor:
    """Gather SP-local log-probs and remove each packed sequence tail token."""

    sp_group, sp_size = _sp_group_and_size(sp_info)
    if sp_size <= 1 or sp_group is None:
        return shifted_packed_tensor(logprobs_local, cu_seqlens, 0, 1)

    logprobs_full = sp_gather_and_unpad(
        logprobs_local,
        gather_dim=0,
        unpad_dim=0,
        padding_size=sp_pad_size,
        sp_group=sp_group,
        grad_scaler=grad_scaler,
    )
    return shifted_packed_tensor(logprobs_full, cu_seqlens, 0, 1)


def gather_entropy_and_shift(
    entropy_local: torch.Tensor,
    sp_pad_size: int,
    cu_seqlens: torch.Tensor | None,
    sp_info: dict[str, Any],
    grad_scaler: bool = True,
) -> torch.Tensor:
    """Gather SP-local entropy and remove each packed sequence tail token."""

    sp_group, sp_size = _sp_group_and_size(sp_info)
    if sp_size <= 1 or sp_group is None:
        return shifted_packed_tensor(entropy_local, cu_seqlens, 0, 1)

    entropy_full = sp_gather_and_unpad(
        entropy_local,
        gather_dim=0,
        unpad_dim=0,
        padding_size=sp_pad_size,
        sp_group=sp_group,
        grad_scaler=grad_scaler,
    )
    return shifted_packed_tensor(entropy_full, cu_seqlens, 0, 1)


def gather_vocab_logits_stats(
    logits_local: torch.Tensor,
    sp_pad_size: int,
    cu_seqlens: torch.Tensor | None,
    sp_info: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather vocab min/max stats across SP ranks and align to shifted tokens."""

    vocab_min_local = logits_local.detach().min(-1).values.float()
    vocab_max_local = logits_local.detach().max(-1).values.float()

    sp_group, sp_size = _sp_group_and_size(sp_info)
    if sp_size <= 1 or sp_group is None:
        return (
            shifted_packed_tensor(vocab_min_local, cu_seqlens, 0, 1),
            shifted_packed_tensor(vocab_max_local, cu_seqlens, 0, 1),
        )

    vocab_min_full = sp_gather_and_unpad(
        vocab_min_local,
        gather_dim=0,
        unpad_dim=0,
        padding_size=sp_pad_size,
        sp_group=sp_group,
        grad_scaler=False,
    )
    vocab_max_full = sp_gather_and_unpad(
        vocab_max_local,
        gather_dim=0,
        unpad_dim=0,
        padding_size=sp_pad_size,
        sp_group=sp_group,
        grad_scaler=False,
    )
    return (
        shifted_packed_tensor(vocab_min_full, cu_seqlens, 0, 1),
        shifted_packed_tensor(vocab_max_full, cu_seqlens, 0, 1),
    )


def pad_and_slice_packed_inputs(
    padded_input_ids: torch.Tensor,
    padded_position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sp_info: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, torch.Tensor]:
    """Pad/slice packed model inputs for Ulysses SP.

    Returns local ``input_ids``, local ``position_ids``, padded attention mask,
    SP padding size, and local rolled labels.
    """

    sp_group, sp_size = _sp_group_and_size(sp_info)
    sp_rank = int(sp_info.get("sp_rank", 0))
    if sp_size <= 1 or sp_group is None:
        return (
            padded_input_ids,
            padded_position_ids,
            attention_mask,
            0,
            torch.roll(padded_input_ids, shifts=-1, dims=1),
        )

    input_ids_rolled = torch.roll(padded_input_ids, shifts=-1, dims=1)
    local_input_ids, local_position_ids, _, sp_pad_size = ulysses_pad_and_slice_inputs(
        padded_input_ids,
        position_ids_rmpad=padded_position_ids,
        sp_size=sp_size,
        sp_group=sp_group,
        sp_rank=sp_rank,
    )
    input_ids_rolled, _, _, _ = ulysses_pad_and_slice_inputs(
        input_ids_rolled,
        position_ids_rmpad=None,
        sp_size=sp_size,
        sp_group=sp_group,
        sp_rank=sp_rank,
    )
    if attention_mask is not None:
        attention_mask, _, _, _ = ulysses_pad(
            attention_mask,
            position_ids_rmpad=None,
            sp_size=sp_size,
            pad_value=0,
        )

    return (
        local_input_ids,
        local_position_ids,
        attention_mask,
        sp_pad_size,
        input_ids_rolled,
    )