from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def create_causal_mask(
    config, input_embeds, attention_mask, cache_position, past_key_values, position_ids
):
    # do not split attention mask into chunks under ulyssess sequence parallel's condition
    return attention_mask


def split_into_micro_batches(batch, micro_batch_size):
    """
    Split a batch dictionary into multiple micro batches.

    Args:
        batch: Input dictionary {key: tensor}, where all tensors must have
               the same size in the first dimension (batch_size).
        micro_batch_size: Number of samples per micro batch.

    Returns:
        List[Dict]: List of split micro batches, each being a dictionary
                   with the same structure as input.
    """
    if not batch:
        return []

    # Get global batch size (assuming all tensors have same first dim)
    batch_size = next(iter(batch.values())).size(0)
    # Calculate number of micro batches needed
    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    # Split each tensor in the batch dictionary
    micro_batches = []
    for i in range(num_micro_batches):
        start = i * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        micro_batch = {
            key: tensor[start:end]  # Handles tensors of any dimension
            for key, tensor in batch.items()
        }
        micro_batches.append(micro_batch)
    return micro_batches


def slice_input_tensor(
    x: Tensor,
    dim: int,
    group: ProcessGroup,
    rank: int,
    padding: bool = True,
    sp_mode="ulysses",
) -> Tensor:
    sp_world_size = dist.get_world_size(group)
    sp_rank = rank
    dim_size = x.size(dim)
    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = pad_tensor(x, dim, padding_size)
    # slice the input tensor
    if sp_mode == "ulysses":
        parts = x.size(dim) // sp_world_size
        slc = [slice(None)] * len(x.shape)
        slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
        return x[tuple(slc)].contiguous()
    else:  # zigzag_ring
        # zigzag ring attention requires that the chunks of the sequence be concatenated according to zigzag, for example:
        # s: [0, 1, 2, 3, 4, 5, 6, 7], sp_size=2
        # g0: [0, 1, 6, 7], g1: [2, 3, 4, 5]
        parts = x.size(dim) // (sp_world_size * 2)
        slc = [slice(None)] * len(x.shape)
        forward_start = sp_rank * parts
        backward_start = (2 * sp_world_size - sp_rank - 1) * parts
        slc1 = slc.copy()
        slc1[dim] = slice(forward_start, forward_start + parts)
        slc2 = slc.copy()
        slc2[dim] = slice(backward_start, backward_start + parts)
        return torch.cat([x[slc1], x[slc2]], dim=dim).contiguous()


def ulysses_pad(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    sp_size: int = 1,
    sp_mode: str = "ulysses",
    pad_value: int = 0,
):
    if position_ids_rmpad is not None:
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(
            -1
        ), f"input_ids shape: {input_ids_rmpad.shape}, position_ids shape: {position_ids_rmpad.shape}"
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, labels, 0
    _, total_seq_len = input_ids_rmpad.shape
    if sp_mode == "ulysses":
        pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    else:  # zigzag_ring
        pad_size = (2 * sp_size - total_seq_len % (2 * sp_size)) % (2 * sp_size)
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(
            input_ids_rmpad, (0, pad_size), value=pad_value
        )
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(
                pad_size, device=position_ids_rmpad.device
            ).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.expand(
                    *position_ids_rmpad.shape[:-1], -1
                )
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
        if labels is not None:
            labels = torch.nn.functional.pad(labels, (0, pad_size), value=-100)

    return input_ids_rmpad, position_ids_rmpad, labels, pad_size


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    labels: torch.Tensor = None,
    sp_size: int = 1,
    sp_group: ProcessGroup = None,
    sp_rank: int = 0,
    sp_mode: str = "ulysses",
    pad_value: int = 0,
):
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size
    """
    input_ids_rmpad, position_ids_rmpad, labels, pad_size = ulysses_pad(
        input_ids_rmpad,
        position_ids_rmpad,
        labels,
        sp_size,
        sp_mode=sp_mode,
        pad_value=pad_value,
    )
    input_ids_rmpad = slice_input_tensor(
        input_ids_rmpad,
        dim=1,
        group=sp_group,
        rank=sp_rank,
        padding=False,
        sp_mode=sp_mode,
    )
    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(
            position_ids_rmpad,
            dim=1,
            group=sp_group,
            rank=sp_rank,
            padding=False,
            sp_mode=sp_mode,
        )
    if labels is not None:
        labels = slice_input_tensor(
            labels, dim=1, group=sp_group, rank=sp_rank, padding=False, sp_mode=sp_mode
        )

    return input_ids_rmpad, position_ids_rmpad, labels, pad_size
