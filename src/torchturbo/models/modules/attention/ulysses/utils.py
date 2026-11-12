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
    config, inputs_embeds, attention_mask, cache_position, past_key_values, position_ids
):
    # Attention mask passthrough for flash-attention style dispatch: the
    # kernels take causality (and sliding windows) from their own kwargs and
    # padding/packing from the 2D mask, so no 4D mask is built here. Models
    # call this with keyword arguments, so the parameter names must match the
    # upstream factories.
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


def slice_input_tensor(x: Tensor, dim: int, group: ProcessGroup, rank: int) -> Tensor:
    """Return this rank's contiguous shard of a sequence-aligned tensor.

    Args:
        x: Global tensor whose ``dim`` is a multiple of the SP size.
        dim: Sequence dimension.
        group: Sequence parallel process group.
        rank: Rank within ``group``.

    Returns:
        The local shard along ``dim``.
    """
    sp_size = dist.get_world_size(group)
    seq_len = x.size(dim)
    if seq_len % sp_size:
        raise ValueError(
            f"Sequence length {seq_len} is not divisible by SP size {sp_size}; "
            "pad the batch with ulysses_pad_inputs() before slicing."
        )
    local_len = seq_len // sp_size
    return x.narrow(dim, rank * local_len, local_len).contiguous()
