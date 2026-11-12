import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torchturbo.utils.functools import cache


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r"""Get the sequnce lengths in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [2, 3, 1, 2, 3]
    ```
    """
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts: torch.Tensor = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    return seqlens


# packing attention mask copied from llamma-factory
def get_unpad_data_lf(
    attention_mask: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    r"""Prepare the indices and seqlens for flash attn varlen function.

    Returns:
        indices: indices of non-masked tokens from the flattened sequence.
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        max_seqlen_in_batch: the largest seqlen in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # output
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 2, 5, 6, 8, 11]
    3
    ```

    """
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


# non-packing attention mask
def get_unpad_data_normal(attention_mask):
    r"""Prepare the indices and seqlens for flash attn varlen function.

    Returns:
        indices: indices of non-masked tokens from the flattened sequence.
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        max_seqlen_in_batch: the largest seqlen in the current batch.

    e.g.
    ```python
    # input
    [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    # output
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 5, 11]
    6
    ```

    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

@cache
def _get_unpad_data(
    attention_mask: torch.Tensor, _version: int | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if torch.max(attention_mask) > 1:
        return get_unpad_data_lf(attention_mask)
    else:
        return get_unpad_data_normal(attention_mask)


def get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Reuse unpadding metadata until the padding or segment mask changes."""
    if torch.is_inference_mode_enabled() or attention_mask.is_inference():
        return _get_unpad_data.__wrapped__(attention_mask, None)
    return _get_unpad_data(attention_mask, attention_mask._version)


@cache
def _packed_token_counts(
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    _query_version: int | None,
    _key_version: int | None,
) -> tuple[int, int]:
    num_query_tokens = int(cu_seq_lens_q[-1])
    num_key_tokens = (
        num_query_tokens if cu_seq_lens_k is cu_seq_lens_q else int(cu_seq_lens_k[-1])
    )
    return num_query_tokens, num_key_tokens


def unpad_varlen_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Trim BSND attention inputs to the valid prefixes described by cu.

    Returns:
        Query, key and value tensors, and the number of query tail tokens
        removed. Trimming combines batch rows into one; inputs without
        padding retain their original shape and identity.
    """
    if (
        torch.is_inference_mode_enabled()
        or cu_seq_lens_q.is_inference()
        or cu_seq_lens_k.is_inference()
    ):
        num_query_tokens, num_key_tokens = _packed_token_counts.__wrapped__(
            cu_seq_lens_q, cu_seq_lens_k, None, None,
        )
    else:
        num_query_tokens, num_key_tokens = _packed_token_counts(
            cu_seq_lens_q, cu_seq_lens_k, cu_seq_lens_q._version, cu_seq_lens_k._version,
        )
    query_padding = query.shape[0] * query.shape[1] - num_query_tokens
    key_padding = key.shape[0] * key.shape[1] - num_key_tokens
    if query_padding == 0 and key_padding == 0:
        return query, key, value, 0
    query = query.flatten(0, 1)[:num_query_tokens].unsqueeze(0)
    key = key.flatten(0, 1)[:num_key_tokens].unsqueeze(0)
    value = value.flatten(0, 1)[:num_key_tokens].unsqueeze(0)
    return query, key, value, query_padding


@cache
def _cu_seqlens_to_segment_ids(
    cu_seqlens: torch.Tensor, num_tokens: int, _version: int | None,
) -> torch.Tensor:
    token_indices = torch.arange(num_tokens, device=cu_seqlens.device)
    segment_ids = torch.bucketize(token_indices, cu_seqlens[1:], right=True) + 1
    return segment_ids.masked_fill(token_indices >= cu_seqlens[-1], 0).unsqueeze(0)


def cu_seqlens_to_segment_ids(cu_seqlens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Derive per-token segment IDs from cu boundaries.

    Args:
        cu_seqlens: Cumulative lengths addressing flattened batch tokens.
        num_tokens: Physical token count, including any tail padding.

    Returns:
        Integer segment IDs shaped [1, num_tokens] (1..n per packed sample,
        0 beyond the last boundary). Not an attention mask: consumers use
        ID equality to build block-diagonal packing structure.
    """
    if torch.is_inference_mode_enabled() or cu_seqlens.is_inference():
        return _cu_seqlens_to_segment_ids.__wrapped__(cu_seqlens, num_tokens, None)
    return _cu_seqlens_to_segment_ids(cu_seqlens, num_tokens, cu_seqlens._version)


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    # dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def _unpad_input_core(input, indices):
    return index_first_axis(rearrange(input, "b s ... -> (b s) ..."), indices)


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (
        (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    )
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        _unpad_input_core(hidden_states, indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def packing_mask_to_causal_mask(attention_mask: "torch.Tensor") -> "torch.Tensor":
    r"""Convert packing attention mask to causal attention mask.

    Args:
        attention_mask: (batch, seqlen), packing mode where values 1, 2, 3... indicate
                        different sequences within each batch, and 0 indicates padding.

    Returns:
        causal_mask: (total_len, total_len) where total_len is the total number of
                     non-padding tokens. Each sequence block is a lower triangular matrix.

    e.g.
    ```python
    # input (batch=2, seqlen=6)
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    # sequences: [seq1(len=2), seq2(len=3), seq3(len=1), seq4(len=2), seq5(len=3)]
    # total_len = 2 + 3 + 1 + 2 + 3 = 11
    # output shape: (11, 11)
    # Each sequence block is a lower triangular matrix:
    # seq1: positions 0-1, can see 0-1 within seq1
    # seq2: positions 2-4, can see 2-4 within seq2
    # etc.
    ```
    """
    assert attention_mask.size(0) == 1, "packing mode only support bs==1"
    seqlens = get_seqlens_in_batch(attention_mask)  # (num_sequences,)
    total_len = seqlens.sum().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )  # (num_sequences + 1,)

    # Create causal mask: each sequence block is lower triangular
    causal_mask = torch.zeros(
        total_len, total_len, dtype=torch.bool, device=attention_mask.device
    )

    for i in range(len(seqlens)):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        # Lower triangular matrix for this sequence
        causal_mask[start:end, start:end] = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device)
        )

    return causal_mask
