import inspect
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch import Tensor
from torch.distributed import ProcessGroup
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ...comm.comm import gather_from_sequence_parallel_region
from .utils import (
    repeat_kv,
    unpad_tensor,
    pad_tensor,
    slice_input_tensor,
)
from torchturbo.logging import get_logger

logger = get_logger(__name__)


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_group: Optional[dist.ProcessGroup],
    sp_size: int,
    async_op: bool = False,
):
    input_list = [
        t.contiguous() for t in torch.tensor_split(local_input, sp_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(sp_size)]
    comm = dist.all_to_all(output_list, input_list, group=sp_group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        sp_group: dist.ProcessGroup,
        sp_size: int,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        ctx.sp_group = sp_group
        ctx.sp_size = sp_size
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(
            local_input, scatter_dim, gather_dim, sp_group, sp_size, async_op
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: Tensor
    ) -> Tuple[None, None, Tensor, None, None, None]:
        input_t = (
            torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
            if ctx.async_op
            else grad_output[0]
        )
        return (
            None,
            None,
            all_to_all_tensor(
                input_t,
                ctx.gather_dim,
                ctx.scatter_dim,
                ctx.sp_group,
                ctx.sp_size,
                False,
            ),
            None,
            None,
            None,
        )


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    sp_group: ProcessGroup,
    sp_size: int,
    unpadded_dim_size: int = 0,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    if not sp_group:
        return x
    x = SeqAllToAll.apply(sp_group, sp_size, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_size != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(
    x: Tensor, head_dim: int, seq_dim: int, sp_group: ProcessGroup, sp_size: int
) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    if not sp_group:
        return x
    dim_size = x.size(seq_dim)
    if dim_size % sp_size != 0:
        padding_size = sp_size - (dim_size % sp_size)
        x = pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(sp_group, sp_size, x, seq_dim, head_dim, False)


def ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    query_length,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    sp_group: ProcessGroup,
    sp_size: int,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ########## AlltoAll for Ulysses ##########
    if sp_size > 1:
        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(
            query_states, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
        )
        key_states = gather_seq_scatter_heads(
            key_states, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
        )
        value_states = gather_seq_scatter_heads(
            value_states, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
        )

        # all_gather position_ids when available
        if position_ids is not None:
            # TODO: eliminate this all_gather by passing cu_seq_lens explicitly.
            # https://github.com/huggingface/transformers/pull/33932
            position_ids_list = [torch.empty_like(position_ids) for _ in range(sp_size)]
            torch.distributed.all_gather(position_ids_list, position_ids, group=sp_group)
            position_ids = torch.concat(position_ids_list, dim=-1)
        if query_length != query_states.size(1):
            assert query_length * sp_size == query_states.size(
                1
            ), "ulysses fa only pass 1/sp query here"
            query_length = query_states.size(1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        *args,
        position_ids=position_ids,
        **kwargs,
    )

    ########## AlltoAll for Ulysses ##########
    if sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(
            attn_output, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
        )

    return attn_output


def _forward_arg_index(forward, names: tuple[str, ...], skip_first: int):
    try:
        parameters = list(inspect.signature(forward).parameters.values())
    except (TypeError, ValueError):
        return None

    positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    positional_index = 0
    for parameter in parameters:
        if parameter.kind in positional_kinds:
            if parameter.name in names:
                arg_index = positional_index - skip_first
                return arg_index if arg_index >= 0 else None
            positional_index += 1
    return None


def patch_kda_attention_forward_for_sp(
    attn_cls: type,
    sp_group: ProcessGroup,
    sp_size: int,
    sp_rank: int,
):
    """Patch KDA/DSA-style sequence-state attention for correctness under SP.

    These layers carry state along the sequence dimension, so local shard-only
    execution would break causality. This wrapper gathers the full sequence,
    runs the original forward, then scatters the output sequence back.
    """

    if sp_size <= 1 or sp_group is None:
        return

    origin_attr = "_torchturbo_origin_forward"
    origin_forward = getattr(attn_cls, origin_attr, None)
    if origin_forward is None:
        origin_forward = attn_cls.forward
        setattr(attn_cls, origin_attr, origin_forward)

    cache_arg_index = _forward_arg_index(
        origin_forward, ("past_key_values", "past_key_value"), skip_first=2
    )

    def sp_forward(self, hidden_states, *args, **kwargs):
        return kda_attention_sp_forward(
            self,
            origin_forward,
            hidden_states,
            args,
            kwargs,
            cache_arg_index,
            sp_group,
            sp_size,
            sp_rank,
        )

    attn_cls.forward = sp_forward


def kda_attention_sp_forward(
    module,
    origin_forward,
    hidden_states: Tensor,
    args,
    kwargs,
    cache_arg_index: Optional[int],
    sp_group: ProcessGroup,
    sp_size: int,
    sp_rank: int,
):
    past = kwargs.get("past_key_values", kwargs.get("past_key_value", None))
    if past is None and cache_arg_index is not None and len(args) > cache_arg_index:
        past = args[cache_arg_index]
    if past is not None:
        raise NotImplementedError(
            "Sequence-parallel KDA/DSA attention does not support KV cache."
        )

    local_seq_len = hidden_states.size(1)
    full_hidden_states = all_gather_sp_with_grad_reduce(
        hidden_states, dim=1, sp_group=sp_group, sp_size=sp_size
    )
    result = origin_forward(module, full_hidden_states, *args, **kwargs)
    is_tuple = isinstance(result, (tuple, list))
    if is_tuple:
        output, *rest = result
    else:
        output = result
    output = slice_input_tensor(
        output,
        dim=1,
        group=sp_group,
        rank=sp_rank,
        padding=False,
    )
    if output.size(1) != local_seq_len:
        output = output[:, :local_seq_len].contiguous()
    if is_tuple:
        return (output, *rest)
    return output


# copied from verl
def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            position_ids = kwargs.get("position_ids")
            visual_pos_masks = kwargs.get("visual_pos_masks")
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(
                    inputs_embeds, dim=1, padding=False
                )
                call_kwargs["position_ids"] = slice_input_tensor(
                    position_ids, dim=-1, padding=False
                )
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models
                if visual_pos_masks is not None:
                    original_visual_mask = visual_pos_masks
                    sliced_visual_mask = slice_input_tensor(
                        visual_pos_masks, dim=1, padding=False
                    )
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    if deepstack_visual_embeds is not None:
                        sliced_embeds = []

                        num_visual_before = original_visual_mask.sum().item()
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard
                            rank = get_ulysses_sequence_parallel_rank()
                            seq_len = original_visual_mask.shape[1]
                            local_seq_len = seq_len // current_ulysses_sp_size
                            start_idx = rank * local_seq_len
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            visual_start = (
                                original_visual_mask[:, :start_idx].sum().item()
                                if start_idx > 0
                                else 0
                            )
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[visual_start:visual_end])
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[:0])
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds

                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    logger.info_rank0("Monkey patch %s.forward for Ulysses SP input slicing.", model_class.__name__)


def patch_forward_for_sp_input_slicing(
    model_class: type, sp_group: dist.ProcessGroup, sp_size: int
):
    if sp_size <= 1:
        return

    original_forward = model_class.forward
    sp_rank = dist.get_rank(sp_group)

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            position_ids = kwargs.get("position_ids")
            visual_pos_masks = kwargs.get("visual_pos_masks")
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = sp_size

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(
                    inputs_embeds, dim=1, group=sp_group, rank=sp_rank, padding=False
                )
                call_kwargs["position_ids"] = slice_input_tensor(
                    position_ids, dim=-1, group=sp_group, rank=sp_rank, padding=False
                )
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models
                if visual_pos_masks is not None:
                    original_visual_mask = visual_pos_masks
                    sliced_visual_mask = slice_input_tensor(
                        visual_pos_masks,
                        dim=1,
                        group=sp_group,
                        rank=sp_rank,
                        padding=False,
                    )
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    if deepstack_visual_embeds is not None:
                        sliced_embeds = []

                        num_visual_before = original_visual_mask.sum().item()
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard
                            rank = sp_rank
                            seq_len = original_visual_mask.shape[1]
                            local_seq_len = seq_len // current_ulysses_sp_size
                            start_idx = rank * local_seq_len
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            visual_start = (
                                original_visual_mask[:, :start_idx].sum().item()
                                if start_idx > 0
                                else 0
                            )
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[visual_start:visual_end])
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[:0])
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds

                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward


class SPAllGather(Function):
    """
    Custom all_gather autograd function for sequence parallelism.
    Supports input shapes:
      - [seq_len]                -> gather on dim=0
      - [bs, seq_len, ...]       -> gather on dim=1

    When ``grad_scaler=True``, the backward pass multiplies each rank's local
    slice by ``sp_size`` before returning it. This is the gradient-equivalent of a
    reduce-scatter when the gathered tensor is consumed identically on every SP
    rank (i.e. every rank produces the same ``grad_output``), replacing the
    cross-rank sum of reduce_scatter with a local scaling factor and thus saving
    one collective.
    """

    @staticmethod
    def forward(ctx, input_tensor, seq_dim, sp_group, sp_size, grad_scaler=False):
        ctx.sp_group = sp_group
        ctx.sp_size = sp_size
        ctx.input_shape = input_tensor.shape
        ctx.seq_dim = seq_dim
        ctx.grad_scaler = grad_scaler

        # Save original shape and flatten non-sequence dims for stability
        orig_size = input_tensor.size()
        if input_tensor.dim() > 2:
            # [bs, seq_len, ...] -> [bs, seq_len, -1]
            input_tensor = input_tensor.view(orig_size[0], orig_size[1], -1)
        input_tensor = input_tensor.contiguous()

        # Prepare tensor list for all_gather
        tensor_list = [torch.empty_like(input_tensor) for _ in range(sp_size)]
        dist.all_gather(tensor_list, input_tensor, group=sp_group)

        # Concatenate along the sequence dimension
        output = torch.cat(tensor_list, dim=seq_dim)

        # Restore original shape (with expanded sequence dim)
        # Original: [bs, L, ...], output: [bs, L*sp_size, -1]
        new_shape = list(orig_size)
        new_shape[seq_dim] = output.size(seq_dim)  # update seq_len
        output = output.view(new_shape)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank(ctx.sp_group)
        seq_dim = ctx.seq_dim

        if ctx.grad_scaler:
            # Equivalent to a reduce-scatter when every rank's grad_output is identical
            grad_output = grad_output * ctx.sp_size

        # Split along sequence dimension
        chunks = torch.chunk(grad_output, ctx.sp_size, dim=seq_dim)
        grad_input = chunks[rank].contiguous()

        # Reshape back to original input shape
        grad_input = grad_input.view(ctx.input_shape)

        return grad_input, None, None, None, None


def all_gather_sp(
    tensor: Tensor,
    dim: int,
    sp_group: ProcessGroup,
    sp_size: int,
    grad_scaler: bool = False,
) -> Tensor:
    """All-gather tensor in SP group and concatenate along specified dimension.

    Args:
        tensor: Input tensor to gather
        dim: Dimension to concatenate along
        sp_group: Process group for SP
        sp_size: SP size
        grad_scaler: If True, scale each rank's local backward gradient by
            ``sp_size``. Use only when the gathered tensor is consumed
            identically on every SP rank (replicated downstream computation).

    Returns:
        Gathered tensor with shape expanded along `dim`
    """
    return SPAllGather.apply(tensor, dim, sp_group, sp_size, grad_scaler)


def all_gather_sp_with_grad_reduce(
    tensor: Tensor, dim: int, sp_group: ProcessGroup, sp_size: int
) -> Tensor:
    """All-gather tensor and reduce gradients for replicated full-sequence use.

    This is for layers that must run on the full sequence on every SP rank and
    then slice their output back to the local shard. Each rank's local loss may
    still depend on hidden states from other sequence shards, so backward must
    sum gradient contributions for each original shard across all ranks.
    """

    if sp_size <= 1 or sp_group is None:
        return tensor
    if dim < 0:
        dim = tensor.dim() + dim
    tensor = tensor.movedim(dim, 0).contiguous()
    tensor = gather_from_sequence_parallel_region(
        tensor, tensor_parallel_output_grad=True, group=sp_group
    )
    return tensor.movedim(0, dim).contiguous()
