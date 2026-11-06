# Chunked all-to-all MoE path with overlapped communication and expert compute.

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Iterator
import functools

import torch
import torch.distributed as dist

from torchturbo.ops import ops
from torchturbo.ops.moe import MoEPermutation

from .moe_config import MoEConfig
from .token_dispatcher import MoEAlltoAllTokenDispatcher
from .utils import RoutingResult, maybe_move_tensor_to_cpu

__all__ = ["MoEChunkedAlltoAllTokenDispatcher", "run_chunked_moe", "shared_recompute_group"]


@dataclass(frozen=True)
class _ChunkHostSplits:
    """Per-chunk all-to-all split sizes and expert counts on the host."""

    input_splits: list[list[int]]
    output_splits: list[list[int]]
    tokens_per_expert: list[torch.Tensor]


@dataclass(frozen=True)
class _ChunkMetadata:
    """Per-chunk routing counts derived once per layer on device."""

    # int64 [chunks, ep_size, num_local_experts]; received rows per local expert.
    received_counts: torch.Tensor
    # int64 counts shaped [chunks, ep_size]; rows this rank sends per chunk.
    send_counts: torch.Tensor
    # int64 [chunks, ep_size]; rows every rank sends to this rank per chunk.
    receive_counts: torch.Tensor
    # int64 counts shaped [chunks, num_local_experts]; expert totals per chunk.
    tokens_per_expert: torch.Tensor


@dataclass
class _ChunkGraph:
    """Per-chunk expert-segment backward handle built during forward."""

    restored: torch.Tensor
    received_leaf: torch.Tensor
    row_indices: torch.Tensor

    def __repr__(self) -> str:
        # Value storages may be freed; repr must never read tensor data.
        return (
            f"_ChunkGraph(restored={tuple(self.restored.shape)}, "
            f"received_leaf={tuple(self.received_leaf.shape)}, "
            f"row_indices={tuple(self.row_indices.shape)})"
        )


def _release_activation_value(tensor: torch.Tensor | None) -> None:
    """Return a tensor storage to the allocator without freeing the autograd object."""
    if tensor is None:
        return
    try:
        storage = tensor.untyped_storage()
        if storage.size() > 0:
            storage.resize_(0)
    except Exception:
        # Pure memory optimization; allocator quirks must never affect correctness.
        pass


@contextmanager
def shared_recompute_group() -> Iterator[None]:
    """Pin a backward pass to one checkpoint-recompute graph id; no-op under reentrant checkpointing."""
    try:
        from torch.utils.checkpoint import GraphExecGroup
    except ImportError:  # pragma: no cover - torch without GraphExecGroup
        yield
        return
    if GraphExecGroup._get_current_group() is not None:
        yield
        return
    with GraphExecGroup():
        yield


_ORIGINAL_AUTOGRAD: dict[str, Any] = {}


def _install_shared_backward_guard() -> None:
    """Route every backward/grad call through one GraphExecGroup per call."""
    if _ORIGINAL_AUTOGRAD:
        return
    try:
        from torch.utils.checkpoint import GraphExecGroup  # noqa: F401
    except ImportError:  # pragma: no cover - torch without GraphExecGroup
        return
    import torch.autograd as _autograd

    _ORIGINAL_AUTOGRAD["backward"] = _autograd.backward
    _ORIGINAL_AUTOGRAD["grad"] = _autograd.grad

    @functools.wraps(_autograd.backward)
    def backward(*args: Any, **kwargs: Any) -> Any:
        with shared_recompute_group():
            return _ORIGINAL_AUTOGRAD["backward"](*args, **kwargs)

    @functools.wraps(_autograd.grad)
    def autograd_grad(*args: Any, **kwargs: Any) -> Any:
        with shared_recompute_group():
            return _ORIGINAL_AUTOGRAD["grad"](*args, **kwargs)

    _autograd.backward = backward
    _autograd.grad = autograd_grad


def _uninstall_shared_backward_guard() -> None:
    """Restore the original autograd entry points (test support)."""
    if not _ORIGINAL_AUTOGRAD:
        return
    import torch.autograd as _autograd

    _autograd.backward = _ORIGINAL_AUTOGRAD["backward"]
    _autograd.grad = _ORIGINAL_AUTOGRAD["grad"]
    _ORIGINAL_AUTOGRAD.clear()


def _fsdp_param_groups(fsdp_state: Any) -> list[Any]:
    """Return the FSDP2 parameter groups as a list across torch layouts."""
    groups = getattr(fsdp_state, "_fsdp_param_groups", None)
    if groups is None:
        group = getattr(fsdp_state, "_fsdp_param_group", None)
        return [] if group is None else [group]
    return list(groups)


def _fsdp_groups(experts: torch.nn.Module) -> list[Any] | None:
    """Return the experts FSDP parameter groups, running lazy init when it exists."""
    try:
        from torch.distributed.fsdp._fully_shard import FSDPModule
    except ImportError:  # pragma: no cover - torch without FSDP2 internals
        return None
    if not isinstance(experts, FSDPModule):
        return None
    fsdp_state = experts._get_fsdp_state()
    lazy_init = getattr(fsdp_state, "_lazy_init", None)
    if lazy_init is not None:
        lazy_init()
    return _fsdp_param_groups(fsdp_state) or None


@contextmanager
def _fsdp_forward_hold(experts: torch.nn.Module) -> Iterator[None]:
    """A single unshard/reshard around all per-chunk expert forward calls."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            TrainingState,
            is_bw,
        )
    except ImportError:
        TrainingState = None  # pragma: no cover - torch without FSDP2 internals

        def is_bw() -> bool:  # fallback: never defer the reshard
            return False
    groups = _fsdp_groups(experts)
    if groups is None:
        yield
        return
    # Mirror pre_forward's state transition because module hooks are bypassed.
    saved_states = [
        getattr(group, "_training_state", None) for group in groups
    ]
    for group in groups:
        if TrainingState is not None:
            group._training_state = TrainingState.FORWARD
        group.unshard()
        group.wait_for_unshard()
    try:
        yield
    finally:
        if not is_bw():
            # Normal forward reshard once; recompute defers reshard to backward.
            for group in groups:
                group.reshard()
            for group, saved in zip(groups, saved_states):
                if saved is not None:
                    group._training_state = saved


def _fsdp_state_is_post_backward(group: Any) -> bool:
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            TrainingState,
        )
    except ImportError:  # pragma: no cover - torch without FSDP2 internals
        return False
    return (
        getattr(group, "_training_state", None) == TrainingState.POST_BACKWARD
    )


def _fsdp_reset_group_state_to_idle(group: Any) -> None:
    """Reset an FSDP group to IDLE after a manual post_backward."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            TrainingState,
        )
    except ImportError:  # pragma: no cover - torch without FSDP2 internals
        return
    if getattr(group, "_training_state", None) == TrainingState.POST_BACKWARD:
        group._training_state = TrainingState.IDLE


@contextmanager
def _fsdp_backward_hold(experts: torch.nn.Module) -> Iterator[None]:
    """A single unshard and post_backward flush around chunked expert backward."""
    groups = _fsdp_groups(experts)
    if groups is None:
        yield
        return
    for group in groups:
        # Idempotent unshard; recomputed forward may leave weights unsharded.
        group.unshard()
        group.wait_for_unshard()
        set_grad_dtypes = getattr(group, "_set_unsharded_grad_dtypes", None)
        if set_grad_dtypes is not None:
            set_grad_dtypes(defer_upcast=True)
    try:
        yield
    finally:
        # Single flush at the end of the backward task: one reduce-scatter plus reshard.
        def _finalize_chunked_backward(fsdp_groups=tuple(groups)):
            for group in fsdp_groups:
                if not _fsdp_state_is_post_backward(group):
                    group.post_backward()
                # Restore IDLE so the next backward's flush is not skipped.
                _fsdp_reset_group_state_to_idle(group)
        try:
            torch.autograd.Variable._execution_engine.queue_callback(
                _finalize_chunked_backward,
            )
        except Exception:  # pragma: no cover - engine callback unavailable
            _finalize_chunked_backward()


class MoEChunkedAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """Chunk-level all-to-all layout and token movement helpers."""

    def __init__(
        self, num_local_experts: int, local_expert_indices: list[int],
        config: MoEConfig, ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(num_local_experts, local_expert_indices, config=config, ep_group=ep_group)
        if config.moe_apply_probs_on_input:
            raise ValueError(
                "alltoall_chunked applies routing weights on expert outputs; "
                "moe_apply_probs_on_input is unsupported."
            )
        _install_shared_backward_guard()
        if self.ep_size <= 1:
            raise ValueError(
                "alltoall_chunked requires expert parallel size > 1; "
                "use the alltoall dispatcher for ep=1."
            )
        if self.device_runtime.current_stream() is None:
            raise ValueError(
                "alltoall_chunked requires a device stream."
            )
        self.num_chunks = max(1, int(config.moe_num_chunks))
        self._comm_stream = self.device_runtime.new_stream()

    def _prepare_chunks(
        self, flat: torch.Tensor, routing: RoutingResult,
    ) -> tuple[
        torch.Tensor, RoutingResult, list[tuple[int, int]],
        "_ChunkMetadata", "_ChunkHostSplits", torch.Tensor,
    ]:
        """Derive chunk bounds and host splits for contiguous token chunks."""
        bounds = self._chunk_bounds(flat.shape[0])
        metadata = self._chunk_metadata(routing, bounds)
        host = self._host_metadata(metadata)
        # Output-side weights stay dense on the source rank for the fused merge.
        dense_probs = routing.probs.to(torch.float32)
        return flat, routing, bounds, metadata, host, dense_probs

    def _chunk_bounds(self, num_tokens: int) -> list[tuple[int, int]]:
        """Split token rows into contiguous chunk row intervals."""
        chunk_count = max(1, self.num_chunks)
        base_size = num_tokens // chunk_count
        remainder = num_tokens % chunk_count
        bounds, start = [], 0
        for index in range(chunk_count):
            end = start + base_size + (1 if index < remainder else 0)
            bounds.append((start, end))
            start = end
        return bounds

    def _chunk_metadata(
        self,
        routing: RoutingResult,
        bounds: list[tuple[int, int]],
    ) -> _ChunkMetadata:
        """Build per-chunk count matrices with a fast histogram and one gather."""
        routing_map = routing.routing_map
        chunk_count = len(bounds)
        # Masking reduction avoids the index_put_ accumulate that stalls NPU.
        counts = torch.stack(
            [routing_map[start:end].sum(dim=0) for start, end in bounds]
        ).long()
        local_start = self.local_expert_indices[0]
        local_end = self.local_expert_indices[-1] + 1
        gathered = self._gather_counts(counts, chunk_count)
        local_counts = gathered[:, :, local_start:local_end]
        received_counts = local_counts.permute(1, 0, 2).contiguous()
        send_counts = counts.view(
            chunk_count, self.ep_size, self.num_local_experts
        ).sum(dim=2)
        receive_counts = received_counts.sum(dim=2).contiguous()
        return _ChunkMetadata(
            received_counts,
            send_counts,
            receive_counts,
            received_counts.sum(dim=1),
        )

    def _gather_counts(self, counts: torch.Tensor, chunk_count: int) -> torch.Tensor:
        """All-gather the chunk histogram on the communication stream."""
        gathered = counts.new_empty(self.ep_size, chunk_count, self.num_experts)
        runtime = self.device_runtime
        compute = runtime.current_stream()
        counts_ready = runtime.new_event()
        counts_ready.record(compute)
        comm = self._comm_stream
        comm.wait_event(counts_ready)
        with runtime.stream(comm):
            dist.all_gather_into_tensor(
                gathered.view(-1), counts.view(-1), group=self.ep_group,
            )
        gathered_ready = runtime.new_event()
        gathered_ready.record(comm)
        runtime.record_stream(gathered, comm)
        compute.wait_event(gathered_ready)
        return gathered

    def _host_metadata(self, metadata: _ChunkMetadata) -> _ChunkHostSplits:
        """Copy every chunk split in one transfer and synchronize once."""
        chunk_count = metadata.send_counts.shape[0]
        pack = torch.cat([
            metadata.send_counts.reshape(-1),
            metadata.receive_counts.reshape(-1),
            metadata.tokens_per_expert.reshape(-1),
        ])
        current_stream = self.device_runtime.current_stream()
        on_side_stream = (
            current_stream is not None
            and self.dtoh_stream is not None
            and current_stream != self.dtoh_stream
        )
        if on_side_stream:
            self.dtoh_stream.wait_stream(current_stream)
        with self.device_runtime.stream(self.dtoh_stream):
            pack = maybe_move_tensor_to_cpu(pack, record_stream=on_side_stream)
        self.device_runtime.synchronize_stream(self.dtoh_stream)
        ranks = self.ep_size
        size = chunk_count * ranks
        send = pack[:size].view(chunk_count, ranks)
        receive = pack[size : 2 * size].view(chunk_count, ranks)
        tpe_size = chunk_count * self.num_local_experts
        tokens_per_expert = pack[2 * size : 2 * size + tpe_size].view(
            chunk_count, self.num_local_experts
        )
        return _ChunkHostSplits(
            [row.tolist() for row in send],
            [row.tolist() for row in receive],
            [tokens_per_expert[index] for index in range(chunk_count)],
        )

    def _permute_chunk(
        self, flat: torch.Tensor, routing: RoutingResult, start: int, end: int,
    ) -> MoEPermutation:
        return ops.moe_permutation(
            flat[start:end],
            routing.routing_map[start:end],
            probs=None,
            num_out_tokens=(end - start) * self.config.moe_router_topk,
            expert_indices=routing.expert_indices[start:end],
        )

    def _sort_received(
        self, received: torch.Tensor, received_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Transpose received chunks from source-rank order to expert order."""
        if self.num_local_experts == 1:
            return received
        sorted_tokens, _ = ops.moe_sort_chunks(
            received, received_counts.ravel(), self.sort_input_by_local_experts,
        )
        return sorted_tokens

    def _restore_order(
        self, expert_output: torch.Tensor, received_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Transpose expert outputs back to source-rank chunk order."""
        if self.num_local_experts == 1:
            return expert_output
        restored, _ = ops.moe_sort_chunks(
            expert_output,
            received_counts.t().ravel(),
            self.restore_output_by_local_experts,
        )
        return restored

    def _merge_chunk(
        self,
        combined: torch.Tensor,
        row_indices: torch.Tensor,
        expert_indices: torch.Tensor,
        dense_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Weight and sum the chunk expert rows back into source tokens with the fused unpermutation kernel."""
        restore_shape = (row_indices.shape[0], combined.shape[-1])
        chunk_permutation = MoEPermutation(
            combined, None, row_indices, expert_indices,
        )
        return ops.moe_unpermutation(
            chunk_permutation, restore_shape=restore_shape, probs=dense_probs,
        )

    def _a2a_into(
        self,
        input_: torch.Tensor,
        output: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        comm: Any = None,
    ) -> torch.Tensor:
        """Run a bare all-to-all writing into a preallocated output buffer."""
        input_ = input_.contiguous()
        if comm is not None:
            self.device_runtime.record_stream(input_, comm)
        dist.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.ep_group,
        )
        return output

    # Orchestrated backward (owned by _ChunkedMoEFunction)

    def _backward_merge_chunk(
        self,
        grad_out_chunk: torch.Tensor,
        graph: "_ChunkGraph",
        combined_slice: torch.Tensor,
        dense_probs_chunk: torch.Tensor,
        expert_indices_chunk: torch.Tensor,
        need_input_grads: bool,
        need_probs_grad: bool,
        g_combined_out: torch.Tensor | None = None,
    ) -> tuple["torch.Tensor | None", "torch.Tensor | None"]:
        """Manual adjoint of the weighted merge computed with gather/multiply/scatter primitives."""
        chunk_tokens = grad_out_chunk.shape[0]
        topk = graph.row_indices.shape[1]
        row_indices = graph.row_indices.to(torch.long)
        packed = row_indices.reshape(-1)
        num_rows = packed.numel()
        # Broadcast g_merged into per-(token, slot) order.
        g_merged_ij = (
            grad_out_chunk.unsqueeze(1)
            .expand(chunk_tokens, topk, grad_out_chunk.shape[-1])
            .reshape(num_rows, grad_out_chunk.shape[-1])
        )
        g_combined: "torch.Tensor | None" = None
        if need_input_grads:
            weights = dense_probs_chunk.gather(1, expert_indices_chunk)
            # Mixed dtype promotes to FP32 in one multiply.
            g_flat = g_merged_ij * weights.flatten().unsqueeze(-1)
            # Row indices are a bijection over the chunk rows.
            if g_combined_out is None:
                g_combined = grad_out_chunk.new_zeros(
                    num_rows, grad_out_chunk.shape[-1],
                )
                g_combined[packed] = g_flat
            else:
                g_combined = g_combined_out
                g_combined.zero_()
                g_combined[packed] = g_flat
        g_probs: "torch.Tensor | None" = None
        if need_probs_grad:
            # g_probs[i, e] = sum_j g_merged[i] . combined[inv[i, j]].
            combined_ij = combined_slice.index_select(0, packed)
            d_weights = (
                g_merged_ij.to(torch.float32)
                * combined_ij.to(torch.float32)
            ).sum(dim=-1)
            g_probs = torch.zeros(
                chunk_tokens, self.num_experts, dtype=torch.float32,
                device=grad_out_chunk.device,
            )
            g_probs.scatter_add_(
                1, expert_indices_chunk, d_weights.view(chunk_tokens, topk),
            )
        return g_combined, g_probs

    def _backward_permute_chunk(
        self,
        grad_flat_slice: torch.Tensor,
        grad_permuted: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> None:
        """Scatter permuted gradients back into the flat chunk slice using the saved indices."""
        indices = row_indices.to(torch.long)
        gathered = grad_permuted[indices.reshape(-1)]
        grad_flat_slice += gathered.view(
            indices.shape[0], indices.shape[1], grad_flat_slice.shape[-1],
        ).sum(dim=1)


def _expert_chunk(
    dispatcher, received, received_counts, tokens_per_expert, experts,
):
    """Expert segment with an autograd graph and FSDP hooks bypassed."""
    with torch.enable_grad():
        rcv_leaf = received.detach().requires_grad_(True)
        expert_input = dispatcher._sort_received(rcv_leaf, received_counts)
        expert_output, bias = experts.forward(
            expert_input, tokens_per_expert, None,
        )
        assert bias is None, "Bias is not supported in MoEChunkedAlltoAllTokenDispatcher"
        restored = dispatcher._restore_order(expert_output, received_counts)
    return restored, rcv_leaf


def _backward_experts_grad(
    graph: "_ChunkGraph", grad_restored: torch.Tensor,
) -> torch.Tensor:
    """Run expert backward through the forward-built graph and return the received-token gradient."""
    torch.autograd.backward(graph.restored, grad_restored)
    grad = graph.received_leaf.grad
    if grad is None:
        grad = torch.zeros_like(graph.received_leaf)
    graph.received_leaf.grad = None
    # Free consumed graph values so backward peak tracks the chunk in flight.
    _release_activation_value(graph.restored)
    _release_activation_value(graph.received_leaf)
    return grad


def _forward_chunked_pipeline(
    dispatcher, flat, routing, experts, bounds, host, received_counts,
    dense_probs, graphs,
):
    """Overlap chunk transfers with expert compute on two streams."""
    runtime = dispatcher.device_runtime
    compute = runtime.current_stream()
    comm = dispatcher._comm_stream
    chunk_count = len(bounds)
    topk = dispatcher.config.moe_router_topk
    merged = flat.new_empty(flat.shape)
    combined_buffer = flat.new_empty(
        (flat.shape[0] * topk, flat.shape[-1]))
    combined_bounds = [(start * topk, end * topk) for start, end in bounds]
    recv_rows = [sum(splits) for splits in host.output_splits]
    recv_offsets = [0]
    for rows in recv_rows[:-1]:
        recv_offsets.append(recv_offsets[-1] + rows)
    dispatch_buffer = flat.new_empty((sum(recv_rows), flat.shape[-1]))

    ev_permute = [runtime.new_event() for _ in range(chunk_count)]
    ev_dispatch = [runtime.new_event() for _ in range(chunk_count)]
    ev_experts = [runtime.new_event() for _ in range(chunk_count)]
    ev_combine = [runtime.new_event() for _ in range(chunk_count)]
    permuted = [None] * chunk_count
    received = [None] * chunk_count
    perm_row_indices = [None] * chunk_count
    perm_expert_indices = [None] * chunk_count

    def enqueue_permute(index):
        start, end = bounds[index]
        permutation = dispatcher._permute_chunk(flat, routing, start, end)
        permuted[index] = permutation
        # Copy merge metadata before releasing the packed source rows.
        perm_row_indices[index] = permutation.row_indices
        perm_expert_indices[index] = permutation.expert_indices
        ev_permute[index].record(compute)

    def enqueue_dispatch(index):
        comm.wait_event(ev_permute[index])
        offset = recv_offsets[index]
        row_count = recv_rows[index]
        slot = dispatch_buffer[offset:offset + row_count]
        received[index] = slot
        with runtime.stream(comm):
            dispatcher._a2a_into(
                permuted[index].tokens, slot,
                host.output_splits[index], host.input_splits[index],
                comm,
            )
        runtime.record_stream(permuted[index].tokens, comm)
        runtime.record_stream(slot, comm)
        # Free packed source rows once dispatch has consumed them.
        permuted[index] = None
        ev_dispatch[index].record(comm)

    def enqueue_combine(index, expert_output):
        comm.wait_event(ev_experts[index])
        c_start, c_end = combined_bounds[index]
        with runtime.stream(comm):
            dispatcher._a2a_into(
                expert_output, combined_buffer[c_start:c_end],
                host.input_splits[index], host.output_splits[index],
                comm,
            )
        runtime.record_stream(combined_buffer, comm)
        runtime.record_stream(expert_output, comm)
        ev_combine[index].record(comm)

    enqueue_permute(0)
    enqueue_dispatch(0)
    if chunk_count > 1:
        enqueue_permute(1)
        enqueue_dispatch(1)

    # Phase 1: run every chunk's experts and enqueue its combine.
    for index in range(chunk_count):
        compute.wait_event(ev_dispatch[index])
        runtime.record_stream(received[index], compute)
        expert_output, rcv_leaf = _expert_chunk(
            dispatcher, received[index], received_counts[index],
            host.tokens_per_expert[index], experts,
        )
        graphs.append(
            _ChunkGraph(expert_output, rcv_leaf, perm_row_indices[index])
        )
        ev_experts[index].record(compute)
        if index + 2 < chunk_count:
            enqueue_permute(index + 2)
            enqueue_dispatch(index + 2)
        enqueue_combine(index, expert_output)
        expert_output = None
        received[index] = None

    # Phase 2: wait for each combine on demand and merge in order.
    for index in range(chunk_count):
        compute.wait_event(ev_combine[index])
        start, end = bounds[index]
        c_start, c_end = combined_bounds[index]
        combined_chunk = combined_buffer[c_start:c_end]
        runtime.record_stream(combined_chunk, compute)
        merged_chunk = dispatcher._merge_chunk(
            combined_chunk, perm_row_indices[index],
            perm_expert_indices[index], dense_probs[start:end],
        )
        merged[start:end].copy_(merged_chunk)
        merged_chunk = None
        perm_row_indices[index] = None
        perm_expert_indices[index] = None
    return merged, combined_buffer


def _backward_chunked_pipeline(
    dispatcher, *, bounds, combined_bounds, host, graphs, dense_probs,
    expert_indices, combined, grad_flat_in, grad_flat, grad_probs,
    need_input_grads, need_probs_grad,
):
    """Mirror of the forward pipeline, run in reverse per chunk."""
    runtime = dispatcher.device_runtime
    compute = runtime.current_stream()
    comm = dispatcher._comm_stream
    chunk_count = len(bounds)
    topk = dispatcher.config.moe_router_topk
    hidden_cols = grad_flat_in.shape[-1]
    combined_rows = grad_flat_in.shape[0] * topk
    recv_rows = [sum(splits) for splits in host.output_splits]
    recv_offsets = [0]
    for rows in recv_rows[:-1]:
        recv_offsets.append(recv_offsets[-1] + rows)
    # Preallocate one contiguous workspace per boundary instead of one output per chunk.
    grad_combined_buffer = (
        grad_flat_in.new_empty((combined_rows, hidden_cols))
        if need_input_grads else None
    )
    grad_restored_buffer = (
        grad_flat_in.new_empty((sum(recv_rows), hidden_cols))
        if need_input_grads else None
    )
    # Reuse the forward combine rows as the dispatch-backward output.
    grad_permuted_buffer = combined if need_input_grads else None
    ev_merge = [runtime.new_event() for _ in range(chunk_count)]
    ev_combine_bwd = [runtime.new_event() for _ in range(chunk_count)]
    ev_experts = [runtime.new_event() for _ in range(chunk_count)]
    ev_dispatch = [runtime.new_event() for _ in range(chunk_count)]
    grad_restored: list = [None] * chunk_count
    grad_combined: list = [None] * chunk_count
    grad_received: list = [None] * chunk_count
    grad_permuted: list = [None] * chunk_count
    row_indices = [None] * chunk_count

    def run_merge_backward(index):
        start, end = bounds[index]
        c_start, c_end = combined_bounds[index]
        g_combined, g_probs = dispatcher._backward_merge_chunk(
            grad_flat_in[start:end], graphs[index],
            combined[c_start:c_end], dense_probs[start:end],
            expert_indices[start:end], need_input_grads, need_probs_grad,
            g_combined_out=(
                grad_combined_buffer[c_start:c_end]
                if need_input_grads else None
            ),
        )
        grad_combined[index] = g_combined
        if grad_probs is not None:
            grad_probs[start:end] = g_probs
        if not need_input_grads:
            # No expert backward follows; drop the graph after merge adjoint.
            graphs[index] = None
        ev_merge[index].record(compute)

    def enqueue_combine_backward(index):
        comm.wait_event(ev_merge[index])
        offset = recv_offsets[index]
        row_count = recv_rows[index]
        slot = grad_restored_buffer[offset:offset + row_count]
        grad_restored[index] = slot
        with runtime.stream(comm):
            dispatcher._a2a_into(
                grad_combined[index], slot,
                host.output_splits[index], host.input_splits[index],
                comm,
            )
        runtime.record_stream(grad_combined[index], comm)
        runtime.record_stream(slot, comm)
        grad_combined[index] = None
        ev_combine_bwd[index].record(comm)

    def run_experts_backward(index):
        compute.wait_event(ev_combine_bwd[index])
        runtime.record_stream(grad_restored[index], compute)
        row_indices[index] = graphs[index].row_indices
        grad_received[index] = _backward_experts_grad(
            graphs[index], grad_restored[index],
        )
        grad_restored[index] = None
        graphs[index] = None  # free the chunk graph promptly
        ev_experts[index].record(compute)

    def enqueue_dispatch_backward(index):
        comm.wait_event(ev_experts[index])
        c_start, c_end = combined_bounds[index]
        slot = grad_permuted_buffer[c_start:c_end]
        grad_permuted[index] = slot
        with runtime.stream(comm):
            dispatcher._a2a_into(
                grad_received[index], slot,
                host.input_splits[index], host.output_splits[index],
                comm,
            )
        runtime.record_stream(grad_received[index], comm)
        runtime.record_stream(slot, comm)
        grad_received[index] = None
        ev_dispatch[index].record(comm)

    def run_permute_backward(index):
        compute.wait_event(ev_dispatch[index])
        runtime.record_stream(grad_permuted[index], compute)
        start, end = bounds[index]
        if grad_flat is not None:
            dispatcher._backward_permute_chunk(
                grad_flat[start:end], grad_permuted[index],
                row_indices[index],
            )
        row_indices[index] = None
        grad_permuted[index] = None

    for index in reversed(range(chunk_count)):
        run_merge_backward(index)
    if not need_input_grads:
        return
    for index in reversed(range(chunk_count)):
        enqueue_combine_backward(index)
    for index in reversed(range(chunk_count)):
        run_experts_backward(index)
        enqueue_dispatch_backward(index)
    for index in reversed(range(chunk_count)):
        run_permute_backward(index)


def run_chunked_moe(
    dispatcher: MoEChunkedAlltoAllTokenDispatcher,
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    routing: RoutingResult,
) -> torch.Tensor:
    """Run chunk dispatch, expert compute, and combine through one autograd function."""
    hidden_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_shape[-1])
    merged = _ChunkedMoEFunction.apply(
        dispatcher, experts, flat, routing.probs,
        routing.routing_map, routing.expert_indices,
    )
    return merged.view(hidden_shape)


class _ChunkedMoEFunction(torch.autograd.Function):
    """Autograd function that owns the chunked MoE forward and backward."""

    @staticmethod
    def forward(
        ctx, dispatcher, experts, flat, probs, routing_map, expert_indices,
    ):
        routing = RoutingResult(
            probs=probs, routing_map=routing_map,
            expert_indices=expert_indices,
        )
        flat, routing, bounds, metadata, host, dense_probs = \
            dispatcher._prepare_chunks(flat, routing)
        graphs: "list[_ChunkGraph]" = []
        with _fsdp_forward_hold(experts):
            merged, combined = _forward_chunked_pipeline(
                dispatcher, flat, routing, experts, bounds, host,
                metadata.received_counts, dense_probs, graphs,
            )
        ctx.dispatcher = dispatcher
        ctx.experts = experts
        ctx.bounds = bounds
        ctx.host = host
        ctx.flat_shape = flat.shape
        # Keep per-chunk expert graphs as plain attributes; they die with the backward node.
        ctx.chunk_graphs = graphs
        # Backward consumes the expert graphs structurally, so their value storages can be released here.
        for graph in graphs:
            _release_activation_value(graph.restored)
            _release_activation_value(graph.received_leaf)
        # Merge inputs travel through save_for_backward and rehydrate through the checkpoint.
        ctx.save_for_backward(
            probs, dense_probs, routing.expert_indices, combined,
        )
        return merged

    @staticmethod
    def backward(ctx, grad_output):
        dispatcher = ctx.dispatcher
        (
            probs, dense_probs, expert_indices, combined,
        ) = ctx.saved_tensors
        # needs_input_grad indexes: (dispatcher, experts, flat, probs, routing_map, expert_indices).
        need_input_grads = ctx.needs_input_grad[2] or any(
            param.requires_grad for param in ctx.experts.parameters()
        )
        need_probs_grad = ctx.needs_input_grad[3]
        hidden_cols = ctx.flat_shape[-1]
        grad_flat_in = grad_output.reshape(-1, hidden_cols)
        grad_flat: "torch.Tensor | None" = (
            torch.zeros(
                ctx.flat_shape, dtype=grad_flat_in.dtype,
                device=grad_flat_in.device,
            )
            if need_input_grads else None
        )
        grad_probs: "torch.Tensor | None" = (
            torch.zeros(
                ctx.flat_shape[0], dispatcher.num_experts, dtype=torch.float32,
                device=probs.device,
            )
            if need_probs_grad else None
        )
        topk = dispatcher.config.moe_router_topk
        combined_bounds = [
            (start * topk, end * topk) for start, end in ctx.bounds
        ]
        kwargs = dict(
            bounds=ctx.bounds,
            combined_bounds=combined_bounds,
            host=ctx.host,
            graphs=ctx.chunk_graphs,
            dense_probs=dense_probs,
            expert_indices=expert_indices,
            combined=combined,
            grad_flat_in=grad_flat_in,
            grad_flat=grad_flat,
            grad_probs=grad_probs,
            need_input_grads=need_input_grads,
            need_probs_grad=need_probs_grad,
        )
        with _fsdp_backward_hold(ctx.experts):
            _backward_chunked_pipeline(dispatcher, **kwargs)
        grad_probs_out: "torch.Tensor | None" = None
        if grad_probs is not None:
            grad_probs_out = grad_probs.to(probs.dtype)
        return None, None, grad_flat, grad_probs_out, None, None
