# Chunked all-to-all token dispatcher for MoE layers.
#
# Selected via ``moe_token_dispatcher_type = "alltoall_chunked"`` and kept in
# a dedicated module so the default ``alltoall`` dispatch path imports and
# executes exactly as before. Local tokens are split into contiguous chunks;
# chunk-level all-to-all transfers run on a dedicated communication stream so
# chunk ``c + 1`` transfers while chunk ``c`` computes.

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Iterator
import functools

import torch
import torch.distributed as dist

from torchturbo.ops import ops
from torchturbo.ops.moe import MoEPermutation

from ..comm.comm import all_to_all
from .moe_config import MoEConfig
from .token_dispatcher import MoEAlltoAllTokenDispatcher
from .utils import RoutingResult, maybe_move_tensor_to_cpu

__all__ = ["MoEChunkedAlltoAllTokenDispatcher", "shared_recompute_group"]


@dataclass(frozen=True)
class _ChunkHostSplits:
    """Per-chunk all-to-all split sizes and expert counts on the host."""

    input_splits: list[list[int]]
    output_splits: list[list[int]]
    tokens_per_expert: list[torch.Tensor]


@dataclass(frozen=True)
class _ChunkMetadata:
    """Per-chunk routing counts derived once per layer on device."""

    # int64 counts shaped [chunks, ep_size, num_local_experts]; per-chunk
    # source-rank chunks received for every local expert.
    received_counts: torch.Tensor
    # int64 counts shaped [chunks, ep_size]; rows this rank sends per chunk.
    send_counts: torch.Tensor
    # int64 counts shaped [chunks, ep_size]; rows every rank sends per chunk
    # to this rank.
    receive_counts: torch.Tensor
    # int64 counts shaped [chunks, num_local_experts]; expert totals per chunk.
    tokens_per_expert: torch.Tensor


@dataclass
class _ChunkGraph:
    """Per-chunk expert-segment backward handle built during forward.

    The chunked forward runs only the expert segment under
    ``torch.enable_grad()`` (the dispatch all-to-all output becomes the
    detached ``received_leaf``), so the backward consumes that graph
    directly instead of re-executing the experts. The merge segment is
    *not* part of the graph: its adjoint is computed from the saved
    combine outputs (see ``_backward_merge_chunk``), so this record
    holds no tensor whose value must survive the forward -- activation
    retention stays identical to the plain dispatcher under any
    external recomputation scheme. The value storages of ``restored``
    and ``received_leaf`` are released once the forward pipeline has
    consumed them; only the autograd structure is kept.
    """

    restored: torch.Tensor
    received_leaf: torch.Tensor
    row_indices: torch.Tensor

    def __repr__(self) -> str:
        # Value storages may have been returned to the allocator; a repr
        # must never read tensor data (debuggers/pytest would crash).
        return (
            f"_ChunkGraph(restored{tuple(self.restored.shape)}, "
            f"received_leaf{tuple(self.received_leaf.shape)}, "
            f"row_indices{tuple(self.row_indices.shape)})"
        )


def _release_activation_value(tensor: torch.Tensor | None) -> None:
    """Return *tensor*'s memory to the allocator, keeping the object alive.

    The orchestrated backward consumes the expert-segment graphs
    structurally -- grad_fn chains and leaf ``.grad`` accumulation work
    with an empty storage -- and nothing on the backward path reads the
    released values: the sort/restore permutation nodes keep only index
    metadata, and the expert nodes keep their own (sorted) inputs. The
    release is therefore mode-independent -- eager training, external
    activation recomputation and their combinations all stay correct.
    Only tensors that no autograd node has saved may be released; freeing
    a storage that a SavedVariable aliases would corrupt backward.
    """
    if tensor is None:
        return
    try:
        storage = tensor.untyped_storage()
        if storage.size() > 0:
            storage.resize_(0)
    except Exception:
        # Pure memory optimization: allocator quirks on some backends must
        # never affect correctness.
        pass


@contextmanager
def shared_recompute_group() -> Iterator[None]:
    """Pin a backward pass to a single checkpoint-recompute graph id.

    Non-reentrant ``torch.utils.checkpoint`` keys saved-tensor rehydration
    by backward graph-task id: for every distinct id it encounters while
    running a frame's backward, it re-runs the region's ``recompute_fn``,
    re-entering the recompute context that ``context_fn`` built exactly
    once per ``checkpoint()`` call. When that context is one-shot (any
    ``@contextmanager``-built CM), the second entry crashes inside
    contextlib with ``AttributeError: '_GeneratorContextManager' object
    has no attribute 'args'`` (or ``RuntimeError: generator didn't
    yield`` on older Pythons).

    Wrapping the whole ``loss.backward()`` in one
    ``torch.utils.checkpoint.GraphExecGroup`` gives every checkpointed
    frame exactly one id, so each region recomputes once per backward
    and every consumer -- including the nested backwards issued by the
    chunked MoE dispatcher -- shares the same rehydrated values.

    No-op under ``use_reentrant=True`` (reentrant checkpointing ignores
    groups), with checkpointing disabled, or on torch builds without
    ``GraphExecGroup``. An already-active ambient group is reused, since
    groups cannot nest.
    """
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


@contextmanager
def _shared_recompute_group() -> Iterator[None]:
    """One recomputation for the whole chunked backward.

    Under a non-reentrant ``torch.utils.checkpoint``, saved-value
    rehydration is keyed by the backward graph-task id: every nested
    ``torch.autograd.backward`` the chunked backward issues gets a fresh
    id, and each id would pay a full layer recomputation (C chunks -> C+1
    expert forwards instead of 2). A ``GraphExecGroup`` pins all those
    backwards to one id, so the checkpoint recomputes exactly once and
    every nested backward consumes the same rehydrated values.

    The same id-split bites the other way around too: any node of the
    checkpointed layer that backward-passes *before* this dispatcher and
    unpacks a saved tensor pays a recompute under the outer backward id,
    and this dispatcher's group id would then pay a second one (crashing
    into one-shot recompute contexts). The dispatcher therefore installs
    the autograd guard below: every ``torch.autograd.backward``/``grad``
    call runs under one shared group, and this scope reuses that ambient
    group instead of opening a second one (groups cannot nest).
    """
    with shared_recompute_group():
        yield


_ORIGINAL_AUTOGRAD: dict[str, Any] = {}


def _install_shared_backward_guard() -> None:
    """Route every backward/grad call through one GraphExecGroup per call.

    Installed once when the first chunked dispatcher is built. The backward
    engine snapshots the thread-local autograd state -- including the
    graph-exec group -- into the GraphTask at the ``backward()`` call and
    re-installs it around every node (per-node ``ThreadLocalStateGuard`` in
    engine.cpp), so a group must be active *before* the call to cover the
    whole graph; anything set mid-backward (e.g. from tensor hooks) is
    wiped at the next node boundary. Wrapping the autograd entry points is
    the single chokepoint that provably puts one group id over every node
    of every checkpoint frame, so each frame recomputes exactly once even
    though this dispatcher's backward adds its own nested backwards.
    Nested calls find the ambient group and pass through unchanged; the
    guard is a no-op without checkpoint hooks, under ``use_reentrant=True``
    and on torch builds without ``GraphExecGroup``.
    """
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


def _fsdp_module(module: torch.nn.Module) -> bool:
    """True when the module is wrapped by FSDP2 (``fully_shard``)."""
    try:
        from torch.distributed.fsdp._fully_shard import FSDPModule
    except ImportError:
        return False  # pragma: no cover - torch without FSDP2 internals
    return isinstance(module, FSDPModule)


def _fsdp_param_groups(fsdp_state: Any) -> list[Any]:
    """Parameter groups of an FSDP2 state across torch layouts.

    torch 2.10 keeps a single ``_fsdp_param_group`` per state while 2.11+
    stores ``_fsdp_param_groups``; normalize both to a list.
    """
    groups = getattr(fsdp_state, "_fsdp_param_groups", None)
    if groups is None:
        group = getattr(fsdp_state, "_fsdp_param_group", None)
        return [] if group is None else [group]
    return list(groups)


def _fsdp_groups(experts: torch.nn.Module) -> list[Any] | None:
    """FSDP parameter groups of *experts*; None for non-FSDP modules.

    Runs FSDP's lazy initialisation (comm-stream setup, root detection)
    when the installed torch exposes it on the state: the chunked
    forward bypasses the module hooks that normally perform it, and
    this may be the module's very first forward.
    """
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
    """One weight all-gather covering every per-chunk forward expert call.

    The chunked forward calls ``experts.forward`` directly (bypassing
    ``__call__`` so no FSDP forward hooks fire once per chunk and no
    ``RegisterPostBackwardFunction`` node is inserted into the graph),
    so it unshards explicitly once on entry -- idempotent when the
    weights are already unsharded -- and reshards on exit to keep the
    normal memory lifecycle. During activation recomputation the
    forward runs inside the backward pass: resharding there would only
    make the chunked backward hold unshard again, so the reshard is
    deferred to the backward hold's single reduce/reshard flush.
    Non-FSDP modules pass through untouched.
    """
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
    # Mirror ``pre_forward``'s state transition -- module hooks are
    # bypassed -- so the unshard picks FSDP's forward all-gather
    # streams and the exit reshard follows the forward reshard policy.
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
            # Normal forward: mirror ``post_forward`` -- reshard once
            # after all chunks and restore the training state. Under
            # activation recomputation the forward runs inside the
            # backward pass and FSDP's own post_forward skips the
            # reshard (is_bw()), and so do we: the chunked backward
            # hold reuses the unsharded weights and its single
            # post_backward performs the one reduce-scatter and
            # reshard, so a recomputed forward plus backward costs one
            # experts all-gather instead of two.
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
    """Reset a group's training state to IDLE after a manual flush.

    ``FSDPParamGroup.post_backward()`` leaves ``_training_state`` at
    ``POST_BACKWARD``; in FSDP's own wiring the root post-backward final
    callback -- which the chunked path never triggers, since it bypasses
    the module hooks -- resets it to ``IDLE``. Without this reset, the
    next chunked backward's flush sees the stale ``POST_BACKWARD`` and
    skips the gradient reduce-scatter and reshard entirely: silently
    unsynchronized expert gradients from the second backward on.
    """
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
    """One all-gather and one reduce-scatter for chunked experts backward.

    FSDP's backward wiring -- pre-backward unshard, gradient reduction,
    reshard -- is installed by its forward hooks, which the chunked
    forward bypasses, so nothing FSDP-side fires per chunk here. This
    hold supplies the wiring explicitly: unshard once on entry, let the
    per-chunk backwards accumulate into the unsharded ``.grad`` buffers,
    then invoke ``post_backward`` exactly once so the accumulated
    gradients reduce-scatter and the weights reshard a single time.
    The flush is queued at the end of this backward task: under a
    non-reentrant checkpoint the rest of the checkpointed layer
    rehydrates after this module's backward, and that recomputation
    re-runs the forward hold, which would re-unshard whatever this hold
    resharded eagerly. Non-FSDP modules pass through untouched.
    """
    groups = _fsdp_groups(experts)
    if groups is None:
        yield
        return
    for group in groups:
        # Idempotent when the (recomputed) forward left the group
        # unsharded; otherwise performs the one backward all-gather.
        group.unshard()
        group.wait_for_unshard()
        set_grad_dtypes = getattr(group, "_set_unsharded_grad_dtypes", None)
        if set_grad_dtypes is not None:
            set_grad_dtypes(defer_upcast=True)
    try:
        yield
    finally:
        # Single flush at the end of the backward task (see docstring):
        # one reduce-scatter over every chunk's accumulated gradients
        # plus one reshard, honoring the module's original policies.
        def _finalize_chunked_backward(fsdp_groups=tuple(groups)):
            for group in fsdp_groups:
                if not _fsdp_state_is_post_backward(group):
                    group.post_backward()
                # post_backward() parks the group in POST_BACKWARD; restore
                # the IDLE invariant FSDP's own final callback maintains, so
                # the next backward's flush is not skipped (see the reset
                # helper's docstring).
                _fsdp_reset_group_state_to_idle(group)
        try:
            torch.autograd.Variable._execution_engine.queue_callback(
                _finalize_chunked_backward,
            )
        except Exception:  # pragma: no cover - engine callback unavailable
            _finalize_chunked_backward()


class MoEChunkedAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """All-to-all dispatcher that pipelines token chunks with expert compute.

    Local tokens are split into ``moe_num_chunks`` contiguous chunks along the
    token dimension. Every chunk runs an independent dispatch/experts/combine
    sequence, and chunk-level all-to-all transfers run on a dedicated
    communication stream so that chunk ``c + 1`` transfers while chunk ``c``
    computes. Routing weights stay on the source ranks and are applied during
    the final weighted merge (output-side weights), so the separate
    probability all-to-all of the plain dispatcher disappears.

    Backward is owned by ``_ChunkedMoEFunction`` and reorders the per-chunk
    expert backward and dispatch all-to-all across the two streams so chunk
    ``c + 1`` transfers while chunk ``c`` backpropagates.
    """

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
        self.num_chunks = max(1, int(config.moe_num_chunks))
        # Stream-based pipelining requires device streams; CPU test runtimes
        # and single-rank EP fall back to the serial chunk loop.
        overlap_supported = self.device_runtime.current_stream() is not None
        self._comm_stream = (
            self.device_runtime.new_stream()
            if overlap_supported and self.ep_size > 1
            else None
        )

    def execute(
        self,
        hidden_states: torch.Tensor,
        routing: RoutingResult,
        experts: torch.nn.Module,
    ) -> torch.Tensor:
        hidden_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_shape[-1])
        if self._needs_custom_backward(flat, routing, experts):
            merged = _ChunkedMoEFunction.apply(
                self, experts, flat, routing.probs,
                routing.routing_map, routing.expert_indices,
            )
            return merged.view(hidden_shape)
        flat, routing, bounds, metadata, host, dense_probs = \
            self._prepare_chunks(flat, routing)
        merged, _ = self._forward_core(
            flat, routing, experts, bounds, host, metadata, dense_probs,
        )
        return merged.view(hidden_shape)

    def _needs_custom_backward(
        self, flat: torch.Tensor, routing: RoutingResult,
        experts: torch.nn.Module,
    ) -> bool:
        """True when gradients may flow and the chunked backward must run.

        The orchestrating autograd function owns backward so per-chunk
        FSDP all-gathers/reduces collapse to one each and chunk backwards
        overlap the communication stream. Under inference or a no-grad
        caller the plain forward loop is enough.
        """
        if not torch.is_grad_enabled():
            return False
        if flat.requires_grad or routing.probs.requires_grad:
            return True
        return any(
            param.requires_grad for param in experts.parameters()
        )

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
        # Output-side routing weights stay dense on the source rank; the
        # fused unpermute gathers and applies them during each chunk merge.
        dense_probs = routing.probs.to(torch.float32)
        return flat, routing, bounds, metadata, host, dense_probs

    def _forward_core(
        self,
        flat: torch.Tensor,
        routing: RoutingResult,
        experts: torch.nn.Module,
        bounds: list[tuple[int, int]],
        host: "_ChunkHostSplits",
        metadata: "_ChunkMetadata",
        dense_probs: torch.Tensor,
        graphs: "list[_ChunkGraph] | None" = None,
    ) -> tuple[torch.Tensor, "torch.Tensor | None"]:
        """Run the per-chunk forward pipeline, concatenated.

        Returns ``(merged, combined)``: the token-ordered MoE output and,
        on the grad-tracked (``graphs``) path, the concatenated combine
        rows -- the merge inputs that the manual merge adjoint consumes
        in backward; ``combined`` is ``None`` on the plain path.

        One expert-weight all-gather per layer regardless of chunk count:
        FSDP-wrapped experts are invoked through ``forward`` directly in
        both paths (per-chunk FSDP hooks are bypassed), and the forward
        hold unshards once on entry and reshards once on exit.
        """
        with _fsdp_forward_hold(experts):
            if self._comm_stream is None:
                outputs, combined_parts = self._execute_serial(
                    flat, routing, experts, bounds, host,
                    metadata.received_counts, dense_probs, graphs,
                )
            else:
                outputs, combined_parts = self._execute_pipelined(
                    flat, routing, experts, bounds, host,
                    metadata.received_counts, dense_probs, graphs,
                )
        combined: "torch.Tensor | None" = (
            torch.cat(combined_parts, dim=0)
            if combined_parts is not None else None
        )
        return torch.cat(outputs, dim=0), combined

    def _chunk_bounds(self, num_tokens: int) -> list[tuple[int, int]]:
        """Split token rows into contiguous chunk row intervals.

        Every EP rank builds exactly ``num_chunks`` intervals, possibly
        empty, so all ranks issue the same collective sequence even when
        their token counts differ.
        """
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
        # Per-chunk routing histogram via the same masking reduction the
        # plain dispatcher uses, replacing the atomic index_put_
        # accumulate that NPU serialized and that, issued inline before
        # the chunk transfers, amplified into a multi-ms stall.
        counts = torch.stack(
            [routing_map[start:end].sum(dim=0) for start, end in bounds]
        ).long()
        local_start = self.local_expert_indices[0]
        local_end = self.local_expert_indices[-1] + 1
        if self.ep_size > 1:
            gathered = self._gather_counts(counts, chunk_count)
            local_counts = gathered[:, :, local_start:local_end]
            received_counts = local_counts.permute(1, 0, 2).contiguous()
        else:
            received_counts = counts[:, local_start:local_end].unsqueeze(1).contiguous()
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
        """All-gather the chunk histogram across the EP group.

        When chunk transfers are pipelined on a side stream, run this gather on
        that same stream too, so every EP collective stays ordered on a single
        HCCL stream. Issuing the gather on the compute stream while chunk
        all-to-all runs on the communication stream made the shared communicator
        serialize and cost ~30 ms in NPU measurement.
        """
        gathered = counts.new_empty(self.ep_size, chunk_count, self.num_experts)
        if self._comm_stream is None:
            dist.all_gather_into_tensor(
                gathered.view(-1), counts.view(-1), group=self.ep_group,
            )
            return gathered
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
        permutation: MoEPermutation,
        dense_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Weight and sum the chunk's expert rows back into source tokens.

        Delegates to the fused ``moe_unpermutation`` operator
        (``npu_moe_token_unpermute`` on NPU), which gathers the chunk's dense
        routing weights and accumulates the weighted merge in FP32 in a single
        kernel with autograd, replacing the separate gather/mul/sum sequence.
        """
        restore_shape = (permutation.row_indices.shape[0], combined.shape[-1])
        chunk_permutation = MoEPermutation(
            combined, None, permutation.row_indices, permutation.expert_indices,
        )
        return ops.moe_unpermutation(
            chunk_permutation, restore_shape=restore_shape, probs=dense_probs,
        )

    # ------------------------------------------------------------------
    # Orchestrated backward (owned by _ChunkedMoEFunction)
    # ------------------------------------------------------------------

    def _a2a_no_grad(
        self, grad: torch.Tensor,
        output_split_sizes: list[int], input_split_sizes: list[int],
    ) -> torch.Tensor:
        """Bare collective without an autograd node.

        The chunked backward is orchestrated by ``_ChunkedMoEFunction``
        and moves the all-to-all gradients on the communication stream
        explicitly instead of through autograd nodes.
        """
        with torch.no_grad():
            return all_to_all(
                self.ep_group, grad, output_split_sizes, input_split_sizes,
            )

    def _backward_merge_chunk(
        self,
        grad_out_chunk: torch.Tensor,
        graph: "_ChunkGraph",
        combined_slice: torch.Tensor,
        dense_probs_chunk: torch.Tensor,
        expert_indices_chunk: torch.Tensor,
        need_input_grads: bool,
        need_probs_grad: bool,
    ) -> tuple["torch.Tensor | None", "torch.Tensor | None"]:
        """Manual adjoint of the weighted merge (unpermutation) segment.

        The forward merged ``merged[i] = sum_j w[i, j] * combined[inv[i, j]]``
        with ``w[i, j] = probs[i, expert_indices[i, j]]`` and ``inv`` the
        bijective packed-row map ``row_indices``. The adjoints are
        therefore

        * ``g_combined[inv[i, j]] = w[i, j] * g_merged[i]`` (one row per
          assignment, no accumulation), and
        * ``g_probs[i, e] = sum_j (g_merged[i] . combined[inv[i, j]])``
          over the slots assigned to expert ``e``,

        computed with gather/multiply/scatter primitives instead of an
        autograd graph. That keeps this segment's forward values out of
        the chunk graphs (they travel through ``save_for_backward`` like
        any plain activation), so the dispatcher needs no knowledge of
        whatever external recomputation wraps the layer. Accumulation
        runs in FP32 to mirror the merge kernel.
        """
        chunk_tokens = grad_out_chunk.shape[0]
        topk = graph.row_indices.shape[1]
        row_indices = graph.row_indices.to(torch.long)
        packed = row_indices.reshape(-1)
        num_rows = packed.numel()
        # g_merged broadcast into per-(token, slot) order, so every entry
        # lines up with packed[k] = row_indices[i, j].
        g_merged_ij = (
            grad_out_chunk.unsqueeze(1)
            .expand(chunk_tokens, topk, grad_out_chunk.shape[-1])
            .reshape(num_rows, grad_out_chunk.shape[-1])
        )
        g_combined: "torch.Tensor | None" = None
        if need_input_grads:
            weights = dense_probs_chunk.gather(1, expert_indices_chunk)
            g_flat = (
                g_merged_ij.to(torch.float32)
                * weights.flatten().unsqueeze(-1)
            ).to(g_merged_ij.dtype)
            # Scatter into packed (assignment) row order; ``row_indices``
            # is a bijection over the chunk rows, so the assignment is
            # deterministic.
            g_combined = grad_out_chunk.new_zeros(
                num_rows, grad_out_chunk.shape[-1],
            )
            g_combined[packed] = g_flat
        g_probs: "torch.Tensor | None" = None
        if need_probs_grad:
            # Combine rows gathered into per-(token, slot) order, then
            # g_probs[i, e] = sum_j (g_merged[i] . combined[i, j]).
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

    def _backward_experts_grad(
        self,
        graph: "_ChunkGraph",
        grad_restored: torch.Tensor,
    ) -> torch.Tensor:
        """Expert-segment backward through the forward-built graph.

        Runs ``backward`` on the graph captured during the (possibly
        recomputed) forward so parameter gradients accumulate through
        the normal autograd path -- the expert backward kernels consume
        the forward-saved activations and the experts are never run
        forward a second time. Returns the received-token gradient.
        """
        torch.autograd.backward(graph.restored, grad_restored)
        grad = graph.received_leaf.grad
        if grad is None:
            grad = torch.zeros_like(graph.received_leaf)
        graph.received_leaf.grad = None
        # The expert graph is consumed; free the segment values eagerly so
        # the backward peak tracks only the chunk in flight.
        _release_activation_value(graph.restored)
        _release_activation_value(graph.received_leaf)
        return grad

    def _backward_permute_chunk(
        self,
        grad_flat_slice: torch.Tensor,
        grad_permuted: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> None:
        """Scatter permuted gradients back into the flat chunk slice.

        The permute forward gathered source-token rows into expert order;
        ``row_indices[t]`` lists the permuted positions of token ``t``'s
        top-k copies, so the adjoint gathers those rows and sums them.
        No re-execution of the permutation operator is needed, so the
        flat input does not have to be persisted for backward.
        """
        indices = row_indices.to(torch.long)
        gathered = grad_permuted[indices.reshape(-1)]
        grad_flat_slice += gathered.view(
            indices.shape[0], indices.shape[1], grad_flat_slice.shape[-1],
        ).sum(dim=1)

    def _execute_backward(
        self,
        grad_output: torch.Tensor,
        experts: torch.nn.Module,
        flat_shape: torch.Size,
        probs: torch.Tensor,
        dense_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        combined: torch.Tensor,
        bounds: list[tuple[int, int]],
        host: "_ChunkHostSplits",
        graphs: list["_ChunkGraph"],
        need_input_grads: bool,
        need_probs_grad: bool,
    ) -> tuple["torch.Tensor | None", "torch.Tensor | None"]:
        """Chunked backward: one FSDP all-gather/reduce, overlapped chunks.

        ``graphs`` holds the per-chunk expert graphs captured by the
        forward (or whatever external recomputation re-ran it); the
        backward orchestrates the adjoints -- merge adjoints first
        (feeding combine-backwards to the communication stream), then
        expert backwards and dispatch-backwards -- and never re-executes
        the experts, so expert compute runs exactly once per training
        step. The merge adjoint is a manual gather/multiply/scatter over
        the saved combine outputs and routing weights; the permute
        backward is a gather-sum scatter keyed on the saved permutation
        indices; the flat input itself is not persisted.
        """
        if not need_input_grads and not need_probs_grad:
            return None, None
        hidden_cols = flat_shape[-1]
        grad_flat_in = grad_output.reshape(-1, hidden_cols)
        grad_flat: "torch.Tensor | None" = (
            torch.zeros(
                flat_shape, dtype=grad_flat_in.dtype,
                device=grad_flat_in.device,
            )
            if need_input_grads else None
        )
        grad_probs: "torch.Tensor | None" = (
            torch.zeros(
                flat_shape[0], self.num_experts, dtype=torch.float32,
                device=probs.device,
            )
            if need_probs_grad else None
        )
        topk = self.config.moe_router_topk
        combined_bounds = [(start * topk, end * topk) for start, end in bounds]
        kwargs = dict(
            bounds=bounds,
            combined_bounds=combined_bounds,
            host=host,
            graphs=graphs,
            dense_probs=dense_probs,
            expert_indices=expert_indices,
            combined=combined,
            grad_flat_in=grad_flat_in,
            grad_flat=grad_flat,
            grad_probs=grad_probs,
            need_input_grads=need_input_grads,
            need_probs_grad=need_probs_grad,
        )
        with _fsdp_backward_hold(experts):
            if self._comm_stream is None:
                self._backward_serial(**kwargs)
            else:
                self._backward_pipelined(**kwargs)
        grad_probs_out: "torch.Tensor | None" = None
        if grad_probs is not None:
            grad_probs_out = grad_probs.to(probs.dtype)
        return grad_flat, grad_probs_out

    def _backward_serial(
        self, *, bounds, combined_bounds, host, graphs, dense_probs,
        expert_indices, combined, grad_flat_in, grad_flat, grad_probs,
        need_input_grads, need_probs_grad,
    ) -> None:
        for index in reversed(range(len(bounds))):
            start, end = bounds[index]
            c_start, c_end = combined_bounds[index]
            graph = graphs[index]
            g_combined, g_probs = self._backward_merge_chunk(
                grad_flat_in[start:end], graph, combined[c_start:c_end],
                dense_probs[start:end], expert_indices[start:end],
                need_input_grads, need_probs_grad,
            )
            if grad_probs is not None:
                grad_probs[start:end] = g_probs
            if not need_input_grads:
                graphs[index] = None
                continue
            g_restored = self._a2a_no_grad(
                g_combined, host.output_splits[index], host.input_splits[index],
            )
            g_received = self._backward_experts_grad(graph, g_restored)
            g_permuted = self._a2a_no_grad(
                g_received, host.input_splits[index], host.output_splits[index],
            )
            if grad_flat is not None:
                self._backward_permute_chunk(
                    grad_flat[start:end], g_permuted,
                    graph.row_indices,
                )
            graphs[index] = None  # free the chunk graph promptly

    def _backward_pipelined(
        self, *, bounds, combined_bounds, host, graphs, dense_probs,
        expert_indices, combined, grad_flat_in, grad_flat, grad_probs,
        need_input_grads, need_probs_grad,
    ) -> None:
        """Mirror of the forward pipeline, run in reverse per chunk.

        Merge adjoints run first (pure gather/multiply/scatter over the
        saved combine outputs), feeding combine-backwards to the
        communication stream back to back; every expert backward then
        overlaps the remaining combine-backwards, and each
        dispatch-backward overlaps the next expert backward. Only the
        trailing dispatch-backward has no compute left to hide behind.
        """
        runtime = self.device_runtime
        compute = runtime.current_stream()
        comm = self._comm_stream
        chunk_count = len(bounds)
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
            g_combined, g_probs = self._backward_merge_chunk(
                grad_flat_in[start:end], graphs[index],
                combined[c_start:c_end], dense_probs[start:end],
                expert_indices[start:end], need_input_grads, need_probs_grad,
            )
            grad_combined[index] = g_combined
            if grad_probs is not None:
                grad_probs[start:end] = g_probs
            ev_merge[index].record(compute)

        def enqueue_combine_backward(index):
            comm.wait_event(ev_merge[index])
            with runtime.stream(comm):
                grad_restored[index] = self._a2a_no_grad(
                    grad_combined[index],
                    host.output_splits[index], host.input_splits[index],
                )
            runtime.record_stream(grad_combined[index], comm)
            grad_combined[index] = None
            ev_combine_bwd[index].record(comm)

        def run_experts_backward(index):
            compute.wait_event(ev_combine_bwd[index])
            runtime.record_stream(grad_restored[index], compute)
            row_indices[index] = graphs[index].row_indices
            grad_received[index] = self._backward_experts_grad(
                graphs[index], grad_restored[index],
            )
            graphs[index] = None  # free the chunk graph promptly
            ev_experts[index].record(compute)

        def enqueue_dispatch_backward(index):
            comm.wait_event(ev_experts[index])
            with runtime.stream(comm):
                grad_permuted[index] = self._a2a_no_grad(
                    grad_received[index],
                    host.input_splits[index], host.output_splits[index],
                )
            runtime.record_stream(grad_received[index], comm)
            grad_received[index] = None
            ev_dispatch[index].record(comm)

        def run_permute_backward(index):
            compute.wait_event(ev_dispatch[index])
            runtime.record_stream(grad_permuted[index], compute)
            start, end = bounds[index]
            if grad_flat is not None:
                self._backward_permute_chunk(
                    grad_flat[start:end], grad_permuted[index],
                    row_indices[index],
                )
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

    def _expert_chunk(
        self,
        received: torch.Tensor,
        received_counts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        experts: torch.nn.Module,
    ) -> torch.Tensor:
        expert_input = self._sort_received(received, received_counts)
        if _fsdp_module(experts):
            # Bypass ``__call__`` like the graph path: per-chunk FSDP
            # forward hooks would unshard/reshard once per chunk; the
            # enclosing forward hold does it once instead.
            expert_output, bias = experts.forward(
                expert_input, tokens_per_expert, None,
            )
        else:
            expert_output, bias = experts(expert_input, tokens_per_expert, None)
        assert bias is None, "Bias is not supported in MoEChunkedAlltoAllTokenDispatcher"
        return self._restore_order(expert_output, received_counts)

    def _expert_chunk_graph(
        self,
        received: torch.Tensor,
        received_counts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        experts: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert segment with an autograd graph, FSDP hooks bypassed.

        Returns ``(restored, received_leaf)``: the chunk's expert output
        with its backward graph and the detached received-token leaf.
        The surrounding forward hold keeps the weights unsharded for all
        chunks; the chunked backward consumes the returned graph
        directly, so the experts run forward exactly once per training
        step (no re-execution under ``_execute_backward``, and during
        activation recomputation this *is* the recomputation).
        FSDP-wrapped experts are called through ``forward`` directly:
        going through ``__call__`` would re-fire FSDP's forward hooks
        once per chunk (extra all-gathers plus, when this runs during
        recomputation inside backward, forward hooks in the middle of
        the backward pass disturbing ``FSDP::backward_prefetch``).
        """
        with torch.enable_grad():
            rcv_leaf = received.detach().requires_grad_(True)
            expert_input = self._sort_received(rcv_leaf, received_counts)
            if _fsdp_module(experts):
                expert_output, bias = experts.forward(
                    expert_input, tokens_per_expert, None,
                )
            else:
                expert_output, bias = experts(
                    expert_input, tokens_per_expert, None,
                )
            assert bias is None, "Bias is not supported in MoEChunkedAlltoAllTokenDispatcher"
            restored = self._restore_order(expert_output, received_counts)
        return restored, rcv_leaf

    def _execute_serial(
        self, flat, routing, experts, bounds, host, received_counts,
        dense_probs, graphs: "list[_ChunkGraph] | None" = None,
    ) -> tuple[list[torch.Tensor], "list[torch.Tensor] | None"]:
        outputs = []
        combined_parts: "list[torch.Tensor] | None" = (
            [] if graphs is not None else None
        )
        for index, (start, end) in enumerate(bounds):
            permutation = self._permute_chunk(flat, routing, start, end)
            received = all_to_all(
                self.ep_group, permutation.tokens,
                host.output_splits[index], host.input_splits[index],
            )
            if graphs is None:
                expert_output = self._expert_chunk(
                    received, received_counts[index],
                    host.tokens_per_expert[index], experts,
                )
                combined = all_to_all(
                    self.ep_group, expert_output,
                    host.input_splits[index], host.output_splits[index],
                )
                merged = self._merge_chunk(
                    combined, permutation, dense_probs[start:end],
                )
            else:
                restored, rcv_leaf = self._expert_chunk_graph(
                    received, received_counts[index],
                    host.tokens_per_expert[index], experts,
                )
                combined = all_to_all(
                    self.ep_group, restored,
                    host.input_splits[index], host.output_splits[index],
                )
                # The merge segment stays out of the autograd graph: its
                # adjoint is computed manually in backward from the saved
                # combine outputs, so no leaf values are retained here.
                merged = self._merge_chunk(
                    combined, permutation, dense_probs[start:end],
                )
                graphs.append(
                    _ChunkGraph(restored, rcv_leaf, permutation.row_indices)
                )
                combined_parts.append(combined)
            outputs.append(merged)
        return outputs, combined_parts

    def _execute_pipelined(
        self, flat, routing, experts, bounds, host, received_counts,
        dense_probs, graphs: "list[_ChunkGraph] | None" = None,
    ) -> tuple[list[torch.Tensor], "list[torch.Tensor] | None"]:
        """Overlap chunk transfers with expert compute on two streams.

        The communication stream is fed as ``D0, D1, D2, C0, D3, C1, ...``:
        dispatch chunk ``c + 2`` enters before combine chunk ``c`` so the
        queue never idles while some chunk is computable.

        The compute stream runs in two phases. Phase 1 issues every expert
        computation back to back (each depending only on its own dispatch),
        feeding each chunk's combine to the communication stream as soon as
        its experts finish. Phase 2 waits for each combine and merges. This
        matters most for C=2: the second chunk's GMM overlaps the first
        chunk's combine all-to-all, instead of stalling until that combine
        completes (a merge-wait inside a single loop serialized them).
        """
        runtime = self.device_runtime
        compute = runtime.current_stream()
        comm = self._comm_stream
        chunk_count = len(bounds)

        ev_permute = [runtime.new_event() for _ in range(chunk_count)]
        ev_dispatch = [runtime.new_event() for _ in range(chunk_count)]
        ev_experts = [runtime.new_event() for _ in range(chunk_count)]
        ev_combine = [runtime.new_event() for _ in range(chunk_count)]
        permuted = [None] * chunk_count
        received = [None] * chunk_count
        outputs = [None] * chunk_count

        def enqueue_permute(index):
            start, end = bounds[index]
            permuted[index] = self._permute_chunk(flat, routing, start, end)
            ev_permute[index].record(compute)

        def enqueue_dispatch(index):
            comm.wait_event(ev_permute[index])
            with runtime.stream(comm):
                received[index] = all_to_all(
                    self.ep_group, permuted[index].tokens,
                    host.output_splits[index], host.input_splits[index],
                )
            runtime.record_stream(permuted[index].tokens, comm)
            ev_dispatch[index].record(comm)

        def enqueue_combine(index, expert_output):
            comm.wait_event(ev_experts[index])
            with runtime.stream(comm):
                combined = all_to_all(
                    self.ep_group, expert_output,
                    host.input_splits[index], host.output_splits[index],
                )
            runtime.record_stream(expert_output, comm)
            ev_combine[index].record(comm)
            return combined

        enqueue_permute(0)
        enqueue_dispatch(0)
        if chunk_count > 1:
            enqueue_permute(1)
            enqueue_dispatch(1)

        combined = [None] * chunk_count
        rcv_leaves = [None] * chunk_count
        restored_graphs = [None] * chunk_count
        # Phase 1: expert compute for every chunk; combines are fed to the
        # communication stream as each chunk's experts finish. The
        # grad-tracked path builds the expert graphs from detached
        # received leaves (FSDP hooks bypassed; the forward hold
        # unshards once).
        for index in range(chunk_count):
            compute.wait_event(ev_dispatch[index])
            runtime.record_stream(received[index], compute)
            if graphs is None:
                expert_output = self._expert_chunk(
                    received[index], received_counts[index],
                    host.tokens_per_expert[index], experts,
                )
            else:
                expert_output, rcv_leaves[index] = self._expert_chunk_graph(
                    received[index], received_counts[index],
                    host.tokens_per_expert[index], experts,
                )
                restored_graphs[index] = expert_output
            ev_experts[index].record(compute)
            if index + 2 < chunk_count:
                enqueue_permute(index + 2)
                enqueue_dispatch(index + 2)
            combined[index] = enqueue_combine(index, expert_output)

        # Phase 2: wait for each combine on demand and merge in order;
        # the grad-tracked path merges the combine rows as plain values
        # (the merge segment stays out of the autograd graph -- its
        # adjoint is computed manually in ``_backward_merge_chunk``) and
        # records the expert-segment graphs for the orchestrated
        # backward.
        combined_parts: "list[torch.Tensor] | None" = (
            [] if graphs is not None else None
        )
        for index in range(chunk_count):
            compute.wait_event(ev_combine[index])
            runtime.record_stream(combined[index], compute)
            start, end = bounds[index]
            outputs[index] = self._merge_chunk(
                combined[index], permuted[index], dense_probs[start:end],
            )
            if graphs is not None:
                graphs.append(
                    _ChunkGraph(
                        restored_graphs[index], rcv_leaves[index],
                        permuted[index].row_indices,
                    )
                )
                combined_parts.append(combined[index])
        return outputs, combined_parts

class _ChunkedMoEFunction(torch.autograd.Function):
    """Owns both forward and backward of the chunked MoE dispatcher.

    Forward runs the pipelined chunk loop inside ``Function.forward``
    with only the expert segment re-enabled for autograd (all-to-all
    boundaries stay detached leaves, moved explicitly on the
    communication stream); the merge segment stays out of the graph --
    its adjoint is computed manually in backward from the saved combine
    rows. The resulting per-chunk expert graphs are held on ``ctx`` in
    *structure only*: their activation values are returned to the
    allocator as soon as the merged output is produced, so retention is
    identical to the plain dispatcher under any external recomputation
    scheme. FSDP hooks never fire for the per-chunk expert calls
    (direct ``forward`` under the explicit unshard hold).

    Recomputation support, both reentrant modes: the merge inputs are
    carried by ``save_for_backward``, so an external non-reentrant
    checkpoint packs/releases them here and rebuilds them during its own
    recomputation pass (the reentrant pass rebuilds them by re-running
    this very ``forward``); the expert graphs either are the recomputed
    segment themselves (reentrant) or are consumed structurally by the
    original node after the checkpoint rehydrates their packed saved
    tensors (non-reentrant). Either way the experts never run an extra
    forward for the chunked backward.

    Backward only orchestrates: manual merge adjoints (feeding
    combine-backwards to the communication stream), expert backwards
    through the saved expert graphs (expert parameter gradients
    accumulate through the normal autograd path), manual dispatch
    gradient all-to-alls, and a gather-sum scatter for the permute
    backward. ``_fsdp_backward_hold`` collapses the per-chunk FSDP
    weight all-gathers and gradient reduce-scatters into one of each
    per layer (reusing the unsharded weights across a recomputation
    pass).
    """

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
        graphs: list[_ChunkGraph] = []
        merged, combined = dispatcher._forward_core(
            flat, routing, experts, bounds, host, metadata, dense_probs,
            graphs,
        )
        ctx.dispatcher = dispatcher
        ctx.experts = experts
        ctx.bounds = bounds
        ctx.host = host
        ctx.flat_shape = flat.shape
        # Per-chunk expert graphs are internal to this node; keep them as
        # plain attributes (saved_tensors is for node inputs/outputs
        # only). They die with the backward node.
        ctx.chunk_graphs = graphs
        # The merged output and the saved combine rows below have copied
        # every per-chunk value the forward still owed its consumers, so
        # the expert segments are needed *structurally* only from here
        # on: the backward walks their grad_fn chains and reads the leaf
        # ``.grad`` slots, never the released values. Returning those
        # storages to the allocator right after the forward is therefore
        # mode-independent -- eager training, external recomputation and
        # both reentrant modes all stay correct -- and it is what keeps
        # activation retention identical to the plain dispatcher under
        # layer-level checkpointing.
        for graph in graphs:
            _release_activation_value(graph.restored)
            _release_activation_value(graph.received_leaf)
        # The merge inputs travel through save_for_backward like any
        # plain activation: under an external non-reentrant checkpoint
        # they are packed and released here and rebuilt by the checkpoint
        # recomputation; under reentrant recomputation this very call
        # *is* the rebuild. Either way ``backward`` sees rehydrated
        # values and does no reconstruction of its own.
        ctx.save_for_backward(
            probs, dense_probs, routing.expert_indices, combined,
        )
        return merged

    @staticmethod
    def backward(ctx, grad_output):
        dispatcher = ctx.dispatcher
        # Everything that may trigger a checkpoint rehydration -- the
        # saved-tensor unpack and every per-chunk expert backward -- runs
        # inside one shared recomputation group (see its docstring).
        with _shared_recompute_group():
            return _ChunkedMoEFunction._backward_impl(ctx, grad_output)

    @staticmethod
    def _backward_impl(ctx, grad_output):
        dispatcher = ctx.dispatcher
        (
            probs, dense_probs, expert_indices, combined,
        ) = ctx.saved_tensors
        # ``needs_input_grad``: (dispatcher, experts, flat, probs,
        # routing_map, expert_indices); the backward consumes the
        # forward-built chunk graphs and needs no saved flat.
        need_input_grads = ctx.needs_input_grad[2] or any(
            param.requires_grad for param in ctx.experts.parameters()
        )
        need_probs_grad = ctx.needs_input_grad[3]
        grad_flat, grad_probs = dispatcher._execute_backward(
            grad_output, ctx.experts, ctx.flat_shape, probs, dense_probs,
            expert_indices, combined, ctx.bounds, ctx.host,
            ctx.chunk_graphs, need_input_grads, need_probs_grad,
        )
        return None, None, grad_flat, grad_probs, None, None
