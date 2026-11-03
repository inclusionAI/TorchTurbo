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

import torch
import torch.distributed as dist

from torchturbo.ops import ops
from torchturbo.ops.moe import MoEPermutation

from ..comm.comm import all_to_all
from .moe_config import MoEConfig
from .token_dispatcher import MoEAlltoAllTokenDispatcher
from .utils import RoutingResult, maybe_move_tensor_to_cpu

__all__ = ["MoEChunkedAlltoAllTokenDispatcher"]


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


@contextmanager
def _hold_experts_unsharded(experts: torch.nn.Module) -> Iterator[None]:
    """Keep FSDP-sharded expert weights unsharded across the per-chunk calls.

    The chunked dispatcher invokes the experts module once per chunk. Under
    FSDP2 (torch >= 2.10), every call unshards (all-gathers) the expert
    weights in its pre-forward hook and reshards right after, so a C-chunk
    layer pays C weight all-gathers per forward, plus one more per chunk
    again during recomputation under ``fsdp_recompute_context``. Both
    reentrant modes funnel every recompute call through the same idempotent
    ``FSDPParamGroup.unshard`` (``_unshard_before_forward`` explicitly and
    ``_restart_reentrant_forward`` via FSDP's own pre-forward), so holding
    the group unsharded deduplicates all of them. Disable post-forward
    resharding for the duration of the chunk loop so the first call unshards
    once and later calls reuse the unsharded weights, then restore the
    original policy and reshard on exit to keep the normal memory lifecycle.
    Non-FSDP modules pass through untouched.
    """
    try:
        from torch.distributed.fsdp._fully_shard import FSDPModule
    except ImportError:
        FSDPModule = None  # pragma: no cover - torch without FSDP2 internals
    if FSDPModule is None or not isinstance(experts, FSDPModule):
        yield
        return
    fsdp_state = experts._get_fsdp_state()
    groups = _fsdp_param_groups(fsdp_state)
    if not groups:
        yield
        return
    saved_auto = fsdp_state._auto_reshard_after_forward
    saved_mesh_info = [group.post_forward_mesh_info for group in groups]
    try:
        experts.set_reshard_after_forward(False, recurse=False)
        yield
    finally:
        fsdp_state._auto_reshard_after_forward = saved_auto
        for group, mesh_info in zip(groups, saved_mesh_info):
            group.post_forward_mesh_info = mesh_info
        experts.reshard()


class MoEChunkedAlltoAllTokenDispatcher(MoEAlltoAllTokenDispatcher):
    """All-to-all dispatcher that pipelines token chunks with expert compute.

    Local tokens are split into ``moe_num_chunks`` contiguous chunks along the
    token dimension. Every chunk runs an independent dispatch/experts/combine
    sequence, and chunk-level all-to-all transfers run on a dedicated
    communication stream so that chunk ``c + 1`` transfers while chunk ``c``
    computes. Routing weights stay on the source ranks and are applied during
    the final weighted merge (output-side weights), so the separate
    probability all-to-all of the plain dispatcher disappears.

    Backward executes the same per-chunk operators through their autograd
    definitions without stream pipelining.
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
        bounds = self._chunk_bounds(flat.shape[0])
        metadata = self._chunk_metadata(routing, bounds)
        host = self._host_metadata(metadata)
        # Output-side routing weights stay dense on the source rank; the fused
        # unpermute gathers and applies them during each chunk merge.
        dense_probs = routing.probs.to(torch.float32)
        # One expert-weight all-gather per layer regardless of chunk count:
        # FSDP must not re-unshard between the per-chunk experts calls.
        with _hold_experts_unsharded(experts):
            if self._comm_stream is None:
                outputs = self._execute_serial(
                    flat, routing, experts, bounds, host,
                    metadata.received_counts, dense_probs,
                )
            else:
                outputs = self._execute_pipelined(
                    flat, routing, experts, bounds, host,
                    metadata.received_counts, dense_probs,
                )
        return torch.cat(outputs, dim=0).view(hidden_shape)

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
        self, routing: RoutingResult, bounds: list[tuple[int, int]],
    ) -> _ChunkMetadata:
        """Build per-chunk count matrices with a fast histogram and one gather."""
        chunk_count = len(bounds)
        routing_map = routing.routing_map
        # Per-chunk routing histogram via the same masking reduction the plain
        # dispatcher uses, replacing the atomic index_put_ accumulate that NPU
        # serialized and that, issued inline before the chunk transfers,
        # amplified into a multi-ms stall.
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
        pack = torch.cat(
            (
                metadata.send_counts.reshape(-1),
                metadata.receive_counts.reshape(-1),
                metadata.tokens_per_expert.reshape(-1),
            )
        )
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
        tokens_per_expert = pack[2 * size :].view(chunk_count, self.num_local_experts)
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

    def _expert_chunk(
        self,
        received: torch.Tensor,
        received_counts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        experts: torch.nn.Module,
    ) -> torch.Tensor:
        expert_input = self._sort_received(received, received_counts)
        expert_output, bias = experts(expert_input, tokens_per_expert, None)
        assert bias is None, "Bias is not supported in MoEChunkedAlltoAllTokenDispatcher"
        return self._restore_order(expert_output, received_counts)

    def _execute_serial(
        self, flat, routing, experts, bounds, host, received_counts, dense_probs,
    ) -> list[torch.Tensor]:
        outputs = []
        for index, (start, end) in enumerate(bounds):
            permutation = self._permute_chunk(flat, routing, start, end)
            received = all_to_all(
                self.ep_group, permutation.tokens,
                host.output_splits[index], host.input_splits[index],
            )
            expert_output = self._expert_chunk(
                received, received_counts[index], host.tokens_per_expert[index],
                experts,
            )
            combined = all_to_all(
                self.ep_group, expert_output,
                host.input_splits[index], host.output_splits[index],
            )
            outputs.append(
                self._merge_chunk(combined, permutation, dense_probs[start:end])
            )
        return outputs

    def _execute_pipelined(
        self, flat, routing, experts, bounds, host, received_counts, dense_probs,
    ) -> list[torch.Tensor]:
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
        # Phase 1: expert compute for every chunk; combines are fed to the
        # communication stream as each chunk's experts finish.
        for index in range(chunk_count):
            compute.wait_event(ev_dispatch[index])
            runtime.record_stream(received[index], compute)
            expert_output = self._expert_chunk(
                received[index], received_counts[index],
                host.tokens_per_expert[index], experts,
            )
            ev_experts[index].record(compute)
            if index + 2 < chunk_count:
                enqueue_permute(index + 2)
                enqueue_dispatch(index + 2)
            combined[index] = enqueue_combine(index, expert_output)

        # Phase 2: wait for each combine on demand and merge in order.
        for index in range(chunk_count):
            compute.wait_event(ev_combine[index])
            runtime.record_stream(combined[index], compute)
            start, end = bounds[index]
            outputs[index] = self._merge_chunk(
                combined[index], permuted[index], dense_probs[start:end],
            )
        return outputs
