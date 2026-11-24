# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Modified by the TorchTurbo contributors.

"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from math import prod
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist

from torchturbo.ops import ops
from torchturbo.ops.moe import VirtualExpertPlan as _RoutingPlan
from torchturbo.ops.moe import VirtualExpertPlannerWorkspace
from torchturbo.runtime import DeviceRuntime, get_device_runtime
from torchturbo.utils.torch_utils import torch_version

from .grouped_linear import grouped_linear
from .moe_config import MoEConfig

if TYPE_CHECKING:
    from .experts import GroupedMLP


class WeightDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(slots=True)
class VirtualExpertPlan:
    """One forward's routes and in-flight state, retained through its backward.

    Native autograd supplies gradients per execution instead of persistent
    main_grad buffers. Events also belong to the plan so live graphs and
    checkpoint replay can share the runtime without replacing each other's state.
    """

    routing: _RoutingPlan
    push_in_flight: bool = False
    ready: object | None = None
    prefetch_done: object | None = None
    grad_reduce_done: object | None = None
    slot_weights: tuple[torch.Tensor, torch.Tensor] | None = None
    grad_results: list[torch.Tensor | None] = field(default_factory=lambda: [None, None])


def plan_virtual_expert_routes(
    top_indices: torch.Tensor, workspace: VirtualExpertPlannerWorkspace, alignment: int = 1,
) -> VirtualExpertPlan:
    """Plan deterministic virtual-expert placement and map this rank's compact routes."""
    if top_indices.dim() != 2 or top_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Virtual-expert planner takes int32/int64 [num_tokens, topk] expert ids.")
    if alignment != 1:
        raise ValueError("Compact virtual-expert transport requires alignment=1.")
    plan = ops.plan_virtual_expert_routes(
        top_indices.contiguous(), workspace=workspace, async_finish=True,
    )
    return VirtualExpertPlan(routing=plan)


class _VirtualExpertRuntime:
    """Planner scratch and communication streams shared by compatible layers."""

    def __init__(
        self, runtime: DeviceRuntime, num_experts: int, device: torch.device,
        group: dist.ProcessGroup | None, peer_access: bool,
    ) -> None:
        self.planner = VirtualExpertPlannerWorkspace(
            num_experts=num_experts, device=device, group=group, runtime=runtime, peer_access=peer_access,
        )
        self.planner_stream: object | None = None
        self.weight_streams: tuple[object, object] | None = None
        self.grad_stream: object | None = None
        if runtime.supports_async_streams:
            self.planner_stream = runtime.new_stream()
            self.weight_streams = (runtime.new_stream(), runtime.new_stream())
            self.grad_stream = runtime.new_stream()

    def destroy(self) -> None:
        """Release planner resources before the process group is destroyed."""
        if self.planner is not None:
            self.planner.destroy()
        self.planner = self.planner_stream = self.weight_streams = self.grad_stream = None


@dataclass(frozen=True, slots=True)
class _VirtualExpertConfig:
    """Storage layout shared by compatible layers in one expert group."""

    group: dist.ProcessGroup | None
    device: torch.device
    num_local_experts: int
    member_shapes: tuple[torch.Size, torch.Size]
    dtype: torch.dtype
    grad_dtype: torch.dtype
    weight_peer_access: bool = True
    grad_peer_access: bool = True


class _VirtualExpertStorage:
    """Reusable replica weights and gradient transport scratch for one layout."""

    def __init__(self, config: _VirtualExpertConfig) -> None:
        self.config = config
        self._runtime = get_device_runtime()
        self.weight_arena = self.grad_arena = self.weight_handle = self.grad_handle = None
        self.slot_weights = ()
        self.last_use: object | None = None
        try:
            self._allocate(config)
        except Exception:
            self.destroy()
            raise

    def _allocate(self, config: _VirtualExpertConfig) -> None:
        self.member_numels = tuple(prod(shape) for shape in config.member_shapes)
        sections = tuple(config.num_local_experts * size for size in self.member_numels)
        runtime = self._runtime
        with torch.inference_mode(False):
            self.weight_arena, self.weight_handle = runtime.allocate_communication_buffer(
                sum(sections), dtype=config.dtype, device=config.device, group=config.group,
                peer_access=config.weight_peer_access,
            )
            self.grad_arena, self.grad_handle = runtime.allocate_communication_buffer(
                sum(sections), dtype=config.grad_dtype, device=config.device, group=config.group,
                peer_access=config.grad_peer_access,
            )
        self.slot_weights = tuple(
            section.view(config.num_local_experts, *shape)
            for section, shape in zip(self.weight_arena.split(sections), config.member_shapes)
        )

    def destroy(self) -> None:
        """Drop model-owned arena bindings after all graphs have finished."""
        if self.weight_arena is None and self.grad_arena is None:
            return
        self._runtime.synchronize()
        with torch.no_grad():
            for slot in self.slot_weights:
                slot.set_()
        self.slot_weights = ()
        self.weight_arena = self.grad_arena = self.weight_handle = self.grad_handle = None
        self.last_use = None

    def record_use(self) -> None:
        """Keep subsequent pushes behind all readers of the current contents."""
        runtime = get_device_runtime()
        if runtime.supports_async_streams:
            self.last_use = runtime.new_event()
            self.last_use.record(runtime.current_stream())


class _VirtualExpertHook(torch.autograd.Function):
    """Run a plan's callback during backward and pass through the tensor gradient."""

    @staticmethod
    def forward(ctx, tensor, method, plan):
        ctx.method, ctx.plan = method, plan
        return tensor

    @staticmethod
    def backward(ctx, grad):
        ctx.method(ctx.plan)
        return grad, None, None


class _VirtualExpertGradComplete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight1, weight2, hidden, probs, plan, load_balancer):
        ctx.plan, ctx.load_balancer = plan, load_balancer
        return hidden, probs

    @staticmethod
    def backward(ctx, hidden_grad, probs_grad):
        previous_plan = ctx.load_balancer._plan
        ctx.load_balancer._plan = ctx.plan
        grad1, grad2 = ctx.load_balancer._finish_grad_reduce()
        if previous_plan is not ctx.plan:
            ctx.load_balancer._plan = previous_plan
        return grad1, grad2, hidden_grad, probs_grad, None, None


class VirtualExpertLoadBalancer:
    """Mixin that plans virtual-expert routes and coordinates replica execution."""

    storages: ClassVar[WeakValueDictionary[_VirtualExpertConfig, _VirtualExpertStorage]] = (
        WeakValueDictionary()
    )
    runtimes: ClassVar[WeakValueDictionary[
        tuple[DeviceRuntime, dist.ProcessGroup | None, torch.device, int, int],
        _VirtualExpertRuntime,
    ]] = WeakValueDictionary()

    # A rank-local GC cycle must not evict one side of a WORLD-collective
    # allocation. Native-only layouts can still expire with their last layer.
    _peer_resources: ClassVar[list[_VirtualExpertRuntime | _VirtualExpertStorage]] = []

    def initialize_virtual_expert_load_balancer(
        self,
        *,
        group: dist.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: MoEConfig,
    ) -> None:
        """Initialize virtual-expert state without initializing the transport parent."""
        self.ep_size = dist.get_world_size(group)
        if not (
            self.ep_size == config.expert_model_parallel_size
            and num_experts == config.num_moe_experts
            and num_experts == self.ep_size * num_local_experts
            and router_topk == config.moe_router_topk
        ):
            raise ValueError(
                "Virtual-expert runtime layout must match MoEConfig; got "
                f"EP={self.ep_size}, experts={num_experts}, local={num_local_experts}, "
                f"topk={router_topk}."
            )
        self.group = group
        self.config = config
        self.router_topk = router_topk
        self.num_owned_experts = num_local_experts
        self._plan: VirtualExpertPlan | None = None
        self.planner_done: object | None = None
        self._experts: GroupedMLP | None = None
        self.weight_streams: tuple[object, object] | None = None
        self.grad_stream: object | None = None
        self.planner_stream: object | None = None
        self.planner: VirtualExpertPlannerWorkspace | None = None
        self._shared_runtime: _VirtualExpertRuntime | None = None
        self.storage: _VirtualExpertStorage | None = None

    def bind_experts(self, experts: GroupedMLP) -> None:
        """Bind the expert module without retaining its current weight views."""
        self._experts = experts
        experts.bind_virtual_experts(self)

    def _unshard_source_weights(self) -> None:
        """Unshard source weights on every EDP peer, including idle EP groups."""
        if torch_version() >= (2, 6, 0):
            from torch.distributed.fsdp._fully_shard import FSDPModule
        else:
            from torch.distributed._composable.fsdp.fully_shard import FSDPModule
        if isinstance(self._experts, FSDPModule):
            unshard = self._experts.unshard(async_op=True)
            if unshard is not None:
                unshard.wait()

    def _runtime_init(self, hidden_states: torch.Tensor) -> None:
        """Initialize reusable resources without retaining FSDP weight views."""
        if self.planner is not None and self.planner.device == hidden_states.device:
            return
        self._unshard_source_weights()
        self._bind_runtime(hidden_states.device)

    def _bind_runtime(self, device: torch.device) -> None:
        """Bind shared buffers using the current unsharded expert layout."""
        runtime = get_device_runtime()
        key = (
            runtime, self.group, device,
            self.ep_size * self.num_owned_experts, self.ep_size,
        )
        shared = self.runtimes.get(key)
        planner_peer_access = ops.requires_peer_memory("plan_virtual_expert_routes")
        histogram_bytes = weight_bytes = grad_bytes = None
        if shared is None and planner_peer_access:
            histogram_bytes = self.ep_size * key[3] * torch.int64.itemsize
        storage = None
        if self._experts is not None:
            w1, w2 = self._experts.native_weights()
            config = _VirtualExpertConfig(
                self.group, w1.device, self.num_owned_experts,
                (w1.shape[1:], w2.shape[1:]), w1.dtype,
                torch.float32 if w1.dtype in (torch.float16, torch.bfloat16) else w1.dtype,
                ops.requires_peer_memory("virtual_expert_weight_exchange"),
                ops.requires_peer_memory("virtual_expert_grad_reduce"),
            )
            storage = self.storages.get(config)
            if storage is None:
                elements = w1.numel() + w2.numel()
                if config.weight_peer_access:
                    weight_bytes = elements * config.dtype.itemsize
                if config.grad_peer_access:
                    grad_bytes = elements * config.grad_dtype.itemsize
        if shared is None or (self._experts is not None and storage is None):
            # Native-only ranks must agree with their SHMEM peers before any allocation.
            runtime.shmem.prepare_heap([histogram_bytes, weight_bytes, grad_bytes])
        if shared is None:
            shared = _VirtualExpertRuntime(
                runtime, key[3], device, self.group, planner_peer_access,
            )
            self.runtimes[key] = shared
            if planner_peer_access:
                self._peer_resources.append(shared)
        self._shared_runtime = shared
        self.planner_stream = shared.planner_stream
        self.weight_streams = shared.weight_streams
        self.grad_stream = shared.grad_stream
        if self._experts is not None:
            if storage is None:
                storage = _VirtualExpertStorage(config)
                self.storages[config] = storage
                if config.weight_peer_access or config.grad_peer_access:
                    self._peer_resources.append(storage)
            self.storage = storage
        self.planner = shared.planner

    @classmethod
    def finalize(cls) -> None:
        """Release shared model resources before the process group is destroyed.

        Call after training and all live graphs have finished. Bound layers
        must not execute again; construct new layers to initialize new resources.
        All WORLD ranks using NPU SHMEM call finalize before destroying their group.
        """
        for storage in tuple(cls.storages.values()):
            storage.destroy()
        for runtime in tuple(cls.runtimes.values()):
            runtime.destroy()
        cls.storages.clear()
        cls.runtimes.clear()
        cls._peer_resources.clear()
        get_device_runtime().shmem.release_unused()

    @property
    def num_runtime_experts(self) -> int:
        """Natives plus virtual-expert slots per rank."""
        return 2 * self.num_owned_experts

    @property
    def prefetch_done(self) -> object | None:
        """Completion event for the execution currently bound to the manager."""
        return None if self._plan is None else self._plan.prefetch_done

    @contextmanager
    def _on_side_stream(self, stream: object | None, done: object | None):
        """Wait on the caller and record completion, including failed submissions."""
        runtime = get_device_runtime()
        caller = runtime.current_stream()
        if isinstance(stream, tuple):
            stream = stream[1] if stream[0] == caller else stream[0]
        if stream is not None and stream != caller:
            stream.wait_stream(caller)
        with runtime.stream(stream):
            try:
                yield stream
            finally:
                if done is not None:
                    done.record(stream)

    # ---- forward: dispatcher hooks -----------------------------------------------------------

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Initialize the runtime before routing.

        Native gradient completion is attached inside the expert FSDP boundary.
        """
        # Replay must join the previous push before replacing its plan, so
        # planner and weight kernels preserve the SHMEM team's barrier order.
        self._wait_weight_push()
        self._runtime_init(hidden_states)
        self._plan = self.planner_done = None
        return hidden_states

    def plan_dispatch(self, top_indices: torch.Tensor) -> None:
        """Overlap planning with independent shared-expert compute."""
        runtime = get_device_runtime()
        done = runtime.new_event() if self.planner_stream is not None else None
        # A failed launch still records done; cleanup must join that submission.
        self.planner_done = done
        with self._on_side_stream(self.planner_stream, self.planner_done):
            runtime.record_stream(top_indices, self.planner_stream)
            self._plan = plan_virtual_expert_routes(top_indices, self.planner)
        self._plan.ready = self.planner_done
        self._start_weight_push(WeightDirection.FORWARD)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Join planned routes without waiting for the replica weight push."""
        runtime = get_device_runtime()
        routing = self._plan.routing.wait()
        if self._plan.ready is not None:
            current = runtime.current_stream()
            current.wait_event(self._plan.ready)
            runtime.record_stream(routing.virtual_experts, current)
        virtual_experts, routing.virtual_experts = routing.virtual_experts, None
        return hidden_states, virtual_experts

    def prepare_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the backward weight-push wait before transport combine."""
        return _VirtualExpertHook.apply(hidden_states, self._prepare_expert_backward, self._plan)

    def finalize_layer_output(self, output: torch.Tensor) -> torch.Tensor:
        """Hand the plan to the output hook, which starts its backward weight push."""
        plan, self._plan = self._plan, None
        self.planner_done = None
        return _VirtualExpertHook.apply(output, self._start_backward, plan)

    # ---- weight push -------------------------------------------------------------------------

    @torch.no_grad()
    def _start_weight_push(self, direction: WeightDirection) -> None:
        """Push both FCs' weights into this pass's shared slots."""
        runtime = get_device_runtime()
        plan = self._plan
        self._unshard_source_weights()
        if direction == WeightDirection.FORWARD and self.storage is not None:
            # FSDP may use a different dtype while the parameter is sharded.
            weight = self._experts.weight1
            layout = self.storage.config
            if weight.dtype != layout.dtype or weight.device != layout.device:
                self._bind_runtime(weight.device)
        # An unfinished device plan has no host replica count yet. Every EP
        # rank must still submit its push, including ranks with no local work.
        if self.storage is None or plan.routing.has_global_replicas is False:
            return
        w1, w2 = self._experts.native_weights()
        storage = self.storage
        if self.weight_streams is not None:
            plan.prefetch_done = runtime.new_event()
        plan.push_in_flight = True
        with self._on_side_stream(self.weight_streams, self.prefetch_done) as weight_stream:
            if weight_stream is not None:
                if direction == WeightDirection.FORWARD and plan.ready is not None:
                    weight_stream.wait_event(plan.ready)
                if storage.last_use is not None:
                    weight_stream.wait_event(storage.last_use)
            runtime.record_stream(w1, weight_stream)
            runtime.record_stream(w2, weight_stream)
            r1, r2, _, _ = ops.virtual_expert_weight_exchange(
                w1, w2, None, None, plan.routing,
                out=storage.slot_weights, grad_buffer=storage.grad_arena,
                buffer_handles=(storage.weight_handle, storage.grad_handle),
            )
            if plan.slot_weights is None:
                # Other passes overwrite these slots, then backward restores this
                # plan's values. Its saved views need independent version counters.
                plan.slot_weights = tuple(slot.new_empty(0).set_(slot) for slot in (r1, r2))

    @torch.no_grad()
    def _wait_weight_push(self, *_) -> None:
        """Make the current stream wait for the pass's push, once per push."""
        plan = self._plan
        if plan is None or not plan.push_in_flight:
            return
        if self.prefetch_done is not None:
            runtime = get_device_runtime()
            current = runtime.current_stream()
            current.wait_event(self.prefetch_done)
            if plan.slot_weights is not None:
                for weight in plan.slot_weights:
                    runtime.record_stream(weight, current)
        plan.push_in_flight = False

    @torch.no_grad()
    def _start_backward(self, plan: VirtualExpertPlan) -> None:
        """At the layer output's backward, take the plan back and start its push."""
        self._plan = plan
        self._start_weight_push(WeightDirection.BACKWARD)
        # Frozen inputs and partial autograd.grad may never reach the completion
        # gate. A graph-task callback releases only this pass's remaining state.
        torch.autograd.Variable._execution_engine.queue_callback(
            partial(self._finish_expert_backward, plan)
        )

    @torch.no_grad()
    def _prepare_expert_backward(self, plan: VirtualExpertPlan) -> None:
        """Restore this graph's plan before expert backward or checkpoint replay."""
        if self._plan is not plan:
            self._start_backward(plan)
        self._wait_weight_push()

    # ---- gradient reduction ------------------------------------------------------------------

    @torch.no_grad()
    def _start_grad_reduce(
        self, plan: VirtualExpertPlan, fc_layer: int, wgrad: torch.Tensor,
    ) -> None:
        """Reduce this FC's completed wgrad back to the owners."""
        storage = self.storage
        runtime = get_device_runtime()
        if self.grad_stream is not None:
            plan.grad_reduce_done = runtime.new_event()
        offset = 0 if fc_layer == 0 else storage.slot_weights[0].numel()
        with self._on_side_stream(self.grad_stream, plan.grad_reduce_done) as stream:
            runtime.record_stream(wgrad, stream)
            plan.grad_results[fc_layer] = ops.virtual_expert_grad_reduce(
                wgrad[:self.num_owned_experts], wgrad[self.num_owned_experts:], plan.routing.weight_context,
                grad_buffer=storage.grad_arena, arena_offset=offset,
                buffer_handle=storage.grad_handle, allow_inplace=True,
            )

    @torch.no_grad()
    def _finish_grad_reduce(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Join reductions and return the original parameter gradients to native autograd."""
        plan, self._plan = self._plan, None
        if plan.grad_reduce_done is not None:
            runtime = get_device_runtime()
            current = runtime.current_stream()
            current.wait_event(plan.grad_reduce_done)
            for result in plan.grad_results:
                if result is not None:
                    runtime.record_stream(result, current)
        grad1, grad2 = plan.grad_results
        plan.grad_results[:] = (None, None)
        plan.grad_reduce_done = None
        return grad1, grad2

    # ---- native expert and execution adapters ------------------------------------------------

    @contextmanager
    def bind_expert_weights(
        self,
        hidden: torch.Tensor, counts: torch.Tensor, probs: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor],
    ) -> Iterator[tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]]:
        """Join the weight push and bind FSDP gradient completion for this forward."""
        self._wait_weight_push()
        plan = self._plan
        try:
            if plan.routing.has_global_replicas:
                hidden, probs = _VirtualExpertGradComplete.apply(*weights, hidden, probs, plan, self)
                num_owned = self.num_owned_experts
                num_replicas = plan.routing.num_local_replicas
                wgrad_counts = counts[:num_owned + num_replicas]
                owned_tokens = int(wgrad_counts[:num_owned].sum())
                has_replica_tokens = owned_tokens < hidden.shape[0]
                counts = wgrad_counts
                if not has_replica_tokens:
                    counts = counts[:num_owned]
                elif not owned_tokens:
                    counts = counts[num_owned:]

                linears = []
                for fc_layer, (owner, slot) in enumerate(zip(weights, plan.slot_weights)):
                    weight = owner
                    if has_replica_tokens:
                        replica = slot[:num_replicas]
                        weight = (owner, replica) if owned_tokens else replica
                    store = partial(self._start_grad_reduce, plan, fc_layer) if owner.requires_grad else None
                    linears.append(partial(
                        grouped_linear, weight=weight, wgrad_counts=wgrad_counts, wgrad_store=store,
                    ))
            else:
                counts = counts[:self.num_owned_experts]
                linears = [partial(grouped_linear, weight=w) for w in weights]
            fc1, fc2 = linears
            yield hidden, counts, probs, fc1, fc2
        finally:
            self.storage.record_use()

    def _finish_expert_backward(self, plan: VirtualExpertPlan) -> None:
        previous_plan = self._plan
        self._plan = plan
        self._wait_weight_push()
        self._finish_grad_reduce()
        if previous_plan is not plan:
            self._plan = previous_plan
        if self.storage is not None:
            self.storage.record_use()

    def _release_virtual_experts(self) -> None:
        """Release the current plan; backward hooks retain their own plans."""
        self._wait_weight_push()
        if self.planner_done is not None:
            get_device_runtime().current_stream().wait_event(self.planner_done)
        self._plan = self.planner_done = None
