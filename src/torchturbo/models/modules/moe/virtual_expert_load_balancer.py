# Adapted from NVIDIA Megatron-LM:
# https://github.com/NVIDIA/Megatron-LM
#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Modified by the TorchTurbo contributors.

"""Virtual-expert load balancing, virtual-expert planning, and runtime weight movement."""

from __future__ import annotations

from collections.abc import Callable
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

from .moe_config import MoEConfig

if TYPE_CHECKING:
    from .experts import GroupedMLP


class WeightDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(slots=True)
class VirtualExpertPlan(_RoutingPlan):
    """One forward's routes and in-flight state, retained through its backward.

    Native autograd supplies gradients per execution instead of persistent
    main_grad buffers. Events also belong to the plan so live graphs and
    checkpoint replay can share the runtime without replacing each other's state.
    """

    push_in_flight: bool = False
    started: set[int] = field(default_factory=set)
    ready: object | None = None
    prefetch_done: object | None = None
    grad_reduce_done: object | None = None
    slot_weights: tuple[torch.Tensor, torch.Tensor] | None = None
    native_grads: list[torch.Tensor | None] = field(default_factory=lambda: [None, None])
    replica_grads: list[torch.Tensor | None] = field(default_factory=lambda: [None, None])
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
        top_indices.contiguous(), workspace.local_counts.numel(), workspace.ep_group,
        workspace=workspace, async_finish=True,
    )
    return VirtualExpertPlan(
        plan.virtual_experts, plan.experts_to_copy, plan.dispatch_context,
        plan.weight_context, plan.has_global_replicas, plan._materialize, plan.ep_group,
    )


class _VirtualExpertRuntime:
    """Planner scratch and communication streams shared by compatible layers."""

    def __init__(
        self, runtime: DeviceRuntime, num_experts: int, ep_size: int, device: torch.device,
    ) -> None:
        self.planner = VirtualExpertPlannerWorkspace.allocate(num_experts, ep_size, device)
        self.planner_stream: object | None = None
        self.weight_streams: tuple[object, object] | None = None
        self.grad_stream: object | None = None
        if runtime.supports_async_streams:
            self.planner_stream = runtime.new_stream()
            self.weight_streams = (runtime.new_stream(), runtime.new_stream())
            self.grad_stream = runtime.new_stream()


@dataclass(frozen=True, slots=True)
class _VirtualExpertConfig:
    """Storage layout shared by compatible layers in one expert group."""

    group: dist.ProcessGroup | None
    device: torch.device
    num_local_experts: int
    member_shapes: tuple[torch.Size, torch.Size]
    dtype: torch.dtype
    grad_dtype: torch.dtype


class _VirtualExpertStorage:
    """Reusable replica weights and gradient transport scratch for one layout."""

    def __init__(self, config: _VirtualExpertConfig) -> None:
        self.config = config
        self.member_numels = tuple(prod(shape) for shape in config.member_shapes)
        sections = tuple(config.num_local_experts * size for size in self.member_numels)
        runtime = get_device_runtime()
        with torch.inference_mode(False):
            self.weight_arena, self.weight_handle = runtime.allocate_communication_buffer(
                sum(sections), dtype=config.dtype, device=config.device, group=config.group,
            )
            self.grad_arena, self.grad_handle = runtime.allocate_communication_buffer(
                sum(sections), dtype=config.grad_dtype, device=config.device, group=config.group,
            )
        self.slot_weights = tuple(
            section.view(config.num_local_experts, *shape)
            for section, shape in zip(self.weight_arena.split(sections), config.member_shapes)
        )
        self.last_use: object | None = None

    def record_use(self) -> None:
        """Keep subsequent pushes behind all readers of the current contents."""
        runtime = get_device_runtime()
        if runtime.supports_async_streams:
            self.last_use = runtime.new_event()
            self.last_use.record(runtime.current_stream())


class _VirtualExperts:
    """One layer's runtime weight bindings over shared storage.

    Native weight views are acquired for each expert call and released before
    FSDP reshard. Plans retain each graph's replica slots and gradient results.
    """

    def __init__(self, storage: _VirtualExpertStorage) -> None:
        self.storage = storage
        self.config = storage.config
        self.runtime_weights: tuple[tuple[torch.Tensor, torch.Tensor | None], ...] | None = None


class _VirtualExpertHook(torch.autograd.Function):
    """Call a method during backward, preserving the tensor gradient.

    Its return values are gradients for inputs. Keep the callback alive so
    native autograd also works after the caller releases the module.
    """

    @staticmethod
    def forward(ctx, tensor, method, args, *inputs):
        ctx.method, ctx.args = method, args
        return tensor

    @staticmethod
    def backward(ctx, grad):
        return (grad, None, None, *(ctx.method(*ctx.args) or ()))


class _VirtualExpertBackward(torch.autograd.Function):
    """Restore the captured plan before the shared expert runtime is used."""

    @staticmethod
    def forward(ctx, hidden_states, plan, load_balancer):
        ctx.plan, ctx.load_balancer = plan, load_balancer
        return hidden_states

    @staticmethod
    def backward(ctx, grad):
        load_balancer = ctx.load_balancer
        if load_balancer._plan is not ctx.plan:
            load_balancer._start_backward(ctx.plan)
        load_balancer._prepare_expert_backward()
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


class _VirtualExpertFCReady(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, probs, native, replica, trainable, fc_layer, plan, load_balancer):
        ctx.set_materialize_grads(False)
        ctx.trainable, ctx.fc_layer = trainable, fc_layer
        ctx.plan, ctx.load_balancer = plan, load_balancer
        # Each graph restores these slots before reading its saved weights.
        # Independent version counters permit shared-arena reuse between passes.
        replica = replica.new_empty(0).set_(
            replica.untyped_storage(), replica.storage_offset(), replica.size(), replica.stride(),
        )
        if not trainable:
            ctx.mark_non_differentiable(native, replica)
        return hidden, probs, native, replica

    @staticmethod
    def backward(ctx, hidden_grad, probs_grad, native_grad, replica_grad):
        if ctx.trainable:
            plan, load_balancer = ctx.plan, ctx.load_balancer
            previous_plan = load_balancer._plan
            load_balancer._plan = plan
            # Both linear backwards recorded unrounded wgrads in this plan.
            if ctx.fc_layer == 1:
                load_balancer.start_fc2_grad_reduce()
            else:
                load_balancer._start_grad_reduce(0)
            if previous_plan is not plan:
                load_balancer._plan = previous_plan
        # FC2 must submit its reduction before releasing FC1's activation gradient.
        return hidden_grad, probs_grad, None, None, None, None, None, None


class VirtualExpertLoadBalancer:
    """Mixin that plans virtual-expert routes and coordinates replica execution."""

    storages: ClassVar[WeakValueDictionary[_VirtualExpertConfig, _VirtualExpertStorage]] = (
        WeakValueDictionary()
    )
    runtimes: ClassVar[WeakValueDictionary[
        tuple[DeviceRuntime, dist.ProcessGroup | None, torch.device, int, int],
        _VirtualExpertRuntime,
    ]] = WeakValueDictionary()

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
        self.ep_size = dist.get_world_size(group) if group is not None else 1
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
        self._alignment = 1
        self._plan: VirtualExpertPlan | None = None
        self.planner_done: object | None = None
        self._experts: GroupedMLP | None = None
        self.weight_streams: tuple[object, object] | None = None
        self.grad_stream: object | None = None
        self.planner_stream: object | None = None
        self.planner: VirtualExpertPlannerWorkspace | None = None
        self._shared_runtime: _VirtualExpertRuntime | None = None
        self.virtual_experts: _VirtualExperts | None = None

    def bind_experts(self, experts: GroupedMLP) -> None:
        """Bind the expert module, acquiring its current weights each forward."""
        self._experts = experts
        # Checkpoint captures this callable, so binding and native autograd
        # consumption replay inside the same boundary as the expert GEMMs.
        experts.forward = partial(self._expert_forward, experts.forward)

    def _runtime_init(self, hidden_states: torch.Tensor) -> None:
        """Initialize reusable resources without retaining FSDP weight views."""
        if self.planner is not None:
            return
        runtime = get_device_runtime()
        key = (
            runtime, self.group, hidden_states.device,
            self.ep_size * self.num_owned_experts, self.ep_size,
        )
        shared = self.runtimes.get(key)
        if shared is None:
            shared = _VirtualExpertRuntime(runtime, key[3], self.ep_size, hidden_states.device)
            shared.planner.ep_group = self.group
            self.runtimes[key] = shared
        self._shared_runtime = shared
        self.planner_stream = shared.planner_stream
        self.weight_streams = shared.weight_streams
        self.grad_stream = shared.grad_stream
        if self._experts is not None:
            # FSDP may cast weights during unshard. Build the layout from the
            # actual compute weights without retaining any unsharded views.
            self._prepare_source_weights()
            w1, w2 = self._native_weights()
            config = _VirtualExpertConfig(
                self.group, w1.device, self.num_owned_experts,
                (w1.shape[1:], w2.shape[1:]), w1.dtype,
                torch.float32 if w1.dtype in (torch.float16, torch.bfloat16) else w1.dtype,
            )
            storage = self.storages.get(config)
            if storage is None:
                storage = _VirtualExpertStorage(config)
                self.storages[config] = storage
            self.virtual_experts = _VirtualExperts(storage)
            self._experts.register_forward_pre_hook(self._wait_weight_push)
        self.planner = shared.planner

    @property
    def num_runtime_experts(self) -> int:
        """Natives plus virtual-expert slots per rank."""
        return 2 * self.num_owned_experts

    def runtime_weights(self, fc_layer: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the graph-bound native and replica matrices for FC1 (0) or FC2 (1)."""
        return self.virtual_experts.runtime_weights[fc_layer]

    def runtime_weight_grads(self) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        """Bind wgrad destinations to the exact execution used by expert forward."""
        return self._plan.native_grads, self._plan.replica_grads

    @property
    def source_parameters(self) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """Read the current optimizer-owned FC1 and FC2 weights after unshard."""
        return self._experts.weight1, self._experts.weight2

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
            self._plan = plan_virtual_expert_routes(top_indices, self.planner, self._alignment)
        self._plan.ready = self.planner_done
        self._start_weight_push(WeightDirection.FORWARD)

    def prepare_virtual_expert_dispatch(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Join planned routes without waiting for the replica weight push."""
        runtime = get_device_runtime()
        self._plan.wait()
        if self._plan.ready is not None:
            current = runtime.current_stream()
            current.wait_event(self._plan.ready)
            runtime.record_stream(self._plan.virtual_experts, current)
            metadata = self._plan.dispatch_context
            if metadata is not None:
                runtime.record_stream(metadata.sorted_expert_indices, current)
                runtime.record_stream(metadata.route_order, current)
        virtual_experts, self._plan.virtual_experts = self._plan.virtual_experts, None
        return hidden_states, virtual_experts

    def prepare_virtual_expert_combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach the backward weight-push wait before transport combine."""
        return _VirtualExpertBackward.apply(hidden_states, self._plan, self)

    def finalize_layer_output(self, output: torch.Tensor) -> torch.Tensor:
        """Hand the plan to the output hook, which starts its backward weight push."""
        plan, self._plan = self._plan, None
        self.planner_done = None
        return _VirtualExpertHook.apply(output, self._start_backward, (plan,))

    # ---- weight push -------------------------------------------------------------------------

    @torch.no_grad()
    def _start_weight_push(self, direction: WeightDirection) -> None:
        """Push both FCs' weights into this pass's shared slots."""
        runtime = get_device_runtime()
        plan = self._plan
        self._prepare_source_weights()
        # An unfinished device plan has no host replica count yet. Every EP
        # rank must still submit its push, including ranks with no local work.
        if self.virtual_experts is None or plan.has_global_replicas is False:
            return
        w1, w2 = self._native_weights()
        storage = self.virtual_experts.storage
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
                w1, w2, None, None, plan,
                out=storage.slot_weights, grad_buffer=storage.grad_arena,
                buffer_handles=(storage.weight_handle, storage.grad_handle),
            )
            if plan.slot_weights is None:
                plan.slot_weights = (r1, r2)

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
        plan.started.clear()
        self._start_weight_push(WeightDirection.BACKWARD)
        # Frozen inputs and partial autograd.grad may never reach the completion
        # gate. A graph-task callback releases only this pass's remaining state.
        torch.autograd.Variable._execution_engine.queue_callback(
            partial(self._finish_expert_backward, plan)
        )

    @torch.no_grad()
    def _prepare_expert_backward(self) -> None:
        """Wait for the backward push before expert backward or checkpoint replay."""
        self._wait_weight_push()

    # ---- gradient reduction ------------------------------------------------------------------

    @torch.no_grad()
    def _start_grad_reduce(self, fc_layer: int) -> None:
        """Enqueue one FC's reduction behind its native and replica wgrad GEMMs."""
        plan = self._plan
        native_grads, replica_grads = plan.native_grads[fc_layer], plan.replica_grads[fc_layer]
        storage = self.virtual_experts.storage
        runtime = get_device_runtime()
        if self.grad_stream is not None:
            plan.grad_reduce_done = runtime.new_event()
        offset = 0 if fc_layer == 0 else storage.slot_weights[0].numel()
        with self._on_side_stream(self.grad_stream, plan.grad_reduce_done) as stream:
            runtime.record_stream(native_grads, stream)
            runtime.record_stream(replica_grads, stream)
            plan.grad_results[fc_layer] = ops.virtual_expert_grad_reduce(
                native_grads, replica_grads, plan.weight_context,
                grad_buffer=storage.grad_arena, arena_offset=offset,
                buffer_handle=storage.grad_handle,
            )
        plan.native_grads[fc_layer] = plan.replica_grads[fc_layer] = None
        plan.started.add(fc_layer)

    def start_fc2_grad_reduce(self) -> None:
        """Start FC2's reduction from expert backward, right behind its wgrad GEMMs."""
        self._start_grad_reduce(1)

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
        results = tuple(plan.grad_results)
        plan.grad_results[:] = [None, None]
        plan.grad_reduce_done = None
        plan.started.clear()
        return results

    # ---- native expert and execution adapters ------------------------------------------------

    def _native_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight1, weight2 = self.source_parameters
        return (
            weight1.view(self.num_owned_experts, self.config.hidden_size, -1),
            weight2.view(self.num_owned_experts, -1, self.config.hidden_size),
        )

    def _expert_forward(
        self,
        forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, None]],
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Prepare and bind runtime tensors inside the expert checkpoint boundary."""
        plan = self._plan
        virtual_experts = self.virtual_experts
        storage = virtual_experts.storage
        # Expert-only checkpoint replay invokes forward without module hooks.
        self._wait_weight_push()
        w1, w2 = self._native_weights()
        replica1 = replica2 = None
        try:
            if plan.has_global_replicas:
                num_replicas = sum(plan.weight_context.receive_splits)
                with torch.no_grad():
                    replica1, replica2, _, _ = ops.virtual_expert_weight_exchange(
                        w1, w2, None, None,
                        plan.weight_context, prefetched=plan.slot_weights,
                        out=storage.slot_weights,
                        grad_buffer=storage.grad_arena,
                        buffer_handles=(storage.weight_handle, storage.grad_handle),
                    )
                replica1, replica2 = replica1[:num_replicas], replica2[:num_replicas]
                # This gate must be inside the expert FSDP input hook. Returning
                # parameter gradients after dispatch backward would miss FSDP's
                # post-backward reduction. Checkpoint replay captures it here.
                permuted_local_hidden_states, permuted_probs = _VirtualExpertGradComplete.apply(
                    w1, w2, permuted_local_hidden_states, permuted_probs, plan, self,
                )
            else:
                tokens_per_expert = tokens_per_expert[:self.num_owned_experts]
            virtual_experts.runtime_weights = ((w1, replica1), (w2, replica2))
            self._experts.bind_virtual_experts(self)
            return forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        finally:
            # Release unsharded native views before FSDP's post-forward hook.
            self._experts._runtime_weights = None
            self._experts._runtime_weight_grads = None
            self._experts._runtime_weight_hook = None
            virtual_experts.runtime_weights = None
            storage.record_use()

    def prepare_fc_weights(
        self, fc_layer: int,
        hidden_states: torch.Tensor, probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attach one FC's readiness gate before its native and replica GEMMs."""
        native, replica = self.virtual_experts.runtime_weights[fc_layer]
        return _VirtualExpertFCReady.apply(
            hidden_states, probs, native.detach(), replica,
            native.requires_grad, fc_layer, self._plan, self,
        )

    def _prepare_source_weights(self) -> None:
        """Make source expert weights available for the current execution stream.

        FSDP preparation includes every EDP peer, including idle EP groups.
        """
        if torch_version() >= (2, 6, 0):
            from torch.distributed.fsdp._fully_shard import FSDPModule
        else:
            from torch.distributed._composable.fsdp.fully_shard import FSDPModule
        if isinstance(self._experts, FSDPModule):
            unshard = self._experts.unshard(async_op=True)
            if unshard is not None:
                unshard.wait()

    def _finish_expert_backward(self, plan: VirtualExpertPlan) -> None:
        previous_plan = self._plan
        self._plan = plan
        self._wait_weight_push()
        self._finish_grad_reduce()
        if previous_plan is not plan:
            self._plan = previous_plan
        if self.virtual_experts is not None:
            self.virtual_experts.storage.record_use()

    def _release_virtual_experts(self) -> None:
        """Release the current plan; backward hooks retain their own plans."""
        self._wait_weight_push()
        if self.planner_done is not None:
            get_device_runtime().current_stream().wait_event(self.planner_done)
        self._plan = self.planner_done = None
