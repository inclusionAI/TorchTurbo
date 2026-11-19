"""Reuse native plans and a single data workspace for dynamic expert counts."""

import time
from dataclasses import replace
from typing import Any

import torch
from tt_megamoe import shmem
from tt_megamoe.modules.mega_moe import function as native
from tt_megamoe.modules.mega_moe import route as native_module
from tt_megamoe.modules.mega_moe.plan import build_mega_moe_plan
from tt_megamoe.modules.mega_moe.workspace import MegaMoeWorkspace
from tt_megamoe.scheduler.config import event_workspace_bytes, mega_moe_event_capacity

from .routing import prepare_route


class SharedPlanWorkspace(MegaMoeWorkspace):
    """Share data and graph leases while keeping plan-specific ready generations."""

    def __init__(
        self,
        parent: MegaMoeWorkspace,
        forward: torch.Tensor,
        backward: torch.Tensor,
        counter_bytes: int,
    ) -> None:
        super().__init__(
            shared=parent.shared,
            dtype=parent.dtype,
            device=parent.device,
            expert_capacity=parent.expert_capacity,
            routed_slots=parent.routed_slots,
            dispatch_mode=parent.dispatch_mode,
            source_buffer=parent.source_buffer,
            expert_buffer=parent.expert_buffer,
            routed_buffer=parent.routed_buffer,
            forward_event_counters=forward,
            backward_event_counters=backward,
            gmm_workspace=parent.gmm_workspace,
            swiglu_grad_workspace=parent.swiglu_grad_workspace,
            completion_event=parent.completion_event,
            event_counter_bytes=counter_bytes,
            capacity_floor=parent.capacity_floor,
            lock=parent.lock,
            graphs=parent.graphs,
        )
        self.parent = parent

    def claim(self) -> None:
        self.parent.claim()
        self.in_use = True

    def release(self) -> None:
        self.parent.release()
        self.in_use = False

    def wait_for_reuse(self) -> None:
        self.parent.wait_for_reuse()


class _RefetchContext:
    """Adapt the native saved-state ABI without retaining shared weight views."""

    def __init__(self, ctx: Any, weights: tuple[torch.Tensor, ...]) -> None:
        self.needs_input_grad = ctx.needs_input_grad
        self.autograd_ctx = ctx
        self.local_weights = weights
        self.saved_tensors: tuple = ()

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        # Native state stores dispatch/up/activation, then weight1/weight2.
        # Save original parameters through autograd so FSDP can reshard them.
        self.autograd_ctx.save_for_backward(
            *tensors[:3], *self.local_weights, *tensors[5:]
        )

    def finish_forward(self) -> None:
        # Do not retain parameters or create a cycle outside saved-tensor hooks.
        del self.autograd_ctx, self.local_weights

    def maybe_clear_saved_tensors(self) -> None:
        self.saved_tensors = ()


class _FusedTransferFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        route: Any,
        plan: Any,
        workspace: MegaMoeWorkspace,
        permutation: tuple,
        topk_weights: torch.Tensor | None,
        workspace_claimed: bool,
        copies: torch.Tensor,
        resources: Any,
        extra: int,
    ) -> torch.Tensor:
        ctx.transfer_resources = resources
        ctx.transfer_copies = copies
        ctx.original_shapes = weight1.shape, weight2.shape
        ctx.original_dtypes = weight1.dtype, weight2.dtype
        ctx.transfer_extra = extra
        native_ctx = ctx
        if resources.refetch_in_backward:
            native_ctx = _RefetchContext(ctx, (weight1, weight2))
            ctx.native_ctx = native_ctx
        resources.claim()
        try:
            weights = resources.prefetch((weight1, weight2), copies, extra)
            return native._MegaMoeFunction.forward(
                native_ctx,
                hidden,
                *weights,
                route,
                plan,
                workspace,
                permutation,
                topk_weights,
                workspace_claimed,
            )
        except Exception:
            if not resources.refetch_in_backward:
                resources.in_use = False
            raise
        finally:
            if resources.refetch_in_backward:
                native_ctx.finish_forward()
                resources.release()

    @staticmethod
    def backward(ctx: Any, output_gradient: torch.Tensor) -> tuple:
        resources = ctx.transfer_resources
        native_ctx = ctx
        if resources.refetch_in_backward:
            # Unpack before claiming: non-reentrant hooks may rerun forward.
            # FSDP's pre-backward hook has restored the local parameters here.
            saved = ctx.saved_tensors
            ctx.maybe_clear_saved_tensors()
            native_ctx = ctx.native_ctx
            resources.claim()
        try:
            if resources.refetch_in_backward:
                weights = resources.prefetch(
                    saved[3:5], ctx.transfer_copies, ctx.transfer_extra
                )
                native_ctx.saved_tensors = (*saved[:3], *weights, *saved[5:])
                del saved
            gradients = native._MegaMoeFunction.backward(native_ctx, output_gradient)
            # The native backward owns these gradients; reduce into them in place.
            owners = resources.return_generated_gradients(
                gradients[1:3], ctx.transfer_copies
            )
        finally:
            if resources.refetch_in_backward:
                native_ctx.maybe_clear_saved_tensors()
                resources.release()
        owners = tuple(
            gradient.to(dtype).reshape(shape)
            for gradient, dtype, shape in zip(
                owners, ctx.original_dtypes, ctx.original_shapes
            )
        )
        return (gradients[0], *owners, *gradients[3:], None, None, None)


class SharedMegaMoePlanPool:
    """Cache small native plans over one maximum-width data workspace per mode."""

    def __init__(
        self, group: Any, tokens: int, resources: Any, executors: dict
    ) -> None:
        self.group = group
        self.tokens = tokens
        self.resources = resources
        self.executors = executors
        self.states: dict[tuple[str, float], dict] = {}
        self.statistics = {
            "plan_builds": 0,
            "plan_build_seconds": 0.0,
            "data_workspaces": 0,
            "event_pool_bytes": 0,
        }

    def _state(self, hidden: torch.Tensor, config: Any, limit: float) -> dict:
        maximum = 2 * self.resources.root_size * self.resources.local_experts
        mode = "pre_activation" if limit else "none"
        key = (mode, limit)
        if key in self.states:
            return self.states[key]
        executor = self.executors[key]
        owner = executor
        owner.workspace.ensure(owner.spec, hidden.dtype, hidden.device)
        event_bytes = event_workspace_bytes(owner.spec.ep_size, maximum)
        stride = (event_bytes + 511) // 512 * 512
        slots = self.resources.local_experts + 1
        pools = tuple(
            shmem.empty((stride * slots,), dtype=torch.uint8, alignment=512)
            for _ in range(2)
        )
        state = {
            "owner": owner,
            "pools": pools,
            "stride": stride,
            "plans": {},
            "workspaces": {},
        }
        self.states[key] = state
        self.statistics["data_workspaces"] += 1
        self.statistics["event_pool_bytes"] += 2 * stride * slots
        return state

    def _plan(
        self, state: dict, experts: int, extra: int, device: torch.device, nominal: bool = False, batch: bool = False
    ) -> tuple[Any, MegaMoeWorkspace]:
        key = (extra, nominal, batch)
        if key not in state["plans"]:
            started = time.perf_counter()
            with torch.profiler.record_function(f"MoonEP::build_plan_C{extra}"):
                owner = state["owner"]
                spec = replace(owner.spec, num_experts=experts, nominal_schedule=nominal, batch_experts=batch)
                plan = build_mega_moe_plan(spec, device)
                size = event_workspace_bytes(spec.ep_size, spec.num_experts)
                begin = extra * state["stride"]
                view = SharedPlanWorkspace(
                    owner.workspace,
                    state["pools"][0][begin : begin + size],
                    state["pools"][1][begin : begin + size],
                    mega_moe_event_capacity(spec.num_experts, spec.ep_size) * 4,
                )
                state["plans"][key] = plan
                state["workspaces"][key] = view
            self.statistics["plan_builds"] += 1
            self.statistics["plan_build_seconds"] += time.perf_counter() - started
        return state["plans"][key], state["workspaces"][key]

    def prepare_all(self, hidden: torch.Tensor, config: Any, limit: float) -> None:
        """Build every topology-valid width before measured training starts."""
        state = self._state(hidden, config, limit)
        local = self.resources.local_experts
        for extra in self.extra_widths():
            experts = self.resources.root_size * (local + extra)
            for batch in (False, True):
                self._plan(state, experts, extra, hidden.device, batch=batch)
        for batch in (False, True):
            self._plan(state, self.resources.root_size * local, 0, hidden.device, nominal=True, batch=batch)

    def extra_widths(self) -> range | tuple[int, ...]:
        """Return the legal execution widths prepared before training."""
        local = self.resources.local_experts
        return tuple(
            sorted({bucket_extra_width(value, local) for value in range(local + 1)})
        )

    def execute(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        probs: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor],
        config: Any,
        limit: float,
        copies: torch.Tensor,
        extra: int,
        planned: Any,
    ) -> torch.Tensor:
        state = self._state(hidden, config, limit)
        owner = state["owner"]
        plan, workspace = self._plan(
            state, config.num_moe_experts, extra, hidden.device, nominal=planned.nominal_schedule, batch=planned.batch_experts
        )
        with torch.no_grad():
            route = prepare_route(hidden, ids, plan.spec, workspace, planned)
        if route.maximum_received_slots > owner.workspace.capacity_floor:
            raise RuntimeError("MoonEP route exceeds the reserved receive capacity")
        if extra == 0:
            local = self.resources.local_experts
            original = (
                weights[0].view(
                    local, config.hidden_size, 2 * config.moe_ffn_hidden_size
                ),
                weights[1].view(local, config.moe_ffn_hidden_size, config.hidden_size),
            )
            if not torch.is_grad_enabled():
                return native._execute_mega_moe_without_autograd(
                    probs, *original, route, plan, workspace
                )
            output = native._MegaMoeFunction.apply(
                hidden,
                *original,
                route.metadata,
                plan,
                workspace,
                (route.routed_tokens, ids, route.unpermute_mapping),
                None,
                False,
            )
            return native_module.restore_topk_output(
                output, route.unpermute_mapping, probs
            )
        if not torch.is_grad_enabled():
            self.resources.claim()
            try:
                prefetched = self.resources.prefetch(weights, copies, extra)
                return native._execute_mega_moe_without_autograd(
                    probs, *prefetched, route, plan, workspace
                )
            finally:
                self.resources.release()
        output = _FusedTransferFunction.apply(
            hidden,
            *weights,
            route.metadata,
            plan,
            workspace,
            (route.routed_tokens, ids, route.unpermute_mapping),
            None,
            False,
            copies,
            self.resources,
            extra,
        )
        return native_module.restore_topk_output(output, route.unpermute_mapping, probs)

    def close(self) -> None:
        for state in self.states.values():
            assert (
                not state["owner"].workspace.in_use
                and not state["owner"].workspace.graphs
            )
            for pool in state["pools"]:
                shmem.free(pool)


def bucket_extra_width(needed: int, local_experts: int) -> int:
    """Pad only empty virtual slots using topology-derived power-of-two widths."""
    return 0 if needed == 0 else min(local_experts, 1 << (needed - 1).bit_length())
