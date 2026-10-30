# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Differentiable autograd function for the low-level MegaMoe ABI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch_npu

from torchturbo.optimizations.mega_moe.runtime.torch import ops as multicore_ops

from .plan import MegaMoePlan
from .route import PreparedTopKRoute, RouteMetadata, restore_topk_output
from .workspace import MegaMoeWorkspace


@dataclass(frozen=True)
class _ForwardIntermediates:
    """Forward tensors owned by one autograd invocation."""

    up_proj: Any
    activation: Any
    down_proj: Any


@dataclass(frozen=True)
class _SavedBackwardState:
    """Forward tensors and route metadata restored by backward."""

    dispatch: Any
    up_proj: Any
    activation: Any
    weight1: Any
    weight2: Any
    group_list: Any
    dispatch_src_off: Any
    dispatch_target_off: Any
    dispatch_size: Any
    combine_src_off: Any
    combine_target_off: Any
    combine_size: Any


@dataclass(frozen=True)
class _BackwardIntermediates:
    """Backward tensors owned by one autograd invocation."""

    grad_weight1: Any
    grad_weight2: Any
    act_grad: Any
    swiglu_grad: Any
    gate_dx: Any


@dataclass(frozen=True)
class _ForwardExecution:
    """Borrowed workspace and synchronization state for one forward launch."""

    dispatch: Any
    source: Any
    combine: Any
    event_counters: Any
    gmm_workspace: Any
    intermediates: _ForwardIntermediates


@dataclass(frozen=True)
class _BackwardExecution:
    """Borrowed workspace and synchronization state for one backward launch."""

    dispatch: Any
    grad_x: Any
    event_counters: Any
    gmm_workspace: Any
    swiglu_workspace: Any
    intermediates: _BackwardIntermediates


def _workspace_tensor(tensor: Any | None, name: str) -> Any:
    """Return an initialized workspace tensor or raise an invariant error."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe workspace {name} is not initialized.")
    return tensor


def _dispatch_and_source(spec: Any, workspace: Any, rows: Any, capacity: int) -> tuple[Any, Any]:
    """Select transport storage while retaining a source-ready stream dependency."""
    if getattr(spec, "dispatch_mode", "push") == "push":
        return _workspace_tensor(workspace.expert_buffer, "expert_buffer"), rows
    source = _workspace_tensor(workspace.source_buffer, "source_buffer")
    if rows is not source:
        source.copy_(rows)
    dispatch = torch.empty((capacity, spec.hidden_size), dtype=rows.dtype, device=rows.device)
    return dispatch, source


def _stage_backward_source(ctx: Any, grad_output: Any, permutation_inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    """Write pull dY into SHMEM before expert scratch allocation."""
    if getattr(ctx.plan.spec, "dispatch_mode", "push") == "push":
        return grad_output, None
    source = _workspace_tensor(ctx.workspace.source_buffer, "source_buffer")
    if not ctx.has_unpermute:
        source.copy_(grad_output)
        return source, None
    mapping, expert_output, probs = permutation_inputs
    grad_probs = torch.empty_like(probs)
    multicore_ops.mega_moe_unpermute_grad_out(expert_output, grad_output, mapping, probs, source, grad_probs)
    return source, grad_probs.to(ctx.topk_dtype) if ctx.needs_input_grad[7] else None


def _allocate_forward_intermediates(
    spec: Any,
    capacity: int,
    routed_tokens: Any,
    dispatch: Any,
) -> _ForwardIntermediates:
    """Allocate outputs whose received rows are overwritten before use."""
    # GMM and SwiGLU cover the received prefix; an empty rank reads no rows.
    up_proj = torch.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    activation = torch.empty(
        (capacity, spec.intermediate_size),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    return _ForwardIntermediates(up_proj, activation, torch.empty_like(dispatch[:capacity]))


def _allocate_backward_intermediates(
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
    reusable_dispatch: Any | None = None,
    group_list: torch.Tensor | None = None,
) -> _BackwardIntermediates:
    """Allocate overwritten activations and zero-safe expert gradients."""
    grad_weight2 = torch.empty_like(weight2, memory_format=torch.contiguous_format)
    act_grad = torch.empty(
        (capacity, spec.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    gate_dx = reusable_dispatch if reusable_dispatch is not None else torch.empty(
        (capacity, spec.hidden_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    # Empty experts must retain exact zero gradients on the baseline kernel.
    grad_weight1 = torch.empty_like(weight1, memory_format=torch.contiguous_format)
    from .gradient import zero_empty_expert_gradients
    zero_empty_expert_gradients(grad_weight1, grad_weight2, group_list)
    return _BackwardIntermediates(grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx)


def _saved_backward_state(saved_tensors: tuple[Any, ...]) -> _SavedBackwardState:
    """Restore named kernel state from the flat autograd tensor tuple."""
    return _SavedBackwardState(*saved_tensors[:12])


def _prepare_forward_execution(
    workspace: MegaMoeWorkspace, plan: MegaMoePlan, routed_tokens: Any, capacity: int,
) -> _ForwardExecution:
    """Resolve shared buffers and event counters for forward."""
    dispatch, source = _dispatch_and_source(plan.spec, workspace, routed_tokens, capacity)
    return _ForwardExecution(
        dispatch=dispatch, source=source,
        combine=_workspace_tensor(workspace.routed_buffer, "routed_buffer"),
        event_counters=workspace.prepare_event_counters(forward=True),
        gmm_workspace=_workspace_tensor(workspace.gmm_workspace, "gmm_workspace"),
        intermediates=_allocate_forward_intermediates(plan.spec, capacity, routed_tokens, dispatch),
    )


def _prepare_backward_execution(
    workspace: MegaMoeWorkspace, plan: MegaMoePlan, saved: _SavedBackwardState, grad_output: Any,
) -> _BackwardExecution:
    """Resolve shared buffers and event counters for backward."""
    dispatch = (torch.empty_like(saved.dispatch) if plan.spec.dispatch_mode == "pull"
                else _workspace_tensor(workspace.expert_buffer, "expert_buffer"))
    return _BackwardExecution(
        dispatch=dispatch,
        grad_x=_workspace_tensor(workspace.routed_buffer, "routed_buffer"),
        event_counters=workspace.prepare_event_counters(forward=False),
        gmm_workspace=_workspace_tensor(workspace.gmm_workspace, "gmm_workspace"),
        swiglu_workspace=_workspace_tensor(workspace.swiglu_grad_workspace, "swiglu_grad_workspace"),
        intermediates=_allocate_backward_intermediates(
            plan.spec, saved.dispatch.shape[0], grad_output, saved.weight1, saved.weight2,
            dispatch[:saved.dispatch.shape[0]] if plan.reuse_backward_dispatch else None,
            group_list=saved.group_list,
        ),
    )


def _restore_input_gradient(ctx: Any, grad_x: Any, permutation_inputs: tuple[Any, ...]) -> Any:
    """Return an owned input gradient while the workspace lease is held."""
    if not ctx.needs_input_grad[0]:
        return None
    if not ctx.has_permutation:
        return grad_x.clone()
    (unpermute_mapping,) = permutation_inputs
    spec = ctx.plan.spec
    # Consume the shared gradient before release records completion.
    # The permutation gradient owns its reduced [T, H] output.
    return multicore_ops.moe_token_permute_grad(
        grad_x, unpermute_mapping, spec.local_num_tokens, spec.top_k
    )


def _save_forward_state(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    saved_dispatch: Any,
    up_proj: Any,
    activation: Any,
    weight1: Any,
    weight2: Any,
    metadata: RouteMetadata,
    permutation_inputs: tuple[Any, ...] = (),
) -> None:
    """Save owned forward tensors and route state for backward."""
    ctx.plan = plan
    ctx.workspace = workspace
    ctx.save_for_backward(
        saved_dispatch,
        up_proj,
        activation,
        weight1,
        weight2,
        metadata.group_list,
        metadata.dispatch_src_off,
        metadata.dispatch_target_off,
        metadata.dispatch_size,
        metadata.combine_src_off,
        metadata.combine_target_off,
        metadata.combine_size,
        *permutation_inputs,
    )
    workspace.track_graph(ctx)


def _release_completed_graph(ctx: Any, workspace: MegaMoeWorkspace) -> None:
    """Keep retained graphs alive; older engines conservatively wait for context GC."""
    # The engine's retention flag avoids unpacking saved tensors or guessing from output references.
    keep_graph = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)  # pylint: disable=protected-access
    if keep_graph is not None and not keep_graph():
        workspace.release_graph(ctx)


def _launch_forward_kernel(
    plan: MegaMoePlan,
    metadata: RouteMetadata,
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    execution: _ForwardExecution,
) -> None:
    """Launch the internal forward ABI with prepared buffers and metadata."""
    spec = plan.spec
    multicore_ops.mega_moe(
        execution.dispatch,
        metadata.dispatch_target_off,
        routed_tokens.contiguous(),
        metadata.dispatch_src_off,
        metadata.dispatch_size,
        weight1,
        metadata.group_list,
        execution.intermediates.up_proj,
        execution.intermediates.activation,
        weight2,
        metadata.group_list,
        execution.intermediates.down_proj,
        execution.combine,
        metadata.combine_target_off,
        metadata.combine_src_off,
        metadata.combine_size,
        execution.gmm_workspace,
        plan.up_proj_tiling,
        plan.swiglu_tiling,
        plan.down_proj_tiling,
        plan.fwd_runtime,
        execution.event_counters,
        spec.rank_id,
        spec.ep_size,
        spec.num_experts,
        spec.hidden_size,
        spec.local_num_tokens,
    )


def _launch_backward_kernel(
    plan: MegaMoePlan,
    saved: _SavedBackwardState,
    grad_output: Any,
    execution: _BackwardExecution,
) -> None:
    """Launch the internal backward ABI with prepared buffers and metadata."""
    spec = plan.spec
    gradients = execution.intermediates
    multicore_ops.mega_moe_grad(
        execution.dispatch,
        saved.dispatch_target_off,
        grad_output.contiguous(),
        saved.dispatch_src_off,
        saved.dispatch_size,
        saved.activation,
        gradients.grad_weight2,
        saved.weight2,
        gradients.act_grad,
        saved.up_proj,
        gradients.swiglu_grad,
        saved.weight1,
        gradients.gate_dx,
        execution.grad_x,
        saved.combine_target_off,
        saved.combine_src_off,
        saved.combine_size,
        saved.dispatch,
        gradients.grad_weight1,
        saved.group_list,
        plan.act_grad_tiling,
        plan.gate_grad_tiling,
        plan.w1_grad_tiling,
        plan.w2_grad_tiling,
        plan.swiglu_grad_tiling,
        execution.gmm_workspace,
        execution.swiglu_workspace,
        plan.bwd_runtime,
        execution.event_counters,
        spec.rank_id,
        spec.ep_size,
        spec.num_experts,
        spec.hidden_size,
        spec.local_num_tokens,
    )


# Torch declares variadic autograd hooks; concrete functions use operator-specific signatures.
class _MegaMoeFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Autograd bridge for the in-place forward/backward custom ops."""

    @staticmethod
    # pylint: disable-next=arguments-differ
    def forward(
        ctx: Any,
        routed_tokens: Any,
        weight1: Any,
        weight2: Any,
        route: RouteMetadata,
        plan: MegaMoePlan,
        workspace: MegaMoeWorkspace,
        permutation: tuple[Any, Any, Any] | None,
        topk_weights: torch.Tensor | None,
        workspace_claimed: bool,
    ) -> Any:
        """Launch the legacy forward op and save owned backward inputs.

        Args:
            ctx: Autograd context for saved backward state.
            routed_tokens: Expert-major token rows.
            weight1: Local gate and up-projection weights.
            weight2: Local down-projection weights.
            route: Route offsets, sizes and group metadata.
            plan: Shape-specific forward and backward descriptors.
            workspace: Reusable directional execution buffers.
            permutation: Optional routed rows, expert IDs and inverse mapping
                when the first tensor argument contains original token rows.
            topk_weights: Optional Router weights for pull source-output gradients.
            workspace_claimed: Whether the caller owns the forward workspace lease.

        Returns:
            Owned expert-major output rows.
        """
        spec = plan.spec
        if topk_weights is not None and (spec.dispatch_mode != "pull" or permutation is None):
            raise ValueError("Integrated output unpermutation requires pull dispatch and an input permutation.")
        metadata = route
        permutation_inputs = ()
        if permutation is not None:
            routed_tokens, _, unpermute_mapping = permutation
            if ctx.needs_input_grad[0] or topk_weights is not None:
                permutation_inputs = (unpermute_mapping,)
        ctx.has_permutation = permutation is not None
        ctx.has_unpermute = topk_weights is not None
        if workspace_claimed:
            if not workspace.in_use:
                raise RuntimeError("MegaMoe forward requires the caller-held workspace lease.")
        else:
            workspace.ensure(spec, routed_tokens.dtype, routed_tokens.device)
            workspace.claim()
        try:
            # Dispatch and combine overwrite disjoint route ranges before consumers run.
            capacity = metadata.expert_capacity
            execution = _prepare_forward_execution(
                workspace,
                plan,
                routed_tokens,
                capacity,
            )
            _launch_forward_kernel(
                plan,
                metadata,
                execution.source,
                weight1,
                weight2,
                execution,
            )
            output = execution.combine.clone()
            if spec.dispatch_mode == "pull":
                saved_dispatch = execution.dispatch
            else:
                # Combine has consumed down_proj before SHMEM dispatch is reused.
                execution.intermediates.down_proj.copy_(execution.dispatch[:capacity])
                saved_dispatch = execution.intermediates.down_proj
            if ctx.has_unpermute:
                probs = topk_weights.float().contiguous()
                ctx.topk_dtype = topk_weights.dtype
                permutation_inputs += (output, probs)
                output = torch_npu.npu_moe_token_unpermute(output, unpermute_mapping, probs=probs)
            _save_forward_state(
                ctx,
                plan,
                workspace,
                saved_dispatch,
                execution.intermediates.up_proj,
                execution.intermediates.activation,
                weight1,
                weight2,
                metadata,
                permutation_inputs,
            )
            return output
        finally:
            if not workspace_claimed:
                workspace.release()

    @staticmethod
    # pylint: disable-next=arguments-differ
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        """Launch the legacy backward op into zero-safe gradient buffers.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_output: Expert-major output gradient rows.

        Returns:
            Gradients aligned with the differentiable forward inputs.
        """
        plan = ctx.plan
        workspace = ctx.workspace
        # Non-reentrant checkpoint hooks allow each saved tensor to be unpacked only once.
        saved_tensors = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        saved = _saved_backward_state(saved_tensors)
        permutation_inputs = saved_tensors[12:]
        del saved_tensors
        workspace.claim()
        try:
            source, grad_topk_weights = _stage_backward_source(ctx, grad_output, permutation_inputs)
            permutation_inputs = permutation_inputs[:1]
            execution = _prepare_backward_execution(
                workspace,
                plan,
                saved,
                grad_output,
            )
            _launch_backward_kernel(
                plan,
                saved,
                source,
                execution,
            )
            # Release ordinary scratch before allocating the owned token gradient;
            # SHMEM remains leased until the current-stream work is recorded.
            execution = replace(execution, dispatch=None, intermediates=_BackwardIntermediates(
                execution.intermediates.grad_weight1,
                execution.intermediates.grad_weight2,
                None, None, None,
            ))
            grad_input = _restore_input_gradient(ctx, execution.grad_x, permutation_inputs)
            _release_completed_graph(ctx, workspace)
            return (
                grad_input,
                execution.intermediates.grad_weight1,
                execution.intermediates.grad_weight2,
                None,
                None,
                None,
                None,
                grad_topk_weights,
                None,
            )
        finally:
            workspace.release()






def _execute_mega_moe_without_autograd(
    topk_weights: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    route: PreparedTopKRoute,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    *,
    workspace_claimed: bool = False,
) -> torch.Tensor:
    """Restore owned token outputs before releasing inference-only scratch.

    Original reentrant-checkpoint forwards and inference need no backward-owned
    dispatch copy. Output restoration consumes SHMEM on the leased stream and
    returns ordinary owned storage, so later calls cannot overwrite the result.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("The no-autograd execution path requires a no_grad context")
    if workspace_claimed:
        if not workspace.in_use:
            raise RuntimeError("Inference requires the caller-held workspace lease")
    else:
        workspace.ensure(plan.spec, route.routed_tokens.dtype, route.routed_tokens.device)
        workspace.claim()
    try:
        execution = _prepare_forward_execution(workspace, plan, route.routed_tokens, route.metadata.expert_capacity)
        _launch_forward_kernel(plan, route.metadata, execution.source,
                               weight1.contiguous(), weight2.contiguous(), execution)
        return restore_topk_output(execution.combine, route.unpermute_mapping, topk_weights)
    finally:
        if not workspace_claimed:
            workspace.release()
