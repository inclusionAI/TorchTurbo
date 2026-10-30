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
"""Owned directional workspace for managed Torch MegaMoe execution."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from torchturbo.optimizations.mega_moe.runtime import _automatic, shmem
from torchturbo.optimizations.mega_moe.runtime.scheduler.config import (
    MIN_EVENT_CAPACITY,
    event_workspace_bytes,
    mega_moe_event_capacity,
)

from .spec import MegaMoeSpec, _resolve_receive_capacity

# The composed Cube-only GMM and SwiGLU-grad kernels do not use these legacy
# tensor arguments. Keep non-empty ABI placeholders; native tiling reserves
# the separate CANN library workspace needed by the enclosing operator.
_GMM_WORKSPACE_BYTES = 512
_SWIGLU_GRAD_WORKSPACE_BYTES = 512
_WORKSPACE_ALIGNMENT = 512
_SYMMETRIC_FIELDS = ("source_buffer", "expert_buffer", "routed_buffer",
                     "forward_event_counters", "backward_event_counters")


def _spec_workspace_bytes(
    specification: Mapping[str, Any], element_size: int, *, receive_capacity: int | None = None,
) -> int:
    """Return planned symmetric bytes for one execution-resource group."""
    local_tokens = specification["local_num_tokens"]
    routed_slots = local_tokens * specification["top_k"]
    capacity = _resolve_receive_capacity(
        specification["initial_capacity_factor"],
        routed_slots,
        specification["ep_size"],
    )
    if receive_capacity is not None:
        capacity = receive_capacity
    if specification.get("dispatch_mode", "push") == "pull":
        capacity = routed_slots
    tensor_bytes = (
        (capacity + routed_slots) * specification["hidden_size"] * element_size
    )
    event_bytes = event_workspace_bytes(specification["ep_size"], specification["num_experts"])
    return tensor_bytes + 2 * event_bytes + 4 * (_WORKSPACE_ALIGNMENT - 1)


@dataclass
class MegaMoeWorkspace:
    """Buffers owned by one standalone or explicitly shared resource group."""

    shared: bool
    dtype: Any | None = None
    device: Any | None = None
    expert_capacity: int = 0
    routed_slots: int = 0
    dispatch_mode: str = "push"
    source_buffer: Any | None = None
    expert_buffer: Any | None = None
    routed_buffer: Any | None = None
    forward_event_counters: Any | None = None
    backward_event_counters: Any | None = None
    gmm_workspace: Any | None = None
    swiglu_grad_workspace: Any | None = None
    completion_event: Any | None = None
    in_use: bool = False
    closing: bool = False
    used: bool = False
    event_counter_bytes: int = MIN_EVENT_CAPACITY * 4
    forward_ready_initialized: bool = False
    backward_ready_initialized: bool = False
    capacity_floor: int = 0
    lock: Any = field(default_factory=threading.Lock, repr=False)
    graphs: Any = field(default_factory=weakref.WeakSet, repr=False)


    def track_graph(self, context: Any) -> None:
        """Retain a weak lease while this context can still run backward.

        Args:
            context: Autograd context that may still access workspace buffers.
        """
        self.graphs.add(context)

    def can_close(self) -> bool:
        """Check active calls and weak graph leases without unpacking saved tensors."""
        return not self.in_use and not self.graphs

    def release_graph(self, context: Any) -> None:
        """Retire this context's lease after successful non-retained backward.

        Args:
            context: Autograd context whose final backward completed.
        """
        self.graphs.discard(context)

    def ensure(self, spec: MegaMoeSpec, dtype: Any, device: Any) -> None:
        """Allocate the fixed configured route capacity once.

        Args:
            spec: First-forward-bound shape and capacity specification.
            dtype: Element dtype for symmetric data buffers.
            device: Device that owns local and symmetric buffers.
        """
        with self.lock:
            self._ensure_buffers(spec, dtype, device)

    def _ensure_buffers(self, spec: MegaMoeSpec, dtype: Any, device: Any) -> None:
        """Allocate this workspace at its current monotonic capacity."""
        if self.closing:
            raise RuntimeError("cannot allocate a closing MegaMoe workspace.")
        requested_capacity = max(spec.receive_capacity, self.capacity_floor)
        if self.expert_buffer is not None or self.source_buffer is not None:
            compatible = (
                self.dispatch_mode == spec.dispatch_mode
                and self.dtype == dtype
                and self.device == device
                and self.routed_slots == spec.routed_slots
                and self.expert_capacity == requested_capacity
            )
            if not compatible:
                raise ValueError("MegaMoe workspace cannot change device, dtype, routed shape, or capacity.")
            return

        self.dispatch_mode = spec.dispatch_mode
        self.dtype = dtype
        self.device = device
        self.expert_capacity = requested_capacity
        self.routed_slots = spec.routed_slots
        self.event_counter_bytes = mega_moe_event_capacity(spec.num_experts, spec.ep_size) * 4
        event_bytes = event_workspace_bytes(spec.ep_size, spec.num_experts)
        try:
            field_name = "source_buffer" if self.dispatch_mode == "pull" else "expert_buffer"
            rows = spec.routed_slots if self.dispatch_mode == "pull" else requested_capacity
            setattr(self, field_name, shmem.empty(
                (rows, spec.hidden_size), dtype=dtype, alignment=_WORKSPACE_ALIGNMENT,
            ))
            self.routed_buffer = shmem.empty(
                (spec.routed_slots, spec.hidden_size),
                dtype=dtype,
                alignment=_WORKSPACE_ALIGNMENT,
            )
            self.forward_event_counters = shmem.empty(
                (event_bytes,),
                dtype=torch.uint8,
                alignment=_WORKSPACE_ALIGNMENT,
            )
            self.backward_event_counters = shmem.empty(
                (event_bytes,),
                dtype=torch.uint8,
                alignment=_WORKSPACE_ALIGNMENT,
            )
            self.gmm_workspace = torch.empty(
                (_GMM_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.swiglu_grad_workspace = torch.empty(
                (_SWIGLU_GRAD_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.completion_event = torch.npu.Event()
        except Exception:
            self._free_symmetric_tensors()
            self._free_local_tensors()
            raise

    def prepare_event_counters(self, *, forward: bool) -> Any:
        """Reset per-call counters while preserving the ready generation."""
        if forward:
            field_name = "forward_event_counters"
            events = self.forward_event_counters
            ready_initialized = self.forward_ready_initialized
        else:
            field_name = "backward_event_counters"
            events = self.backward_event_counters
            ready_initialized = self.backward_ready_initialized
        if events is None:
            raise RuntimeError(f"MegaMoe workspace {field_name} is not initialized.")
        events[:self.event_counter_bytes].zero_()
        if events.numel() == self.event_counter_bytes or ready_initialized:
            return events

        # Initialize the persistent generation before any peer can signal it.
        events[self.event_counter_bytes:].zero_()
        shmem.host_barrier()
        if forward:
            self.forward_ready_initialized = True
        else:
            self.backward_ready_initialized = True
        return events

    def wait_for_reuse(self) -> None:
        """Order count exchange after the previous workspace lease."""
        with self.lock:
            if self.used:
                torch.npu.current_stream(self.device).wait_event(self.completion_event)

    def claim(self) -> None:
        """Claim the serial workspace and order it after the previous stream."""
        with self.lock:
            if self.closing:
                raise RuntimeError("cannot claim a closing MegaMoe workspace.")
            if self.in_use:
                raise RuntimeError("MegaMoe execution resources do not support concurrent calls.")
            if self.used:
                torch.npu.current_stream().wait_event(self.completion_event)
            self.in_use = True

    def release(self) -> None:
        """Record completion ordering and release the serial workspace lease."""
        with self.lock:
            if not self.in_use:
                raise RuntimeError("MegaMoe workspace was released without an active lease.")
            self.completion_event.record(torch.npu.current_stream())
            self.in_use = False
            self.used = True

    def _free_symmetric_tensors(self) -> None:
        """Free each unique SHMEM allocation and invalidate its tensor view."""
        if self.closing:
            remaining = tuple(tensor is not None for tensor in (
                self.source_buffer, self.expert_buffer, self.routed_buffer,
                self.forward_event_counters, self.backward_event_counters))
            if any(peer != remaining for peer in _automatic.exchange(remaining)):
                raise RuntimeError("partial symmetric teardown differs across ranks; restart the process")
        for field_name in (
            "source_buffer",
            "expert_buffer",
            "routed_buffer",
            "forward_event_counters",
            "backward_event_counters",
        ):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            shmem.free(tensor)
            setattr(self, field_name, None)
        self.expert_capacity = 0
        self.routed_slots = 0
        self.forward_ready_initialized = False
        self.backward_ready_initialized = False

    def _free_local_tensors(self) -> None:
        """Release local-only workspaces after all queued kernels complete."""
        for field_name in ("gmm_workspace", "swiglu_grad_workspace"):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            tensor.untyped_storage().resize_(0)
            setattr(self, field_name, None)

    def close(self) -> None:
        """Synchronize and release buffers after all backward graph leases end."""
        with self.lock:
            if self.in_use:
                raise RuntimeError("cannot close MegaMoe workspace during an active call.")
            if not self.can_close():
                raise RuntimeError("cannot close MegaMoe workspace while backward graphs still need its buffers.")
            empty = all(tensor is None for tensor in (
                self.source_buffer,
                self.expert_buffer,
                self.routed_buffer,
                self.forward_event_counters,
                self.backward_event_counters,
                self.gmm_workspace,
                self.swiglu_grad_workspace,
                self.completion_event,
            ))
            self.closing = True
        # An empty rank must still participate when peers retain buffers after a failure.
        if all(_automatic.exchange(empty)):
            self.closing = True
            return
        _automatic.collective_call(lambda: None if empty else torch.npu.synchronize(self.device))
        # Workspace teardown requires each collective barrier to complete before
        # the following free or local Tensor release.
        shmem.host_barrier()
        _automatic.collective_call(self._free_symmetric_tensors)
        shmem.host_barrier()
        _automatic.collective_call(self._free_local_tensors)
        self.dtype = None
        self.device = None
        self.completion_event = None
        self.used = False
        self.closing = True
