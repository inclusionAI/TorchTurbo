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
"""Official MoonEP expert banks for ordinary grouped-GEMM executors."""
from __future__ import annotations

import weakref
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore._device import transfer_vector_cores


class ExpertProjectionBuffer:
    """Use one BF16 expert bank from the caller-owned shared SHMEM runtime.

    The caller owns the collective runtime and leases this bank across expert
    compute. Bind after init and release the source tensor only after finalize.
    Forward and backward consumers must finish before another invocation reuses
    the bank; framework autograd contexts must save home parameters, not its views.
    """

    def __init__(self, num_experts: int, ep_size: int, rows: int, columns: int, device: torch.device, *, storage: torch.Tensor) -> None:
        if ep_size <= 0 or num_experts <= 0 or num_experts % ep_size:
            raise ValueError("MoonEP experts must be evenly divisible by EP size")
        if rows <= 0 or columns <= 0:
            raise ValueError("Expert projection dimensions must be positive")
        self.num_experts = num_experts
        self.local_experts = num_experts // ep_size
        if (storage.shape != (2 * self.local_experts, rows, columns)
                or storage.dtype != torch.bfloat16 or storage.device != device):
            raise ValueError("MoonEP projection storage has an incompatible geometry, dtype or device")
        self.source = storage
        self.bank: torch.Tensor | None = None

    def bind(self) -> None:
        """Expose the bank; official grouped APIs own peer-address carriers."""
        self.bank = self.source


class ExpertTransfer:
    """Share staging between layers while retaining each graph's home parameters."""

    @torch.inference_mode(False)
    def __init__(
        self, group: dist.ProcessGroup, num_experts: int,
        projection_shapes: tuple[tuple[int, int], ...], device: torch.device,
        *, allocator: Any, grad_reduce_dtype: torch.dtype = torch.float32,
    ) -> None:
        if grad_reduce_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("MoonEP gradient return supports FP32 or BF16")
        self.grad_reduce_dtype = grad_reduce_dtype
        self.group = group
        self.rank = dist.get_rank(group)
        self.root_size = dist.get_world_size(group)
        self.local_experts = num_experts // self.root_size
        self.allocator = allocator
        self.projections = []
        try:
            for rows, columns in projection_shapes:
                storage = allocator.empty(
                    (2 * self.local_experts, rows, columns), dtype=torch.bfloat16)
                try:
                    projection = ExpertProjectionBuffer(
                        num_experts, self.root_size, rows, columns, device, storage=storage)
                except BaseException:
                    allocator.free(storage)
                    raise
                self.projections.append(projection)
        except BaseException:
            for projection in self.projections:
                allocator.free(projection.source)
            raise
        self.num_cores = int(torch.npu.get_device_properties(device).cube_core_num)
        self.vector_cores = transfer_vector_cores(device, self.num_cores)
        self.completion = torch.npu.Event()
        self.in_use = False
        self.used = False
        self.generation = 0
        self.cache_generation = None
        self.cache_token = None
        self.graphs = weakref.WeakSet()
        self.statistics = {"backward_cache_hits": 0, "backward_prefetches": 0}

    def bind(self) -> None:
        """Bind registered projections once their collective root is active."""
        for projection in self.projections:
            projection.bind()

    def claim(self) -> None:
        """Order shared banks after the preceding call on its actual stream."""
        if self.in_use:
            raise RuntimeError("MoonEP staging does not support concurrent expert calls")
        if self.used:
            torch.npu.current_stream().wait_event(self.completion)
        self.in_use = True

    def release(self) -> None:
        """Record completion before allowing another layer or stream to reuse banks."""
        if not self.in_use:
            raise RuntimeError("MoonEP staging was released without a lease")
        self.completion.record(torch.npu.current_stream())
        self.in_use = False
        self.used = True

    def prefetch(
        self, weights: tuple[torch.Tensor, ...], copies: torch.Tensor,
        extra: int | None = None, *, token: object | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Refresh model-owned parameters using the official remote-weight kernel."""
        if not self.in_use:
            raise RuntimeError("MoonEP prefetch requires an active staging lease")
        if len(weights) != len(self.projections):
            raise ValueError("MoonEP requires one home tensor for each projection")
        if extra is not None and extra != copies.shape[1]:
            raise ValueError("MoonEP borrowed capacity must match this invocation's copies")
        from npu_patch import _C
        from npu_patch.launchers import _env_use_udma

        banks = tuple(projection.bank for projection in self.projections)
        if any(bank is None for bank in banks):
            raise RuntimeError("Bind expert banks before publishing parameters")
        home = tuple(weight.reshape(self.local_experts, *bank.shape[1:])
                     for weight, bank in zip(weights, banks))
        udma = _env_use_udma("prefetch")
        blocks = self.num_cores if udma else self.vector_cores
        result = tuple(_C.prefetch_banks(
            home, banks, copies, self.local_experts, self.rank, self.root_size, blocks, udma))
        self.generation += 1
        self.cache_generation = self.generation
        self.cache_token = token
        return result

    def return_generated_gradients(
        self, gradients: tuple[torch.Tensor, ...], copies: torch.Tensor,
        *, output_dtypes: tuple[torch.dtype, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return native producer dW using the selected communication precision."""
        if not self.in_use:
            raise RuntimeError("MoonEP gradient return requires an active staging lease")
        if len(gradients) != len(self.projections):
            raise ValueError("MoonEP requires one gradient for each projection")
        from npu_patch import _C
        from npu_patch.launchers import _env_use_udma

        banks = tuple(projection.bank for projection in self.projections)
        if any(bank is None for bank in banks):
            raise RuntimeError("Bind expert banks before returning gradients")
        wire = tuple(gradient.to(self.grad_reduce_dtype) for gradient in gradients)
        udma = _env_use_udma("grad_reduce")
        blocks = self.num_cores if udma else self.vector_cores
        result = tuple(_C.reduce_banks(
            wire, banks, copies, self.local_experts, self.rank, blocks, udma,
            **({"output_dtypes": output_dtypes} if output_dtypes is not None else {})))
        self.cache_generation = None
        self.cache_token = None
        return result

    def close(self) -> None:
        """Retire executor views; the root owner finalizes registered backing."""
        if self.in_use or self.graphs:
            raise RuntimeError("MoonEP staging is still needed by an active call or backward graph")
        for projection in self.projections:
            self.allocator.free(projection.source)
