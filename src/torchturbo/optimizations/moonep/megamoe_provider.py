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
"""TorchTurbo-owned official MoonEP provider injected into HMC.

The caller owns the HMC runtime, native schedule and autograd. This provider
adds only official planning/transfer calls and a managed planning workspace.
The logical carrier preserves the official local-base/remote-owner contract;
its metadata is consumed only by the official prefetch launcher.
"""
from typing import Any

import torch

from hyper_multicore.integrations.mega_moe.planning import PlanningWorkspace
from hyper_multicore.moonep.placement import ExpertPlacement

from npu_patch.launchers import (
    planning_workspace_sizes, _env_use_udma,
)


def build_official_placement(
    hidden: torch.Tensor, ids: torch.Tensor, counts: torch.Tensor, target: int, rank: int,
    *, resources: Any, source_rows: tuple[int, ...],
    source_inverse: torch.Tensor | None = None,
    source_counts: tuple[tuple[int, ...], ...] | None = None,
) -> ExpertPlacement | None:
    """Bind managed scratch to the execution-independent official planner."""
    from .official_planner import OfficialPlanner

    engine = resources
    ranks, experts = counts.shape
    tokens, top_k = ids.shape
    workspace = engine.placement_workspace
    if workspace is None or workspace.capacity < tokens:
        if workspace is not None:
            workspace.manager.close_workspace(workspace)
        capacity = 1 << max(0, (tokens - 1).bit_length())
        local, symmetric = planning_workspace_sizes(
            capacity, top_k, experts, capacity * top_k, ranks, engine.spec.num_cube_cores)
        workspace = PlanningWorkspace(engine, capacity, local, symmetric,
                                      (experts, top_k), hidden.device)
        engine.placement_workspace = workspace
    planner = workspace.provider_state
    if planner is None:
        planner = OfficialPlanner(experts, top_k, ranks, rank, engine.spec.num_cube_cores)
        workspace.provider_state = planner
    workspace.in_use = True
    try:
        return planner(hidden, ids, counts, target, rank,
                       resources=(workspace.local, workspace.symmetric.data_ptr(), workspace.symmetric.numel()),
                       source_rows=source_rows, source_inverse=source_inverse, source_counts=source_counts)
    finally:
        workspace.in_use = False


def prefetch(
    pool: Any, weights: tuple[torch.Tensor, ...], copies: torch.Tensor,
    extra: int, token: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Use official prefetch on the existing shared weight-bank lease."""
    if not pool.in_use:
        raise RuntimeError("Official prefetch requires an active upstream bank lease")
    pool.cache_token = pool.cache_generation = None
    from npu_patch import _C

    if extra:
        with pool.transfer_context():
            udma = _env_use_udma('prefetch')
            blocks = pool.blocks if udma else pool.transfer_blocks
            prefetched = tuple(_C.prefetch_banks(
                weights, pool.weights, copies[:, :extra].contiguous(), pool.local_experts,
                pool.rank, pool.ranks, blocks, udma))
    else:
        for bank, weight in zip(pool.weights, weights):
            bank[:pool.local_experts].copy_(weight)
        prefetched = tuple(bank[:pool.local_experts].detach() for bank in pool.weights)
    pool.generation += 1
    pool.cache_generation, pool.cache_token = pool.generation, token
    pool.statistics["prefetches"] += 1
    pool.statistics["remote_prefetches"] += int(bool(extra))
    return prefetched


def return_gradients(
    pool: Any, gradients: tuple[torch.Tensor, ...], copies: torch.Tensor,
    active: bool = True, *, output_dtypes: tuple[torch.dtype, ...] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Publish FP32/BF16 partial gradients and reduce with FP32 accumulation."""
    from npu_patch import _C

    if active:
        with pool.transfer_context():
            udma = _env_use_udma('grad_reduce')
            blocks = pool.blocks if udma else pool.transfer_blocks
            home = _C.reduce_banks(gradients, pool.weights, copies, pool.local_experts,
                                   pool.rank, blocks, udma,
                                   **({'output_dtypes': output_dtypes} if output_dtypes is not None else {}))
    else:
        home = tuple(gradient[:pool.local_experts] for gradient in gradients)
    pool.statistics["gradient_returns"] += 1
    pool.statistics["remote_gradient_returns"] += int(active)
    pool.cache_token = pool.cache_generation = None
    return tuple(home)


class OfficialTransferBackend:
    """Select official transfers without changing other MoonEP executors globally."""

    prefetch = staticmethod(prefetch)
    return_gradients = staticmethod(return_gradients)
    return_gradients_as = return_gradients
