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
"""Build the static graph and tiling plan for managed MegaMoe execution."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.backward.graph import (
    build_backward_graph,
)
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from torchturbo.optimizations.mega_moe.runtime.scheduler.runtime import serialize_runtime_config
from torchturbo.optimizations.mega_moe.runtime.scheduler.config import TaskSplitValue

from .spec import MegaMoeSpec
from .batched import collapse_batched_tasks
from .compact import compact_graph, pipeline_vector_queue, pipeline_cube_queue


@dataclass(frozen=True)
class MegaMoePlan:
    """Rank-local schedules and tiling tensors for forward and backward."""

    spec: MegaMoeSpec
    fwd_runtime: torch.Tensor
    up_proj_tiling: Any
    swiglu_tiling: Any
    down_proj_tiling: Any
    bwd_runtime: torch.Tensor
    act_grad_tiling: Any
    gate_grad_tiling: Any
    w1_grad_tiling: Any
    w2_grad_tiling: Any
    swiglu_grad_tiling: Any
    reuse_backward_dispatch: bool = False

    @property
    def fwd_runtime_config(self) -> Any:
        """Return the disabled forward RuntimeConfig for compatibility."""
        return self.fwd_runtime

    @property
    def bwd_runtime_config(self) -> Any:
        """Return the disabled backward RuntimeConfig for compatibility."""
        return self.bwd_runtime


def _tensor_from_bytes(data: bytes, device: Any) -> torch.Tensor:
    """Copy serialized runtime data into a device uint8 tensor."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def _resize_swiglu_tiling(data: bytes, num_cube_cores: int) -> bytes:
    """Resize the baseline 24-core repeated struct for the active workers."""
    baseline_workers = 49
    if len(data) % baseline_workers:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    struct_size = len(data) // baseline_workers
    return data[:struct_size] * (2 * num_cube_cores + 1)


def _build_task_values(spec: MegaMoeSpec) -> TaskSplitValue:
    """Translate rank-local token shape into legacy scheduler coordinates.

    Args:
        spec: Validated local-token and expert topology specification.

    Returns:
        Scheduler values whose single partition is the rank-local input.
    """
    return TaskSplitValue(
        tp=1,
        ep=spec.ep_size,
        seq_size=math.ceil(spec.local_num_tokens / math.lcm(spec.dispatch_split, spec.combine_split))
        * math.lcm(spec.dispatch_split, spec.combine_split),
        all_expert_num=spec.num_experts,
        top_k=spec.top_k,
        dispatch_mode=spec.dispatch_mode,
    )


def _build_runtime_artifacts(spec: MegaMoeSpec) -> tuple[Any, Any, Any, Any]:
    """Build forward/backward graphs and their serialized RuntimeConfig objects."""
    task_values = _build_task_values(spec)
    forward_graph = build_forward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        swiglu_mode=spec.swiglu_mode,
    )
    compact = spec.dispatch_mode == "push"
    if compact:
        compact = compact_graph(forward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule)
    forward_graph.propagate_splits(task_values)
    forward_data = build_forward_config(
        forward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    backward_graph = build_backward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        swiglu_mode=spec.swiglu_mode,
    )
    if compact:
        compact_graph(backward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule)
    backward_graph.propagate_splits(task_values)
    backward_data = build_backward_config(
        backward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    if spec.batch_experts:
        # Keep complete stages ordered before consumers that wait for every expert.
        for config in (forward_data, backward_data):
            config._padding[0] = (3 if compact else 0) | 8
            for task_id in list(config.cube_task_indices[:config.task_index_num[0]]) + list(config.vector_task_indices[:config.task_index_num[1]]):
                config.all_tasks[task_id].extra_value_2 = 5
            collapse_batched_tasks(config, spec.num_cube_cores)
    elif compact:
        forward_data._padding[0] = 7
        backward_data._padding[0] = 7
        pipeline_vector_queue(forward_data, forward_graph, task_values)
        pipeline_vector_queue(backward_data, backward_graph, task_values)
        pipeline_cube_queue(forward_data, forward_graph, task_values, spec.num_cube_cores)
        pipeline_cube_queue(backward_data, backward_graph, task_values, spec.num_cube_cores)
    return forward_graph, forward_data, backward_graph, backward_data


def _prepare_runtimes(
    spec: MegaMoeSpec, forward_config: Any, backward_config: Any, device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy forward and backward runtime images to the execution device."""
    return tuple(_tensor_from_bytes(serialize_runtime_config(config), device)
                 for config in (forward_config, backward_config))


def build_mega_moe_plan(spec: MegaMoeSpec, device: Any) -> MegaMoePlan:
    """Build trimmed dense forward/backward runtime images without fusion slots.

    Args:
        spec: Validated local-token and expert topology specification.
        device: NPU device receiving serialized descriptors and tiling tensors.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(spec)
    fwd_runtime, bwd_runtime = _prepare_runtimes(spec, forward_config, backward_config, device)
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": 1 if forward_config._padding[0] & 4 else spec.local_experts,
        "num_cube_cores": spec.num_cube_cores,
    }
    return MegaMoePlan(
        spec=spec,
        reuse_backward_dispatch=can_reuse_backward_dispatch(
            backward_config, spec.local_experts, spec.num_cube_cores,
        ),
        fwd_runtime=fwd_runtime,
        up_proj_tiling=_tensor_from_bytes(
            get_up_proj_tiling_bytes(forward_graph.get_op("up_proj").split_value, **gmm_options),
            device,
        ),
        swiglu_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_tiling_bytes(
                    forward_graph.get_op("swiglu").split_value,
                    intermediate_size=spec.intermediate_size,
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
        down_proj_tiling=_tensor_from_bytes(
            get_down_proj_tiling_bytes(forward_graph.get_op("down_proj").split_value, **gmm_options),
            device,
        ),
        bwd_runtime=bwd_runtime,
        act_grad_tiling=_tensor_from_bytes(
            get_act_grad_tiling_bytes(backward_graph.get_op("act_grad").split_value, **gmm_options),
            device,
        ),
        gate_grad_tiling=_tensor_from_bytes(
            get_gate_grad_tiling_bytes(backward_graph.get_op("gate_grad").split_value, **gmm_options),
            device,
        ),
        w1_grad_tiling=_tensor_from_bytes(
            get_w1_grad_tiling_bytes(backward_graph.get_op("w1_grad").split_value, **gmm_options),
            device,
        ),
        w2_grad_tiling=_tensor_from_bytes(
            get_w2_grad_tiling_bytes(backward_graph.get_op("w2_grad").split_value, **gmm_options),
            device,
        ),
        swiglu_grad_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_grad_tiling_bytes(
                    backward_graph.get_op("swiglu_grad").split_value,
                    intermediate_size=spec.intermediate_size,
                    clipped=spec.swiglu_limit is not None,
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
    )
