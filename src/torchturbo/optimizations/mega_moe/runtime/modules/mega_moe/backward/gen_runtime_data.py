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
"""Construct the backward device task graph runtime."""

from torchturbo.optimizations.mega_moe.runtime.scheduler.builder import allocate_graph_config
from torchturbo.optimizations.mega_moe.runtime.scheduler.config import RuntimeConfigC, TaskSplitValue, configure_ready_handshake, init_task_split_value, validate_runtime_config
from torchturbo.optimizations.mega_moe.runtime.scheduler.graph import ComputeGraph
from torchturbo.optimizations.mega_moe.runtime.scheduler.scheduler import revise_gmm_task_queue_bwd, revise_task_queue
from torchturbo.optimizations.mega_moe.runtime.tasks.utils import add_dynamic_data, add_terminate


def build_config_for_rank(graph: ComputeGraph, tsv: TaskSplitValue, rank_id: int,
                          num_cube_cores: int = 24) -> RuntimeConfigC:
    """Build backward RuntimeConfig for a single rank.

    Args:
        graph: Propagated backward compute graph.
        tsv: Task split values for the target topology.
        rank_id: Rank-local identifier.
        num_cube_cores: Available cube-core count.

    Returns:
        Validated runtime image including transport handshakes.
    """
    cfg = allocate_graph_config(graph, tsv)
    cfg.num_workers    = 2 * num_cube_cores   # NUM_WORKERS_VECTOR = 2 × NUM_WORKERS_CUBE

    # Reset counters
    init_task_split_value(tsv)
    tsv.rank_id = rank_id

    # Fill tasks in topological order (dispatch, act_grad, w2_grad, swiglu_grad, gate_grad, combine, w1_grad)
    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, tsv)

    # task_num_all = sum of all op tasks + 1 (terminate)
    task_num_all = sum(op.task_num for op in graph.topological_sort()) + 1

    dispatch_op    = graph.get_op("dispatch")
    swiglu_grad_op = graph.get_op("swiglu_grad")
    act_grad_op    = graph.get_op("act_grad")
    w1_grad_op     = graph.get_op("w1_grad")
    w2_grad_op     = graph.get_op("w2_grad")
    combine_op     = graph.get_op("combine")

    add_terminate(cfg, tsv,
                  w2_grad_op.task_num + w1_grad_op.task_num
                  + combine_op.task_num // tsv.ep * tsv.ep)
    revise_task_queue(cfg, tsv, dispatch_op.task_num, swiglu_grad_op.task_num, combine_op.task_num)
    revise_gmm_task_queue_bwd(cfg, tsv, act_grad_op.task_num, num_cube_cores=num_cube_cores)
    add_dynamic_data(cfg, tsv, dynamic_input_position=19)

    cfg.task_num = task_num_all
    cfg.atomic_add_values[0] = 1
    configure_ready_handshake(cfg, tsv)
    validate_runtime_config(cfg, tsv, num_cube_cores)
    return cfg
