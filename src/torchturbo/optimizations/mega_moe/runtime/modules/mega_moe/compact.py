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
"""Bound descriptor count by hardware parallelism, not route capacity."""
import math
from collections import defaultdict

from torchturbo.optimizations.mega_moe.runtime.scheduler.config import RuntimeConfigC, TaskSplitValue
from torchturbo.optimizations.mega_moe.runtime.scheduler.graph import ComputeGraph, OpType

COMPACT_MARKER = 0x434F4D50


def compact_graph(graph: ComputeGraph, topology: TaskSplitValue, workers: int, nominal: bool = False) -> bool:
    """Partition actual device-side rows across bounded persistent lanes."""
    operators = graph.topological_sort()
    original = 0
    compact = 0
    for op in operators:
        if op.op_type == OpType.ALLTOALL:
            original += topology.all_expert_num * (topology.per_expert_seq_to_other // op.split_value)
            compact += topology.all_expert_num * min(max(1, math.ceil(workers / topology.ep)),
                                                    topology.per_expert_seq_to_other // op.split_value)
        elif op.op_type in (OpType.SWIGLU, OpType.SWIGLU_GRAD):
            original += topology.single_rank_expert_num * (topology.per_expert_seq // op.split_value)
            compact += topology.single_rank_expert_num * min(workers, topology.per_expert_seq // op.split_value)
    # Persistent partitions add loop and pipeline overhead; small graphs retain the dense path.
    if not nominal and original < 4 * compact:
        return False
    for op in operators:
        if op.op_type == OpType.ALLTOALL:
            lanes = min(math.ceil(workers / topology.ep), topology.per_expert_seq_to_other // op.split_value)
            groups = topology.all_expert_num
        elif op.op_type in (OpType.SWIGLU, OpType.SWIGLU_GRAD):
            lanes = min(workers, topology.per_expert_seq // op.split_value)
            groups = topology.single_rank_expert_num
        else:
            continue
        op.compact_lanes = max(1, lanes)
        if nominal:
            nominal_tiles = math.ceil(topology.per_rank_seq / (groups * op.split_value))
            op.compact_lanes = min(op.compact_lanes, max(1, nominal_tiles))
        op.split_spec.task_num_fn = lambda unused, n=groups * op.compact_lanes: n
    return True


def pipeline_vector_queue(config: RuntimeConfigC, graph: ComputeGraph, topology: TaskSplitValue) -> None:
    """Pipeline dispatch, activation and return with two experts in flight."""
    dispatch = graph.get_op("dispatch")
    activation = graph.get_op("swiglu_grad" if any(op.name == "swiglu_grad" for op in graph.topological_sort())
                              else "swiglu")
    combine = graph.get_op("combine")
    ids = list(config.vector_task_indices[:config.task_index_num[1]])
    experts = topology.single_rank_expert_num
    dispatch_ids = ids[:dispatch.task_num]
    begin = dispatch.task_num
    act_ids = ids[begin:begin + activation.task_num]
    com_ids = ids[begin + activation.task_num:begin + activation.task_num + combine.task_num]
    dis_groups = [[] for _ in range(experts)]
    act_groups = [[] for _ in range(experts)]
    com_groups = [[] for _ in range(experts)]
    for task_id in dispatch_ids:
        task = config.all_tasks[task_id]
        owner = (task.task_index // dispatch.compact_lanes) % experts
        dis_groups[owner].append(task_id)
    for task_id in act_ids:
        task = config.all_tasks[task_id]
        act_groups[task.task_index // activation.compact_lanes].append(task_id)
    for task_id in com_ids:
        task = config.all_tasks[task_id]
        owner = (task.task_index // combine.compact_lanes) % experts
        com_groups[owner].append(task_id)
    reordered = []
    for expert in range(experts):
        reordered.extend(dis_groups[expert])
        if expert >= 1:
            reordered.extend(act_groups[expert - 1])
        if expert >= 2:
            reordered.extend(com_groups[expert - 2])
    reordered.extend(act_groups[-1])
    if experts >= 2:
        reordered.extend(com_groups[-2])
    reordered.extend(com_groups[-1])
    reordered.extend(ids[begin + activation.task_num + combine.task_num:])
    if sorted(reordered) != sorted(ids):
        raise RuntimeError("DAG pipeline lost or duplicated task descriptors")
    for index, task_id in enumerate(reordered):
        config.vector_task_indices[index] = task_id


def pipeline_cube_queue(
    config: RuntimeConfigC, graph: ComputeGraph, topology: TaskSplitValue, workers: int,
) -> None:
    """Keep producer/consumer projections near each other for bounded expert lookahead."""
    ids = list(config.cube_task_indices[:config.task_index_num[0]])
    stages = defaultdict(lambda: defaultdict(list))
    for task_id in ids:
        task = config.all_tasks[task_id]
        stages[task.outputs[0].input_position][task.task_index // workers].append(task_id)
    backward = any(op.name == "swiglu_grad" for op in graph.topological_sort())
    first_ops = ("w2_grad", "act_grad") if backward else ("up_proj",)
    second_ops = ("gate_grad", "w1_grad") if backward else ("down_proj",)
    first = tuple(graph.get_op(name).param_positions[-1] for name in first_ops)
    second = tuple(graph.get_op(name).param_positions[-1] for name in second_ops)
    reordered = []
    for expert in range(topology.single_rank_expert_num):
        for stage in first:
            reordered.extend(stages[stage][expert])
        if expert:
            for stage in second:
                reordered.extend(stages[stage][expert-1])
    for stage in second:
        reordered.extend(stages[stage][topology.single_rank_expert_num-1])
    if sorted(reordered) != sorted(ids):
        raise RuntimeError("Expert Cube pipeline lost or duplicated tasks")
    for index, task_id in enumerate(reordered):
        config.cube_task_indices[index] = task_id
    # With few persistent partitions the legacy polling delay dominates short producer stages.
    for task_id in ids + list(config.vector_task_indices[:config.task_index_num[1]]):
        config.all_tasks[task_id].extra_value_2 = 5
