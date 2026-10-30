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
"""Allocate MegaMoE task graphs and their event storage."""
from torchturbo.optimizations.mega_moe.runtime.scheduler.config import (
    RuntimeConfigC,
    TaskSplitValue,
    mega_moe_event_capacity,
)
from torchturbo.optimizations.mega_moe.runtime.scheduler.graph import ComputeGraph
from torchturbo.optimizations.mega_moe.runtime.scheduler.runtime import allocate_runtime_config


def allocate_graph_config(graph: ComputeGraph, tsv: TaskSplitValue) -> RuntimeConfigC:
    """Allocate graph tasks plus the terminal before any fill operation.

    Args:
        graph: Graph after split propagation.
        tsv: Topology whose experts determine event storage.

    Returns:
        Zero-initialized runtime with sufficient task, queue and event slots.
    """
    required = 1 + sum(op.task_num for op in graph.topological_sort())
    return allocate_runtime_config(
        required, mega_moe_event_capacity(tsv.all_expert_num, tsv.ep), tsv.single_rank_expert_num
    )
