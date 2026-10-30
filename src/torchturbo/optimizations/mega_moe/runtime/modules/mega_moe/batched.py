"""Collapse batched GMM stages while preserving their event dependencies."""
from collections import Counter
from typing import Any
from torchturbo.optimizations.mega_moe.runtime.scheduler.config import TaskType


def collapse_batched_tasks(config: Any, num_cube_cores: int) -> None:
    """Keep one GMM per stage and account for every remaining producer."""
    ids = list(config.cube_task_indices[:config.task_index_num[0]])
    vectors = list(config.vector_task_indices[:config.task_index_num[1]])
    stages, mapping, retained = {}, {}, []
    removed = Counter()
    for task_id in ids:
        task = config.all_tasks[task_id]
        assert task.task_type == TaskType.TASK_GROUPED_MATMUL
        stage = task.tiling_data_position
        if task.task_index < num_cube_cores:
            retained.append(task_id)
            stages.setdefault(stage, task.trigger_event)
        else:
            removed[task.trigger_event] += 1
    for task_id in ids:
        task = config.all_tasks[task_id]
        mapping[task.trigger_event] = stages[task.tiling_data_position]
    # Weight-gradient stages share the final event with Vector completion.
    # Subtract only removed producers, preserving all remaining producers.
    for event, count in removed.items():
        assert config.all_event_num_triggers[event] >= count
        config.all_event_num_triggers[event] -= count
    for task_id in retained + vectors:
        task = config.all_tasks[task_id]
        if task.dependent_event in mapping:
            task.dependent_event = mapping[task.dependent_event]
            if task.task_type == TaskType.TASK_GROUPED_MATMUL:
                task.extra_value_1 = 0x42415443
    config.task_index_num[0] = len(retained)
    for index, task_id in enumerate(retained):
        config.cube_task_indices[index] = task_id
    for task_id in retained + vectors:
        task = config.all_tasks[task_id]
        if task.dependent_event:
            assert config.all_event_num_triggers[task.dependent_event] > 0
