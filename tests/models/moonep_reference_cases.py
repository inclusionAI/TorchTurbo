"""Hand-derived alignment-1 cases for the Megatron MoonEP planning contract.

Source: NVIDIA/Megatron-LM commit 928ce7ce256b734f00f1c81ddd3a8958d6b61ebf,
megatron/core/transformer/moe/ops/virtual_expert_triton.py, placement/remapping.
These are manually checked expectations, not captured CUDA execution results.
Each rank supplies the same number of routes, as required by that implementation.
"""

MEGATRON_PLANNER_CASES = [
    {
        "name": "hot-owner",
        "num_local_experts": 2,
        "topk": 2,
        "top_indices": [
            [[0, 1], [1, 0]],
            [[0, 1], [1, 0]],
        ],
        "copies": [[-1, -1], [0, -1]],
        "runtime_ids": [
            [[6, 1], [1, 6]],
            [[6, 1], [1, 6]],
        ],
    },
    {
        "name": "replica-slot-ties",
        "num_local_experts": 4,
        "topk": 2,
        "top_indices": [
            [[0, 1], [2, 3], [0, 1], [2, 3]],
            [[0, 1], [2, 3], [0, 1], [2, 3]],
        ],
        "copies": [[-1, -1, -1, -1], [1, 0, -1, -1]],
        "runtime_ids": [
            [[13, 12], [2, 3], [13, 12], [2, 3]],
            [[13, 12], [2, 3], [13, 12], [2, 3]],
        ],
    },
    {
        "name": "cross-source-occurrences",
        "num_local_experts": 2,
        "topk": 2,
        "top_indices": [
            [[0, 2], [0, 1]],
            [[0, 1], [0, 1]],
        ],
        "copies": [[-1, -1], [0, -1]],
        "runtime_ids": [
            [[0, 4], [6, 1]],
            [[6, 1], [6, 1]],
        ],
    },
    {
        "name": "whole-receiver-deficit",
        "num_local_experts": 2,
        "topk": 1,
        "top_indices": [
            [[0], [1], [2], [3], [4], [5]],
            [[0], [1], [2], [3], [4], [5]],
            [[0], [1], [2], [3], [6], [2]],
            [[0], [1], [0], [2], [3], [0]],
        ],
        "copies": [[3, -1], [-1, -1], [2, -1], [0, -1]],
        "runtime_ids": [
            [[0], [1], [4], [2], [8], [9]],
            [[14], [1], [4], [5], [8], [9]],
            [[14], [1], [4], [5], [12], [10]],
            [[14], [1], [14], [10], [5], [14]],
        ],
    },
    {
        "name": "empty",
        "num_local_experts": 2,
        "topk": 2,
        "top_indices": [[], []],
        "copies": [[-1, -1], [-1, -1]],
        "runtime_ids": [[], []],
    },
]
