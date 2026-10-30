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
"""Prepare trusted Router output for managed Torch MegaMoe execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch_npu

from torchturbo.optimizations.mega_moe.runtime.torch import ops as multicore_ops

from .spec import MegaMoeSpec

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace


@dataclass(frozen=True)
class RouteMetadata:
    """Element offsets and lengths, plus expert boundaries in token rows."""

    dispatch_src_off: torch.Tensor
    dispatch_target_off: torch.Tensor
    dispatch_size: torch.Tensor
    combine_src_off: torch.Tensor
    combine_target_off: torch.Tensor
    combine_size: torch.Tensor
    group_list: torch.Tensor
    expert_capacity: int


@dataclass(frozen=True)
class PreparedTopKRoute:
    """Permuted payload and exact metadata for one Router decision."""

    routed_tokens: torch.Tensor
    unpermute_mapping: torch.Tensor
    tokens_per_expert: torch.Tensor
    received_counts: torch.Tensor
    metadata: RouteMetadata
    maximum_received_slots: int = 0
    nominal_schedule: bool = False














def _permute_topk_input_out(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the inverse mapping owned while writing rows into leased SHMEM."""
    mapping = torch.empty(topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    multicore_ops.moe_token_permute_out(hidden_states.contiguous(), topk_ids.contiguous(), output, mapping)
    return output, mapping






def restore_topk_output(
    expert_output: torch.Tensor,
    unpermute_mapping: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Restore token order and apply Router weights with the native NPU op."""
    return torch_npu.npu_moe_token_unpermute(
        expert_output,
        unpermute_mapping,
        probs=topk_weights.float().contiguous(),
    )
