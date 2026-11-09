# SPDX-License-Identifier: Apache-2.0
"""TorchTurbo adapter for the external hyper-multicore MegaMoE package."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from torchturbo.optimizations.routed_experts import (
    RoutedExpertsBackend,
    check_shared_shmem_runtime,
)


class MegaMoeBackend(RoutedExpertsBackend):
    """Pass current model-owned weights to the upstream parameterless executor.

    Executors own only execution resources. Parameter identity, FSDP hooks,
    optimizer placement and checkpoint ownership stay with GroupedMLP.
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        token_bound: int,
        modes: tuple[tuple[str, float], ...],
        *,
        enable_moonep: bool = False,
        grad_reduce_dtype: str = "fp32",
    ) -> None:
        super().__init__(ep_group, token_bound)
        self.modes = modes
        self.enable_moonep = enable_moonep
        self.grad_reduce_dtype = grad_reduce_dtype
        self.executors: dict[tuple[str, float], Any] = {}

    def _executor(self, experts: Any) -> Any:
        check_shared_shmem_runtime(require_moonep=self.enable_moonep)
        try:
            from hyper_multicore import MegaMoeExperts
        except ModuleNotFoundError as error:
            if error.name == "hyper_multicore":
                raise ImportError(
                    "MegaMoE requires hyper-multicore; run scripts/runs/deps/mega_moe.sh."
                ) from error
            raise
        planner = None
        transfer_backend = None
        if self.enable_moonep:
            from torchturbo.optimizations.moonep.megamoe_provider import (
                OfficialTransferBackend,
                build_official_placement,
            )

            planner = build_official_placement
            transfer_backend = OfficialTransferBackend
        from torchturbo.optimizations import _expert_cleanup

        if not self.executors:
            config = experts.config
            # Register all supported layer modes before the first NPU call so
            # the upstream heap manager sees the full reservation manifest.
            for mode, limit in self.modes:
                self.executors[mode, limit] = MegaMoeExperts(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_ffn_hidden_size,
                    num_experts=config.num_moe_experts,
                    top_k=config.moe_router_topk,
                    swiglu_limit=limit or None,
                    ep_group=self.ep_group,
                    create_parameters=False,
                    apply_probs_before_fc2=True,
                    enable_hot_replica=self.enable_moonep,
                    moonep_planner=planner,
                    moonep_transfer_backend=transfer_backend,
                    moonep_grad_reduce_dtype=self.grad_reduce_dtype,
                    expert_weight_layout="EIO",
                    dispatch_mode="push",
                    **(
                        {"clamp_after_activation": True}
                        if mode == "post_activation"
                        else {}
                    ),
                )
            _expert_cleanup.register(self.close, self.ep_group)
        limit = float(experts.swiglu_limit or 0.0)
        mode = (
            (
                "post_activation"
                if experts.config.moe_swiglu_clamp_gate_after_activation
                else "pre_activation"
            )
            if limit
            else "none"
        )
        return self.executors[mode, limit]

    def execute(
        self,
        experts: Any,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Run upstream autograd with views of the current unsharded parameters."""
        self._check_token_capacity(indices.shape[0])
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "MegaMoE requires BF16 activations; configure mixed_precision: bf16."
            )
        if (
            experts.weight1.dtype != torch.bfloat16
            or experts.weight2.dtype != torch.bfloat16
        ):
            raise ValueError(
                "MegaMoE requires BF16 expert parameters at the FSDP forward boundary."
            )
        config = experts.config
        ranks = dist.get_world_size(self.ep_group)
        local = config.num_moe_experts // ranks
        weights = (
            experts.weight1.view(
                local, config.hidden_size, 2 * config.moe_ffn_hidden_size
            ),
            experts.weight2.view(local, config.moe_ffn_hidden_size, config.hidden_size),
        )
        flat = hidden_states.reshape(-1, config.hidden_size)
        ids = indices.reshape(-1, config.moe_router_topk).int()
        executor = self._executor(experts)
        if executor.training != experts.training:
            executor.train(experts.training)
        return executor(
            flat, ids, probs.reshape(-1, config.moe_router_topk), expert_weights=weights
        ).reshape_as(hidden_states)

    def runtime_statistics(self) -> tuple[dict[str, Any], ...]:
        """Read public upstream diagnostic snapshots without a device readback."""
        return tuple(
            executor.moonep_statistics() for executor in self.executors.values()
        )

    def close(self) -> None:
        """Release resources; upstream owns synchronization and live-graph checks."""
        for executor in self.executors.values():
            executor.close()
        self.executors.clear()
