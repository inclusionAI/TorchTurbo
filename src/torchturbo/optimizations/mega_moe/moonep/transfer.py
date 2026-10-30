"""Device expert transfers using the active MegaMoE symmetric heap."""

import torch
import torch.distributed as dist
from torchturbo.optimizations.mega_moe.runtime.moonep import load_bridge

from .plans import SharedMegaMoePlanPool


class SharedExpertTransfer:
    """Lease process-shared weight slots for each forward or backward call."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        local_experts: int,
        hidden: int,
        intermediate: int,
        *,
        reuse_weight_slots: bool = False,
    ) -> None:
        from torchturbo.optimizations.mega_moe.runtime import shmem

        self.group = group
        self.local_experts = local_experts
        self.world_rank = dist.get_rank()
        self.rank = dist.get_rank(group)
        self.root_size = dist.get_world_size(group)
        self.in_use = False
        self.cache_token = None
        self.generation = 0
        self.cache_generation = None
        self.reuse_weight_slots = reuse_weight_slots
        self.execution_pools: list[SharedMegaMoePlanPool] = []
        self.statistics = {
            "prefetches": 0,
            "gradient_returns": 0,
            "host_plan_readbacks": 0,
            "backward_prefetches": 0,
            "backward_cache_hits": 0,
        }
        shmem.acquire(root_group=group)
        bridge = load_bridge()
        assert bridge.runtime_state() == [self.rank, self.root_size], (
            "expert kernels must share the active MegaMoE EP root"
        )
        print(
            "MOONEP_SHARED_RUNTIME", self.world_rank, bridge.runtime_state(), flush=True
        )
        self.weights = (
            shmem.empty(
                (2 * local_experts, hidden, 2 * intermediate), dtype=torch.bfloat16
            ),
            shmem.empty(
                (2 * local_experts, intermediate, hidden), dtype=torch.bfloat16
            ),
        )
        # Backward consumes all borrowed weights before staging their gradients.
        # The transfer kernel barriers keep the slots live through all peer reads.
        self.slots = (
            tuple(weight[local_experts:] for weight in self.weights)
            if reuse_weight_slots
            else tuple(
                shmem.empty((local_experts, *weight.shape[1:]), dtype=torch.bfloat16)
                for weight in self.weights
            )
        )

    def claim(self) -> None:
        if self.in_use:
            raise RuntimeError(
                "Shared MoonEP weight slots do not support concurrent execution"
            )
        self.in_use = True

    def prefetch(
        self, weights: tuple[torch.Tensor, ...], copies: torch.Tensor, extra: int,
        *, token: object | None = None,
    ) -> tuple[torch.Tensor, ...]:
        bridge = load_bridge()
        assert self.in_use
        self.cache_token = None
        self.cache_generation = None
        for pool, weight in zip(self.weights, weights):
            pool[: self.local_experts].copy_(
                weight.reshape_as(pool[: self.local_experts])
            )
        for pool in self.weights:
            bridge.prefetch(
                pool,
                pool[self.local_experts :],
                copies[self.rank],
                self.rank,
                self.root_size,
            )
        self.statistics["prefetches"] += 1
        self.cache_token = token
        self.generation += 1
        self.cache_generation = self.generation
        # Every call gets distinct autograd metadata while sharing the leased storage.
        return tuple(
            weight[: self.local_experts + extra].detach() for weight in self.weights
        )

    def return_generated_gradients(
        self, gradients: tuple[torch.Tensor, ...], copies: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Accumulate into gradients owned by the native backward producer."""
        bridge = load_bridge()
        home = []
        for gradient, slots in zip(gradients, self.slots):
            owner = gradient[: self.local_experts]
            extra = gradient.shape[0] - self.local_experts
            slots[:extra].copy_(gradient[self.local_experts :])
            bridge.reduce(owner, owner, slots, copies, self.rank)
            home.append(owner)
        self.statistics["gradient_returns"] += 1
        self.cache_token = None
        self.cache_generation = None
        self.in_use = False
        return tuple(home)

    def close(self) -> None:
        from torchturbo.optimizations.mega_moe.runtime import shmem

        assert not self.in_use
        torch.npu.synchronize()
        dist.barrier(group=self.group)
        for pool in self.execution_pools:
            pool.close()
        allocations = (
            self.weights if self.reuse_weight_slots else (*self.weights, *self.slots)
        )
        for value in allocations:
            shmem.free(value)
        shmem.release()
