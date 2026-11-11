"""FSDP2-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShardingStrategy(str, Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "full_shard"


class FSDPPrefetchStrategy(str, Enum):
    """MoE-aware FSDP2 prefetch topology (who prefetches whom).

    ``NATIVE`` keeps FSDP's implicit-only behavior. ``AUTO`` applies
    ``EXPERT_LOOKAHEAD`` to discovered MoE chains and otherwise leaves FSDP's
    native prefetching untouched.
    """

    NATIVE = "native"
    AUTO = "auto"
    EXPERT_WITH_LAYER = "expert_with_layer"
    EXPERT_LOOKAHEAD = "expert_lookahead"


class FSDPPrefetchDirection(str, Enum):
    """Which pass(es) a non-native prefetch strategy configures."""

    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"


@dataclass
class FSDPConfig:
    """FSDP2-specific configuration.

    Attributes
    ----------
    sharding_strategy:
        How to shard model parameters.
    reshard_after_forward:
        Whether to reshard params after forward (trade memory for speed).
    cpu_offload:
        Whether to apply CPU offload to all FSDP-wrapped units.
    pin_memory:
        Pin CPU offloaded memory for faster transfers.
    prefetch_strategy:
        MoE-aware explicit prefetch topology. ``NATIVE`` keeps FSDP's
        implicit-only behavior.
    prefetch_direction:
        Which pass(es) the prefetch strategy configures. Only takes effect when
        ``prefetch_strategy`` is not ``NATIVE``.
    forward_prefetch_depth:
        Rolling prefetch horizon in the forward pass. ``1`` (default)
        prefetches only the next layer; larger values keep more future layers
        in flight without re-scheduling a layer that an earlier source already
        covered, trading peak memory for more overlap.
    backward_prefetch_depth:
        Same as ``forward_prefetch_depth`` but for the backward pass. Only takes
        effect when ``prefetch_direction`` is ``FORWARD_BACKWARD``.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    reshard_after_forward: bool = True
    cpu_offload: bool = False
    pin_memory: bool = True
    prefetch_strategy: FSDPPrefetchStrategy = FSDPPrefetchStrategy.AUTO
    prefetch_direction: FSDPPrefetchDirection = FSDPPrefetchDirection.FORWARD_BACKWARD
    forward_prefetch_depth: int = 1
    backward_prefetch_depth: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.sharding_strategy, str):
            self.sharding_strategy = ShardingStrategy(self.sharding_strategy)
        if isinstance(self.prefetch_strategy, str):
            self.prefetch_strategy = FSDPPrefetchStrategy(self.prefetch_strategy)
        if isinstance(self.prefetch_direction, str):
            self.prefetch_direction = FSDPPrefetchDirection(self.prefetch_direction)

        self.forward_prefetch_depth = int(self.forward_prefetch_depth)
        self.backward_prefetch_depth = int(self.backward_prefetch_depth)
        if self.forward_prefetch_depth < 1 or self.backward_prefetch_depth < 1:
            raise ValueError(
                "prefetch depth must be >= 1, got forward=%d backward=%d"
                % (self.forward_prefetch_depth, self.backward_prefetch_depth)
            )
        if (
            self.prefetch_direction == FSDPPrefetchDirection.FORWARD
            and self.backward_prefetch_depth != 1
        ):
            raise ValueError(
                "backward_prefetch_depth has no effect when "
                "prefetch_direction=forward; got backward_prefetch_depth=%d"
                % self.backward_prefetch_depth
            )
