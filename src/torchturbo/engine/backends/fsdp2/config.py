"""FSDP2-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


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


class CPUOffloadMode(str, Enum):
    """Execution modes selected by :class:`CPUOffloadConfig`.

    ``FULL`` offloads parameters, gradients, and optimizer state.
    ``OPTIMIZER`` keeps parameters and gradients on the device while optimizer state stays on CPU.
    """

    FULL = "full"
    OPTIMIZER = "optimizer"


@dataclass
class CPUOffloadConfig:
    """FSDP2 CPU offload policy used during training.

    Optimizer steps and checkpoints remain available with this policy enabled.
    Temporary suspension via engine.offload_optimizer() is controlled separately.

    ``enabled`` controls CPU offload. When enabled, ``mode`` selects either
    full offload or optimizer-state-only offload. ``optimizer_memory_fraction``
    bounds optimizer workspaces: AdamWSwap sizes its staging slab from that
    fraction of free device memory, while Muon checks its complete-matrix
    compute and exchange buffers against that fraction of free device memory
    plus blocks cached by the device allocator. Muon stages offloaded tensors
    with blocking copies, so ``overlap_comm_compute`` gives little overlap
    there. Zero selects synchronous CPU updates, which for Muon includes CPU
    collectives in either mode. Full-mode native AdamW retains its existing
    execution policy based on each parameter's device.
    """

    enabled: Optional[bool] = None
    mode: CPUOffloadMode = CPUOffloadMode.FULL
    optimizer_memory_fraction: float = 0.8

    def __post_init__(self) -> None:
        if self.enabled is None:
            return
        if isinstance(self.enabled, str):
            # Lazy import: engine.args imports this module.
            from ...args import string_to_bool

            self.enabled = string_to_bool(self.enabled)
        else:
            self.enabled = bool(self.enabled)
        if not self.enabled:
            self.mode = CPUOffloadMode.FULL
            self.optimizer_memory_fraction = 0.8
            return
        self.mode = CPUOffloadMode(self.mode)
        self.optimizer_memory_fraction = float(self.optimizer_memory_fraction)
        if not 0 <= self.optimizer_memory_fraction <= 1:
            raise ValueError(
                "optimizer_memory_fraction must be in [0, 1], got "
                f"{self.optimizer_memory_fraction}"
            )


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
        Structured CPU-offload policy (see :class:`CPUOffloadConfig`). The
        master ``enabled`` switch and ``mode`` select no offload, full FSDP
        offload, or optimizer-state-only offload.
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
    cpu_offload: CPUOffloadConfig = field(
        default_factory=lambda: CPUOffloadConfig(enabled=False)
    )
    prefetch_strategy: FSDPPrefetchStrategy = FSDPPrefetchStrategy.AUTO
    prefetch_direction: FSDPPrefetchDirection = FSDPPrefetchDirection.FORWARD_BACKWARD
    forward_prefetch_depth: int = 1
    backward_prefetch_depth: int = 1

    def __post_init__(self) -> None:
        cpu_offload = self.cpu_offload
        if cpu_offload is None:
            self.cpu_offload = CPUOffloadConfig(enabled=False)
        elif isinstance(cpu_offload, CPUOffloadConfig):
            self.cpu_offload = cpu_offload
        elif isinstance(cpu_offload, bool):
            raise ValueError(
                "cpu_offload must be an object with 'enabled' (e.g. "
                '{"enabled":true, "mode":"optimizer"}); bare true/false is not accepted'
            )
        elif isinstance(cpu_offload, dict):
            if "enabled" not in cpu_offload:
                raise ValueError("cpu_offload requires 'enabled'")
            self.cpu_offload = CPUOffloadConfig(**cpu_offload)
        else:
            try:
                from omegaconf import DictConfig, OmegaConf
            except ImportError as exc:  # pragma: no cover — omegaconf is a core dep
                raise TypeError(
                    f"Cannot interpret {cpu_offload!r} as CPUOffloadConfig"
                ) from exc
            if not isinstance(cpu_offload, DictConfig):
                raise TypeError(
                    f"Cannot interpret {cpu_offload!r} as CPUOffloadConfig"
                )
            config = OmegaConf.to_container(cpu_offload, resolve=True)
            if "enabled" not in config:
                raise ValueError("cpu_offload requires 'enabled'")
            self.cpu_offload = CPUOffloadConfig(**config)
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
