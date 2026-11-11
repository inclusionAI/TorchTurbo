"""FSDP-specific parallel plan and module specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .parallel import ParallelPlan


def check_fqn_match(pattern: str, fqn: str) -> bool:
    """Check if a fully-qualified name matches a pattern.

    ``*`` matches a single dot-separated segment (cannot cross ``.``).
    ``**`` matches zero or more segments (crosses ``.`` boundaries).
    """
    regex_str = re.escape(pattern)
    # Handle ** first (escaped as \*\*), then remaining single *
    regex_str = regex_str.replace(r"\*\*", r".*")
    regex_str = regex_str.replace(r"\*", r"[^.]+")
    return bool(re.match(f"^{regex_str}$", fqn))


@dataclass
class FSDPPlacementSpec:
    """A placement definition optionally tied to a specific mesh dimension for FSDP.

    Attributes
    ----------
    placement:
        The DTensor placement (e.g., ``Shard(0)``, ``Replicate()``).
    mesh_dim:
        Name or index of the sharding mesh dimension (e.g., ``"ep"``).
        If ``None``, the backend decides the default dimension.
    """

    placement: Any
    mesh_dim: str | int | None = None


@dataclass
class FSDPPlanSpec:
    """Unified logical settings for an FSDP module or its layers.

    Attributes
    ----------
    placement:
        The parameter/tensor placement definition.
    no_shard:
        Whether to skip FSDP sharding for this module completely.
    standalone_world_shard:
        Whether to shard this module as a standalone FSDP unit on the world mesh.
    activation_checkpoint:
        Whether to apply whole-module activation checkpointing to this module.
    replace_module:
        A callable to replace the matched module. Signature: `(original_module, **kwargs) -> nn.Module`
    module_kwargs:
        Keyword arguments to pass to the `replace_module` callable.
    """

    placement: FSDPPlacementSpec | None = None
    no_shard: bool = False
    standalone_world_shard: bool = False
    activation_checkpoint: bool = False
    replace_module: Any | None = None
    module_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.no_shard and (
            self.standalone_world_shard or self.placement is not None
        ):
            raise ValueError(
                "FSDPPlanSpec.no_shard is mutually exclusive with "
                "standalone_world_shard and placement."
            )
        if self.standalone_world_shard and self.placement is not None:
            raise ValueError(
                "FSDPPlanSpec.standalone_world_shard is redundant with placement; "
                "use placement for custom mesh sharding."
            )


@dataclass
class FSDPParallelPlan(ParallelPlan):
    """FSDP-specific parallel plan spanning topologies and placements.

    Attributes
    ----------
    plan:
        FQN to FSDPPlanSpec mapping dictionary.
    sp_size:
        Sequence Parallelism group size.
    ep_size:
        Expert Parallelism group size.
    tp_size:
        Tensor Parallelism group size.
    """

    plan: dict[str, FSDPPlacementSpec | FSDPPlanSpec | Any] = field(
        default_factory=dict
    )
    sp_size: int = 1
    ep_size: int = 1
    tp_size: int = 1

    def __post_init__(self) -> None:
        """Normalize plan values to FSDPPlanSpec/FSDPPlacementSpec."""
        for k, v in list(self.plan.items()):
            if isinstance(v, tuple) and len(v) == 2:
                self.plan[k] = FSDPPlanSpec(
                    placement=FSDPPlacementSpec(placement=v[0], mesh_dim=v[1])
                )
            elif isinstance(v, FSDPPlacementSpec):
                self.plan[k] = FSDPPlanSpec(placement=v)
            elif not isinstance(v, FSDPPlanSpec):
                self.plan[k] = FSDPPlanSpec(placement=FSDPPlacementSpec(placement=v))

    def match(self, fqn: str) -> Any | None:
        """Find the specification for a given FQN. Returns None if no match."""
        for pattern, spec in self.plan.items():
            if check_fqn_match(pattern, fqn):
                return spec
        return None
