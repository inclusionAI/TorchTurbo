"""FSDP-specific parallel plan and module specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..engine.utils import match_fqn
from .parallel import ParallelPlan

if TYPE_CHECKING:
    from ..models.modules.vision.io import BaseVisionBalanceIO


class VisionBalanceMode(str, Enum):
    """Weight used to distribute whole media across vision encoder ranks."""

    PATCH = "patch"
    PATCH_SQ = "patch_sq"


@dataclass
class VisionBalanceSpec:
    """Configure media balancing and the model's input/output protocol.

    Attributes:
        mode: Media balancing weight, or None to leave balancing disabled in
            the resolved plan. None inherits the model default in an override.
        io: Model-provided input/output adapter. None inherits the model
            adapter in a partial plan override.
    """

    mode: VisionBalanceMode | None = None
    io: BaseVisionBalanceIO | None = None

    def __post_init__(self) -> None:
        if self.mode is not None and not isinstance(self.mode, VisionBalanceMode):
            raise TypeError("VisionBalanceSpec.mode must be a VisionBalanceMode.")


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
    vision_balance:
        Media balancing mode and the model's input/output adapter. Hooks are
        installed only when the resolved spec has a mode. Non-None fields in
        a partial spec override the corresponding model defaults.
    """

    placement: FSDPPlacementSpec | None = None
    no_shard: bool = False
    standalone_world_shard: bool = False
    activation_checkpoint: bool = False
    replace_module: Any | None = None
    module_kwargs: dict[str, Any] = field(default_factory=dict)
    vision_balance: VisionBalanceSpec | None = None

    def __post_init__(self) -> None:
        if self.vision_balance is not None and not isinstance(
            self.vision_balance, VisionBalanceSpec
        ):
            raise TypeError("FSDPPlanSpec.vision_balance must be a VisionBalanceSpec.")
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
        FQN pattern to FSDPPlanSpec mapping dictionary.
    sp_size:
        Sequence Parallelism group size.
    ep_size:
        Expert Parallelism group size.
    tp_size:
        Tensor Parallelism group size.
    use_fla_kcp:
        Whether to route KDA layers through FLA kernel context parallelism
        (KCP) instead of the default Ulysses sequence parallelism.
    """

    plan: dict[str, FSDPPlacementSpec | FSDPPlanSpec | Any] = field(
        default_factory=dict
    )
    sp_size: int = 1
    ep_size: int = 1
    tp_size: int = 1
    use_fla_kcp: bool = False

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
            if match_fqn(pattern, fqn):
                return spec
        return None
