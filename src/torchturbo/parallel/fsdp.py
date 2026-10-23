"""FSDP-specific parallel plan and module specifications."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan schema, keeping replacement kwargs and IO objects typed."""
        plan = copy.deepcopy(self)
        data = {f.name: getattr(plan, f.name) for f in fields(plan)}
        data["plan"] = {}
        for fqn, spec in plan.plan.items():
            spec_d = {f.name: getattr(spec, f.name) for f in fields(spec)}
            for name in ("placement", "vision_balance"):
                value = spec_d[name]
                if value is not None:
                    spec_d[name] = {f.name: getattr(value, f.name) for f in fields(value)}
            data["plan"][fqn] = spec_d
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FSDPParallelPlan:
        """Rebuild a full plan or partial overrides from a dictionary."""
        spec_fields = [f.name for f in fields(FSDPPlanSpec)]
        plan_entries = {}
        for fqn, spec_d in (data.get("plan") or {}).items():
            # Missing fields must remain None so partial plans inherit model defaults.
            spec_kwargs = {**dict.fromkeys(spec_fields), **spec_d}
            placement_d = spec_kwargs["placement"]
            if isinstance(placement_d, Mapping):
                spec_kwargs["placement"] = FSDPPlacementSpec(**placement_d)
            vision_d = spec_kwargs["vision_balance"]
            if vision_d is not None:
                mode = vision_d.get("mode")
                spec_kwargs["vision_balance"] = VisionBalanceSpec(**{
                    **vision_d,
                    "mode": VisionBalanceMode(mode) if mode is not None else None,
                })
            plan_entries[fqn] = FSDPPlanSpec(**spec_kwargs)
        return cls(**{**data, "plan": plan_entries})

    def match(self, fqn: str) -> Any | None:
        """Find the specification for a given FQN. Returns None if no match."""
        for pattern, spec in self.plan.items():
            if match_fqn(pattern, fqn):
                return spec
        return None
