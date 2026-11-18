"""Backend-independent optimizer plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn
from torch.optim import Optimizer

from ...logging import get_logger


logger = get_logger(__name__)


class ParamGroup:
    """A resolved optimizer parameter group for a prepared model."""

    __slots__ = (
        "optimizer_name",
        "params_by_fqn",
        "optimizer_kwargs",
    )

    optimizer_name: str
    params_by_fqn: dict[str, nn.Parameter]
    optimizer_kwargs: dict[str, Any]

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer_name: str,
        param_names: Sequence[str] | None = None,
        params: Sequence[nn.Parameter] | None = None,
        optimizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("ParamGroup model must be an nn.Module")
        if not isinstance(optimizer_name, str) or not optimizer_name:
            raise ValueError("ParamGroup optimizer_name must be a non-empty string")
        if optimizer_kwargs is not None and not isinstance(
            optimizer_kwargs,
            Mapping,
        ):
            raise TypeError("ParamGroup optimizer_kwargs must be a mapping or None")

        resolved_optimizer_kwargs = dict(optimizer_kwargs or {})
        reserved_keys = {"params", "param_names"} & set(resolved_optimizer_kwargs)
        if reserved_keys:
            raise ValueError(
                "ParamGroup optimizer_kwargs cannot contain reserved keys: "
                f"{sorted(reserved_keys)!r}"
            )

        requested_param_names = () if param_names is None else tuple(param_names)
        requested_params = () if params is None else tuple(params)
        if not requested_param_names and not requested_params:
            raise ValueError("ParamGroup requires param_names, params, or both")
        if not all(isinstance(name, str) for name in requested_param_names):
            raise TypeError("ParamGroup param_names must contain strings")

        named_parameters = tuple(model.named_parameters())
        params_by_fqn = dict(named_parameters)
        fqn_by_param_id = {id(param): fqn for fqn, param in named_parameters}
        selected_fqns: set[str] = set()

        for fqn in requested_param_names:
            try:
                param = params_by_fqn[fqn]
            except KeyError as exc:
                raise ValueError(
                    f"ParamGroup param_name {fqn!r} is not in the model"
                ) from exc
            if not param.requires_grad:
                raise ValueError(
                    f"ParamGroup parameter {fqn!r} does not require gradients"
                )
            selected_fqns.add(fqn)

        for param in requested_params:
            try:
                fqn = fqn_by_param_id[id(param)]
            except KeyError as exc:
                raise ValueError(
                    "ParamGroup params must belong to the prepared model"
                ) from exc
            model_param = params_by_fqn[fqn]
            if model_param is not param:
                raise ValueError(
                    "ParamGroup params must match model parameters by identity"
                )
            if not param.requires_grad:
                raise ValueError(
                    f"ParamGroup parameter {fqn!r} does not require gradients"
                )
            selected_fqns.add(fqn)

        self.optimizer_name = optimizer_name
        self.params_by_fqn = {
            fqn: param for fqn, param in named_parameters if fqn in selected_fqns
        }
        self.optimizer_kwargs = resolved_optimizer_kwargs

    def get_optimizer_param_group(self) -> dict[str, Any]:
        """Return the dict accepted by ``torch.optim.Optimizer``."""
        return {
            **self.optimizer_kwargs,
            "params": tuple(self.params_by_fqn.values()),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"optimizer_name={self.optimizer_name!r}, "
            f"parameters={len(self.params_by_fqn)}, "
            f"optimizer_kwargs={self.optimizer_kwargs!r})"
        )


class OptimizerRecipe:
    """Base type for model-provided optimizer semantics."""

    __slots__ = ()


class OptimizerPlan(ABC):
    """Base class for fully resolved optimizer construction plans."""

    __slots__ = ()

    @abstractmethod
    def get_param_groups(self) -> tuple[ParamGroup, ...]:
        """Return all resolved parameter groups in construction order."""

    @abstractmethod
    def get_param_groups_by_optimizer_name(
        self,
    ) -> dict[str, tuple[ParamGroup, ...]]:
        """Return the resolved groups indexed by optimizer name."""

    def _validate_param_groups(
        self,
        model: nn.Module,
        param_groups: Sequence[ParamGroup],
    ) -> None:
        """Validate identity, uniqueness, and trainable-parameter coverage."""
        if not isinstance(model, nn.Module):
            raise TypeError("OptimizerPlan model must be an nn.Module")
        groups = tuple(param_groups)
        if not groups:
            raise ValueError("optimizer plan requires parameter groups")
        if not all(isinstance(group, ParamGroup) for group in groups):
            raise TypeError("param_groups must contain ParamGroup")

        named_parameters = tuple(
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and not getattr(param, "_engram_host_offload", False)
        )
        model_params_by_fqn = dict(named_parameters)
        claimed_param_ids: dict[int, tuple[str, str]] = {}
        for group in groups:
            if not group.params_by_fqn:
                raise ValueError("optimizer parameter groups cannot be empty")
            for fqn, param in group.params_by_fqn.items():
                model_param = model_params_by_fqn.get(fqn)
                if model_param is None:
                    raise ValueError(
                        f"optimizer parameter {fqn!r} is not a trainable model "
                        "parameter"
                    )
                if model_param is not param:
                    raise ValueError(
                        f"optimizer parameter {fqn!r} does not match the "
                        "prepared model by identity"
                    )
                previous = claimed_param_ids.get(id(param))
                if previous is not None:
                    previous_fqn, previous_optimizer = previous
                    raise ValueError(
                        f"optimizer parameter {fqn!r} is assigned to multiple "
                        "parameter groups: "
                        f"{previous_optimizer!r} ({previous_fqn!r}) and "
                        f"{group.optimizer_name!r}"
                    )
                claimed_param_ids[id(param)] = (fqn, group.optimizer_name)

        unclaimed = [
            fqn for fqn, param in named_parameters if id(param) not in claimed_param_ids
        ]
        if unclaimed:
            raise ValueError(
                "optimizer plan did not claim every trainable parameter; "
                f"first unclaimed FQNs: {unclaimed[:8]}"
            )

    def _group_param_groups_by_optimizer_name(
        self,
        param_groups: Sequence[ParamGroup],
    ) -> dict[str, tuple[ParamGroup, ...]]:
        """Group resolved parameter groups while preserving their order."""
        groups_by_optimizer_name: dict[str, list[ParamGroup]] = defaultdict(list)
        for group in param_groups:
            if not isinstance(group, ParamGroup):
                raise TypeError("param_groups must contain ParamGroup")
            groups_by_optimizer_name[group.optimizer_name].append(group)
        return {
            optimizer_name: tuple(groups)
            for optimizer_name, groups in groups_by_optimizer_name.items()
        }


class OptimizerClassPlan(OptimizerPlan):
    """Plan for an optimizer class supplied by the training framework."""

    __slots__ = (
        "_param_groups",
        "_param_groups_by_optimizer_name",
        "optimizer_cls",
        "optimizer_kwargs",
        "optimizer_name",
    )

    optimizer_cls: type
    optimizer_kwargs: Mapping[str, Any]
    optimizer_name: str

    def __init__(
        self,
        model: nn.Module,
        optimizer_cls: type,
        optimizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        raw_optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self._validate(model, optimizer_cls, raw_optimizer_kwargs)

        resolved_kwargs = dict(raw_optimizer_kwargs)
        resolved_kwargs.pop("params", None)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = resolved_kwargs
        self.optimizer_name = self.optimizer_cls.__name__
        self._param_groups = self._prepare_param_groups(model)
        self._validate_param_groups(model, self._param_groups)
        self._param_groups_by_optimizer_name = (
            self._group_param_groups_by_optimizer_name(self._param_groups)
        )

    def get_param_groups(self) -> tuple[ParamGroup, ...]:
        return self._param_groups

    def get_param_groups_by_optimizer_name(
        self,
    ) -> dict[str, tuple[ParamGroup, ...]]:
        return self._param_groups_by_optimizer_name

    def _prepare_param_groups(
        self,
        model: nn.Module,
    ) -> tuple[ParamGroup, ...]:
        """Split trainable parameters into decay and no-decay optimizer groups."""
        decay: list[str] = []
        no_decay: list[str] = []
        # Megatron Core default: biases and 1-D parameters do not decay.
        # Reference: megatron.core.optimizer._get_param_groups
        for name, param in model.named_parameters():
            if not param.requires_grad or getattr(param, "_engram_host_offload", False):
                continue
            if name.endswith(".bias") or param.ndim == 1:
                no_decay.append(name)
            else:
                decay.append(name)

        param_groups = []
        if decay:
            param_groups.append(
                ParamGroup(
                    model,
                    optimizer_name=self.optimizer_name,
                    param_names=decay,
                )
            )
        if no_decay:
            param_groups.append(
                ParamGroup(
                    model,
                    optimizer_name=self.optimizer_name,
                    param_names=no_decay,
                    optimizer_kwargs={"weight_decay": 0.0},
                )
            )
        logger.info_rank0(
            "Prepared optimizer parameter groups: decay=%d, no_decay=%d, "
            "weight_decay=%s",
            len(decay),
            len(no_decay),
            self.optimizer_kwargs["weight_decay"],
        )
        return tuple(param_groups)

    @staticmethod
    def _validate(
        model: nn.Module,
        optimizer_cls: type,
        optimizer_kwargs: Mapping[str, Any],
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("OptimizerClassPlan model must be an nn.Module")
        if not isinstance(optimizer_cls, type):
            raise TypeError("OptimizerClassPlan.optimizer_cls must be a class")
        if not issubclass(optimizer_cls, Optimizer):
            raise TypeError(
                "OptimizerClassPlan.optimizer_cls must be a torch.optim.Optimizer "
                "subclass"
            )
        if not isinstance(optimizer_kwargs, Mapping):
            raise TypeError("OptimizerClassPlan.optimizer_kwargs must be a mapping")

        optimizer_kwargs = dict(optimizer_kwargs)
        required_kwargs = ("lr", "weight_decay")
        missing_kwargs = [
            name
            for name in required_kwargs
            if name not in optimizer_kwargs or optimizer_kwargs[name] is None
        ]
        if missing_kwargs:
            raise ValueError(
                "OptimizerClassPlan requires explicit values for "
                f"{', '.join(missing_kwargs)}"
            )

    def __repr__(self) -> str:
        optimizer_name = (
            f"{self.optimizer_cls.__module__}.{self.optimizer_cls.__qualname__}"
        )
        return (
            f"{type(self).__name__}("
            f"optimizer_cls={optimizer_name}, "
            f"optimizer_kwargs={dict(self.optimizer_kwargs)!r})"
        )


__all__ = [
    "OptimizerClassPlan",
    "OptimizerPlan",
    "OptimizerRecipe",
    "ParamGroup",
]
