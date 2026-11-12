"""Distributed backend base class — the interface each backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from transformers import PretrainedConfig

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ...logging import get_logger
from ...runtime import DeviceRuntime
from ..model_fqn import ModelFQNMap
from ..checkpoint.format import ComponentStatus

if TYPE_CHECKING:
    from ...models.weight_converter import BaseWeightConverter
    from ..config import EngineConfig
    from ..optimizers.plan import OptimizerPlan


class BaseCheckpointManager(ABC):
    """Backend-specific checkpoint tensor I/O.

    Directory arguments identify one concrete checkpoint component. The
    engine-owned coordinator manages the checkpoint root and its lifecycle.
    """

    @abstractmethod
    def native_model_dir(self, checkpoint_dir: str) -> str:
        """Return the directory that stores exact model state."""

    @abstractmethod
    def optimizer_dir(self, checkpoint_dir: str) -> str:
        """Return the directory that stores optimizer state."""

    @abstractmethod
    def native_model_status(self, checkpoint_dir: str) -> ComponentStatus:
        """Return whether exact model state is missing, incomplete, or complete."""

    @abstractmethod
    def optimizer_status(self, checkpoint_dir: str) -> ComponentStatus:
        """Return whether optimizer state is missing, incomplete, or complete."""

    def validate_optimizer_checkpoint(self) -> None:
        """Check local optimizer readiness before checkpoint I/O on any rank."""

    @abstractmethod
    def load_hf(
        self,
        model: nn.Module,
        weights_dir: str,
        *,
        weight_converter: BaseWeightConverter,
        assign: bool = False,
    ) -> None:
        """Load HF weights into a model prepared by the backend."""

    @abstractmethod
    def save_hf(
        self,
        model: nn.Module,
        output_dir: str,
        *,
        source_path: str,
        weight_converter: BaseWeightConverter,
    ) -> None:
        """Export HF weights under ``output_dir``."""

    @abstractmethod
    def save_native(self, model: nn.Module, output_dir: str) -> None:
        """Save exact model state in the backend's native format."""

    @abstractmethod
    def load_native(self, model: nn.Module, checkpoint_dir: str) -> None:
        """Restore exact model state from the backend's native format."""

    @abstractmethod
    def save_optimizer(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        output_dir: str,
    ) -> None:
        """Save optimizer state under ``output_dir``."""

    @abstractmethod
    def load_optimizer(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        checkpoint_dir: str,
    ) -> None:
        """Restore optimizer state from ``checkpoint_dir``."""


class BaseBackend(ABC):
    """Abstract base for a distributed backend.

    Subclass this and implement ``shard_model`` / ``prepare_optimizer``
    to add a new backend (e.g. Megatron).
    """

    name: str = ""

    def __init__(self, config: EngineConfig, device_runtime: DeviceRuntime) -> None:
        self.config = config
        self.logger = get_logger(type(self).__module__)
        if not isinstance(device_runtime, DeviceRuntime):
            raise TypeError(
                "device_runtime must be a DeviceRuntime. "
                "Create backends through TorchTurboEngine so the engine can "
                "select the process device runtime first."
            )
        self._device_runtime = device_runtime
        self._model_fqn_map: ModelFQNMap | None = None

    @property
    def device_runtime(self) -> DeviceRuntime:
        return self._device_runtime

    @property
    def device(self) -> torch.device:
        return self._device_runtime.device

    def get_model_fqn_map(self) -> ModelFQNMap:
        """Return the FQN map captured while preparing the current model."""
        if self._model_fqn_map is None:
            raise RuntimeError(
                "Model FQN map is unavailable. Call prepare_model() first."
            )
        return self._model_fqn_map

    @abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        parallel_plan: Any = None,
        peft_config: Any = None,
    ) -> nn.Module:
        """Wrap / shard the model according to this backend's strategy.

        Parameters
        ----------
        model:
            The raw model to shard.
        parallel_plan:
            Optional ``ParallelPlan`` with layer-name → placement mapping.
            If provided, applies per-layer sharding based on FQN patterns.
        peft_config:
            Optional PEFT/LoRA configuration (``LoraConfig`` or ``dict``).
            When provided, LoRA adapters are applied **after** module
            replacement but **before** sharding.  Backends that don't
            support PEFT should raise an error if this is non-None.
        """

    @abstractmethod
    def materialize(self, model: nn.Module) -> nn.Module:
        """Materialize a prepared model without loading source weights."""

    def finalize_model_load(self, model: nn.Module) -> nn.Module:
        """Run backend setup that requires materialized model parameters."""
        return model

    @abstractmethod
    def create_checkpoint_manager(self) -> BaseCheckpointManager:
        """Create checkpoint I/O after the backend has prepared its topology."""

    @abstractmethod
    def prepare_optimizer(
        self,
        plan: OptimizerPlan,
    ) -> Optimizer:
        """Return a validated optimizer with all plan-defined step hooks installed."""

    def offload_optimizer(self) -> None:
        """Release prepared optimizer device resources at a cleared-gradient boundary."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support optimizer offload"
        )

    def restore_optimizer(self) -> None:
        """Restore the prepared optimizer's configured training state locations."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support optimizer restore"
        )

    @abstractmethod
    def get_dp_info(self) -> dict[str, Any]:
        """Compute data-parallel rank and size.

        Override in subclass for distributed backends.
        """

    @abstractmethod
    def get_ep_info(self) -> dict[str, Any]:
        """Compute expert-parallel rank, size and group.

        Override in subclass for distributed backends.
        """

    @abstractmethod
    def get_sp_info(self) -> dict[str, Any]:
        """Compute sequence-parallel rank, size and group.

        Override in subclass for distributed backends.
        """

    @abstractmethod
    def get_parallel_context(self) -> dict[str, Any]:
        """Return parallel-context dict of EP/SP/DP info.

        Override in subclass for distributed backends.
        """

    def prepare_inputs(self, model: nn.Module, inputs: Any) -> Any:
        """Apply backend-specific input layout requirements before a step."""
        return inputs

    @abstractmethod
    def compute_loss(
        self, model: nn.Module, inputs: Any, return_outputs: bool = False, **kwargs
    ):
        """Compute Loss.

        Override in subclass for distributed backends.
        """

    def compute_diffusion_loss(
        self,
        model: nn.Module,
        inputs: Any,
        ga: int = 1,
        return_outputs: bool = False,
        **kwargs,
    ):
        """Compute block-diffusion loss for ``is_diffusion_model`` models.

        Default is unsupported; a backend opts in by overriding this. The engine
        routes here only when ``model.is_diffusion_model`` is True, so a backend
        that never trains diffusion models can leave the default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support diffusion training "
            "(compute_diffusion_loss). Use a backend that implements it, e.g. FSDP2."
        )

    def clip_grad_norm_(self, parameters, max_norm=1.0, norm_type=2.0):
        """Clip the gradient norm of ``parameters`` in place; return the total norm.

        Default delegates to :func:`torch.nn.utils.clip_grad_norm_`. Distributed
        backends (e.g. FSDP2 with DTensor / expert-parallel grads) override this
        with a sharding-aware implementation.
        """
        return nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
