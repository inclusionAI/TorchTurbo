"""Checkpoint tensor I/O of the FSDP2 backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from torch.optim import Optimizer

from ...base import BaseCheckpointManager
from ....checkpoint.format import ComponentStatus
from .layout import MODEL_DCP_DIR, OPTIMIZER_DCP_DIR, dcp_status
from .dcp import DcpCheckpoint
from .weight_loader import DistParamLoader
from .weight_saver import ShardedParamManager

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from torchturbo.models.weight_converter import BaseWeightConverter
    from torchturbo.runtime import DeviceRuntime


class FSDP2CheckpointManager(BaseCheckpointManager):
    """Move checkpoint tensors between storage and an FSDP2 model.

    The engine-owned checkpoint coordinator handles save policy, component
    layout and scheduler state. This manager owns only operations that require
    FSDP2 mesh and device knowledge.
    """

    def __init__(
        self,
        *,
        rank: int,
        mesh: DeviceMesh,
        world_mesh: DeviceMesh,
        parallel_context: dict[str, Any],
        device_runtime: DeviceRuntime,
        cpu_offload: bool,
        require_optimizer_not_suspended: Callable[[], None],
    ) -> None:
        self._rank = rank
        self._mesh = mesh
        self._world_mesh = world_mesh
        self._device_runtime = device_runtime
        self._cpu_offload = cpu_offload
        self._require_optimizer_not_suspended = require_optimizer_not_suspended
        self._dcp = DcpCheckpoint(
            ep_rank=parallel_context["ep"]["ep_rank"],
            ep_size=parallel_context["ep"]["ep_size"],
            edp_mesh=mesh["edp"],
        )

    def native_model_dir(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, MODEL_DCP_DIR)

    def optimizer_dir(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, OPTIMIZER_DCP_DIR)

    def native_model_status(self, checkpoint_dir: str) -> ComponentStatus:
        return dcp_status(self.native_model_dir(checkpoint_dir))

    def optimizer_status(self, checkpoint_dir: str) -> ComponentStatus:
        return dcp_status(self.optimizer_dir(checkpoint_dir))

    def validate_optimizer_checkpoint(self) -> None:
        """Reject checkpoint I/O during temporary optimizer suspension."""
        self._require_optimizer_not_suspended()

    def load_hf(
        self,
        model: nn.Module,
        weights_dir: str,
        *,
        weight_converter: BaseWeightConverter,
        assign: bool = False,
    ) -> None:
        self._hf_loader(model, weights_dir, weight_converter).load_weights(assign=assign)

    def save_hf(
        self,
        model: nn.Module,
        output_dir: str,
        *,
        source_path: str,
        weight_converter: BaseWeightConverter,
    ) -> None:
        self._hf_saver(model, source_path, weight_converter).save_model(output_dir)

    def save_native(self, model: nn.Module, output_dir: str) -> None:
        self._dcp.save_model(model, output_dir)

    def load_native(self, model: nn.Module, checkpoint_dir: str) -> None:
        self._dcp.load_model(model, checkpoint_dir)

    def save_optimizer(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        output_dir: str,
    ) -> None:
        self._dcp.save_optimizer(optimizer, model, output_dir)

    def load_optimizer(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        checkpoint_dir: str,
    ) -> None:
        self._dcp.load_optimizer(optimizer, model, checkpoint_dir)

    def _hf_loader(
        self,
        model: nn.Module,
        weights_dir: str,
        converter: BaseWeightConverter,
    ) -> DistParamLoader:
        return DistParamLoader(
            model=model,
            path=weights_dir,
            rank=self._rank,
            mesh=self._mesh,
            world_mesh=self._world_mesh,
            device_runtime=self._device_runtime,
            is_cpu_offload=self._cpu_offload,
            weight_converter=converter,
        )

    def _hf_saver(
        self,
        model: nn.Module,
        source_path: str,
        converter: BaseWeightConverter,
    ) -> ShardedParamManager:
        return ShardedParamManager(
            model=model,
            rank=self._rank,
            mesh=self._mesh,
            weight_converter=converter,
            device_runtime=self._device_runtime,
            path=source_path,
            is_cpu_offload=self._cpu_offload,
        )
