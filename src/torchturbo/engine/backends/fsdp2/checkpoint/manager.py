"""Checkpoint tensor I/O of the FSDP2 backend."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer

from ...base import BaseCheckpointManager
from ....checkpoint.format import ComponentStatus
from ....checkpoint.distributed import rank_zero_call
from ....checkpoint.layout import sync_directory
from ....utils import strip_peft_prefix
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
        hf_export_lora: str = "merge",
    ) -> None:
        if hf_export_lora == "adapter":
            self._save_adapter(model, output_dir)
            return
        self._hf_saver(model, source_path, weight_converter).save_model(output_dir)

    def _save_adapter(self, model: nn.Module, output_dir: str) -> None:
        """Write a standalone PEFT adapter directory on rank zero.

        Adapter tensors are gathered to full tensors on every rank because
        LoRA parameters can be DTensors under the backend's parallel mesh.
        Only rank zero writes the resulting, typically small, state dict.
        """
        from peft import get_peft_model_state_dict

        adapter_state_dict = get_peft_model_state_dict(model)
        expert_tensors = self._grouped_lora_tensors(model)
        gathered_state_dict = {}
        for name, tensor in adapter_state_dict.items():
            if name in expert_tensors:
                # Expert tensors have a two-level layout (EDP shard and EP
                # local experts) and use a dedicated global gather.
                source_tensor, num_global_experts = expert_tensors[name]
                gathered_state_dict[name] = self._gather_expert_lora_tensor(
                    source_tensor, num_global_experts
                )
            else:
                gathered_state_dict[name] = self._full_cpu_tensor(tensor)

        def write_adapter() -> None:
            os.makedirs(output_dir, exist_ok=True)
            save_file(
                gathered_state_dict,
                os.path.join(output_dir, "adapter_model.safetensors"),
            )
            for config in model.peft_config.values():
                config.save_pretrained(output_dir)
            sync_directory(output_dir)

        rank_zero_call(write_adapter)

    def _grouped_lora_tensors(self, model: nn.Module) -> dict[str, tuple[Any, int]]:
        """Map saved GroupedMLPLoRA keys to their source tensors."""
        from ....peft.lora import GroupedMLPLoRA

        experts = {}
        for module_name, module in model.named_modules():
            if not isinstance(module, GroupedMLPLoRA):
                continue
            clean_name = strip_peft_prefix(module_name)
            for layer_name in ("lora_A", "lora_B", "lora_A2", "lora_B2"):
                key = f"base_model.model.{clean_name}.{layer_name}.weight"
                experts[key] = (
                    getattr(module, layer_name)["default"].weight,
                    module.base_layer.config.num_moe_experts,
                )
        return experts

    def _stage_cpu_offload_tensor(self, tensor: Any) -> Any:
        """Stage a CPU-resident DTensor shard onto its mesh's device."""
        if not self._cpu_offload or not isinstance(tensor, DTensor):
            return tensor
        local = tensor.to_local()
        compute_device = self._device_runtime.device
        if local.device.type == compute_device.type:
            return tensor
        return DTensor.from_local(
            local.to(compute_device),
            tensor.device_mesh,
            tensor.placements,
            run_check=False,
            shape=tensor.shape,
            stride=tensor.stride(),
        )

    def _full_cpu_tensor(self, tensor: Any) -> Any:
        """Detach a local or distributed tensor into a CPU safetensors value."""
        if isinstance(tensor, DTensor):
            tensor = self._stage_cpu_offload_tensor(tensor)
            tensor = tensor.full_tensor()
        return tensor.detach().to("cpu", copy=True).contiguous()

    def _gather_expert_lora_tensor(
        self, tensor: Any, num_global_experts: int
    ) -> torch.Tensor:
        """Concatenate GroupedMLPLoRA local expert tensors over the EP mesh."""
        # A GroupedMLPLoRA DTensor is sharded over EDP. full_tensor() first
        # reconstructs this rank's complete set of local experts (for example
        # 32), after which the EP gather below obtains every peer's experts.
        # In full CPU-offload mode the local shard is CPU-resident while its
        # mesh still uses the NPU/HCCL backend, so stage it before gathering.
        if isinstance(tensor, DTensor):
            tensor = self._stage_cpu_offload_tensor(tensor)
            local = tensor.full_tensor()
        else:
            local = tensor
        local = local.detach().contiguous()
        if local.device.type != self._device_runtime.device.type:
            local = local.to(self._device_runtime.device)

        ep_mesh = self._mesh["ep"]
        ep_group = ep_mesh.get_group(0)
        ep_size = dist.get_world_size(group=ep_group)
        if ep_size == 1:
            gathered = [local]
        else:
            gathered = [torch.empty_like(local) for _ in range(ep_size)]
            dist.all_gather(gathered, local, group=ep_group)

        # EP communication must stay on the process-group device; HCCL groups
        # reject CPU inputs. Move only the final concatenated tensor to CPU.
        full = torch.cat(gathered, dim=0).to("cpu", copy=True).contiguous()
        if full.shape[0] != num_global_experts:
            raise ValueError(
                "GroupedMLPLoRA expert gather produced an incomplete adapter: "
                f"expected {num_global_experts} experts, got {full.shape[0]} "
                f"(local={tuple(local.shape)}, ep_size={ep_size}, "
                f"tensor={type(tensor).__name__})"
            )
        return full

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
