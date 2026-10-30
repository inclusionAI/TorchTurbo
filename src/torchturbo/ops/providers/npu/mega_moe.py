"""Ascend fused MoonEP and MegaMoE provider with model-owned execution state."""
from __future__ import annotations

import importlib.util
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import nn

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(op_name="mega_moe", impl_name="moonep_mega_moe", kind=OpImplKind.CUSTOM,
             priority=100, devices=(TurboDevice.NPU,))
class MoonEPMegaMoe(OpImpl):
    def available(self) -> bool:
        try:
            from torchturbo.optimizations.mega_moe.runtime import bootstrap

            if importlib.util.find_spec("torch_npu") is None:
                return False
            spec = importlib.util.find_spec("shmem")
            if spec is None or spec.origin is None or version("cann-shmem") != "1.6.0":
                return False
            package = Path(bootstrap.__file__).parent
            shmem = Path(spec.origin).parent / "backends/910"
            return all((package / name).is_file() for name in bootstrap.NATIVE_LIBRARIES) and all(
                (shmem / name).is_file() for name in ("libshmem.so", "libshmem_utils.so")
            )
        except ImportError:
            return False

    def __call__(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        probs: torch.Tensor,
        *,
        experts: nn.Module,
        backend: Any,
    ) -> torch.Tensor:
        return backend.execute(experts, hidden_states, expert_indices, probs)
