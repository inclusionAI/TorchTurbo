"""Ascend Triton occurrence remapping with fused per-token sorting."""

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="moe_remap_routes",
    impl_name="npu_triton_moe_remap_routes",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuTritonMoERemapRoutes(OpImpl):
    def available(self) -> bool:
        try:
            from triton.backends.ascend.compiler import AscendBackend  # noqa: F401

            from ._kernels.moe_remap_routes import remap_routes  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self, indices: torch.Tensor, route_positions: torch.Tensor,
        remap_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ._kernels.moe_remap_routes import remap_routes

        return remap_routes(indices, route_positions, remap_table)
