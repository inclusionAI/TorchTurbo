"""Native (fla) provider for the causal short-conv1d op.

Reference depthwise causal conv1d (with silu activation + varlen via
``cu_seqlens``) used on non-NPU hosts and as the correctness baseline.
``devices=()`` makes it a candidate everywhere; on NPU the AscendC custom
provider (priority=100) wins under `auto`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...packed_params import PackedParams
from ...registry import register_op


@register_op(
    op_name="causal_conv1d",
    impl_name="fla_causal_conv1d",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class FlaCausalConv1d(OpImpl):
    def available(self) -> bool:
        try:
            from fla.modules.convolution import causal_conv1d  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        x,
        weight,
        bias=None,
        *,
        activation=None,
        initial_state=None,
        output_final_state: bool = False,
        packed_params=None,
        **kwargs,
    ):
        # fla's causal_conv1d is an autograd function returning (y, final_state).
        from fla.modules.convolution import causal_conv1d

        # ``packed_params`` is the provider API. Keep accepting the legacy
        # dispatcher keyword during the transition so older model paths do not
        # lose their varlen metadata.
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        if packed_params is not None:
            if cu_seqlens is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            cu_seqlens = packed_params.cu_seqlens
        y, final_state = causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        return y, final_state


@register_op(
    op_name="causal_conv1d",
    impl_name="torch_causal_conv1d",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(TurboDevice.CPU,),
)
class TorchCausalConv1d(OpImpl):
    """Run depthwise causal convolution on independent CPU training sequences."""

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        activation: str | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        packed_params: PackedParams | None = None,
    ) -> tuple[torch.Tensor, None]:
        if initial_state is not None or output_final_state:
            raise NotImplementedError("CPU causal convolution supports stateless training")
        if activation not in (None, "silu", "swish"):
            raise ValueError(f"Unsupported causal convolution activation: {activation}")
        boundaries = (
            packed_params.cu_seqlens_list
            if packed_params is not None
            else [0, x.size(1)]
        )
        outputs = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            output = F.conv1d(
                x[:, start:stop].transpose(1, 2),
                weight.unsqueeze(1),
                bias,
                padding=weight.size(-1) - 1,
                groups=x.size(-1),
            )[..., : stop - start]
            if activation is not None:
                output = F.silu(output)
            outputs.append(output.transpose(1, 2))
        return torch.cat(outputs, dim=1), None
