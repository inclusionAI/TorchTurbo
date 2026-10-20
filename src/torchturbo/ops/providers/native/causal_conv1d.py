"""Native (fla) provider for the causal short-conv1d op.

Reference depthwise causal conv1d (with silu activation + varlen via
``cu_seqlens``) used on non-NPU hosts and as the correctness baseline.
``devices=()`` makes it a candidate everywhere; on NPU the AscendC custom
provider (priority=100) wins under `auto`.
"""

from __future__ import annotations

from ...base import OpImpl, OpImplKind
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
        cp_context=None,
        **kwargs,
    ):
        # fla's causal_conv1d is an autograd function returning (y, final_state).
        # With cp_context, FLA uses its Triton context-parallel implementation.
        from fla.modules.convolution import causal_conv1d

        # ``packed_params`` is the provider API. Keep accepting the legacy
        # dispatcher keyword during the transition so older model paths do not
        # lose their varlen metadata.
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        if packed_params is not None:
            if cu_seqlens is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            cu_seqlens = packed_params.cu_seqlens
        return causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context,
        )
