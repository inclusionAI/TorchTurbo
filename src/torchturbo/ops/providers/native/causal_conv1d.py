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
        head_num=None,
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
        if head_num is not None and y is not None:
            # Mirror the AscendC provider's head-first spelling; the
            # depthwise conv makes the channel regrouping exact.
            head_num = int(head_num)
            batch, seq_len, dim = y.shape
            if head_num <= 0 or dim % head_num:
                raise ValueError(
                    f"head_num={head_num} must be a positive divisor of D={dim}."
                )
            y = (
                y.view(batch, seq_len, head_num, dim // head_num)
                .transpose(1, 2)
                .contiguous()
            )
        return y, final_state
