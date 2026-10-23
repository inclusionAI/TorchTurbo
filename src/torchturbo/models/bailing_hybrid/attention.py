"""Shared training attention adapters for Bailing text and vision-language models."""

from typing import Any

import torch
from torch import nn
from torch.distributed import ProcessGroup

def patch_rms_norm(norm_cls: type[nn.Module]) -> None:
    """Route the supplied RMSNorm class through the device operator dispatcher."""
    from torchturbo.ops import ops

    def rms_norm_forward(
        module: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return ops.rms_norm(hidden_states, module.weight, module.variance_epsilon)

    norm_cls.forward = rms_norm_forward


def patch_kda_forward(
    attention_cls: type[nn.Module],
    sp_group: ProcessGroup | None,
    sp_size: int,
    sp_rank: int,
) -> None:
    """Install packed KDA training with sequence gathering and head sharding.

    The wrapped forward receives global packed boundaries as the
    ``cu_seqlens`` keyword; convolutions and KDA share them while their
    parameters are sliced by head for sequence parallelism.
    """
    from fla.modules.fused_norm_gate import rms_norm_gated

    from torchturbo.ops import ops
    from torchturbo.ops.packed_params import get_packed_params
    from torchturbo.models.modules.attention.ulysses.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    def get_parameter_local_cp(weight: torch.Tensor, dim: int) -> torch.Tensor:
        if sp_size == 1:
            return weight

        slices = [slice(None)] * weight.dim()
        dim_size = weight.size(dim=dim)
        slices[dim] = slice(
            sp_rank * dim_size // sp_size, (sp_rank + 1) * dim_size // sp_size
        )
        return weight[tuple(slices)]

    def norm_forward(
        module: nn.Module,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm_gated(
            x,
            g,
            module.weight,
            None,
            module.activation,
            residual=residual,
            eps=module.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )

    def sp_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        assert past_key_value is None, "KDA training does not accept a KV cache."
        assert self.mode == "chunk", "Only chunk mode is supported in training."

        # Global packed boundaries ride with the forward kwargs; the a2a
        # gathers below rebuild the full sequence on each rank. Convolutions
        # consume cached metadata with the padded tail as a final sequence.
        batch_size, q_len, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if sp_size > 1:
            q = gather_seq_scatter_heads(
                q, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            k = gather_seq_scatter_heads(
                k, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            v = gather_seq_scatter_heads(
                v, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )

        # Every convolution must reset at the same boundaries as recurrent KDA.
        # ``head_num`` asks the fused convolution provider for its native
        # head-first ``[B, H, T, Dh]`` output, which is the spelling the fused
        # KDA operator and its backward consume -- so Q/K/V never cross a
        # layout transpose.  The convolution is depthwise, so regrouping
        # channels is exact; the triton provider regroups them itself.
        q_heads = q.shape[-1] // self.head_k_dim
        k_heads = k.shape[-1] // self.head_k_dim
        v_heads = v.shape[-1] // self.head_dim
        q, _ = ops.causal_conv1d(
            x=q,
            weight=get_parameter_local_cp(self.q_conv1d.weight, dim=0).squeeze(1),
            bias=get_parameter_local_cp(self.q_conv1d.bias, dim=0)
            if self.q_conv1d.bias is not None
            else None,
            activation=self.q_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            packed_params=packed_params,
            head_num=q_heads,
        )
        k, _ = ops.causal_conv1d(
            x=k,
            weight=get_parameter_local_cp(self.k_conv1d.weight, dim=0).squeeze(1),
            bias=get_parameter_local_cp(self.k_conv1d.bias, dim=0)
            if self.k_conv1d.bias is not None
            else None,
            activation=self.k_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            packed_params=packed_params,
            head_num=k_heads,
        )
        v, _ = ops.causal_conv1d(
            x=v,
            weight=get_parameter_local_cp(self.v_conv1d.weight, dim=0).squeeze(1),
            bias=get_parameter_local_cp(self.v_conv1d.bias, dim=0)
            if self.v_conv1d.bias is not None
            else None,
            activation=self.v_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            packed_params=packed_params,
            head_num=v_heads,
        )

        # Compute g, beta from padded hidden_states (always [B, T/sp, H] format)
        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float().sigmoid()

        # q/k/v already come back head-first [B, H, T, D] from the fused
        # convolution, so only the gate needs its last-dim head split.
        g = g.unflatten(-1, (-1, self.head_dim))

        if sp_size > 1:
            g = gather_seq_scatter_heads(
                g, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            beta = gather_seq_scatter_heads(
                beta, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )
            heads_per_rank = self.num_heads // sp_size
            start_head = sp_rank * heads_per_rank
            end_head = start_head + heads_per_rank
            A_log_local = self.A_log[start_head:end_head]
            start_ch = start_head * self.head_dim
            end_ch = end_head * self.head_dim
            dt_bias_local = self.dt_bias[start_ch:end_ch]
        else:
            A_log_local = self.A_log
            dt_bias_local = self.dt_bias

        # ops dispatcher routes to the native fla triton chunk_kda by
        # default; the AscendC fused fwd+bwd provider is opt-in
        # (TORCHTURBO_KDA_ASCENDC=1 or an explicit impl/override).
        #
        # ``head_major_qkv`` keeps Q/K/V head-first into the operator (the
        # native provider converts internally).  ``disable_recompute=True``
        # is required by the fused AscendC backward, which has no recompute
        # path on A2/A3 -- set it on the triton provider too so both
        # backends run in the same memory regime and stay comparable.
        o, _ = ops.chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log_local,
            dt_bias=dt_bias_local,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
            disable_recompute=True,
            head_major_qkv=True,
        )

        if sp_size > 1:
            # o: [B, T, H/sp, V] -> [B, T/sp, H, V]
            o = gather_heads_scatter_seq(
                o, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )

        # Output gating uses the original local sequence and all of its heads.
        if self.no_kda_lora:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.unflatten(-1, (-1, self.head_dim))
        self.o_norm.forward = norm_forward.__get__(self.o_norm)
        o = self.o_norm(o, g)
        o = self.o_proj(o.flatten(-2))
        return o, None, None

    attention_cls.forward = sp_forward
