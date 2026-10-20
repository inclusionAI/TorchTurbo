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
    use_fla_kcp: bool = False,
) -> None:
    """Install packed KDA training with Ulysses or FLA context parallelism.

    The wrapped forward receives global packed boundaries as the
    ``cu_seqlens`` keyword; convolutions and KDA share them while their
    parameters are sliced by head for Ulysses sequence parallelism.
    FLA context parallelism keeps all heads on each local sequence shard.
    """
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.kda import chunk_kda

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
        )

        # Compute g, beta from padded hidden_states (always [B, T/sp, H] format)
        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))
        beta = self.b_proj(hidden_states).float().sigmoid()

        q = q.unflatten(-1, (-1, self.head_k_dim))
        k = k.unflatten(-1, (-1, self.head_k_dim))
        v = v.unflatten(-1, (-1, self.head_dim))
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

        o, _ = chunk_kda(
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

    def kcp_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        from fla.ops.cp import build_cp_context

        batch_size, q_len, _ = hidden_states.shape
        assert past_key_value is None, "KDA training does not accept a KV cache."
        assert self.mode == "chunk", "Only chunk mode is supported in training."

        if batch_size != 1:
            raise ValueError(
                f"FLA KCP requires batch_size=1, got {batch_size}. "
                "Use a per-device batch size of 1 or disable use_fla_kcp."
            )

        # Use global boundaries and isolate any padded tail as a sequence.
        cu_seqlens = kwargs.get("cu_seqlens")
        packed_params = get_packed_params(cu_seqlens, q_len * sp_size)
        if packed_params is not None:
            cu_seqlens = packed_params.cu_seqlens
        else:
            cu_seqlens = torch.tensor(
                [0, q_len * sp_size],
                dtype=torch.int32,
                device=hidden_states.device,
            )
        cp_context = build_cp_context(
            cu_seqlens,
            group=sp_group,
            conv1d_kernel_size=self.conv_size,
        )

        q = self.q_proj(hidden_states)   # [1, T/sp, H*K]
        k = self.k_proj(hidden_states)   # [1, T/sp, H*K]
        v = self.v_proj(hidden_states)   # [1, T/sp, H*V]

        # Dispatch through the ops layer; cp_context selects FLA's
        # Triton CP path instead of AscendC. d,1,w -> d,w.
        q, _ = ops.causal_conv1d(
            x=q,
            weight=self.q_conv1d.weight.squeeze(1),
            bias=self.q_conv1d.bias,
            activation=self.q_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            cp_context=cp_context,
        )
        k, _ = ops.causal_conv1d(
            x=k,
            weight=self.k_conv1d.weight.squeeze(1),
            bias=self.k_conv1d.bias,
            activation=self.k_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            cp_context=cp_context,
        )
        v, _ = ops.causal_conv1d(
            x=v,
            weight=self.v_conv1d.weight.squeeze(1),
            bias=self.v_conv1d.bias,
            activation=self.v_conv1d.activation,
            initial_state=None,
            output_final_state=False,
            cp_context=cp_context,
        )

        if self.no_kda_lora:
            g = self.f_proj(hidden_states)
        else:
            g = self.f_b_proj(self.f_a_proj(hidden_states))

        beta = self.b_proj(hidden_states).float().sigmoid()

        q = q.unflatten(-1, (-1, self.head_k_dim))
        k = k.unflatten(-1, (-1, self.head_k_dim))
        v = v.unflatten(-1, (-1, self.head_dim))
        g = g.unflatten(-1, (-1, self.head_dim))

        # Full A_log/dt_bias under KCP (no slicing).
        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cp_context=cp_context,
        )

        # o_norm uses g recomputed from hidden_states.
        if self.no_kda_lora:
            g = self.g_proj(hidden_states)
        else:
            g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = g.unflatten(-1, (-1, self.head_dim))

        self.o_norm.forward = norm_forward.__get__(self.o_norm)
        o = self.o_norm(o, g)

        o = o.flatten(-2)
        o = self.o_proj(o)
        return o, None, None

    attention_cls.forward = kcp_forward if use_fla_kcp else sp_forward
