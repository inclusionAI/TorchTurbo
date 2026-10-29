"""Head-parallel GatedDeltaNet forward for sequence-sharded inputs."""

from collections.abc import Callable
from typing import Any

import torch
from torch.distributed import ProcessGroup

from torchturbo.runtime import get_device_runtime


def _slice_param(weight: torch.Tensor, sp_size: int, sp_rank: int) -> torch.Tensor:
    num_channels = weight.size(0)
    start = sp_rank * num_channels // sp_size
    stop = (sp_rank + 1) * num_channels // sp_size
    return weight[start:stop]


def _interleave_qkv_conv_parameter(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_key_heads: int
) -> torch.Tensor:
    """Match depthwise-conv channels to the post-all-to-all QKV layout."""
    trailing_shape = q.shape[1:]
    q = q.reshape(num_key_heads, -1, *trailing_shape)
    k = k.reshape(num_key_heads, -1, *trailing_shape)
    v = v.reshape(num_key_heads, -1, *trailing_shape)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing_shape)


def build_gdn_sp_forward(
    sp_group: ProcessGroup | None,
    sp_size: int,
    sp_rank: int,
) -> Callable[..., torch.Tensor]:
    """Build a GDN training forward for local or sequence-sharded inputs.

    SP size one keeps the grouped Q/K/V projections and convolution parameters.
    With multiple ranks, convolution uses the interleaved all-to-all layout.
    The closure owns the stream for overlapping QKV exchange with gate
    projections. Gate projections share an exchange; the output gate stays local.

    Args:
        sp_group: Sequence-parallel process group, or None for SP size one.
        sp_size: Number of ranks in the sequence-parallel group.
        sp_rank: Rank within the sequence-parallel group.

    Returns:
        An unbound forward method returning the local sequence shard.
    """
    import torch.nn.functional as F

    from torchturbo.ops import ops

    from ..ulysses.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    a2a_stream = get_device_runtime().new_stream() if sp_size > 1 else None

    def sp_forward(
        self_attn: torch.nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        # Global packed boundaries ride with the forward kwargs (passed by
        # the decoder layer as cu_seq_lens); the a2a gathers below rebuild
        # the full sequence on each rank.
        cu_seq_lens = kwargs.get("cu_seq_lens")
        B, local_T, _ = hidden_states.shape

        nk_full, nv_full = self_attn.num_k_heads, self_attn.num_v_heads
        dk, dv = self_attn.head_k_dim, self_attn.head_v_dim

        mixed_qkv = self_attn.in_proj_qkv(hidden_states)

        sp_repeat = max(1, sp_size // nk_full)
        nk = nk_full * sp_repeat // sp_size
        nv = nv_full // sp_size
        if sp_size > 1:
            q_raw, k_raw, v_raw = torch.split(
                mixed_qkv,
                [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim],
                dim=-1,
            )

            sh = nk_full * sp_repeat
            G_per_sh = nv_full // sh

            q_shaped = q_raw.reshape(B, local_T, nk_full, dk)
            k_shaped = k_raw.reshape(B, local_T, nk_full, dk)
            if sp_repeat > 1:
                q_shaped = q_shaped.repeat_interleave(sp_repeat, dim=2)
                k_shaped = k_shaped.repeat_interleave(sp_repeat, dim=2)

            v_shaped = v_raw.reshape(B, local_T, sh, G_per_sh * dv)
            interleaved_qkv = torch.cat([q_shaped, k_shaped, v_shaped], dim=-1)

            interleaved_qkv = gather_seq_scatter_heads(
                interleaved_qkv,
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
                a2a_stream=a2a_stream,
            )
            local_qkv = interleaved_qkv.flatten(2)
        else:
            local_qkv = mixed_qkv

        z = self_attn.in_proj_z(hidden_states)
        b = self_attn.in_proj_b(hidden_states)
        a = self_attn.in_proj_a(hidden_states)
        if a2a_stream is not None:
            get_device_runtime().current_stream().wait_stream(a2a_stream)
        T = local_qkv.shape[1]

        if sp_size > 1:
            # Depthwise convolution preserves the exchanged [Q, K, grouped-V]
            # order by reordering only the short filters, not the activation.
            w = self_attn.conv1d.weight
            wq, wk, wv = torch.split(
                w, [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim], dim=0
            )

            K = self_attn.conv_kernel_size
            if sp_repeat > 1:
                wq = (
                    wq.reshape(nk_full, dk, 1, K)
                    .repeat_interleave(sp_repeat, dim=0)
                    .reshape(-1, 1, K)
                )
                wk = (
                    wk.reshape(nk_full, dk, 1, K)
                    .repeat_interleave(sp_repeat, dim=0)
                    .reshape(-1, 1, K)
                )

            local_conv_weight = _interleave_qkv_conv_parameter(
                _slice_param(wq, sp_size, sp_rank),
                _slice_param(wk, sp_size, sp_rank),
                _slice_param(wv, sp_size, sp_rank),
                nk,
            )

            local_conv_bias = None
            if self_attn.conv1d.bias is not None:
                bq, bk, bv = torch.split(
                    self_attn.conv1d.bias,
                    [self_attn.key_dim, self_attn.key_dim, self_attn.value_dim],
                    dim=0,
                )
                if sp_repeat > 1:
                    bq = (
                        bq.reshape(nk_full, dk)
                        .repeat_interleave(sp_repeat, dim=0)
                        .reshape(-1)
                    )
                    bk = (
                        bk.reshape(nk_full, dk)
                        .repeat_interleave(sp_repeat, dim=0)
                        .reshape(-1)
                    )
                local_conv_bias = _interleave_qkv_conv_parameter(
                    _slice_param(bq, sp_size, sp_rank),
                    _slice_param(bk, sp_size, sp_rank),
                    _slice_param(bv, sp_size, sp_rank),
                    nk,
                )
        else:
            local_conv_weight = self_attn.conv1d.weight
            local_conv_bias = self_attn.conv1d.bias

        local_qkv, _ = ops.causal_conv1d(
            x=local_qkv,
            weight=local_conv_weight.squeeze(1),
            bias=local_conv_bias,
            activation=self_attn.activation,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seq_lens,
        )

        if sp_size > 1:
            local_qkv = local_qkv.reshape(B, T, nk, -1)
            q = local_qkv[..., :dk]
            k = local_qkv[..., dk : 2 * dk]
            v = local_qkv[..., 2 * dk :].reshape(B, T, nv, dv)
        else:
            q, k, v = torch.split(local_qkv, [nk * dk, nk * dk, nv * dv], dim=-1)
            q = q.reshape(B, T, nk, dk)
            k = k.reshape(B, T, nk, dk)
            v = v.reshape(B, T, nv, dv)

        if nk < nv:
            r = nv // nk
            q = q.repeat_interleave(r, dim=2)
            k = k.repeat_interleave(r, dim=2)

        # Z is consumed only after ``out`` returns to local-sequence/full-head
        # layout, which is already Z's projection layout. Keep it local and
        # avoid a redundant round trip through sequence parallelism.
        local_z = z.reshape(B, local_T, nv_full, dv)
        if sp_size > 1:
            packed_ba = gather_seq_scatter_heads(
                torch.stack((b, a), dim=-1),
                seq_dim=1,
                head_dim=2,
                sp_group=sp_group,
                sp_size=sp_size,
            )
            b, a = packed_ba.unbind(dim=-1)

        beta = b.sigmoid()
        A_log_local = _slice_param(self_attn.A_log, sp_size, sp_rank)
        dt_bias_local = _slice_param(self_attn.dt_bias, sp_size, sp_rank)
        g = -A_log_local.float().exp() * F.softplus(a.float() + dt_bias_local)

        out, _ = ops.gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seq_lens,
        )

        if sp_size > 1:
            out = gather_heads_scatter_seq(
                out, seq_dim=1, head_dim=2, sp_group=sp_group, sp_size=sp_size
            )

        out = self_attn.norm(out.reshape(-1, dv), local_z.reshape(-1, dv))
        return self_attn.out_proj(out.reshape(B, local_T, -1))

    return sp_forward
