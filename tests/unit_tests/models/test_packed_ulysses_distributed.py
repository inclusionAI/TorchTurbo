"""Packed attention outputs and gradients across real Ulysses collectives."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from itertools import product
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import transformers.modeling_flash_attention_utils as flash_utils
from torch import Tensor

from torchturbo.models.modules.attention import interface
from torchturbo.models.modules.attention.qk_clip import QKClipState
from torchturbo.models.modules.attention.ulysses.ulysses import (
    ulysses_flash_attention_forward,
)
from torchturbo.ops import OpDispatcher, registry
from torchturbo.runtime.cpu import CpuRuntime

from .flash_attention_fakes import lazy_flash_attention_kernels, process_flash_kwargs


def _cpu_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    *,
    causal: bool,
    softmax_scale: float,
    **kwargs,
) -> Tensor:
    # Only replace the unavailable FlashAttention kernel; HF dispatch stays real.
    assert query.shape[0] == int(cu_seqlens_q[-1])
    assert key.shape[0] == value.shape[0] == int(cu_seqlens_k[-1])
    outputs = []
    for q_start, q_end, k_start, k_end in zip(
        cu_seqlens_q[:-1], cu_seqlens_q[1:],
        cu_seqlens_k[:-1], cu_seqlens_k[1:],
        strict=True,
    ):
        output = F.scaled_dot_product_attention(
            query[q_start:q_end].transpose(0, 1),
            key[k_start:k_end].transpose(0, 1),
            value[k_start:k_end].transpose(0, 1),
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(output.transpose(0, 1))
    return torch.cat(outputs)


def _reference_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    lengths: tuple[int, ...],
    is_causal: bool,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Evaluate samples independently, without production packing utilities."""
    outputs, maxima = [], []
    start = 0
    for length in lengths:
        q, k, v = (
            tensor[:, start:start + length].transpose(1, 2)
            for tensor in (query, key, value)
        )
        outputs.append(F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale,
        ).transpose(1, 2))
        scores = (q.detach() @ k.detach().transpose(-1, -2)) * scale
        if is_causal:
            future = torch.ones(length, length, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(future, -torch.inf)
        maxima.append(scores.amax(dim=(0, 2, 3)))
        start += length
    output = torch.cat(outputs, dim=1)
    output = F.pad(output, (0, 0, 0, 0, 0, query.shape[1] - start))
    return output, torch.stack(maxima).amax(dim=0)


def _check_packed_attention(
    rank: int, num_valid_tokens: int, is_causal: bool, collect_maxima: bool,
) -> None:
    torch.manual_seed(num_valid_tokens)
    query, key, value = (torch.randn(1, 8, 4, dim) for dim in (2, 2, 3))
    for tensor in (query, key, value):
        # A leaked tail would dominate both attention and maximum-logit statistics.
        tensor[:, num_valid_tokens:] = 1000.0
        tensor.requires_grad_()
    lengths = (2, num_valid_tokens - 2)
    expected, expected_maxima = _reference_attention(
        query, key, value, lengths, is_causal, scale=0.7,
    )
    output_gradient = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(
        expected, (query, key, value), output_gradient,
    )

    local_slice = slice(rank * 4, (rank + 1) * 4)
    local_inputs = [
        tensor[:, local_slice].detach().clone().requires_grad_()
        for tensor in (query, key, value)
    ]
    state = QKClipState(torch.full((4,), -torch.inf)) if collect_maxima else None
    cu_seqlens = torch.tensor([0, 2, num_valid_tokens], dtype=torch.int32)
    actual = ulysses_flash_attention_forward(
        *local_inputs,
        None,
        4,
        is_causal,
        position_ids=torch.arange(4).unsqueeze(0),
        sp_group=dist.group.WORLD,
        sp_size=2,
        qk_clip_state=state,
        softmax_scale=0.7,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max(lengths),
        max_length_k=max(lengths),
    )
    context = (
        f"rank={rank}, valid={num_valid_tokens}, "
        f"causal={is_causal}, maxima={collect_maxima}"
    )

    def failure_message(message: str) -> str:
        return f"{context}\n{message}"

    torch.testing.assert_close(
        actual, expected[:, local_slice], msg=failure_message,
    )
    actual.backward(output_gradient[:, local_slice])
    for local, expected_gradient in zip(local_inputs, expected_gradients, strict=True):
        torch.testing.assert_close(
            local.grad, expected_gradient[:, local_slice], msg=failure_message,
        )
    if state is not None:
        assert state.observed
        local_maxima = torch.full((4,), -torch.inf)
        head_slice = slice(rank * 2, (rank + 1) * 2)
        local_maxima[head_slice] = expected_maxima[head_slice]
        torch.testing.assert_close(state.max_logits, local_maxima, msg=failure_message)
        dist.all_reduce(state.max_logits, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(state.max_logits, expected_maxima, msg=failure_message)


def _run_packed_attention(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo",
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        cpu_ops = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
        flash_kernels = lazy_flash_attention_kernels(None, _cpu_varlen_attention)
        with (
            patch.object(interface, "ops", cpu_ops),
            patch.object(
                flash_utils, "lazy_import_flash_attention",
                return_value=(flash_kernels, process_flash_kwargs),
            ),
        ):
            # Cover an entirely padded rank, a sequence crossing ranks, and no padding.
            for num_valid_tokens, is_causal, collect_maxima in product(
                (3, 7, 8), (False, True), (False, True),
            ):
                _check_packed_attention(rank, num_valid_tokens, is_causal, collect_maxima)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
def test_packed_ulysses_matches_independent_samples_across_ranks(tmp_path) -> None:
    mp.spawn(
        _run_packed_attention,
        args=(str(tmp_path / "packed-attention"),),
        nprocs=2,
        join=True,
    )
