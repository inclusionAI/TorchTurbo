"""Packed QSA outputs and gradients across independent SP process groups."""

from __future__ import annotations

import copy
import os
import sys
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers.modeling_flash_attention_utils as flash_utils
from torch.utils.checkpoint import checkpoint

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.models.modules.attention.ulysses import all_gather_sp_with_grad_reduce
from torchturbo.models.qwen4_exp import qsa
from torchturbo.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention
from torchturbo.runtime.npu import NpuRuntime

from ..flash_attention_fakes import lazy_flash_attention_kernels
from .test_model import _small_config
from .test_qsa_attention import _cpu_varlen_kernel, _flash_kwargs
from .test_qsa_varlen import _reference_attention


def _check_qsa(
    group: dist.ProcessGroup,
    group_id: int,
    lengths: tuple[int, ...],
    *,
    ratio12: bool = False,
    checkpoint_mode: bool | None = None,
) -> None:
    torch.manual_seed(30 + group_id)
    config = _small_config().text_config
    config._attn_implementation = "sdpa"
    # Neither the indexer nor the main KV heads need to divide the SP size.
    config.indexer_n_heads = 5
    if ratio12:
        config.num_attention_heads = 24
        config.num_key_value_heads = 2
    reference = Qwen4ExpTextAttention(config, layer_idx=3).double()
    parallel = copy.deepcopy(reference)
    global_length = 32
    local_length = global_length // dist.get_world_size(group)
    local_rank = dist.get_rank(group)
    local_slice = slice(local_rank * local_length, (local_rank + 1) * local_length)
    hidden = torch.randn(1, global_length, config.hidden_size, dtype=torch.double)
    hidden.requires_grad_()
    local_hidden = hidden[:, local_slice].detach().clone().requires_grad_()
    phases = torch.randn(1, global_length, config.head_dim // 2).repeat(1, 1, 2)
    cos, sin = phases.cos().double(), phases.sin().double()

    # Build the reference mask directly, without production packing helpers.
    mask = torch.zeros(1, 1, global_length, global_length, dtype=torch.bool)
    boundaries = [0]
    for length in lengths:
        start = boundaries[-1]
        stop = start + length
        mask[:, :, start:stop, start:stop] = torch.ones(length, length).tril().bool()
        boundaries.append(stop)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    expected, _ = reference(hidden, (cos, sin), mask)
    probe = torch.randn_like(expected)
    expected.backward(probe)

    def forward(hidden_states):
        return qsa.qsa_forward(
            parallel,
            hidden_states,
            (cos[:, local_slice], sin[:, local_slice]),
            None,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            sp_group=group,
            sp_size=dist.get_world_size(group),
            sp_rank=local_rank,
        )[0]

    actual = (
        forward(local_hidden)
        if checkpoint_mode is None
        else checkpoint(forward, local_hidden, use_reentrant=checkpoint_mode)
    )
    context = f"group={group_id}, local_rank={local_rank}, lengths={lengths}"

    def failure_message(message: str) -> str:
        return f"{context}\n{message}"

    torch.testing.assert_close(actual, expected[:, local_slice], msg=failure_message)
    actual.backward(probe[:, local_slice])
    torch.testing.assert_close(
        local_hidden.grad,
        hidden.grad[:, local_slice],
        msg=failure_message,
    )
    for (name, parameter), (_, reference_parameter) in zip(
        parallel.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        if name.startswith("indexer."):
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, (context, name)
        dist.all_reduce(parameter.grad, group=group)
        # RMSNorm computes in FP32 even when the model's parameters are double.
        tolerances = {"atol": 1e-6, "rtol": 1e-5} if "norm.weight" in name else {}
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            msg=failure_message,
            **tolerances,
        )


def _run_qsa(rank: int, init_file: str, ratio12: bool) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME",
        "lo0" if sys.platform == "darwin" else "lo",
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        group_id = rank % 2
        kernels = lazy_flash_attention_kernels(None, _cpu_varlen_kernel)
        with (
            patch.object(
                flash_utils,
                "lazy_import_flash_attention",
                return_value=(kernels, _flash_kwargs),
            ),
            patch.object(qsa, "_QSA_INDEXER_QUERY_CHUNK_SIZE", 3),
            patch.object(qsa, "_QSA_ATTENTION_MAX_KV_BYTES", 8192),
        ):
            # Cross-rank samples, an entirely padded rank, and no padding.
            checkpoint_modes = (None, False, True)
            for checkpoint_mode in checkpoint_modes:
                for lengths in ((3, 24, 2), (3, 8), (5, 27)):
                    _check_qsa(
                        groups[group_id],
                        group_id,
                        lengths,
                        ratio12=ratio12,
                        checkpoint_mode=checkpoint_mode,
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize("ratio12", [False, True], ids=["gqa4", "gqa12"])
def test_qsa_matches_transformers_across_sp_groups(tmp_path, ratio12) -> None:
    mp.spawn(
        _run_qsa,
        args=(str(tmp_path / "qsa-sp"), ratio12),
        nprocs=4,
        join=True,
    )


def _run_native_qsa_varlen(rank: int, init_file: str, dtype: torch.dtype) -> None:
    import torch_npu  # noqa: F401 - registers the NPU device and HCCL backend.

    torch.set_num_threads(1)
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AI_CPU")
    torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        with patch.object(runtime_impl, "_SELECTED_RUNTIME", NpuRuntime()):
            for checkpoint_mode in (None, False, True):
                for lengths in ((3, 24, 2), (3, 8), (5, 27)):
                    _check_native_qsa_varlen(rank, dtype, lengths, checkpoint_mode)
            # Keep the faster rank from tearing down HCCL while its peer is
            # still reading asynchronous reduce-scatter results.
            torch.npu.synchronize()
            dist.barrier()
            torch.npu.synchronize()
    finally:
        dist.destroy_process_group()


def _check_native_qsa_varlen(
    rank: int,
    dtype: torch.dtype,
    lengths: tuple[int, ...],
    checkpoint_mode: bool | None,
) -> None:
    torch.manual_seed(64)
    inputs = [
        torch.randn(1, 32, heads, 256).to(dtype).float().requires_grad_()
        for heads in (24, 2, 2)
    ]
    local_slice = slice(rank * 16, (rank + 1) * 16)
    local_inputs = [
        tensor[:, local_slice].detach().to("npu", dtype=dtype).requires_grad_()
        for tensor in inputs
    ]
    ids = torch.full((1, 32, 2051), -1, dtype=torch.int32)
    start = 0
    for length in lengths:
        for row in range(start, start + length):
            ids[0, row, : row - start + 1] = torch.arange(start, row + 1)
        start += length
    expected = _reference_attention(inputs, ids) * inputs[0].sigmoid()
    probe = torch.randn_like(expected).to(dtype).float()
    expected.backward(probe)
    local_ids = ids[:, local_slice].to("npu")

    def forward(query, key, value):
        key_value = all_gather_sp_with_grad_reduce(
            torch.cat((key, value), dim=-1), 1, dist.group.WORLD, 2
        )
        global_key, global_value = key_value.chunk(2, dim=-1)
        output = qsa.qsa_attention_forward(query, global_key, global_value, local_ids)
        # Production QSA gates every local row before its output projection.
        # Retain that activation dependency on padding-only ranks too, so
        # non-reentrant checkpointing recomputes the collective on both peers.
        return output * query.sigmoid()

    actual = (
        forward(*local_inputs)
        if checkpoint_mode is None
        else checkpoint(forward, *local_inputs, use_reentrant=checkpoint_mode)
    )
    actual.backward(probe[:, local_slice].to("npu", dtype=dtype))
    context = (
        f"rank={rank}, dtype={dtype}, lengths={lengths}, checkpoint={checkpoint_mode}"
    )
    torch.testing.assert_close(
        actual.float().cpu(),
        expected[:, local_slice],
        rtol=0.02,
        atol=0.02,
        msg=context,
    )
    for local_input, reference in zip(local_inputs, inputs, strict=True):
        torch.testing.assert_close(
            local_input.grad.float().cpu(),
            reference.grad[:, local_slice],
            rtol=0.03,
            atol=0.03,
            msg=context,
        )


@pytest.mark.skipif(
    os.environ.get("TORCH_TURBO_TEST_QSA_VARLEN_NPU") != "1",
    reason="opt-in native NPU varlen attention with HCCL sequence parallelism",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_varlen_ratio12_packed_sp2(tmp_path, dtype) -> None:
    pytest.importorskip("torch_npu")
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPUs")
    mp.spawn(
        _run_native_qsa_varlen,
        args=(str(tmp_path / "qsa-native-sp"), dtype),
        nprocs=2,
        join=True,
    )
