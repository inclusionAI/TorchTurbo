"""GDN sequence-parallel values and gradients against serial packed samples."""

from __future__ import annotations

import copy
import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import torchturbo.ops as turbo_ops
from torchturbo.models.qwen3_5 import modeling_qwen3_5 as dense
from torchturbo.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
from torchturbo.ops.dispatcher import OpDispatcher
from torchturbo.runtime.cpu import CpuRuntime


def _cpu_dispatch(op_name, *args, packed_params=None, **kwargs):
    boundaries = (
        packed_params.cu_seqlens_list
        if packed_params is not None
        else [0, args[0].size(1)]
    )
    outputs = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if op_name == "causal_conv1d":
            x, weight, bias = args
            output = F.conv1d(
                x[:, start:stop].transpose(1, 2),
                weight.unsqueeze(1),
                bias,
                padding=weight.size(-1) - 1,
                groups=x.size(-1),
            )[..., : stop - start]
            output = F.silu(output).transpose(1, 2)
        else:
            assert op_name == "gated_delta_rule"
            q, k, v, g, beta = args
            output, _ = dense.torch_recurrent_gated_delta_rule(
                q[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                g[:, start:stop],
                beta[:, start:stop],
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"],
            )
        outputs.append(output)
    return torch.cat(outputs, dim=1), None


def _check_forward(factory, group, num_key_heads, attention_cls):
    torch.manual_seed(31 + num_key_heads)
    config = SimpleNamespace(
        hidden_size=8,
        linear_num_value_heads=4,
        linear_num_key_heads=num_key_heads,
        linear_key_head_dim=3,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    reference = attention_cls(config, layer_idx=0)
    reference.conv1d.bias = torch.nn.Parameter(torch.randn(reference.conv_dim))
    parallel = copy.deepcopy(reference)
    sp_size = dist.get_world_size(group) if group is not None else 1
    sp_rank = dist.get_rank(group) if group is not None else 0
    forward = factory(group, sp_size, sp_rank)

    hidden = torch.randn(1, 8, config.hidden_size, requires_grad=True)
    local_slice = slice(sp_rank * 8 // sp_size, (sp_rank + 1) * 8 // sp_size)
    local_hidden = hidden[:, local_slice].detach().clone().requires_grad_()
    # A sample crosses the SP boundary, and the final token is padding.
    # The serial reference resets both convolution and recurrence per sample.
    expected = torch.cat(
        [reference(hidden[:, start:stop]) for start, stop in ((0, 3), (3, 7), (7, 8))],
        dim=1,
    )
    probe = torch.randn_like(expected)
    expected.backward(probe)
    actual = forward(
        parallel,
        local_hidden,
        cu_seq_lens=torch.tensor([0, 3, 7], dtype=torch.int32),
    )
    torch.testing.assert_close(actual, expected[:, local_slice], atol=1e-6, rtol=1e-5)
    actual.backward(probe[:, local_slice])
    torch.testing.assert_close(
        local_hidden.grad, hidden.grad[:, local_slice], atol=1e-6, rtol=1e-5
    )
    for (name, parameter), (_, reference_parameter) in zip(
        parallel.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None, name
        if group is not None:
            dist.all_reduce(parameter.grad, group=group)
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
            msg=lambda message: f"{factory.__name__}, {name}: {message}",
        )


def _run_cases(group):
    from torchturbo.models.modules.attention.gdn import build_gdn_sp_forward
    from torchturbo.models.modules.attention.gdn import forward as gdn_forward
    from torchturbo.models.modules.attention.ulysses import ulysses

    dispatcher = OpDispatcher.__new__(OpDispatcher)
    runtime = CpuRuntime()
    exchanges = []
    gather_sequence = ulysses.gather_seq_scatter_heads
    gather_heads = ulysses.gather_heads_scatter_seq

    def exchange_sequence(x, *args, **kwargs):
        exchanges.append("sequence")
        return gather_sequence(x, *args, **kwargs)

    def exchange_heads(x, *args, **kwargs):
        exchanges.append("heads")
        return gather_heads(x, *args, **kwargs)

    # Only accelerator kernels are replaced; collectives and the
    # public dispatcher's packed metadata normalization remain real.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(dispatcher, "_dispatch", _cpu_dispatch)
        monkeypatch.setattr(turbo_ops, "ops", dispatcher)
        monkeypatch.setattr(dense, "FusedRMSNormGated", None)
        monkeypatch.setattr(moe, "FusedRMSNormGated", None)
        monkeypatch.setattr(gdn_forward, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(ulysses, "gather_seq_scatter_heads", exchange_sequence)
        monkeypatch.setattr(ulysses, "gather_heads_scatter_seq", exchange_heads)
        for attention_cls in (
            dense.Qwen3_5GatedDeltaNet,
            moe.Qwen3_5MoeGatedDeltaNet,
        ):
            for num_key_heads in (1, 2, 4):
                exchanges.clear()
                _check_forward(build_gdn_sp_forward, group, num_key_heads, attention_cls)
                expected = ["sequence", "sequence", "heads"] if group is not None else []
                assert exchanges == expected


def test_sp1_reuses_projection_and_conv_parameters_without_streams(monkeypatch):
    from torchturbo.models.modules.attention.gdn import build_gdn_sp_forward
    from torchturbo.models.modules.attention.gdn import forward as gdn_forward

    config = SimpleNamespace(
        hidden_size=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    monkeypatch.setattr(dense, "FusedRMSNormGated", None)
    attention = dense.Qwen3_5GatedDeltaNet(config, layer_idx=0)
    attention.conv1d.bias = torch.nn.Parameter(torch.randn(attention.conv_dim))
    projected = []
    attention.in_proj_qkv.register_forward_hook(
        lambda module, inputs, output: projected.append(output)
    )

    def dispatch(op_name, *args, **kwargs):
        if op_name == "causal_conv1d":
            x, weight, bias = args
            assert x is projected[0]
            assert weight.data_ptr() == attention.conv1d.weight.data_ptr()
            assert bias is attention.conv1d.bias
        return _cpu_dispatch(op_name, *args, **kwargs)

    def unexpected_runtime():
        pytest.fail("SP=1 must not access communication streams")

    dispatcher = OpDispatcher.__new__(OpDispatcher)
    monkeypatch.setattr(dispatcher, "_dispatch", dispatch)
    monkeypatch.setattr(turbo_ops, "ops", dispatcher)
    monkeypatch.setattr(gdn_forward, "get_device_runtime", unexpected_runtime)
    forward = build_gdn_sp_forward(None, 1, 0)
    output = forward(attention, torch.randn(1, 8, config.hidden_size))
    assert output.shape == (1, 8, config.hidden_size)


def _worker(rank, init_file):
    torch.set_num_threads(1)
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        _run_cases(dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_gdn_sp1_matches_independent_packed_samples():
    _run_cases(None)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_gdn_sp2_matches_serial_outputs_and_gradients(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "init"),), nprocs=2, join=True)
