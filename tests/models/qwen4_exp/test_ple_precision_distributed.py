"""Low-precision cancellation at real sequence-parallel convolution boundaries."""

from __future__ import annotations

from functools import partial
from types import MethodType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from torchturbo.models.qwen4_exp import ple as impl
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids

from .test_ple import adapted, new_reference
from .test_ple_distributed import _init_process_group


def _check_cancellation(rank, dtype, length, boundaries, reentrant):
    total = length * 4
    local_slice = slice(rank * length, (rank + 1) * length)
    reference = adapted(new_reference().to(dtype))
    with torch.no_grad():
        reference.value_proj.weight.zero_()
        reference.key_proj.weight.zero_()
        # Keep the large cancellation probe finite through RMSNorm in FP16.
        reference.norm_conv.weight.fill_(-0.99)
        reference.conv1d.weight.zero_()
        reference.conv1d.weight[:, 0, -2] = 1.0
        reference.conv1d.weight[:, 0, -1] = (
            1.0078125 if dtype == torch.bfloat16 else 1.0009765625
        )
    layer = adapted(new_reference().to(dtype))
    layer.load_state_dict(reference.state_dict())
    layer.forward = MethodType(
        partial(impl.ple_forward, sp_group=dist.group.WORLD, sp_size=4, sp_rank=rank),
        layer,
    )
    segments = cu_seqlens_to_segment_ids(torch.tensor(boundaries), total)
    rows = torch.zeros(1, total, reference.value_proj.in_features, dtype=dtype)
    rows[:, length - 1, 0] = 1
    batch = impl.PLEBatch((rows,), segments)
    local_batch = impl.PLEBatch((rows[:, local_slice],), segments)
    hidden = torch.zeros(1, total, 16, dtype=dtype, requires_grad=True)
    local_hidden = hidden[:, local_slice].detach().clone().requires_grad_()
    captured = {}

    def retain(name):
        def hook(module, args, output):
            if output.requires_grad and name not in captured:
                output.retain_grad()
                captured[name] = output

        return hook

    reference.norm_conv.register_forward_hook(retain("reference"))
    layer.norm_conv.register_forward_hook(retain("actual"))
    expected = reference(hidden, batch, None)
    fn = partial(layer, ple_inputs=local_batch, past_key_values=None)
    actual = (
        fn(local_hidden)
        if reentrant is None
        else checkpoint(fn, local_hidden, use_reentrant=reentrant)
    )
    torch.testing.assert_close(actual, expected[:, local_slice], atol=0, rtol=0)
    probe = torch.zeros_like(expected)
    probe[:, length - 1, 0] = 510 if dtype == torch.bfloat16 else 4094
    probe[:, length + 2, 0] = -512 if dtype == torch.bfloat16 else -4096
    expected.backward(probe)
    actual.backward(probe[:, local_slice])
    context = f"rank={rank}, dtype={dtype}, length={length}, checkpoint={reentrant}"
    if rank == 0:
        expected_boundary = 0.9921875 if dtype == torch.bfloat16 else 0.9990234375
        assert captured["reference"].grad[0, length - 1, 0].item() == expected_boundary
    torch.testing.assert_close(
        captured["actual"].grad,
        captured["reference"].grad[:, local_slice],
        atol=0,
        rtol=0,
        msg=context,
    )
    torch.testing.assert_close(local_hidden.grad, hidden.grad[:, local_slice])
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in layer.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            continue
        gradient = parameter.grad.float()
        dist.all_reduce(gradient)
        torch.testing.assert_close(
            gradient,
            reference_parameters[name].grad.float(),
            atol=0,
            rtol=0,
            msg=lambda detail, name=name: f"{context}, parameter={name}\n{detail}",
        )


def _run_cancellation(rank, init_file, dtype):
    _init_process_group(rank, 4, init_file)
    try:
        for length, boundaries in ((12, (0, 5, 30)), (4, (0, 2, 11)), (1, (0, 4))):
            for reentrant in (None, False, True):
                _check_cancellation(rank, dtype, length, boundaries, reentrant)
    finally:
        dist.destroy_process_group()


def _check_conv_gradients(rank, dtype, length, needs_grad):
    total = length * 4
    local_slice = slice(rank * length, (rank + 1) * length)
    torch.manual_seed(47)
    x = torch.randn(2, total, 6, dtype=dtype)[..., ::2]
    weight = torch.randn(3, 1, 8, dtype=dtype)[..., ::2] * 0.1
    segments = torch.tensor(
        [[1] * 2 + [2] * (total - 2), [1] * (total - 1) + [2]], dtype=torch.long
    )
    x.requires_grad_(needs_grad[0])
    weight.requires_grad_(needs_grad[1])
    local_x = x[:, local_slice].detach().requires_grad_(needs_grad[0])
    local_weight = weight.detach().clone().requires_grad_(needs_grad[1])
    saved_storage = {}

    def record(tensor):
        if tensor.is_floating_point():
            assert tensor.dtype == dtype
            storage = tensor.untyped_storage()
            saved_storage[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        actual = impl.segment_causal_conv(
            local_x,
            segments,
            local_weight,
            3,
            sp_group=dist.group.WORLD,
            sp_size=4,
            sp_rank=rank,
        )
    # The convolution saves local inputs plus at most H=9 left tokens; SiLU
    # saves a local output. Gathered tensors from unrelated ranks cannot remain.
    saved_elements = (2 * length + min(9, rank * length)) * 2 * 3 + weight.numel()
    assert sum(saved_storage.values()) <= saved_elements * x.element_size()
    expected = impl.segment_causal_conv(x, segments, weight, 3)
    torch.testing.assert_close(actual, expected[:, local_slice])
    gradient = torch.randn_like(expected)
    actual.backward(gradient[:, local_slice])
    expected.backward(gradient)
    if needs_grad[0]:
        torch.testing.assert_close(local_x.grad, x.grad[:, local_slice])
    else:
        assert local_x.grad is None
    if needs_grad[1]:
        # The SP weight gradient must count exactly the outputs on this rank.
        local_probe = torch.zeros_like(gradient)
        local_probe[:, local_slice] = gradient[:, local_slice]
        weight.grad = None
        impl.segment_causal_conv(x, segments, weight, 3).backward(local_probe)
        torch.testing.assert_close(local_weight.grad, weight.grad)
    else:
        assert local_weight.grad is None


def _run_conv_gradients(rank, init_file):
    _init_process_group(rank, 4, init_file)
    try:
        impl._PLE_CONV_WINDOW_TOKENS = 5
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            for length in (12, 4, 1):
                for needs_grad in ((True, True), (True, False), (False, True)):
                    _check_conv_gradients(rank, dtype, length, needs_grad)
            _check_conv_gradients(rank, dtype, 4, (rank % 2 == 0, True))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="requires Gloo"
)
def test_sp_conv_owned_gradients_and_saved_storage(tmp_path):
    mp.spawn(
        _run_conv_gradients,
        args=(str(tmp_path / "ple-conv-gradients"),),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="requires Gloo"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ple_sp_boundary_cancellation_matches_production_sp1(tmp_path, dtype):
    mp.spawn(
        _run_cancellation,
        args=(str(tmp_path / "ple-precision"), dtype),
        nprocs=4,
        join=True,
    )
