"""Compare independent FC reductions with the native all-to-all reference."""

import pytest
import torch
import torch.distributed as dist

from tests.ops.npu.test_virtual_expert_weight_exchange import (
    _assert_equal_on_all_ranks,
    _copies,
    weight_env,
)
from torchturbo.ops.providers.native.virtual_expert_weight_exchange import build_weight_exchange_context
from torchturbo.runtime import is_npu_available

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["ring", "multiple", "sparse", "empty"])
@pytest.mark.parametrize("shapes", [
    ((5, 7), (7, 4)),
    ((64, 64), (64, 32)),
    ((67, 65), (65, 64)),
], ids=["tail", "aligned", "full_and_tail"])
def test_single_fc_owner_gradients_and_arena_offsets(weight_env, dtype, case, shapes):
    dispatcher, group = weight_env
    runtime = dispatcher.device_runtime
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    copies = _copies(size, case)
    context = build_weight_exchange_context(copies, group)
    native = []
    for rows, columns in shapes:
        storage = torch.empty((3, rows * 2, columns * 2), device=runtime.device, dtype=dtype)
        view = storage[:, ::2, ::2]
        values = torch.arange(3 * rows * columns, device=runtime.device).reshape(3, rows, columns)
        view.copy_((values % 17 + rank).to(dtype))
        native.append(view)
    sizes = [value.numel() for value in native]
    scratch, handle = runtime.allocate_communication_buffer(
        sum(sizes), dtype=dtype, device=native[0].device, group=group,
    )
    scratch.fill_(-17)
    replicas = sum(context.receive_splits)
    # FC2 and FC1 use separate arena sections and may be submitted independently.
    for fc in (1, 0):
        shape = (replicas, *native[fc].shape[1:])
        replica_grad = torch.arange(replicas, device=native[fc].device, dtype=dtype)
        replica_grad = (replica_grad + rank + 1)[:, None, None].expand(shape)
        offset = sizes[0] if fc else 0
        expected = dispatcher.virtual_expert_grad_reduce(
            native[fc], replica_grad, context, grad_buffer=torch.empty_like(scratch),
            arena_offset=offset, impl="torch_virtual_expert_grad_reduce",
        )
        actual = dispatcher.virtual_expert_grad_reduce(
            native[fc], replica_grad, context, grad_buffer=scratch,
            arena_offset=offset, buffer_handle=handle,
            impl="npu_shmem_virtual_expert_grad_reduce",
        )
        _assert_equal_on_all_ranks(actual, expected, group)
        if fc == 1:
            _assert_equal_on_all_ranks(scratch[:sizes[0]], torch.full_like(scratch[:sizes[0]], -17), group)
