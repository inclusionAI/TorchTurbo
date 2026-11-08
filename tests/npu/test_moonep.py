"""Official MoonEP planning and FP32 return contracts.

Run with ``torchrun --standalone --nproc-per-node=4 -m pytest -q
 tests/npu/test_moonep.py`` after installing the optional dependencies.
"""
import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch_npu")
    pytest.importorskip("npu_patch")
    pytest.importorskip("hyper_multicore")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("Launch with torchrun and at least two EP ranks")
    from hyper_multicore import shmem
    from torchturbo.optimizations.moonep.official_planner import OfficialPlanner, PlanningScratch
    from torchturbo.optimizations.moonep.official_transfer import ExpertTransfer

    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", rank)
    size = dist.get_world_size()
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    planner = OfficialPlanner(size * 3, 2, size, dist.get_rank(), cores)
    _, scratch_bytes = planner.workspace_sizes(1024)
    bank_bytes = 2 * 3 * 7 * 13 * 2
    page_bytes = 2 * 1024 * 1024
    required = (scratch_bytes + bank_bytes + 1024 + page_bytes - 1) // page_bytes * page_bytes
    shmem.acquire(dist.group.WORLD, heap_size_bytes=required)
    scratch = PlanningScratch(planner, 1024, device, allocator=shmem)
    # Odd expert counts and matrix dimensions must not change FP32 slot aliasing.
    transfer = ExpertTransfer(
        dist.group.WORLD, size * 3, ((7, 13),), device, allocator=shmem
    )
    transfer.bind()
    try:
        yield planner, scratch, transfer, device
    finally:
        torch.npu.synchronize()
        shmem.host_barrier()
        transfer.close()
        scratch.close()
        shmem.release()
        dist.destroy_process_group()


@pytest.mark.parametrize("counts_only", [False, True])
def test_alternating_plans_and_unquantized_owner_gradients(runtime, counts_only):
    planner, scratch, transfer, device = runtime
    rank, size, local = dist.get_rank(), dist.get_world_size(), 3
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    for iteration, tokens in enumerate((257, 1024, 513, 257)):
        ids = torch.arange(2, device=device, dtype=torch.int32).expand(tokens, 2).contiguous()
        counts = torch.bincount(ids.flatten().long(), minlength=size * local).to(torch.int32)
        all_counts = counts.new_empty((size, size * local))
        dist.all_gather_into_tensor(all_counts.view(-1), counts)
        options = {}
        if counts_only:
            options["source_inverse"] = ids.flatten().argsort(stable=True).argsort(stable=True)
            options["source_counts"] = tuple(map(tuple, all_counts.cpu().tolist()))
        plan = planner(None, ids, all_counts, tokens * 2, rank,
                       resources=scratch.inputs(), source_rows=(tokens * 2,) * size,
                       **options)
        assert plan is not None, "Hot routing must exercise actual expert copies"
        virtual = plan.virtual_ids
        table = torch.cat((torch.arange(size * local, device=device).reshape(size, local),
                           plan.copies), dim=1)
        torch.testing.assert_close(table.flatten()[virtual.long()], ids.long(), rtol=0, atol=0)
        histogram = torch.bincount(virtual.flatten().long(), minlength=size * (local + plan.borrowed_capacity))
        dist.all_reduce(histogram)
        assert int(histogram.reshape(size, local + plan.borrowed_capacity).sum(1).max()) <= tokens * 2
        home_ids = torch.arange(rank * local, (rank + 1) * local, device=device)
        weights = (home_ids + iteration).to(torch.bfloat16)[:, None, None].expand(local, 7, 13).contiguous()
        transfer.claim()
        bank = transfer.prefetch((weights,), plan.copies)[0]
        copies = plan.copies.cpu()
        for slot, expert in enumerate(copies[rank].tolist()):
            if expert >= 0:
                torch.testing.assert_close(bank[local + slot],
                    torch.full_like(bank[local + slot], expert + iteration), rtol=0, atol=0)
        # Exact powers-of-two make the added sub-BF16 increments observable.
        value = (rank + 1) / 65536
        gradients = torch.full(bank.shape, value, dtype=torch.float32, device=device)
        returned = transfer.return_generated_gradients(
            (gradients,), plan.copies, output_dtypes=(torch.float32,)
        )[0]
        for expert in range(local):
            expected = value
            global_expert = rank * local + expert
            for peer in range(size):
                expected += int((copies[peer] == global_expert).sum()) * (peer + 1) / 65536
            torch.testing.assert_close(returned[expert],
                torch.full_like(returned[expert], expected), rtol=0, atol=0)
        transfer.release()
