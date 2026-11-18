"""CPU checks of the unmodified internal EP row owner protocol."""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from torchturbo.models.deepseek_v41.host_offload_embedding import TorchHostOffloadEmbedding
from torchturbo.engine.backends.fsdp2.optimizers.engram import EngramOptimizer


def _worker(rank, root):
    dist.init_process_group("gloo", init_method=f"file://{root}/dist", rank=rank, world_size=2)
    groups = [dist.new_group([owner]) for owner in range(2)]
    try:
        table = TorchHostOffloadEmbedding(
            4096, 4, layer_id=0, num_max_tokens_per_rank=32, pin_memory=False,
        ).shard_(dist.group.WORLD)
        initial = torch.arange(4096 * 4, dtype=torch.float32).reshape(4096, 4).div(100).bfloat16()
        with torch.no_grad():
            table.weight.copy_(initial[table.row_start:table.row_start + table.local_rows])
        table.hash_fingerprint = "source-hash-layout"
        optimizer = EngramOptimizer((table,), groups[rank], torch.device("cpu"))
        dense = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.01)
        dense.register_step_pre_hook(lambda *_: optimizer.prepare_gradients())
        dense.register_step_post_hook(optimizer.after_step)
        reference = torch.nn.Parameter(initial.float())
        expected = torch.optim.SparseAdam([reference], lr=0.01)
        for batches in (([2, 2048, 2049], [3, 2048]), ([], [2, 2049])):
            table(torch.tensor(batches[rank], dtype=torch.long)).sum().backward()
            rows = torch.tensor(batches[0] + batches[1])
            gradient = torch.sparse_coo_tensor(rows.unsqueeze(0), torch.ones(rows.numel(), 4) / 2, initial.shape).coalesce()
            reference.grad = gradient
            expected.step()
            with torch.no_grad():
                reference.copy_(reference.bfloat16().float())
            optimizer.prepare_gradients()
            torch.testing.assert_close(optimizer.norm(), gradient.values().norm())
            dense.step()
            torch.testing.assert_close(table.weight, reference.detach()[table.row_start:table.row_start+table.local_rows].bfloat16())
            assert not table.pending_gradients and table.weight.grad is None
        optimizer.save(root)
        assert (Path(root) / f"engram_optimizer_ep{rank}.pt").is_file()
        steps = optimizer.states[0].step_count
        optimizer.load(root)
        assert optimizer.states[0].step_count == steps
        table.weight.requires_grad_(False)
        assert not table(torch.tensor([2, 2048])).requires_grad
    finally:
        dist.destroy_process_group()


def test_ep_owner_gradients_and_optimizer_files(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
