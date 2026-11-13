"""Verify SP>1 KV-gather gradients (reduce-scatter) match the sp=1 reference."""

import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torchturbo.models.modules.attention.ulysses import (  # noqa: E402
    all_gather_sp_with_grad_reduce,
)

D, L = 32, 256
HALF = L // 2
SCALE = D**-0.5
SEED = 1234


def _attn(q, kv, q_start):
    """Causal attention: q [1,Lq,D] local, kv [1,Lkv,D] global, q offset q_start."""
    scores = torch.einsum("bid,bjd->bij", q, kv) * SCALE
    qi = torch.arange(q_start, q_start + q.size(1), device=q.device).unsqueeze(1)
    kj = torch.arange(kv.size(1), device=q.device).unsqueeze(0)
    scores = scores.masked_fill(kj > qi, float("-inf"))
    p = torch.softmax(scores.float(), dim=-1)
    return torch.einsum("bij,bjd->bid", p, kv).to(q.dtype)


def _lin(weight, device):
    m = torch.nn.Linear(D, D, bias=False).to(device)
    with torch.no_grad():
        m.weight.copy_(weight)
    return m


def _reference_grad(x_full, wq_w, wkv_w, device):
    """sp=1 full-graph ground truth, identical on every rank."""
    wq, wkv = _lin(wq_w, device), _lin(wkv_w, device)
    o = _attn(wq(x_full), wkv(x_full), 0)
    ((o.float() ** 2).sum()).backward()
    return wkv.weight.grad.detach().clone()


def _worker(rank, world_size, init_file, device_kind):
    device, backend = "cpu", "gloo"
    if device_kind == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
        device, backend = "npu", "hccl"
    dist.init_process_group(
        backend, init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )

    torch.manual_seed(SEED)
    x_full = torch.randn(1, L, D)
    wq_w = torch.randn(D, D)
    wkv_w = torch.randn(D, D)
    x_local = x_full[:, rank * HALF : (rank + 1) * HALF].clone().to(device)
    x_full_d = x_full.to(device)

    ref = _reference_grad(x_full_d, wq_w, wkv_w, device)

    # sp=2 graph: local queries x gathered KV; all-reduce(SUM) emulates DDP.
    wq, wkv = _lin(wq_w, device), _lin(wkv_w, device)
    kv_full = all_gather_sp_with_grad_reduce(
        wkv(x_local), dim=1, sp_group=dist.group.WORLD, sp_size=world_size
    )
    o = _attn(wq(x_local), kv_full, rank * HALF)
    ((o.float() ** 2).sum()).backward()
    grad = wkv.weight.grad.detach().clone()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    rel_err = ((grad.float() - ref.float()).norm() / ref.float().norm()).item()
    if rank == 0:
        print(f"[sp-kv-grad-ut] device={device} backend={backend} rel_err={rel_err:.6f}")
    assert rel_err < 1e-4, f"sp=2 grads differ from sp=1, rel_err={rel_err}"

    dist.barrier()
    dist.destroy_process_group()


def _npu_available():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.device_count() >= 2
    except (ImportError, Exception):
        return False


def test_sp_kv_grad(tmp_path):
    device_kind = "npu" if _npu_available() else "cpu"
    mp.spawn(
        _worker,
        args=(2, str(tmp_path / "init"), device_kind),
        nprocs=2,
        join=True,
    )
