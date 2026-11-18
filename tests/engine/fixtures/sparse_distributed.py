from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_test_support as h
import torch
import torch.distributed as dist


def worker(args):
    torch.set_num_threads(1)
    if args.backend == "hccl":
        import torch_npu  # noqa: F401 - register the NPU backend.

        torch.npu.set_device(args.rank)
        device = f"npu:{args.rank}"
    else:
        device = "cpu"
    dist.init_process_group(
        args.backend,
        init_method=args.init,
        rank=args.rank,
        world_size=args.world,
        timeout=timedelta(seconds=180),
    )
    group = dist.group.WORLD
    mesh = None
    if args.fsdp:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("npu", (args.world,))
    cases = [
        ((67,), (0, 10, 11, 35, 62, 65), False),
        ((17, 1, 23, 9), (0, 15, 16, 37, 38, 45), True),
        ((19,), (), True),
    ]
    if args.world == 8:
        # Two ranks exceed the retired cap; six ranks must keep dummy gradients.
        active = tuple(range(5117)) + tuple(range(7 * 8192, 7 * 8192 + 4097))
        cases.append(((8 * 8192,), active, True))
    results = []
    for lengths, active, post in cases:
        core, packed, sp, Loss = h.load_runtime()
        torch.manual_seed(42)
        model = (h.small_qwen_model if args.mtp else h.SmallModel)(
            device=device,
            sp_rank=args.rank,
            sp_size=args.world,
            post_embedding=post,
            mix_context=post,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
        mb, info = h.make_mb(
            packed,
            sp,
            lengths=lengths,
            active=active,
            rank=args.rank,
            world=args.world,
            group=group,
            post_embedding=post,
            device=device,
            model=model,
        )
        candidate = copy.deepcopy(model)
        if mesh is not None:
            for m in (model, candidate):
                fully_shard(m.embed, mesh=mesh, reshard_after_forward=True)
                fully_shard(m.lm_head, mesh=mesh, reshard_after_forward=True)
                if args.mtp:
                    fully_shard(m.mtp.proj, mesh=mesh, reshard_after_forward=True)
                fully_shard(m, mesh=mesh, reshard_after_forward=True)
        a = h.run_batch(
            h.engine(core, model, info), copy.deepcopy(mb), Loss
        )
        b = h.run_batch(
            h.engine(core, candidate, info, sparse=True),
            copy.deepcopy(mb),
            Loss,
        )
        if mesh is not None:
            for out in (a, b):
                out["grads"] = {
                    k: (v.full_tensor() if v is not None else None)
                    for k, v in out["grads"].items()
                }
        h.check_equal(
            a,
            b,
            mb["orig_mb"]["ppo_loss_mask"],
            atol=2e-3 if args.bf16 else 3e-6,
            rtol=0.02 if args.bf16 else 3e-5,
        )
        row = dict(
            rank=args.rank,
            lengths=lengths,
            active_global=len(active),
            post_embedding=post,
            dense_rows=model.seen[-1][1],
            sparse_rows=candidate.seen[-1][1],
            loss=float(b["loss"].item()),
            max_gradient_abs_error=max(
                float((a["grads"][k] - b["grads"][k]).abs().max().item())
                for k in a["grads"]
                if a["grads"][k] is not None
            ),
            fsdp=args.fsdp,
            bf16=args.bf16,
            mtp=args.mtp,
        )
        if args.mtp:
            assert model.mtp.calls == 1 and candidate.mtp.calls == 0
            assert a["grads"]["mtp.proj.weight"] is None
            assert b["grads"]["mtp.proj.weight"] is None
            assert len(candidate.seen) == 1  # no extra dense lm_head
        results.append(row)
        # Exercise an optimizer step as well as backward. This is still a tiny
        # model test, not full-size policy training acceptance.
        opt = torch.optim.SGD(candidate.parameters(), lr=1e-3)
        before_mtp = candidate.mtp.proj.weight.detach().clone() if args.mtp else None
        opt.step()
        opt.zero_grad(set_to_none=True)
        if args.mtp:
            # Reuse FSDP after skipping its MTP child in the prior iteration.
            again = h.run_batch(
                h.engine(core, candidate, info, sparse=True),
                copy.deepcopy(mb),
                Loss,
            )
            assert candidate.mtp.calls == 0 and len(candidate.seen) == 2
            assert again["grads"]["mtp.proj.weight"] is None
            opt.step()
            opt.zero_grad(set_to_none=True)
            old = before_mtp.full_tensor() if mesh is not None else before_mtp
            new = (
                candidate.mtp.proj.weight.full_tensor()
                if mesh is not None
                else candidate.mtp.proj.weight
            )
            torch.testing.assert_close(old, new)
        dist.barrier()
    Path(args.out, f"rank{args.rank}.json").write_text(json.dumps(results, indent=2))
    dist.destroy_process_group()


def launch(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    with tempfile.TemporaryDirectory(prefix="native-sp-test-") as temp:
        init = "file://" + str(Path(temp, "rendezvous"))
        try:
            for rank in range(args.world):
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--rank",
                    str(rank),
                    "--world",
                    str(args.world),
                    "--backend",
                    args.backend,
                    "--init",
                    init,
                    "--out",
                    str(out),
                ]
                if args.fsdp:
                    cmd.append("--fsdp")
                if args.bf16:
                    cmd.append("--bf16")
                if args.mtp:
                    cmd.append("--mtp")
                log = open(out / f"rank{rank}.log", "w")
                logs.append(log)
                env = dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    HCCL_CONNECT_TIMEOUT="120",
                    HCCL_EXEC_TIMEOUT="120",
                )
                procs.append(
                    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                )
            deadline = time.monotonic() + 300
            while any(p.poll() is None for p in procs):
                bad = [p.returncode for p in procs if p.poll() not in (None, 0)]
                if bad:
                    raise RuntimeError(f"worker failed: {bad}; see {out}")
                if time.monotonic() > deadline:
                    raise TimeoutError("distributed test exceeded 300 s")
                time.sleep(1)
            assert all(p.returncode == 0 for p in procs)
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for log in logs:
                log.close()
    results = [
        row
        for rank in range(args.world)
        for row in json.loads((out / f"rank{rank}.json").read_text())
    ]
    summary = dict(
        status="PASS",
        backend=args.backend,
        world=args.world,
        fsdp=args.fsdp,
        bf16=args.bf16,
        mtp=args.mtp,
        cases=len(results),
        max_gradient_abs_error=max(r["max_gradient_abs_error"] for r in results),
        dummy_row_cases=sum(r["sparse_rows"] == 1 for r in results),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--init")
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    if a.rank < 0:
        launch(a)
    else:
        worker(a)
