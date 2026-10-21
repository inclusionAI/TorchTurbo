"""Compare standalone MegaMoE forward/backward with ordinary NPU experts."""
import json
import argparse
import os

import torch
import torch.distributed as dist
import torch_npu

from tt_megamoe import MegaMoeExperts


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.detach().float(), expected.detach().float()
    return {
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
        "max_abs": (a - b).abs().max().item(),
        "finite": bool(torch.isfinite(a).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "pre_activation", "post_activation"], default="none")
    parser.add_argument("--limit", type=float, default=0.)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--routing", choices=["uniform", "hot"], default="uniform")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    t, h, intermediate, experts, k = 128, 256, 256, 8, 2
    limit = args.limit
    mode = args.mode
    identity = args.identity
    if identity:
        h = 2 * intermediate
    local_experts = experts // world
    device = torch.device("npu", rank)
    torch.manual_seed(8181)
    x_cpu = (torch.randn(world * t, h) * 0.5).bfloat16()
    ids_cpu = torch.rand(world * t, experts).topk(k, dim=-1).indices.sort(-1).values
    if args.routing == "hot":
        ids_cpu[:] = torch.arange(k)
    probs_cpu = torch.randn(world * t, k).softmax(-1)
    w1_cpu = (torch.randn(experts, h, 2 * intermediate) * 0.03).bfloat16()
    w2_cpu = (torch.randn(experts, intermediate, h) * 0.03).bfloat16()
    if identity:
        w1_cpu = torch.eye(h).bfloat16().repeat(experts, 1, 1)
        w2_cpu = torch.eye(h).bfloat16()[:intermediate].repeat(experts, 1, 1)
    if identity:
        values = torch.tensor([-4.03125, -4., -0.251953125, -0.25, -0.248046875,
                               0., 0.248046875, 0.25, 0.251953125,
                               0.4140625, 0.416015625, 0.41796875, 4., 4.03125]).bfloat16()
        x_cpu[:14, :intermediate] = values[:, None]
        x_cpu[:14, intermediate:] = values.flip(0)[:, None]
    dy_cpu = (torch.randn(world * t, h) * 0.1).bfloat16()
    sl = slice(rank * t, (rank + 1) * t)
    es = slice(rank * local_experts, (rank + 1) * local_experts)
    x = x_cpu[sl].to(device).requires_grad_()
    probs = probs_cpu[sl].to(device).requires_grad_()
    w1 = w1_cpu[es].to(device).requires_grad_()
    w2 = w2_cpu[es].to(device).requires_grad_()
    module = MegaMoeExperts(
        local_num_tokens=t, hidden_size=h, intermediate_size=intermediate,
        num_experts=experts, top_k=k, ep_size=world, ep_group=dist.group.WORLD,
        create_parameters=False,
        swiglu_limit=limit if limit else None,
        swiglu_mode=mode,
    )
    print(f"rank={rank} native_forward_start", flush=True)
    if args.checkpoint:
        from torch.utils.checkpoint import checkpoint
        output = checkpoint(lambda a,p,b,c: module(a, ids_cpu[sl].to(device), p, expert_weights=(b,c)), x, probs, w1, w2, use_reentrant=True)
    else:
        output = module(x, ids_cpu[sl].to(device), probs, expert_weights=(w1, w2))
    torch.npu.synchronize()
    print(f"rank={rank} native_backward_start", flush=True)
    output.backward(dy_cpu[sl].to(device))
    torch.npu.synchronize()
    print(f"rank={rank} reference_start", flush=True)
    xr = x_cpu.to(device).requires_grad_()
    pr = probs_cpu.to(device).requires_grad_()
    wr1 = w1_cpu.to(device).requires_grad_()
    wr2 = w2_cpu.to(device).requires_grad_()
    ref = torch.zeros_like(xr)
    for expert in range(experts):
        rows, slots = (ids_cpu == expert).nonzero(as_tuple=True)
        rows, slots = rows.to(device), slots.to(device)
        up = xr[rows] @ wr1[expert]
        if limit:
            gate, up_branch = up.chunk(2, dim=-1)
            if mode == "pre_activation":
                act = torch.nn.functional.silu(gate.clamp(max=limit)) * up_branch.clamp(-limit, limit)
            else:
                act = torch.nn.functional.silu(gate).clamp(max=limit) * up_branch.clamp(-limit, limit)
        else:
            act = torch_npu.npu_swiglu(up, dim=-1)
        y = act @ wr2[expert]
        ref = ref.index_add(0, rows, (y.float() * pr[rows, slots, None]).to(y.dtype))
    ref.backward(dy_cpu.to(device))
    result = {
        "rank": rank, "mode": mode, "identity": identity, "limit": limit,
        "output": compare(output, ref[sl]),
        "dx": compare(x.grad, xr.grad[sl]),
        "dprobs": compare(probs.grad, pr.grad[sl]),
        "dw1": compare(w1.grad, wr1.grad[es]),
        "dw2": compare(w2.grad, wr2.grad[es]),
    }
    if identity:
        result["dx_gate"] = compare(x.grad[:, :intermediate], xr.grad[sl, :intermediate])
        result["dx_up"] = compare(x.grad[:, intermediate:], xr.grad[sl, intermediate:])
    passed = all(value["finite"] and value["relative_l2"] < 0.02
                 for value in result.values() if isinstance(value, dict))
    result.update(status="PASS" if passed else "FAIL", routing=args.routing,
                  checkpoint=args.checkpoint, world_size=world)
    os.write(1, ("PRECISION_RESULT " + json.dumps(result) + "\n").encode())
    module.close()
    dist.destroy_process_group()
    assert all(v["finite"] and v["relative_l2"] < 0.02
               for v in result.values() if isinstance(v, dict)), result


if __name__ == "__main__":
    main()
