"""Compare standalone MegaMoE forward/backward with ordinary NPU experts."""
import json
import argparse
import os

import torch
import torch.distributed as dist
import torch_npu



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
    parser.add_argument("--checkpoint-mode", choices=["none", "reentrant", "nonreentrant"])
    parser.add_argument("--checkpoint-scope", choices=["layer", "stack"], default="layer")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--require-copies", action="store_true")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    checkpoint_mode = args.checkpoint_mode or ("reentrant" if args.checkpoint else "none")
    assert args.layers > 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    rank = dist.get_rank()
    t, h, intermediate, experts, k = args.tokens, 256, 256, args.experts, args.topk
    assert experts % world == 0 and 0 < k <= experts
    limit = args.limit
    mode = args.mode
    identity = args.identity
    if identity:
        h = 2 * intermediate
    local_experts = experts // world
    device = torch.device("npu", local_rank)
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
    # Distinct parameters exercise reuse of the same staging pool across layers.
    all_w1 = [w1_cpu] + [(torch.randn_like(w1_cpu.float()) * .03).bfloat16()
                         for _ in range(args.layers - 1)]
    all_w2 = [w2_cpu] + [(torch.randn_like(w2_cpu.float()) * .03).bfloat16()
                         for _ in range(args.layers - 1)]
    weights = [(a[es].to(device).requires_grad_(), b[es].to(device).requires_grad_())
               for a, b in zip(all_w1, all_w2)]
    from types import SimpleNamespace
    from torchturbo.models.modules.moe.moe_config import MoEConfig
    from torchturbo.optimizations.mega_moe.moonep.backend import MoonEPMegaMoeBackend

    config = MoEConfig(hidden_size=h, moe_ffn_hidden_size=intermediate,
                      num_moe_experts=experts, moe_router_topk=k,
                      gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                      moe_swiglu_clamp_gate_after_activation=mode == "post_activation")
    backend = MoonEPMegaMoeBackend(dist.group.WORLD, t, ((mode, limit),))

    def module(hidden, ids, weights, *, expert_weights):
        owner = SimpleNamespace(config=config, weight1=expert_weights[0],
                                weight2=expert_weights[1], swiglu_limit=limit)
        return backend.execute(owner, hidden, ids, weights)

    close = backend.close
    print(f"rank={rank} native_forward_start", flush=True)
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    ids = ids_cpu[sl].to(device)

    def layer(hidden, probabilities, a, b):
        return module(hidden, ids, probabilities, expert_weights=(a, b))

    def stack(hidden, probabilities, *parameters):
        for index in range(args.layers):
            hidden = layer(hidden, probabilities, *parameters[2 * index:2 * index + 2])
        return hidden

    parameters = tuple(value for pair in weights for value in pair)
    with set_checkpoint_early_stop(not args.disable_early_stop):
        if checkpoint_mode == "none":
            output = stack(x, probs, *parameters)
        elif args.checkpoint_scope == "stack":
            output = checkpoint(stack, x, probs, *parameters,
                                use_reentrant=checkpoint_mode == "reentrant")
        else:
            output = x
            for pair in weights:
                output = checkpoint(layer, output, probs, *pair,
                                    use_reentrant=checkpoint_mode == "reentrant")
    torch.npu.synchronize()
    print(f"rank={rank} native_backward_start", flush=True)
    output.backward(dy_cpu[sl].to(device))
    torch.npu.synchronize()
    print(f"rank={rank} reference_start", flush=True)
    xr = x_cpu.to(device).requires_grad_()
    pr = probs_cpu.to(device).requires_grad_()
    reference_weights = [(a.to(device).requires_grad_(), b.to(device).requires_grad_())
                         for a, b in zip(all_w1, all_w2)]
    ref = xr
    for wr1, wr2 in reference_weights:
        next_hidden = torch.zeros_like(ref)
        for expert in range(experts):
            rows, slots = (ids_cpu == expert).nonzero(as_tuple=True)
            rows, slots = rows.to(device), slots.to(device)
            up = ref[rows] @ wr1[expert]
            if limit:
                gate, up_branch = up.float().chunk(2, dim=-1)
                if mode == "pre_activation":
                    act = torch.nn.functional.silu(gate.clamp(max=limit)) * up_branch.clamp(-limit, limit)
                else:
                    act = torch.nn.functional.silu(gate).clamp(max=limit) * up_branch.clamp(-limit, limit)
            else:
                act = torch_npu.npu_swiglu(up, dim=-1)
            act = act.to(up.dtype)
            y = act @ wr2[expert]
            next_hidden = next_hidden.index_add(0, rows, (y.float() * pr[rows, slots, None]).to(y.dtype))
        ref = next_hidden
    ref.backward(dy_cpu.to(device))
    result = {
        "rank": rank, "mode": mode, "identity": identity, "limit": limit,
        "output": compare(output, ref[sl]),
        "dx": compare(x.grad, xr.grad[sl]),
        "dprobs": compare(probs.grad, pr.grad[sl]),
    }
    for index, ((w1, w2), (wr1, wr2)) in enumerate(zip(weights, reference_weights)):
        suffix = "" if index == 0 else f"_layer{index}"
        result["dw1" + suffix] = compare(w1.grad, wr1.grad[es])
        result["dw2" + suffix] = compare(w2.grad, wr2.grad[es])
    if args.require_copies:
        assert backend.resources is not None, "Borrowed-expert test requires EP > 1."
    if backend.resources is not None:
        assert not backend.resources.in_use
        if args.require_copies:
            assert backend.resources.statistics["gradient_returns"] >= args.layers, (
                "Hot-routing regression did not exercise borrowed experts."
            )
    if identity:
        result["dx_gate"] = compare(x.grad[:, :intermediate], xr.grad[sl, :intermediate])
        result["dx_up"] = compare(x.grad[:, intermediate:], xr.grad[sl, intermediate:])
    passed = all(value["finite"] and value["relative_l2"] < 0.02
                 for value in result.values() if isinstance(value, dict))
    result.update(status="PASS" if passed else "FAIL", routing=args.routing,
                  checkpoint=checkpoint_mode != "none", checkpoint_mode=checkpoint_mode,
                  checkpoint_scope=args.checkpoint_scope, layers=args.layers,
                  early_stop=not args.disable_early_stop, world_size=world)
    os.write(1, ("PRECISION_RESULT " + json.dumps(result) + "\n").encode())
    close()
    dist.destroy_process_group()
    assert all(v["finite"] and v["relative_l2"] < 0.02
               for v in result.values() if isinstance(v, dict)), result


if __name__ == "__main__":
    main()
