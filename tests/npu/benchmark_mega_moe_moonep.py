"""Dataset-independent EP MoE benchmark against the current TorchTurbo dispatcher."""
import argparse
import json
import os
import time

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[2]/"src/torchturbo/optimizations/mega_moe/runtime/bootstrap.py"))["activate"]()

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime import get_device_runtime


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.float(), expected.float()
    return dict(relative_l2=((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
                max_abs=(a-b).abs().max().item() if a.numel() else 0.0, finite=bool(a.isfinite().all() and b.isfinite().all()), reference_l2=b.norm().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--mutation-check', action='store_true')
    p.add_argument('--profile-root', default=None)
    p.add_argument('--capacity-factor', type=float, default=1.3)
    p.add_argument('--hidden', type=int, default=2560)
    p.add_argument('--intermediate', type=int, default=768)
    p.add_argument('--experts', type=int, default=512)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--seed', type=int, default=91027)
    p.add_argument('--routing', choices=['uniform','balanced','hot','local','remote','zipf','half_hot','alternating','zero_probs','zero_sources','reverse'], default='uniform')
    p.add_argument('--checkpoint', action='store_true')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeat', type=int, default=5)
    args = p.parse_args()
    device_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(device_rank)
    dist.init_process_group('hccl')
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device('npu', device_rank)
    get_or_create_ops(device_runtime=get_device_runtime())
    cfg = MoEConfig(hidden_size=args.hidden, moe_ffn_hidden_size=args.intermediate,
                    num_moe_experts=args.experts, moe_router_topk=args.topk,
                    gated_linear_unit=True, activation_func=F.silu,
                    params_dtype=torch.bfloat16, expert_model_parallel_size=world,
                    moe_token_dispatcher_type='alltoall', moe_router_score_function='sigmoid')
    local = args.experts // world
    experts = GroupedMLP(local, False, cfg).to(device)
    torch.manual_seed(args.seed + rank)
    with torch.no_grad():
        experts.weight1.normal_(0, .02)
        experts.weight2.normal_(0, .02)
    x0 = (torch.randn(args.tokens, args.hidden, device=device)*.5).bfloat16()
    scores = torch.rand(args.tokens, args.experts, device=device)
    if args.routing in ('hot','alternating'):
        scores[:, :args.topk] += 2
    elif args.routing in ('local','remote'):
        owner = rank if args.routing == 'local' else (rank + 1) % world
        scores[:, owner*local:(owner+1)*local] += 2
    elif args.routing == 'zipf':
        scores *= torch.arange(1, args.experts+1, device=device).float().reciprocal()
    elif args.routing == 'half_hot' and rank < max(1, world//2):
        scores[:, :args.topk] += 2
    ids = scores.topk(args.topk, dim=-1).indices.sort(-1).values
    if args.routing == 'balanced':
        ids = (torch.arange(args.tokens*args.topk, device=device).reshape(args.tokens,args.topk) + rank*args.topk) % args.experts
    if args.routing == 'reverse':
        ids = ids.flip(-1)
    hot_ids = ids
    uniform_ids = torch.rand(args.tokens, args.experts, device=device).topk(args.topk, dim=-1).indices.sort(-1).values
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, ids, True)
    p0 = torch.randn(args.tokens, args.topk, device=device).softmax(-1)
    if args.routing == 'zero_probs' or (args.routing == 'zero_sources' and rank != 0):
        p0.zero_()
    dy = (torch.randn_like(x0)*.1).bfloat16()
    dispatcher = MoEAlltoAllTokenDispatcher(local, list(range(rank*local,(rank+1)*local)), cfg, dist.group.WORLD)
    from torchturbo.optimizations.mega_moe.moonep.backend import MoonEPMegaMoeBackend
    reference = None
    reference_phases = {}
    results = []
    initial_weights = (experts.weight1.detach().clone(), experts.weight2.detach().clone()) if args.mutation_check else None
    for variant in ('reference','moonep'):
        mega = (MoonEPMegaMoeBackend(dist.group.WORLD, args.tokens, (("none", 0.0),),
                receive_capacity_factor=args.capacity_factor) if variant == 'moonep' else None)
        print(f'BENCH_START rank={rank} variant={variant}', flush=True)
        def run(x, probs):
            if variant == 'reference':
                dense = torch.zeros_like(scores).scatter(1, ids, probs)
                dispatched, counts, permuted_probs = dispatcher.token_permutation(
                    x, dense, mask
                )
                out, bias = experts(dispatched, counts, permuted_probs)
                if dispatcher.deferred_probs_to_unpermutation:
                    selected = dense.masked_select(mask).reshape_as(probs)
                    return dispatcher.token_unpermutation(out, bias, probs=selected)[0]
                out = out * permuted_probs.unsqueeze(-1)
                return dispatcher.token_unpermutation(out, bias)[0]
            # The NPU dispatcher casts selected routing probabilities to BF16.
            weights = probs.bfloat16()
            return mega.execute(experts, x, ids, weights)
        times = []
        for iteration in range(args.warmup + args.repeat):
            if args.routing == 'alternating':
                ids = uniform_ids if iteration % 2 else hot_ids
                mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, ids, True)
            x = x0.detach().requires_grad_()
            probs = p0.detach().requires_grad_()
            experts.zero_grad(set_to_none=True)
            dist.barrier()
            torch.npu.synchronize()
            start_event = torch.npu.Event(enable_timing=True)
            forward_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            start = time.perf_counter()
            start_event.record()
            if args.checkpoint:
                from torch.utils.checkpoint import checkpoint
                y = checkpoint(run, x, probs, use_reentrant=True)
            else:
                y = run(x, probs)
            forward_event.record()
            y.backward(dy)
            end_event.record()
            torch.npu.synchronize()
            end = time.perf_counter()
            if iteration >= args.warmup:
                times.append([start_event.elapsed_time(forward_event)/1000, forward_event.elapsed_time(end_event)/1000, end-start])
        timing = torch.tensor(times, dtype=torch.float32, device=device)
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        phase_precision = {}
        if args.routing == 'alternating' or args.mutation_check:
            for phase, phase_ids in [('uniform',uniform_ids),('hot',hot_ids),('uniform_again',uniform_ids),('hot_again',hot_ids)]:
                if args.mutation_check:
                    scale = {'uniform':1.0, 'hot':1.125, 'uniform_again':0.75, 'hot_again':1.375}[phase]
                    with torch.no_grad():
                        for parameter, original in zip((experts.weight1, experts.weight2), initial_weights):
                            parameter.copy_(original * scale)
                ids = phase_ids
                mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, ids, True)
                x = x0.detach().requires_grad_()
                probs = p0.detach().requires_grad_()
                experts.zero_grad(set_to_none=True)
                dist.barrier()
                y = checkpoint(run,x,probs,use_reentrant=True) if args.checkpoint else run(x,probs)
                y.backward(dy)
                torch.npu.synchronize()
                snapshot = {'output':y.detach().cpu(),'dx':x.grad.cpu(),'dprobs':probs.grad.cpu(),
                            'dw1':experts.weight1.grad.cpu(),'dw2':experts.weight2.grad.cpu()}
                if variant == 'reference':
                    reference_phases[phase] = snapshot
                else:
                    phase_precision[phase] = {key:compare(value,reference_phases[phase][key]) for key,value in snapshot.items()}
        if args.mutation_check:
            with torch.no_grad():
                for parameter, original in zip((experts.weight1, experts.weight2), initial_weights):
                    parameter.copy_(original)
        actual = {'output': y.detach().cpu(), 'dx': x.grad.cpu(), 'dprobs': probs.grad.cpu(),
                  'dw1': experts.weight1.grad.cpu(), 'dw2': experts.weight2.grad.cpu()}
        row = {'timing': 'continuous host submission; NPU event phases; wall total', 'variant': variant, 'rank':rank, 'config':vars(args),
               'median_ms': dict(zip(['forward','backward','total'], (timing.median(dim=0).values*1000).cpu().tolist())),
               'iteration_ms':(timing[:,2]*1000).cpu().tolist()}
        if reference is None:
            reference = actual
        else:
            row['precision'] = {key:compare(value,reference[key]) for key,value in actual.items()}
            row['precision_thresholds'] = {'output':0.001,'dx':0.001,'dprobs':0.001,'dw1':0.005,'dw2':0.005}
            row['precision_pass'] = all(item['finite'] and item['relative_l2'] <= row['precision_thresholds'][key]
                                        for key,item in row['precision'].items())
            row['transition_precision'] = phase_precision
            row['transition_precision_pass'] = all(metric['finite'] and metric['relative_l2'] <= row['precision_thresholds'][key]
                for phase in phase_precision.values() for key,metric in phase.items())
            row['precision_pass'] &= row['transition_precision_pass']
            if variant == 'moonep' and mega.plan_pool is not None:
                row['reserved_receive_capacities'] = [state['owner'].workspace.capacity_floor
                                                      for state in mega.plan_pool.states.values()]
        results.append(row)
        os.write(1, ('BENCH_RESULT '+json.dumps(row)+'\n').encode())
        del actual, y, x, probs
        if args.profile_root:
            from contextlib import nullcontext
            from torch.profiler import record_function
            context = nullcontext()
            if rank in (0, world - 1):
                context = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False, profile_memory=False, with_stack=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        f'{args.profile_root}/{variant}', worker_name=f'rank{rank}',
                        analyse_flag=False))
            dist.barrier()
            with context:
                for profile_iteration in range(3):
                    x = x0.detach().requires_grad_()
                    probs = p0.detach().requires_grad_()
                    experts.zero_grad(set_to_none=True)
                    dist.barrier()
                    with record_function(f'BENCH_{variant}_forward'):
                        if args.checkpoint:
                            y = checkpoint(run, x, probs, use_reentrant=True)
                        else:
                            y = run(x, probs)
                    with record_function(f'BENCH_{variant}_backward'):
                        y.backward(dy)
                    torch.npu.synchronize()
                    if rank in (0, world - 1):
                        context.step()
            dist.barrier()
            del y, x, probs
        if mega is not None:
            mega.close()
        dist.barrier()
    passed = all(row.get("precision_pass", True) for row in results)
    dist.destroy_process_group()
    if not passed:
        raise AssertionError("MoE output or gradient precision acceptance failed")


if __name__ == '__main__':
    main()
