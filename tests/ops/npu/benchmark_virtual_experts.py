"""Compare native and Triton virtual experts on an idle single-host NPU EP group.

Run with torchrun and PYTHONPATH=.:src. Setup and compilation are excluded.
Each sample measures synchronized wall time and takes the maximum across ranks.
The benchmark changes the process-local ops dispatcher only between variants.
MoE runs size the default SHMEM heap for their weight and gradient arenas;
an explicit NPU_SHMEM_SYMMETRIC_SIZE takes precedence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import time
import traceback
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.registry import registry
from torchturbo.runtime.npu import NpuRuntime
import torchturbo.runtime.runtime as runtime_module
from tests.ops.npu.benchmark_virtual_expert_routes import RoutingCase, routing_cases


PLANNERS = {"native": "torch_plan_virtual_expert_routes", "triton": "npu_triton_plan_virtual_expert_routes"}
MODES = {
    "alltoall": "native",
    "ve_native": "native",
    "ve_planner": "triton",
    "ve_planner_serial": "triton",
    "ve_shmem": "triton",
    "ve_shmem_serial": "triton",
}


class Benchmark:
    def __init__(
        self, args: argparse.Namespace, runtime: NpuRuntime, route_cases: list[RoutingCase],
    ) -> None:
        self.args = args
        self.runtime = runtime
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.route_cases = route_cases
        self.dispatchers = {}
        for mode, planner in MODES.items():
            self.dispatchers[mode] = OpDispatcher(
                device_runtime=runtime, registry=registry, default_policy=OpPolicy.NATIVE,
                overrides={
                    "plan_virtual_expert_routes": PLANNERS[planner],
                    "virtual_expert_weight_exchange": (
                        "npu_shmem_virtual_expert_weight_exchange" if mode.startswith("ve_shmem")
                        else "torch_virtual_expert_weight_exchange"
                    ),
                    "virtual_expert_grad_reduce": (
                        "npu_shmem_virtual_expert_grad_reduce" if mode.startswith("ve_shmem")
                        else "torch_virtual_expert_grad_reduce"
                    ),
                    "grouped_matmul": "torch_npu_grouped_matmul",
                },
            )
        import torch_npu
        import triton

        self.report = {
            "environment": {
                "utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "triton": triton.__version__, "device": torch.npu.get_device_name(),
                "ep_size": self.size, "allocator": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
                "hccl_expansion": os.environ.get("HCCL_OP_EXPANSION_MODE"),
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "shmem_symmetric_size": os.environ.get("NPU_SHMEM_SYMMETRIC_SIZE"),
                "grouped_matmul": "torch_npu_grouped_matmul", "other_ops": "native",
            },
            "method": {
                "warmup": args.warmup, "repetitions_per_round": args.reps,
                "rounds": args.rounds, "inner_iterations": args.inner_iterations,
                "clock": "perf_counter_ns; synchronize before and after; barrier outside timer",
                "aggregation": "maximum rank wall ms per iteration; median and p10/p90 over samples",
                "order": "forward then reverse on alternate rounds",
                "excluded": "compilation, process groups, parity checks, optimizer, FSDP",
                "moe_modes": args.moe_modes,
                "moe_input_seed": "1937 + rank; independent of selected modes",
            },
            "results": [],
            "excluded": [],
            "routing_loads": [],
        }

    def select(self, mode: str) -> None:
        importlib.import_module("torchturbo.ops.ops")._OPS = self.dispatchers[mode]

    def compare(self, case: dict, functions: dict[str, Callable[[], object]]) -> None:
        samples = {name: [] for name in functions}
        for fn in functions.values():
            for _ in range(self.args.warmup):
                fn()
            self.runtime.synchronize()
        for round_id in range(self.args.rounds):
            names = list(functions)
            if round_id % 2:
                names.reverse()
            for name in names:
                fn = functions[name]
                local = []
                for _ in range(self.args.reps):
                    dist.barrier()
                    self.runtime.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(self.args.inner_iterations):
                        fn()
                    self.runtime.synchronize()
                    local.append((time.perf_counter_ns() - start) / 1e6 / self.args.inner_iterations)
                gathered = [None] * self.size
                dist.all_gather_object(gathered, local)
                samples[name].extend(max(values) for values in zip(*gathered))
        if self.rank == 0:
            for name, values in samples.items():
                quantiles = torch.tensor(values, dtype=torch.float64).quantile(torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
                row = dict(case, variant=name, p10_ms=quantiles[0].item(), median_ms=quantiles[1].item(),
                           p90_ms=quantiles[2].item(), samples_ms=values, parity="passed")
                self.report["results"].append(row)
                print("RESULT " + json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.args.output.write_text(json.dumps(self.report, indent=2) + "\n")

    def planner(self) -> None:
        from test_virtual_expert_planner import _assert_plan_equal

        dispatcher = self.dispatchers["alltoall"]
        for tokens in self.args.planner_tokens:
            for route_case in self.route_cases:
                routes = route_case.route_ids(
                    tokens, self.size, self.rank, num_experts=self.args.num_experts, topk=self.args.topk,
                ).to(self.runtime.device)
                functions = {
                    name: lambda impl=impl: dispatcher.plan_virtual_expert_routes(
                        routes, self.args.num_experts, dist.group.WORLD, impl=impl,
                    )
                    for name, impl in PLANNERS.items()
                }
                _assert_plan_equal(functions["triton"](), functions["native"]())
                self.compare(dict(
                    suite="planner", tokens=tokens, dtype="int64",
                    **route_case.metadata(self.size, num_experts=self.args.num_experts, topk=self.args.topk),
                ), functions)

    def moe(self) -> None:
        models = self._create_models()
        for tokens in self.args.moe_tokens:
            for route_case in self.route_cases:
                self._moe_case(tokens, route_case, models)

    def _create_models(self) -> dict[str, MoELayer]:
        """Reuse the same layer and communication storage across route cases."""
        hidden, ffn = self.args.hidden, self.args.ffn
        dtype = torch.bfloat16
        config = MoEConfig(
            hidden_size=hidden, moe_ffn_hidden_size=ffn, num_moe_experts=self.args.num_experts,
            expert_model_parallel_size=self.size, moe_router_topk=self.args.topk, gated_linear_unit=True,
            activation_func=torch.nn.functional.silu, moe_router_dtype="fp32", params_dtype=dtype,
        )
        self.select("alltoall")
        torch.manual_seed(937 + self.rank)
        submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
        reference = MoELayer(config, dist.group.WORLD, submodules=submodules).to(self.runtime.device)
        with torch.no_grad():
            for parameter in reference.experts.parameters():
                parameter.normal_(std=0.015)
            reference.router.weight.zero_()
            for expert in range(config.num_moe_experts):
                reference.router.weight[expert, expert] = 1
        models = {"alltoall": reference}
        virtual_modes = [mode for mode in self.args.moe_modes if mode != "alltoall"]
        for mode in virtual_modes:
            model = MoELayer(
                replace(config, moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True),
                dist.group.WORLD, submodules=submodules,
            ).to(self.runtime.device)
            model.load_state_dict(reference.state_dict())
            models[mode] = model
        return models

    def _moe_case(self, tokens: int, route_case: RoutingCase, models: dict[str, MoELayer]) -> None:
        case_metadata = dict(
            tokens=tokens,
            **route_case.metadata(self.size, num_experts=self.args.num_experts, topk=self.args.topk),
        )
        if self.rank == 0:
            print("CASE_START " + json.dumps(case_metadata), flush=True)
        hidden, ffn = self.args.hidden, self.args.ffn
        dtype = torch.bfloat16
        reference = models["alltoall"]
        config = reference.config
        virtual_modes = [mode for mode in self.args.moe_modes if mode != "alltoall"]
        # Model construction must not change inputs when modes differ.
        torch.manual_seed(1937 + self.rank)
        inputs = torch.rand((1, tokens, hidden), device=self.runtime.device, dtype=dtype) * 0.1
        ids = route_case.route_ids(
            tokens, self.size, self.rank, num_experts=self.args.num_experts, topk=self.args.topk,
        ).to(self.runtime.device)
        inputs[0].scatter_(1, ids, 2)
        inputs.requires_grad_()
        grad_output = torch.randn_like(inputs) * 0.1
        for mode in virtual_modes:
            self.select(mode)
            manager = models[mode].token_dispatcher._comm_manager
            manager._runtime_init(inputs)
            if mode.endswith("_serial"):
                manager.weight_streams = self.runtime.current_stream()

        def run(mode: str, backward: bool) -> torch.Tensor:
            self.select(mode)
            model = models[mode]
            model.zero_grad(set_to_none=True)
            inputs.grad = None
            with torch.set_grad_enabled(backward):
                output = model(inputs)[0]
                if backward:
                    output.backward(grad_output)
            return output

        observed_loads = {}

        def parity_run(mode: str) -> torch.Tensor:
            # Observe the real router and expert inputs outside all timing regions.
            observations = {}

            def record_router(module, args, result):
                observations["routes"] = result[2].detach().reshape(tokens, self.args.topk).cpu().sort(dim=1).values

            def record_experts(module, args):
                observations["expert_tokens"] = args[1].detach().cpu().tolist()

            hooks = (
                models[mode].router.register_forward_hook(record_router),
                models[mode].experts.register_forward_pre_hook(record_experts),
            )
            try:
                output = run(mode, True)
            finally:
                for hook in hooks:
                    hook.remove()
            observations["routes_match_requested"] = torch.equal(observations.pop("routes"), ids.cpu().sort(dim=1).values)
            observations["source_expert_tokens"] = torch.bincount(
                ids.cpu().flatten(), minlength=config.num_moe_experts,
            ).tolist()
            peers = [None] * self.size
            dist.all_gather_object(peers, observations)
            observed_loads[mode] = peers
            assert all(peer["routes_match_requested"] for peer in peers)
            assert sum(sum(peer["expert_tokens"]) for peer in peers) == tokens * self.args.topk * self.size
            if mode == "alltoall":
                global_counts = torch.tensor([peer["source_expert_tokens"] for peer in peers]).sum(0)
                actual_counts = torch.tensor([peer["expert_tokens"] for peer in peers]).reshape(-1)
                torch.testing.assert_close(actual_counts, global_counts)
            return output

        expected = parity_run("alltoall").detach().clone()
        expected_grads = [inputs.grad.clone()] + [p.grad.clone() for p in reference.parameters()]
        max_errors = {}
        parity_failed = False
        for mode in virtual_modes:
            output = parity_run(mode)
            actual_grads = [inputs.grad] + [p.grad for p in models[mode].parameters()]
            error = None
            try:
                torch.testing.assert_close(output, expected, atol=2e-4, rtol=2e-2)
                for a, b in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(a, b, atol=2e-3, rtol=3e-2)
            except AssertionError as exception:
                error = str(exception)
            errors = [None] * self.size
            dist.all_gather_object(errors, error)
            if any(message is not None for message in errors):
                parity_failed = True
                excluded = dict(suite="moe", hidden=hidden, ffn=ffn, **case_metadata,
                                dtype="bf16", failed_variant=mode, rank_errors=errors)
                self.report["excluded"].append(excluded)
                if self.rank == 0:
                    print("EXCLUDED " + json.dumps(excluded), flush=True)
                break
            max_errors[mode] = max((a - b).abs().max().item() for a, b in zip(actual_grads, expected_grads))
        if self.rank == 0:
            loads = dict(hidden=hidden, ffn=ffn, **case_metadata, variants=observed_loads)
            self.report["routing_loads"].append(loads)
        if parity_failed:
            return
        if self.rank == 0:
            print("MOE_PARITY " + json.dumps(dict(**case_metadata, gradient_max_abs_errors=max_errors)), flush=True)
        del expected, expected_grads
        if virtual_modes:
            del actual_grads, output
        for backward in (False, True):
            functions = {mode: lambda mode=mode: run(mode, backward) for mode in self.args.moe_modes}
            self.compare(dict(suite="moe", hidden=hidden, ffn=ffn, **case_metadata, dtype="bf16",
                              phase="fwd_bwd" if backward else "fwd"), functions)
        del models, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("planner", "moe", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--planner-tokens", type=int, nargs="+", default=[256, 2048, 8192])
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--routing", choices=("balanced", "random", "hotspot"), nargs="+", default=["balanced", "hotspot"])
    parser.add_argument("--route-seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--hot-experts", type=int, nargs="+", default=[1])
    hot_group = parser.add_mutually_exclusive_group()
    hot_group.add_argument("--hot-fractions", type=float, nargs="+")
    hot_group.add_argument("--hot-multipliers", type=float, nargs="+")
    parser.add_argument("--hot-layouts", choices=("packed", "spread"), nargs="+", default=["packed"])
    parser.add_argument("--moe-tokens", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--moe-modes", choices=list(MODES), nargs="+", default=list(MODES))
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn", type=int, default=512)
    args = parser.parse_args()
    if args.hot_fractions is None and args.hot_multipliers is None:
        args.hot_multipliers = [2, 4]
    if args.num_experts < 1 or not 1 <= args.topk <= args.num_experts or args.num_experts > args.hidden:
        parser.error("Require 1 <= topk <= num_experts <= hidden for controlled router inputs.")
    ep_size = int(os.environ["WORLD_SIZE"])
    if args.num_experts % ep_size:
        parser.error("num-experts must divide evenly across EP ranks.")
    if args.hot_multipliers is not None and any(m <= 1 for m in args.hot_multipliers):
        parser.error("Hot multipliers must exceed the uniform load (1).")
    if any(count < 1 for count in args.hot_experts) or any(not 0 < p <= 1 for p in (args.hot_fractions or [])):
        parser.error("Hot expert counts must be positive and fractions must be in (0, 1].")
    route_cases = routing_cases(
        ep_size, args.routing, args.route_seeds, args.hot_experts,
        args.hot_fractions or [], args.hot_layouts, num_experts=args.num_experts,
        topk=args.topk, hot_multipliers=args.hot_multipliers,
    )
    if not route_cases:
        parser.error("No valid routing cases remain; adjust --routing or hotspot settings for --num-experts and --topk.")
    if args.suite != "planner" and any(mode != "alltoall" for mode in args.moe_modes):
        # Compatible VE modes share one BF16 weight arena and one FP32 gradient arena.
        num_weight_elements = 3 * args.hidden * args.ffn * (args.num_experts // ep_size)
        arena_bytes = num_weight_elements * (torch.bfloat16.itemsize + torch.float32.itemsize)
        # The standalone planner cache and shared model planner can both stay live.
        num_planners = int(args.suite == "all") + int(any(MODES[mode] == "triton" for mode in args.moe_modes))
        planner_bytes = num_planners * ep_size * args.num_experts * torch.int64.itemsize
        # Round the complete allocation total past a GiB boundary.
        heap_gib = (arena_bytes + planner_bytes) // (1 << 30) + 1
        os.environ.setdefault("NPU_SHMEM_SYMMETRIC_SIZE", f"{heap_gib}G")
    torch.set_num_threads(1)
    runtime = NpuRuntime()
    runtime.initialize()
    runtime.set_device(int(os.environ["LOCAL_RANK"]))
    runtime_module._SELECTED_RUNTIME = runtime
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    benchmark = Benchmark(args, runtime, route_cases)
    try:
        for suite in ("planner", "moe"):
            if args.suite in (suite, "all"):
                {"planner": benchmark.planner, "moe": benchmark.moe}[suite]()
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(benchmark.report, indent=2) + "\n")
            print("BENCHMARK_COMPLETE " + str(args.output), flush=True)
    except Exception:
        # A rank-local assertion must reach torchrun before collective cleanup.
        # Peers may otherwise wait in the next benchmark collective forever.
        traceback.print_exc()
        os._exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
