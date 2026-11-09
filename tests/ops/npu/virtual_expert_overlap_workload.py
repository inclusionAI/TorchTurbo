"""Profile SHMEM or native weight prefetch and token dispatch with a Triton planner.

Run with torchrun EP2 and --output-dir. Inspect the exported Ascend Hardware
kernel and HCCL payload intervals; wait/notify intervals alone prove no overlap.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.npu import NpuRuntime
import torchturbo.runtime.runtime as runtime_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--mixed-hotspot", action="store_true", help="Force token traffic between EP peers.")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--weight-provider", choices=("shmem", "native"), default="shmem")
    args = parser.parse_args()
    import torch_npu

    runtime = NpuRuntime()
    runtime.initialize()
    runtime.set_device(int(os.environ["LOCAL_RANK"]))
    runtime_module._SELECTED_RUNTIME = runtime
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    dispatcher = get_or_create_ops(
        device_runtime=runtime, default_policy=OpPolicy.NATIVE,
        overrides={
            "plan_virtual_expert_routes": "npu_triton_plan_virtual_expert_routes",
            "virtual_expert_weight_exchange": (
                "npu_shmem_virtual_expert_weight_exchange" if args.weight_provider == "shmem"
                else "torch_virtual_expert_weight_exchange"
            ),
            "virtual_expert_grad_reduce": (
                "npu_shmem_virtual_expert_grad_reduce" if args.weight_provider == "shmem"
                else "torch_virtual_expert_grad_reduce"
            ),
            "grouped_matmul": "torch_npu_grouped_matmul",
        },
    )
    try:
        size, rank = dist.get_world_size(), dist.get_rank()
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-4, 2e-2)
        config = MoEConfig(
            hidden_size=args.hidden_size, moe_ffn_hidden_size=args.ffn_size,
            num_moe_experts=2 * size, expert_model_parallel_size=size,
            moe_router_topk=1, gated_linear_unit=True,
            activation_func=torch.nn.functional.silu, moe_router_dtype="fp32", params_dtype=dtype,
        )
        submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
        torch.manual_seed(937 + rank)
        reference = MoELayer(config, dist.group.WORLD, submodules=submodules).to(runtime.device)
        candidate = MoELayer(
            replace(config, moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True),
            dist.group.WORLD, submodules=submodules,
        ).to(runtime.device)

        weight_exchange = dispatcher.virtual_expert_weight_exchange
        token_dispatch = dispatcher.hybrid_ep_dispatch

        def traced_weight_exchange(*call_args, **kwargs):
            phase = "consume" if kwargs.get("prefetched") is not None else "prefetch"
            with torch.profiler.record_function(f"{args.weight_provider}_weight_{phase}"):
                return weight_exchange(*call_args, **kwargs)

        def traced_token_dispatch(*args, **kwargs):
            with torch.profiler.record_function("native_token_dispatch"):
                return token_dispatch(*args, **kwargs)

        dispatcher.virtual_expert_weight_exchange = traced_weight_exchange
        dispatcher.hybrid_ep_dispatch = traced_token_dispatch
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=0.015)
            reference.router.weight.zero_()
            reference.router.weight[0].fill_(0.2 / args.hidden_size)
            candidate.load_state_dict(reference.state_dict())
            inputs = torch.rand((1, args.tokens, args.hidden_size), device=runtime.device, dtype=dtype) + 1
            if args.mixed_hotspot:
                reference.router.weight.zero_()
                reference.router.weight[0, 0] = reference.router.weight[1, 1] = 1
                candidate.load_state_dict(reference.state_dict())
                inputs.mul_(0.05)
                inputs[:, :, 0] = 2
                if rank % 2:
                    inputs[:, args.tokens // 2:, 0] = 0
                    inputs[:, args.tokens // 2:, 1] = 2
            manager = candidate.token_dispatcher._comm_manager
            manager._runtime_init(inputs)
            if args.serial:
                manager.weight_streams = runtime.current_stream()
            expected = reference(inputs)[0]
            for _ in range(4):
                actual = candidate(inputs)[0]
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            runtime.synchronize()
            dist.barrier()
            experimental = torch_npu.profiler._ExperimentalConfig(
                # Level1 records HCCL payload sizes separately from waits.
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                export_type=[torch_npu.profiler.ExportType.Text],
                data_simplification=False,
            )
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output_dir), worker_name=f"rank{rank}"),
                experimental_config=experimental,
            ) as profiler:
                for _ in range(2):
                    with torch.profiler.record_function("virtual_expert_large_weight_forward"):
                        actual = candidate(inputs)[0]
                    runtime.synchronize()
                    profiler.step()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in candidate.experts.parameters())
        print(f"OVERLAP_PARITY_PASS rank={rank} EP={size} hidden={args.hidden_size} "
              f"ffn={args.ffn_size} tokens={args.tokens} native_weight_bytes={weight_bytes} "
              f"dtype={args.dtype} mixed_hotspot={args.mixed_hotspot} "
              f"weight_provider={args.weight_provider} serial={args.serial}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
