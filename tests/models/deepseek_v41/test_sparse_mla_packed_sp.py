"""Check BF16 packed SP against independent per-sample references."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import torch
import torch.distributed as dist

try:
    import torch_npu  # noqa: F401 -- Register the NPU backend.
except ImportError:  # manual NPU harness; skip collection off-device
    pytest.skip("torch_npu not available", allow_module_level=True)

from sparse_mla_test_utils import (
    make_compressed_indices,
    make_far_compressed_indices,
    metrics,
    reference,
    run_sp_attention,
)

from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import SparseAttention
from torchturbo.ops.providers.npu.sparse_flash_mla import _run_sparse_flash_mla


def per_sample(
    inputs: list[torch.Tensor],
    lengths: tuple[int, ...],
    indices: torch.Tensor,
    ratio: int,
    style: str,
) -> torch.Tensor:
    """Evaluate samples independently without SP or packed offset translation."""
    q, ori, cmp, sink = inputs
    outputs = []
    start = offset = 0
    for length in lengths:
        count = length // ratio if ratio else 0
        sq, so = q[start : start + length], ori[start : start + length]
        sc = cmp[offset : offset + count]
        si = indices[start : start + length]
        if style == "fp64":
            output = reference(sq, so, sc, sink, si)
        elif style == "v4":
            config = SimpleNamespace(
                sliding_window=128,
                qk_rope_head_dim=64,
                compress_ratios=[max(ratio, 2) if count else 1],
                num_hidden_layers=1,
                head_dim=512,
            )
            v4_indices = torch.where(si[:, 0] >= 0, si[:, 0] + length, -1).unsqueeze(0)
            output = SparseAttention(config, 0)(
                sq.unsqueeze(0),
                so[:, 0].unsqueeze(0),
                sink,
                sc[:, 0].unsqueeze(0) if count else None,
                v4_indices if count else None,
            ).squeeze(0)
        else:
            cu = torch.tensor([0, length], dtype=torch.int32, device=q.device)
            cu_cmp = (
                torch.tensor([0, count], dtype=torch.int32, device=q.device)
                if count
                else None
            )
            residual = (
                torch.tensor([length % ratio], dtype=torch.int32, device=q.device)
                if count
                else None
            )
            output = _run_sparse_flash_mla(
                sq,
                so,
                sc if count else None,
                si if count else None,
                residual,
                sink,
                512**-0.5,
                max(ratio, 1),
                127,
                0,
                cu_seq_lens_q=cu,
                cu_seq_lens_ori_kv=cu,
                cu_seq_lens_cmp_kv=cu_cmp,
                custom_package=True,
            )
        outputs.append(output)
        start += length
        offset += count
    return torch.cat(outputs) + cmp[:0].sum()


def main() -> None:
    torch.set_num_threads(2)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime import get_device_runtime

    get_or_create_ops(device_runtime=get_device_runtime())
    dist.init_process_group("hccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    # Explicit workaround for the separately documented metadata order issue.
    warmup = torch.ones(1, device="npu")
    dist.all_reduce(warmup)
    torch.npu.synchronize()
    print(f"[rank {rank}] HCCL_WARMUP_DONE", flush=True)
    compare_path = (
        Path(os.environ["OPS_TRANSFORMER_ROOT"])
        / "attention/fused_floyd_attention_grad/tests/assets/impl/compare.py"
    )
    spec = importlib.util.spec_from_file_location("packed_compare", compare_path)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    lengths = tuple(
        int(x) for x in os.environ.get("PACKED_LENGTHS", "3,62,129,1,61").split(",")
    )
    ratios = [int(x) for x in os.environ.get("PACKED_RATIOS", "0,1,2,4").split(",")]
    patterns = os.environ.get("PACKED_PATTERNS", "recent,far").split(",")
    total = sum(lengths)
    if total % size:
        raise ValueError("Packed total length must be divisible by SP size.")
    local = total // size
    begin, end = rank * local, (rank + 1) * local
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    reports = []
    for ratio in ratios:
        for pattern in patterns if ratio else patterns[:1]:
            generator = torch.Generator().manual_seed(42)
            cmp_length = sum(x // ratio for x in lengths) if ratio else 0
            inputs = [
                torch.randn(shape, generator=generator).to(dtype)
                for shape, dtype in (
                    ((total, 2, 512), torch.bfloat16),
                    ((total, 1, 512), torch.bfloat16),
                    ((cmp_length, 1, 512), torch.bfloat16),
                    ((2,), torch.float32),
                )
            ]
            dout = torch.randn((total, 2, 512), generator=generator).bfloat16()
            make_indices = (
                make_far_compressed_indices
                if pattern == "far"
                else make_compressed_indices
            )
            indices = (
                torch.cat(
                    [make_indices(n, ratio, 2, torch.device("cpu")) for n in lengths]
                )
                if ratio
                else torch.full((total, 1, 2), -1, dtype=torch.int32)
            )
            gold_inputs = [x.double().requires_grad_() for x in inputs]
            golden = per_sample(gold_inputs, lengths, indices, ratio, "fp64")
            golden.backward(dout.double())
            bench_inputs = [x.npu().requires_grad_() for x in inputs]
            benchmark = per_sample(bench_inputs, lengths, indices.npu(), ratio, "v4")
            benchmark.backward(dout.npu())
            full_inputs = [x.npu().requires_grad_() for x in inputs]
            full = per_sample(full_inputs, lengths, indices.npu(), ratio, "fused")
            full.backward(dout.npu())
            sq, so = [x[begin:end].npu().requires_grad_() for x in inputs[:2]]
            sc, ss = [x.npu().requires_grad_() for x in inputs[2:]]
            output = run_sp_attention(
                sq,
                so,
                ss,
                cu_seqlens=tuple(boundaries),
                sp_group=dist.group.WORLD,
                cmp_kv_global=sc if ratio else None,
                cmp_indices_local=indices[begin:end].npu() if ratio else None,
                compress_ratio=max(ratio, 1),
            )
            output.backward(dout[begin:end].npu())
            torch.testing.assert_close(output, full[begin:end], rtol=0, atol=0)
            gathered = []
            for tensor in (sq.grad, so.grad):
                pieces = [torch.empty_like(tensor) for _ in range(size)]
                dist.all_gather(pieces, tensor.contiguous())
                gathered.append(torch.cat(pieces))
            for tensor in (sc.grad if ratio else torch.zeros_like(sc), ss.grad):
                reduced = tensor.clone()
                if reduced.numel():
                    dist.all_reduce(reduced)
                gathered.append(reduced)
            checks = {}
            for name, actual, full_input, gold, bench in zip(
                ("dQ", "dOriKV", "dCmpKV", "dSink"),
                gathered,
                full_inputs,
                gold_inputs,
                bench_inputs,
            ):
                if not actual.numel():
                    continue
                metrics(actual, gold.grad)
                metrics(bench.grad, gold.grad)
                check = comparator._atk_precision_check_three_way(
                    actual.cpu(), gold.grad, bench.grad.cpu()
                )
                checks[name] = {
                    "pass": check["pass"],
                    "three_way": check,
                    "vs_full": metrics(actual, full_input.grad.cpu()),
                }
            # Check independent FP64 forward accuracy using the same three-way comparator.
            forward = comparator._atk_precision_check_three_way(
                output.cpu(),
                golden[begin:end].detach(),
                benchmark[begin:end].detach().cpu(),
            )
            # A loss on one sample must not update another sample's Q or KV.
            target = max(range(len(lengths)), key=lambda i: lengths[i])
            target_start, target_end = boundaries[target : target + 2]
            isolated = [x.detach().clone().requires_grad_() for x in (sq, so, sc, ss)]
            iq, io, ic, ist = isolated
            isolated_output = run_sp_attention(
                iq,
                io,
                ist,
                cu_seqlens=tuple(boundaries),
                sp_group=dist.group.WORLD,
                cmp_kv_global=ic if ratio else None,
                cmp_indices_local=indices[begin:end].npu() if ratio else None,
                compress_ratio=max(ratio, 1),
            )
            positions = torch.arange(begin, end, device="npu")
            selected = (positions >= target_start) & (positions < target_end)
            isolated_output.backward(dout[begin:end].npu() * selected[:, None, None])
            for tensor in (iq.grad, io.grad):
                assert (
                    torch.count_nonzero(tensor[~selected]).item() == 0
                ), "Cross-sample gradient leak."
            if ratio and ic.numel():
                cmp_start = sum(n // ratio for n in lengths[:target])
                cmp_end = cmp_start + lengths[target] // ratio
                assert torch.count_nonzero(ic.grad[:cmp_start]).item() == 0
                assert torch.count_nonzero(ic.grad[cmp_end:]).item() == 0
            # Reject malformed indices before fused attention dispatch.
            if ratio:
                try:
                    run_sp_attention(
                        sq,
                        so,
                        ss,
                        cu_seqlens=tuple(boundaries),
                        sp_group=dist.group.WORLD,
                        cmp_kv_global=sc,
                        cmp_indices_local=torch.full_like(indices[begin:end].npu(), -2),
                        compress_ratio=ratio,
                    )
                except ValueError as error:
                    assert "Compressed indices" in str(error)
                else:
                    raise AssertionError("Invalid indices were accepted.")
            passed = forward["pass"] and all(x["pass"] for x in checks.values())
            report = {
                "lengths": lengths,
                "ratio": ratio,
                "pattern": pattern,
                "sp_size": size,
                "rank": rank,
                "dtype": "bf16",
                "forward_exact_vs_full": True,
                "sample_isolation": True,
                "forward_three_way": forward,
                "gradients": checks,
                "pass": passed,
            }
            reports.append(report)
            print(
                f"[rank {rank}] ratio={ratio} pattern={pattern} "
                f"isolation=True forward_exact=True forward_check={forward['pass']} "
                f"gradients={[(name, value['pass']) for name, value in checks.items()]}",
                flush=True,
            )
    Path("work").mkdir(exist_ok=True)
    Path(
        f"work/packed_bf16_{'_'.join(map(str, lengths))}_sp{size}_rank{rank}.json"
    ).write_text(json.dumps(reports, indent=2))
    if not all(x["pass"] for x in reports):
        raise AssertionError("Packed BF16 comparison failed; inspect PACKED_RESULT.")
    print(f"[rank {rank}] PACKED_BF16_ALL_PASSED", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
