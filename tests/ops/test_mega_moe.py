"""Dispatcher policies and EP2 forward/backward checks for routed experts."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.registry import registry
from torchturbo.runtime.cpu import CpuRuntime


def reference(hidden, ids, probs, weight1, weight2, limit, after):
    if hidden.shape[0] == 0:
        return hidden + (weight1.sum() + weight2.sum() + probs.sum()) * 0
    rows = []
    for row in range(hidden.shape[0]):
        routed = []
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            gate, up = (hidden[row] @ weight1[expert]).chunk(2)
            if limit and not after:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            gate = F.silu(gate)
            if limit and after:
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            routed.append(((gate * up) @ weight2[expert]) * probs[row, slot])
        rows.append(torch.stack(routed).sum(0))
    return torch.stack(rows)


def native_worker(rank: int, rendezvous: str, output: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry,
                              default_policy=OpPolicy.NATIVE)
    hidden_size, ffn, num_experts, top_k = 8, 4, 8, 2
    for routing in ("uniform", "hot", "empty_rank", "cold_rank"):
        for limit, after in ((0., False), (.25, False), (.25, True)):
            torch.manual_seed(17)
            weight1 = torch.randn(num_experts, hidden_size, 2 * ffn, dtype=torch.float64, requires_grad=True)
            weight2 = torch.randn(num_experts, ffn, hidden_size, dtype=torch.float64, requires_grad=True)
            config = MoEConfig(hidden_size=hidden_size, moe_ffn_hidden_size=ffn,
                num_moe_experts=num_experts, expert_model_parallel_size=2,
                moe_router_topk=top_k, gated_linear_unit=True, activation_func=F.silu,
                moe_swiglu_clamp_gate_after_activation=after, params_dtype=torch.float64)
            experts = GroupedMLP(4, False, config)
            experts.swiglu_limit = limit
            with torch.no_grad():
                experts.weight1.copy_(weight1.detach()[rank * 4:(rank + 1) * 4].flatten(0, 1))
                experts.weight2.copy_(weight2.detach()[rank * 4:(rank + 1) * 4].flatten(0, 1))
            tokens = 0 if routing == "empty_rank" and rank == 0 else 9 + rank
            torch.manual_seed(31 + rank)
            hidden = torch.randn(tokens, hidden_size, dtype=torch.float64, requires_grad=True)
            ids = torch.randn(tokens, num_experts).topk(top_k, dim=-1).indices
            if routing in ("hot", "cold_rank"):
                ids[:] = torch.tensor([0, 4] if routing == "hot" else [0, 1])
            probs = torch.rand(tokens, top_k, dtype=torch.float64, requires_grad=True)
            if tokens:
                with torch.no_grad():
                    probs[0, 0] = 0
            ref_hidden = hidden.detach().clone().requires_grad_()
            ref_probs = probs.detach().clone().requires_grad_()
            actual = dispatcher.mega_moe(hidden, ids, probs, experts=experts,
                                        backend=SimpleNamespace(ep_group=dist.group.WORLD))
            expected = reference(ref_hidden, ids, ref_probs, weight1, weight2, limit, after)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            gradient = torch.randn_like(actual)
            (actual * gradient).sum().backward()
            (expected * gradient).sum().backward()
            torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(probs.grad, ref_probs.grad, rtol=1e-12, atol=1e-12)
            for parameter, baseline in ((experts.weight1, weight1), (experts.weight2, weight2)):
                if baseline.grad is None:
                    baseline.grad = torch.zeros_like(baseline)
                dist.all_reduce(baseline.grad)
                expected_grad = baseline.grad[rank * 4:(rank + 1) * 4].flatten(0, 1)
                torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-12, atol=1e-12)
    Path(output, f"rank{rank}.passed").write_text("12 EP2 forward/backward cases passed\n")
    dist.destroy_process_group()


def test_native_ep2_outputs_and_gradients(tmp_path):
    mp.spawn(native_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
             nprocs=2, join=True)
    assert all((tmp_path / f"rank{rank}.passed").is_file() for rank in range(2))


def test_custom_policy_does_not_cross_fall_back_to_native():
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry,
                              default_policy=OpPolicy.CUSTOM)
    with pytest.raises(RuntimeError, match="mega_moe.*policy=custom"):
        dispatcher.mega_moe(None, None, None, experts=None, backend=None)


@pytest.mark.parametrize("changes", [
    {"activation_func": F.gelu}, {"gated_linear_unit": False},
    {"moe_apply_probs_on_input": True},
])
def test_native_contract_rejects_unsupported_expert_semantics(changes):
    config = MoEConfig(hidden_size=8, moe_ffn_hidden_size=4,
                       gated_linear_unit=True, activation_func=F.silu)
    for name, value in changes.items():
        setattr(config, name, value)
    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    with pytest.raises(ValueError, match="SwiGLU experts with output-side routing weights"):
        dispatcher.mega_moe(None, None, None, experts=SimpleNamespace(config=config), backend=None)


def test_explicit_npu_implementation_rejects_cpu_runtime():
    from torchturbo.ops.providers.npu.mega_moe import AscendCMegaMoe  # noqa: F401

    dispatcher = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    with pytest.raises(RuntimeError, match="does not support device=cpu"):
        dispatcher.mega_moe(None, None, None, experts=None, backend=None, impl="ascendc_mega_moe")
