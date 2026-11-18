"""Launch with torchrun -m pytest on two A3 ranks (EP) or four (EP plus FSDP)."""

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from torchturbo.engine.config import MegaMoEConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.ops import get_or_create_ops, OpPolicy
from torchturbo.runtime import get_device_runtime


@pytest.fixture(scope="module")
def expert_mesh():
    """Keep one communicator generation for all parameterized device cases."""
    pytest.importorskip("torch_npu")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        get_or_create_ops(device_runtime=get_device_runtime(), default_policy=OpPolicy.AUTO)
        mesh = init_device_mesh("npu", (dist.get_world_size() // 2, 2), mesh_dim_names=("edp", "ep"))
        yield mesh
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", 1)) not in (2, 4), reason="requires two or four torchrun ranks")
@pytest.mark.parametrize(("hot_route", "hot_replica", "reentrant"), [
    (False, False, False), (True, False, False), (True, True, False),
    pytest.param(True, True, True, marks=pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", 1)) != 4, reason="requires sharded expert parameters",
    ), id="hot-replica-fsdp-reentrant"),
])
def test_mega_moe_matches_native_expert_parallel(expert_mesh, hot_route, hot_replica, reentrant):
    from torch.utils.checkpoint import checkpoint

    mesh = expert_mesh
    ep_group = mesh["ep"].get_group()
    cfg = MoEConfig(
        hidden_size=512, moe_ffn_hidden_size=128, num_moe_experts=8,
        expert_model_parallel_size=dist.get_world_size(ep_group),
        moe_router_topk=2, moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        activation_func=F.silu, gated_linear_unit=True, params_dtype=torch.bfloat16,
        moe_swiglu_limit=[0.5], moe_swiglu_clamp_gate_after_activation=True,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    native = MoELayer(cfg, ep_group, submodules, layer_number=0).npu()
    mega_moe = MegaMoEConfig(enable=True, expert_balance="hot_replica" if hot_replica else None)
    candidate = MoELayer(replace(cfg, mega_moe=mega_moe), ep_group, submodules, layer_number=0).npu()
    torch.manual_seed(124 + dist.get_rank(ep_group))
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.normal_(std=0.035)
        if hot_route:
            # Positive inputs select experts 0 and 1 on the same owner.
            native.router.weight.fill_(-0.01)
            native.router.weight[:2].fill_(0.01)
    candidate.load_state_dict(native.state_dict())
    if dist.get_world_size() == 4:
        from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend, _FSDPUnit, _ResolvedFSDPPlan
        from torchturbo.engine.config import EngineConfig
        from torchturbo.parallel import FSDPPlanSpec

        backend = FSDP2Backend(EngineConfig.from_dict({
            "moe": {"mega_moe": {"enable": True, "expert_balance": mega_moe.expert_balance}},
            "activation_checkpointing": {"enabled": reentrant, "use_reentrant": True},
        }), get_device_runtime())
        if reentrant:
            plan = _ResolvedFSDPPlan(modules={"": FSDPPlanSpec(activation_checkpoint=True)}, parameters={})
            backend._apply_activation_checkpointing(candidate, plan)
        backend._apply_fsdp(
            candidate,
            [_FSDPUnit(modules=[candidate.experts], mesh=mesh["edp"])],
            {"mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
            )},
        )
    assert list(native.state_dict()) == list(candidate.state_dict())

    if hot_replica and not reentrant:
        other_submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
        other_native = MoELayer(cfg, ep_group, other_submodules, layer_number=0).npu()
        other_candidate = MoELayer(
            replace(cfg, mega_moe=mega_moe), ep_group, other_submodules, layer_number=0,
        ).npu()
        with torch.no_grad():
            other_native.load_state_dict(native.state_dict())
            other_native.experts.weight1.mul_(1.5)
        other_candidate.load_state_dict(other_native.state_dict())

    torch.manual_seed(2026 + dist.get_rank())
    for rows, recompute in ((128, False), (256, True)):
        x = torch.randn(rows, 1, 512, device="npu", dtype=torch.bfloat16, requires_grad=True)
        if hot_route:
            x = x.detach().abs().requires_grad_()
        other = x.detach().clone().requires_grad_()
        native.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        expected, _ = native(x)
        if recompute and not reentrant:
            actual, _ = checkpoint(candidate, other, use_reentrant=False)
        else:
            actual, _ = candidate(other)
        if hot_replica and not reentrant:
            second_input = torch.randn(192, 1, 512, device="npu", dtype=torch.bfloat16).abs().requires_grad_()
            second_input_copy = second_input.detach().clone().requires_grad_()
            other_native.zero_grad(set_to_none=True)
            other_candidate.zero_grad(set_to_none=True)
            second_expected, _ = other_native(second_input)
            second_actual, _ = other_candidate(second_input_copy)

        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        actual.backward(upstream)
        if hot_replica and not reentrant:
            assert all(parameter.grad is None for parameter in other_candidate.parameters())
            torch.testing.assert_close(second_actual, second_expected, atol=0.012, rtol=0.04)
            second_upstream = torch.randn_like(second_expected)
            second_expected.backward(second_upstream)
            second_actual.backward(second_upstream)
            torch.testing.assert_close(second_input_copy.grad, second_input.grad, atol=0.012, rtol=0.04)
            torch.testing.assert_close(
                other_candidate.experts.weight1.grad, other_native.experts.weight1.grad, atol=0.012, rtol=0.04,
            )
        pairs = [(actual, expected), (other.grad, x.grad)]
        for name, param in candidate.named_parameters():
            reference = native.get_parameter(name).grad
            gradient = param.grad
            if isinstance(gradient, DTensor):
                gradient = gradient.full_tensor()
                # FSDP averages expert gradients over replicas of the same EP rank.
                reference = reference.float()
                dist.all_reduce(reference, group=mesh["edp"].get_group())
                reference = (reference / mesh["edp"].size()).to(gradient.dtype)
            pairs.append((gradient, reference))
        for result, reference in pairs:
            assert result is not None and reference is not None
            torch.testing.assert_close(result, reference, atol=0.012, rtol=0.04)
