"""Exercise first-use initialization without warming a reference collective.

Run this module in a fresh torchrun process with HCCL_OP_EXPANSION_MODE=HOST.
"""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from tests.ops.npu.test_virtual_expert_moe import moe_env
from tests.ops.npu.test_virtual_expert_weight_exchange import weight_env
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.runtime import is_npu_available

pytestmark = pytest.mark.skipif(not is_npu_available(), reason="NPU is unavailable")


def test_virtual_experts_start_before_reference(moe_env):
    runtime, group = moe_env
    config = MoEConfig(
        hidden_size=128, moe_ffn_hidden_size=512, num_moe_experts=2 * dist.get_world_size(group),
        expert_model_parallel_size=dist.get_world_size(group), moe_router_topk=1,
        gated_linear_unit=True, activation_func=torch.nn.functional.silu, moe_router_dtype="fp32",
        moe_token_dispatcher_type="flex", moe_virtual_expert_load_balance=True,
    )
    submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
    torch.manual_seed(370 + dist.get_rank(group))
    candidate = MoELayer(config, group, submodules=submodules).to(runtime.device)
    with torch.no_grad():
        for parameter in candidate.experts.parameters():
            parameter.normal_(std=0.015)
        candidate.router.weight.zero_()
        candidate.router.weight[0, 0] = 1
    inputs = torch.rand((1, 128, 128), device=runtime.device)
    inputs[:, :, 0] = 2
    inputs.requires_grad_()
    output = candidate(inputs)[0]
    output.sum().backward()
    actual = [output.detach().clone(), inputs.grad.clone()] + [p.grad.clone() for p in candidate.parameters()]
    reference = MoELayer(
        replace(config, moe_token_dispatcher_type="alltoall", moe_virtual_expert_load_balance=False),
        group, submodules=submodules,
    ).to(runtime.device)
    reference.load_state_dict(candidate.state_dict())
    inputs.grad = None
    expected = reference(inputs)[0]
    expected.sum().backward()
    wanted = [expected, inputs.grad] + [p.grad for p in reference.parameters()]
    for value, expected_value in zip(actual, wanted):
        torch.testing.assert_close(value, expected_value, atol=2e-5, rtol=2e-4)
