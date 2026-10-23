"""Model replacements honor engine MoE losses across sequence-parallel ranks."""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.models.bailing_hybrid.parallel_plan import _replace_moe_only as replace_bailing
from torchturbo.models.deepseek_v4.parallel_plan import _replace_moe_only as replace_deepseek
from torchturbo.models.glm_moe_dsa.parallel_plan import _replace_moe_only as replace_glm
from torchturbo.models.kimi_k25.parallel_plan import _replace_kimi_moe_only
from torchturbo.models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from torchturbo.models.llada2_moe.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
from torchturbo.models.llada2_moe.parallel_plan import _replace_llada2_moe_only
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoESubmodules
from torchturbo.models.qwen3_5_moe.parallel_plan import _replace_qwen3_5_moe_only
from torchturbo.models.qwen3_moe.parallel_plan import QWen3MoeLayerWrapper
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPParallelPlan, FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime
import torchturbo.runtime.runtime as runtime_module


def _adapter_cases():
    common = SimpleNamespace(
        hidden_size=4, moe_intermediate_size=4, num_experts=4,
        n_routed_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1,
        routed_scaling_factor=1.0, scoring_func="sigmoid", router_dtype="fp32",
        torch_dtype=torch.float32,
    )
    qwen = Qwen3MoeConfig(
        hidden_size=4, moe_intermediate_size=4, num_experts=4,
        num_experts_per_tok=2, torch_dtype=torch.float32,
    )
    llada = LLaDA2MoeConfig(
        hidden_size=4, moe_intermediate_size=4, num_experts=4,
        num_experts_per_tok=2, n_group=1, topk_group=1,
        num_shared_experts=1, torch_dtype=torch.float32,
    )
    deepseek = SimpleNamespace(**{**vars(common), "scoring_func": "sqrtsoftplus"})
    return [
        ("qwen3", qwen, QWen3MoeLayerWrapper, "softmax"),
        ("glm", common, replace_glm, "sigmoid"),
        ("kimi", SimpleNamespace(text_config=common), _replace_kimi_moe_only, "sigmoid"),
        ("deepseek_v4", deepseek, replace_deepseek, "sqrtsoftplus"),
        ("llada2", llada, _replace_llada2_moe_only, "sigmoid"),
        ("qwen3_5", common, _replace_qwen3_5_moe_only, "softmax"),
        ("bailing", common, replace_bailing, "sigmoid"),
    ]


def _reference_loss(weight, inputs, score_function, aux_coeff, z_coeff):
    logits = F.linear(inputs, weight)
    if score_function == "softmax":
        scores = logits.softmax(-1)
    elif score_function == "sigmoid":
        scores = logits.sigmoid()
        scores = scores / scores.sum(-1, keepdim=True)
    else:
        scores = F.softplus(logits).sqrt()
        scores = scores / scores.sum(-1, keepdim=True)
    counts = torch.bincount(scores.topk(2, dim=-1).indices.flatten(), minlength=4)
    aux_loss = 4 * (scores.sum(0) * counts).sum() / (2 * inputs.shape[0] ** 2)
    z_loss = logits.logsumexp(-1).square().mean()
    return aux_coeff * aux_loss + z_coeff * z_loss


def _check_adapter_losses(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        runtime = CpuRuntime()
        runtime_module._SELECTED_RUNTIME = runtime
        get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
        engine_config = EngineConfig()
        backend = FSDP2Backend(engine_config, runtime)
        submodules = MoESubmodules(experts=ModuleSpec(module=GroupedMLP))
        inputs = torch.tensor([
            [1., 2., 3., 4.], [2., 1., 0., -1.],
            [-1., 2., 0.5, 0.], [0.5, -1., 2., 1.],
        ])
        local_inputs = inputs[:1] if rank == 0 else inputs[1:]
        weight = torch.tensor([
            [0.5, -0.3, 0.7, -0.2], [-0.6, 0.2, 0.1, 0.8],
            [0.9, -0.5, 0.4, 0.3], [-0.4, 0.6, -0.7, 0.5],
        ])
        for name, config, replace, score_function in _adapter_cases():
            for aux_coeff, z_coeff in ((0., 0.), (0.2, 0.), (0., 0.3), (0.2, 0.3)):
                engine_config.moe.aux_loss_coeff = aux_coeff
                engine_config.moe.z_loss_coeff = z_coeff
                if name == "llada2":
                    original = LLaDA2MoeSparseMoeBlock(config)
                else:
                    original = nn.Module()
                    original.experts = nn.Identity()
                    original.gate = nn.Identity()
                    original.layer_idx = 0
                model = nn.Module()
                model.config = config
                model.mlp = original
                plan = FSDPParallelPlan(
                    plan={"mlp": FSDPPlanSpec(
                        replace_module=replace,
                        module_kwargs={"moe_submodules": submodules},
                    )},
                    sp_size=2, ep_size=1,
                )
                backend._init_mesh(plan)
                if name == "llada2" and (aux_coeff or z_coeff):
                    with pytest.raises(AssertionError, match="LLaDA2.*aux/z"):
                        backend._replace_modules(model, plan)
                    continue
                backend._replace_modules(model, plan)
                router = model.mlp.router
                with torch.no_grad():
                    router.weight.copy_(weight)
                routing, _ = router(local_inputs.unsqueeze(0))
                # Remove the main-loss contribution to isolate attached router losses.
                (routing.probs.sum() * 0).backward()
                gradient = router.weight.grad.clone()
                dist.all_reduce(gradient)
                gradient /= 2

                ref_weight = weight.clone().requires_grad_()
                expected_loss = _reference_loss(
                    ref_weight, inputs, score_function, aux_coeff, z_coeff,
                )
                expected_loss.backward()
                context = f"{name}, aux={aux_coeff}, z={z_coeff}, rank={rank}"
                torch.testing.assert_close(
                    gradient, ref_weight.grad,
                    msg=lambda message: f"{context}\n{message}",
                )
                metrics = router.pop_metrics()
                assert ("aux_loss" in metrics) == (aux_coeff > 0), context
                assert ("z_loss" in metrics) == (z_coeff > 0), context
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
def test_model_adapters_apply_configured_losses_across_sp(tmp_path) -> None:
    mp.spawn(
        _check_adapter_losses, args=(str(tmp_path / "adapter-sp"),),
        nprocs=2, join=True,
    )
