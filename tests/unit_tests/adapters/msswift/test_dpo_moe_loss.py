"""DPO auxiliary gradients and metrics through the real Qwen3.5 MoE model."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torchturbo.adapters.msswift import engine_registry
from torchturbo.adapters.msswift.trainer import TurboDPOTrainerMixin
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.metrics import MetricTracker
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoESubmodules
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.models.qwen3_5_moe.parallel_plan import _replace_qwen3_5_moe_only
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

from .test_trainer_packed import _packed_batch


@pytest.fixture
def cpu_moe(monkeypatch, tmp_path):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        "torchturbo.models.modules.moe.token_dispatcher.get_device_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(MoEAuxLossAutoScaler, "loss_scale", 1.0)
    get_or_create_ops(device_runtime=runtime)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _model(engine_config, sp_info):
    config = Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 17, "hidden_size": 8, "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8, "num_experts": 2,
            "num_experts_per_tok": 1, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "layer_types": ["full_attention"], "mtp_num_hidden_layers": 0,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000.0,
                "mrope_section": [1, 1, 0],
            },
        },
        vision_config={
            "depth": 1, "hidden_size": 8, "intermediate_size": 16,
            "num_heads": 2, "out_hidden_size": 8, "patch_size": 2,
            "spatial_merge_size": 1, "temporal_patch_size": 1,
        },
    )
    config._attn_implementation = "eager"
    model = TurboQwen3_5MoeForConditionalGeneration(config)
    layer = model.model.language_model.layers[0]
    layer.mlp = _replace_qwen3_5_moe_only(
        layer.mlp, config, sp_info,
        ep_info={"ep_group": dist.group.WORLD, "ep_size": 1},
        moe_submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
        engine_config=engine_config,
    )
    torch.nn.init.normal_(layer.mlp.experts.weight1, std=0.02)
    torch.nn.init.normal_(layer.mlp.experts.weight2, std=0.02)
    return model


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 4])
def test_dpo_uses_engine_auxiliary_gradients_and_metrics(
    monkeypatch, cpu_moe, gradient_accumulation_steps,
):
    torch.manual_seed(7)
    engine = object.__new__(TorchTurboEngine)
    engine.config = EngineConfig()
    engine.config.mfu.enabled = False
    engine.config.moe.aux_loss_coeff = 0.02
    engine.metric_tracker = MetricTracker()
    sp_info = {"sp_size": 1, "sp_rank": 0, "sp_group": None}
    engine.backend = SimpleNamespace(
        get_sp_info=lambda: sp_info,
        get_parallel_context=lambda: {"sp": sp_info},
        prepare_inputs=lambda model, inputs: inputs,
    )
    engine._record_micro = lambda inputs: None
    model = _model(engine.config, sp_info)
    reference = _model(engine.config, sp_info).eval().requires_grad_(False)
    reference.load_state_dict(model.state_dict())

    class SwiftDPOBase:
        def __init__(self, model, ref_model=None):
            # Swift enables its HF auxiliary-loss path for a positive coefficient.
            self.aux_loss_enabled = True
            self.model = model

    class Trainer(TurboDPOTrainerMixin, SwiftDPOBase):
        pass

    monkeypatch.setattr(engine_registry, "get_engine_or_none", lambda: None)
    trainer = Trainer(model, ref_model=reference)
    assert trainer.aux_loss_enabled is False
    assert trainer.ref_model is reference
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    trainer.args = SimpleNamespace(ld_alpha=None)
    trainer.loss_type = ["sigmoid"]
    trainer.label_pad_token_id = -100
    trainer.current_gradient_accumulation_steps = gradient_accumulation_steps
    trainer.get_use_logits_to_keep = lambda: True

    router = model.model.language_model.layers[0].mlp.router
    captured = {}

    def capture_logits(module, args, output):
        captured["logits"] = output[1].reshape(-1, router.num_experts)

    router.register_forward_hook(capture_logits)
    batch = _packed_batch()

    def loss():
        policy = trainer.concatenated_forward(model, batch)
        with torch.no_grad():
            ref = trainer.concatenated_forward(reference, batch, is_ref_model=True)
        assert "aux_loss" not in policy
        assert "aux_loss" not in ref
        logratios = policy["chosen_logps"] - policy["rejected_logps"]
        ref_logratios = ref["chosen_logps"] - ref["rejected_logps"]
        return -F.logsigmoid(0.1 * (logratios - ref_logratios)).mean()

    router.config.moe_aux_loss_coeff = 0.0
    main_loss = loss()
    scores = captured["logits"].softmax(-1)
    assignments = F.one_hot(scores.argmax(-1), router.num_experts).float()
    aux_loss = router.num_experts * (scores.mean(0) * assignments.mean(0)).sum()
    coefficient = engine.config.moe.aux_loss_coeff
    expected_grad = torch.autograd.grad(
        (main_loss + coefficient * aux_loss) / gradient_accumulation_steps,
        router.weight,
        retain_graph=True,
    )[0]
    main_grad = torch.autograd.grad(
        main_loss / gradient_accumulation_steps, router.weight,
    )[0]
    assert not torch.allclose(expected_grad, main_grad)

    router.config.moe_aux_loss_coeff = coefficient
    actual_loss = loss()
    (actual_loss / gradient_accumulation_steps).backward()
    torch.testing.assert_close(actual_loss, main_loss)
    torch.testing.assert_close(router.weight.grad, expected_grad)

    engine.collect_metrics(model)
    logs = {}
    engine.report_metrics(logs, lambda value: value.reshape(1))
    assert logs["aux_loss"] == pytest.approx(aux_loss.item())
    assert logs["aux_loss_scaled"] == pytest.approx(coefficient * aux_loss.item())
    assert reference.model.language_model.layers[0].mlp.router.pop_metrics() == {}
