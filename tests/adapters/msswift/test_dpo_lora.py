"""LoRA DPO uses a fixed base reference and updates only the adapters."""

import copy
import os
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch
from peft import LoraConfig
from torch import nn
from transformers import PretrainedConfig

from torchturbo.adapters.msswift import engine_registry
from torchturbo.adapters.msswift.trainer import make_turbo_dpo_trainer
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.engine.peft import apply_peft
from torchturbo.models.base import TurboModel
from torchturbo.parallel import FSDPPlanSpec

from .test_trainer_packed import _Engine, _packed_batch


class PreferenceTokenModel(TurboModel, nn.Module):
    """Token projection used to exercise real PEFT and Swift DPO loss paths."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = PretrainedConfig()
        self.embedding = nn.Embedding(11, 8)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(8, 11, bias=False)
        self.calls = []
        self.fail_forward = False

    def forward(self, input_ids, position_ids, logits_to_keep=0, **kwargs):
        assert not {"ref_chosen_logps", "ref_rejected_logps"} & kwargs.keys()
        self.calls.append((self.training, torch.is_grad_enabled()))
        if self.fail_forward:
            raise RuntimeError("reference forward failed")
        hidden = self.dropout(self.embedding(input_ids))
        if isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(logits=self.projection(hidden))


def _model(dropout=0.0):
    torch.manual_seed(23)
    return apply_peft(
        PreferenceTokenModel(dropout),
        LoraConfig(r=2, lora_alpha=4, target_modules=["projection"]),
    )


def _trainer(monkeypatch, model):
    engine = _Engine()
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    trainer = object.__new__(make_turbo_dpo_trainer())
    trainer.model = model
    trainer.ref_model = None
    trainer.ref_adapter_name = None
    trainer.model_adapter_name = None
    trainer._peft_has_been_casted_to_bf16 = False
    trainer.label_pad_token_id = -100
    trainer.loss_type = ["sigmoid"]
    trainer.loss_weights = None
    trainer.beta = 0.1
    trainer.label_smoothing = 0.0
    trainer.reference_free = False
    trainer.f_divergence_type = "reverse_kl"
    trainer.use_weighting = False
    trainer.aux_loss_enabled = False
    trainer.current_gradient_accumulation_steps = 4
    trainer.args = SimpleNamespace(ld_alpha=None, rpo_alpha=None)
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda model: model,
        gather_for_metrics=lambda value: value,
    )
    trainer.get_use_logits_to_keep = lambda: True
    return trainer, engine


def _apply_checkpointing(model):
    backend = object.__new__(FSDP2Backend)
    backend.config = EngineConfig()
    backend.config.activation_checkpointing.enabled = True
    backend.config.activation_checkpointing.use_reentrant = False
    backend.prefetch_manager = SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: nullcontext(),
    )
    plan = SimpleNamespace(modules={
        "base_model.model.projection": FSDPPlanSpec(activation_checkpoint=True),
    })
    backend._apply_activation_checkpointing(model, plan)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_lora_dpo_matches_fixed_reference_loss_and_gradients(monkeypatch, checkpointed):
    model = _model()
    expected_model = copy.deepcopy(model)
    trainer, engine = _trainer(monkeypatch, model)
    batch = _packed_batch()
    if checkpointed:
        _apply_checkpointing(model)
    base_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    initial_adapter = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    with torch.no_grad(), expected_model.disable_adapter():
        fixed_reference = trainer.concatenated_forward(
            expected_model.eval(), batch, is_ref_model=True,
        )
    expected_model.train()
    engine.recorded.clear()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.03, weight_decay=0.0,
    )
    for _ in range(2):
        expected_model.load_state_dict(model.state_dict())
        expected_model.zero_grad(set_to_none=True)
        expected_policy = trainer.concatenated_forward(expected_model, batch)
        margin = (
            expected_policy["chosen_logps"] - expected_policy["rejected_logps"]
            - fixed_reference["chosen_logps"] + fixed_reference["rejected_logps"]
        )
        expected_loss = -torch.nn.functional.logsigmoid(trainer.beta * margin).mean()
        (expected_loss / trainer.current_gradient_accumulation_steps).backward()
        engine.recorded.clear()
        model.base_model.model.calls.clear()

        loss, metrics = trainer.get_batch_loss_metrics(model, batch)
        (loss / trainer.current_gradient_accumulation_steps).backward()
        torch.testing.assert_close(loss, expected_loss)
        assert model.base_model.model.calls == [(True, True), (False, False)]
        assert engine.recorded == [(1, 14)]
        assert model.training
        assert not model.base_model.model.projection.disable_adapters
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        expected_parameters = dict(expected_model.named_parameters())
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.dtype == torch.float32
                assert parameter.grad is not None
                torch.testing.assert_close(parameter.grad, expected_parameters[name].grad)
            else:
                assert parameter.grad is None
                torch.testing.assert_close(parameter, base_parameters[name], rtol=0, atol=0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference = trainer.compute_ref_log_probs(batch)
        torch.testing.assert_close(reference[0], fixed_reference["chosen_logps"], rtol=0, atol=0)
        torch.testing.assert_close(reference[1], fixed_reference["rejected_logps"], rtol=0, atol=0)

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, initial_adapter[name])


@pytest.mark.parametrize("training", [False, True])
def test_reference_restores_modes_and_adapters_after_failure(monkeypatch, training):
    model = _model(dropout=0.4)
    model.train(training)
    model.base_model.model.embedding.eval()
    trainer, engine = _trainer(monkeypatch, model)
    modes = [(module, module.training) for module in model.modules()]
    trainable = [parameter.requires_grad for parameter in model.parameters()]
    first = trainer.compute_ref_log_probs(_packed_batch())
    second = trainer.compute_ref_log_probs(_packed_batch())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert [(module, module.training) for module in model.modules()] == modes
    assert engine.recorded == []

    model.base_model.model.fail_forward = True
    with pytest.raises(RuntimeError, match="reference forward failed"):
        trainer.compute_ref_log_probs(_packed_batch())
    assert [(module, module.training) for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == trainable
    assert not model.base_model.model.projection.disable_adapters


def test_cached_reference_is_kept_out_of_forward_and_matches_online_loss(monkeypatch):
    model = _model()
    trainer, _ = _trainer(monkeypatch, model)
    batch = _packed_batch()
    chosen, rejected = trainer.compute_ref_log_probs(batch)
    online_loss, _ = trainer.get_batch_loss_metrics(model, batch)
    cached_batch = dict(batch, ref_chosen_logps=chosen, ref_rejected_logps=rejected)

    def unexpected_reference(batch):
        pytest.fail("A complete reference cache must not trigger an online reference forward")

    trainer.compute_ref_log_probs = unexpected_reference
    cached_loss, _ = trainer.get_batch_loss_metrics(model, cached_batch)
    torch.testing.assert_close(cached_loss, online_loss)
    assert cached_batch["ref_chosen_logps"] is chosen
    assert cached_batch["ref_rejected_logps"] is rejected


@pytest.mark.parametrize("field", ["ref_chosen_logps", "ref_rejected_logps"])
def test_partial_reference_cache_is_rejected_before_forward(monkeypatch, field):
    model = _model()
    trainer, _ = _trainer(monkeypatch, model)
    batch = dict(_packed_batch(), **{field: torch.zeros(2)})
    with pytest.raises(ValueError, match="reference cache requires both"):
        trainer.get_batch_loss_metrics(model, batch)
    assert model.base_model.model.calls == []


def _run_fsdp_lora_dpo(rank, world_size, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank,
        world_size=world_size, timeout=timedelta(seconds=60),
    )
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            model = _model()
            expected_model = copy.deepcopy(model)
            trainer, _ = _trainer(monkeypatch, model)
            batch = _packed_batch()
            with torch.no_grad(), expected_model.disable_adapter():
                reference = trainer.concatenated_forward(
                    expected_model.eval(), batch, is_ref_model=True,
                )
            expected_model.train()
            _apply_checkpointing(model)
            mesh = init_device_mesh("cpu", (world_size,))
            fully_shard(model.base_model.model.projection, mesh=mesh)
            fully_shard(model, mesh=mesh)
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=0.03, weight_decay=0.0,
            )
            expected_optimizer = torch.optim.AdamW(
                (parameter for parameter in expected_model.parameters() if parameter.requires_grad),
                lr=0.03, weight_decay=0.0,
            )
            for _ in range(2):
                policy = trainer.concatenated_forward(expected_model, batch)
                margin = (
                    policy["chosen_logps"] - policy["rejected_logps"]
                    - reference["chosen_logps"] + reference["rejected_logps"]
                )
                expected_loss = -torch.nn.functional.logsigmoid(trainer.beta * margin).mean()
                expected_loss.backward()
                loss, _ = trainer.get_batch_loss_metrics(model, batch)
                loss.backward()
                torch.testing.assert_close(loss, expected_loss)
                expected_parameters = dict(expected_model.named_parameters())
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(
                            parameter.grad.full_tensor(), expected_parameters[name].grad,
                        )
                    else:
                        assert parameter.grad is None
                optimizer.step()
                expected_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                expected_optimizer.zero_grad(set_to_none=True)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.full_tensor(), expected_parameters[name])
                current_reference = trainer.compute_ref_log_probs(batch)
                torch.testing.assert_close(current_reference[0], reference["chosen_logps"])
                torch.testing.assert_close(current_reference[1], reference["rejected_logps"])
    finally:
        dist.destroy_process_group()


def test_lora_dpo_with_fsdp_and_activation_checkpointing(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _run_fsdp_lora_dpo,
        args=(2, str(tmp_path / "lora-dpo")), nprocs=2, join=True,
    )
