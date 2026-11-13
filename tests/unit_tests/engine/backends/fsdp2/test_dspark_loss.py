"""FSDP2 backend contract for model-provided DSpark losses."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import LossChunkConfig
from torchturbo.models.base import TurboModel
from torchturbo.models.deepseek_v4.mtp import DeepseekV4MTPCausalLMOutputWithPast
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

_SCALING = 0.5


@pytest.fixture(autouse=True)
def _process_ops(monkeypatch):
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)


def _backend(dspark_scaling=_SCALING):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.rank = 0
    backend.config = SimpleNamespace(
        loss_chunk=LossChunkConfig(enabled=False),
        mtp=SimpleNamespace(enabled=False, loss_scaling_factor=0.1, log_loss=False),
        dspark=SimpleNamespace(loss_scaling_factor=dspark_scaling),
    )
    backend.get_sp_info = lambda: {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    return backend


class _DSparkModel(TurboModel, nn.Module):
    """Returns fixed logits and a DSpark loss equal to ``dspark_value * weight``."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.logits = nn.Parameter(torch.randn(1, 6, 5))
        self.dspark_weight = nn.Parameter(torch.tensor(1.0))
        self.router_weight = nn.Parameter(torch.tensor(2.0))
        self.dspark_value = 0.0
        self.attach_router_aux_loss = False

    def forward(self, input_ids, position_ids, use_cache, cu_seq_lens_q=None, **kwargs):
        dspark_value = self.dspark_value
        def dspark_loss_fn(labels):
            loss = self.dspark_weight * dspark_value
            if self.attach_router_aux_loss:
                aux_loss = self.router_weight.square() * 0.1
                loss = MoEAuxLossAutoScaler.apply(loss, aux_loss)
            return loss

        return DeepseekV4MTPCausalLMOutputWithPast(
            logits=self.logits,
            dspark_loss_fn=dspark_loss_fn,
        )


def _micro_batch(labels):
    labels = torch.tensor([labels])
    return {
        "input_ids": torch.ones_like(labels),
        "labels": labels,
        "cu_seq_lens_q": torch.tensor([0, labels.size(1)], dtype=torch.int32),
    }


def _lm_loss_sum(model, labels):
    shifted = F.pad(torch.tensor([labels]), (0, 1), value=-100)[:, 1:]
    lm_sum = F.cross_entropy(
        model.logits.detach().flatten(0, 1), shifted.flatten(), reduction="sum"
    )
    return lm_sum, (shifted != -100).sum()


def test_dspark_loss_is_a_microbatch_mean_without_num_items():
    model = _DSparkModel()
    model.dspark_value = 3.0
    labels = [-100, -100, 1, 2, 3, 4]

    total, outputs = _backend().compute_loss(
        model, _micro_batch(labels), return_outputs=True
    )
    total.backward()

    lm_sum, num_tokens = _lm_loss_sum(model, labels)
    torch.testing.assert_close(total, lm_sum / num_tokens + _SCALING * 3.0)
    torch.testing.assert_close(model.dspark_weight.grad, torch.tensor(_SCALING * 3.0))
    assert outputs.dspark_loss_fn is None


def test_zero_dspark_scale_excludes_router_aux_gradients():
    model = _DSparkModel()
    model.dspark_value = 3.0
    model.attach_router_aux_loss = True
    labels = [-100, -100, 1, 2, 3, 4]

    total, _ = _backend(dspark_scaling=0.0).compute_loss(
        model, _micro_batch(labels), return_outputs=True
    )
    total.backward()

    lm_sum, num_tokens = _lm_loss_sum(model, labels)
    torch.testing.assert_close(total, lm_sum / num_tokens)
    assert model.dspark_weight.grad is None
    assert model.router_weight.grad is None


def test_dspark_loss_follows_num_items_across_gradient_accumulation():
    model = _DSparkModel()
    # Two micro-batches of one optimizer step with different token counts.
    micro_batches = [([-100, 1, 2, -100, -100, -100], 1.0), ([1, 2, 3, 4, 0, 1], 4.0)]
    token_counts = [_lm_loss_sum(model, labels)[1] for labels, _ in micro_batches]
    num_items = int(sum(token_counts))

    total = 0
    for labels, dspark_value in micro_batches:
        model.dspark_value = dspark_value
        total = total + _backend().compute_loss(
            model, _micro_batch(labels), num_items_in_batch=num_items
        )

    lm_loss = sum(_lm_loss_sum(model, labels)[0] for labels, _ in micro_batches) / num_items
    # DSpark means are weighted by each micro-batch's share of the LM tokens.
    dspark_mean = sum(
        count * value for count, (_, value) in zip(token_counts, micro_batches)
    ) / num_items
    torch.testing.assert_close(total, lm_loss + _SCALING * dspark_mean)
