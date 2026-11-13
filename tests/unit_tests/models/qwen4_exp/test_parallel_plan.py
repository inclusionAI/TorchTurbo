"""Resolve Qwen4 checkpoint targets and exercise their backend recomputation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import ActivationCheckpointingConfig
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.logging import get_logger
from torchturbo.models.qwen4_exp.model import TurboQwen4ExpForConditionalGeneration

from .test_model import _small_config


class _CheckpointProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.1))
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return (hidden_states * self.scale).sin()


@pytest.mark.parametrize("enabled", [False, True])
def test_default_plan_recomputes_only_decoder_blocks(enabled):
    model = TurboQwen4ExpForConditionalGeneration(_small_config()).train()
    decoder_blocks = list(model.model.language_model.layers)
    vision_blocks = list(model.model.visual.blocks)
    blocks = decoder_blocks + vision_blocks
    for block in blocks:
        block.checkpoint_probe = _CheckpointProbe()
        # Keep the real model tree and plan resolution; a differentiable leaf
        # makes recomputation observable without requiring accelerator kernels.
        block.forward = lambda hidden_states, block=block: block.checkpoint_probe(
            hidden_states
        )
    plan = FSDP2Backend._compile_parallel_plan(
        model.get_default_parallel_plan(), build_model_fqn_map(model)
    )
    targets = {
        model.get_submodule(fqn)
        for fqn, spec in plan.modules.items()
        if spec.activation_checkpoint
    }
    assert targets == set(decoder_blocks)
    backend = object.__new__(FSDP2Backend)
    backend.config = SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=enabled)
    )
    backend.logger = get_logger(__name__)
    backend.prefetch_manager = SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: nullcontext()
    )
    backend._apply_activation_checkpointing(model, plan)

    torch.manual_seed(17)
    hidden_states = torch.randn(2, 3, requires_grad=True)
    reference_input = hidden_states.detach().clone().requires_grad_()
    reference_scales = [
        block.checkpoint_probe.scale.detach().clone().requires_grad_()
        for block in blocks
    ]
    actual, expected = hidden_states, reference_input
    for block, scale in zip(blocks, reference_scales, strict=True):
        actual = block(actual)
        expected = (expected * scale).sin()
    assert [block.checkpoint_probe.calls for block in blocks] == [1] * len(blocks)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    assert [block.checkpoint_probe.calls for block in decoder_blocks] == [
        2 if enabled else 1
    ] * len(decoder_blocks)
    assert [block.checkpoint_probe.calls for block in vision_blocks] == [1] * len(
        vision_blocks
    )
    torch.testing.assert_close(hidden_states.grad, reference_input.grad)
    for block, scale in zip(blocks, reference_scales, strict=True):
        torch.testing.assert_close(block.checkpoint_probe.scale.grad, scale.grad)
