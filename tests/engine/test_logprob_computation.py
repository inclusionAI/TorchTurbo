"""Chunked/checkpointed packed logprobs retain dense values and gradients."""

from argparse import ArgumentTypeError
from unittest.mock import patch

import pytest
import torch

from torchturbo.adapters.rl import packed
from torchturbo.engine.config import EngineConfig, LogprobComputationConfig


@pytest.mark.parametrize("chunk_size", [1, 4, 32])
@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_chunked_logprobs_entropy_and_gradients(chunk_size, checkpoint_chunks):
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(13, 17, generator=generator, requires_grad=True)
    labels = torch.randint(17, (13,), generator=generator)
    candidate = logits.detach().clone().requires_grad_(True)
    dense_lp, dense_ent = packed._gather_logprobs_entropy(logits, labels, 0.7)
    lp, ent = packed.gather_logprobs_entropy(
        candidate,
        labels,
        0.7,
        chunk_size,
        checkpoint_chunks,
    )
    weights = torch.linspace(-1, 1, 13)
    ((dense_lp * weights).sum() + 0.03 * dense_ent.sum()).backward()
    ((lp * weights).sum() + 0.03 * ent.sum()).backward()
    torch.testing.assert_close(lp, dense_lp)
    torch.testing.assert_close(ent, dense_ent)
    torch.testing.assert_close(candidate.grad, logits.grad)


@pytest.mark.parametrize("no_grad,requires_grad", [(True, True), (False, False)])
def test_checkpointing_skipped_without_autograd(no_grad, requires_grad):
    logits = torch.zeros(7, 11, requires_grad=requires_grad)
    labels = torch.arange(7)
    with (
        torch.set_grad_enabled(not no_grad),
        patch.object(packed, "checkpoint") as checkpoint,
    ):
        lp, ent = packed.gather_logprobs_entropy(
            logits,
            labels,
            chunk_size=3,
            checkpoint_chunks=True,
        )
    checkpoint.assert_not_called()
    assert not lp.requires_grad and not ent.requires_grad
    torch.testing.assert_close(lp, torch.full((7,), -torch.tensor(11.0).log()))


def test_checkpoint_chunks_recompute_in_backward():
    logits = torch.randn(9, 11, requires_grad=True)
    labels = torch.arange(9)
    with patch.object(
        packed, "_gather_logprobs_entropy", wraps=packed._gather_logprobs_entropy
    ) as compute:
        lp, ent = packed.gather_logprobs_entropy(
            logits,
            labels,
            chunk_size=4,
            checkpoint_chunks=True,
        )
        assert compute.call_count == 3
        (lp.sum() + ent.sum()).backward()
        assert compute.call_count == 6


@pytest.mark.parametrize(
    "values,error",
    [
        ({"chunk_size": 0}, ValueError),
        ({"chunk_size": -1}, ValueError),
        ({"chunk_size": True}, TypeError),
        ({"chunk_size": 2.5}, TypeError),
        ({"checkpoint_chunks": "true"}, TypeError),
    ],
)
def test_packed_helper_rejects_invalid_options(values, error):
    with pytest.raises(error):
        packed.gather_logprobs_entropy(
            torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), **values
        )


def test_logprob_config_defaults_and_roundtrip():
    defaults = EngineConfig().logprob_computation
    assert defaults == LogprobComputationConfig(
        chunk_size=1024, checkpoint_chunks=False
    )
    assert EngineConfig(logprob_computation=None).logprob_computation == defaults
    config = EngineConfig.from_dict(
        {
            "logprob_computation": {"chunk_size": "8", "checkpoint_chunks": "true"},
        }
    )
    assert config.logprob_computation == LogprobComputationConfig(8, True)
    serialized = config.to_dict()["logprob_computation"]
    assert serialized == {"chunk_size": 8, "checkpoint_chunks": True}
    assert (
        EngineConfig.from_dict({"logprob_computation": serialized}).logprob_computation
        == config.logprob_computation
    )
    assert "logprob_computation" not in config.extra


@pytest.mark.parametrize(
    "values,error",
    [
        ({"chunk_size": 0}, ValueError),
        ({"chunk_size": -1}, ValueError),
        ({"chunk_size": True}, TypeError),
        ({"chunk_size": "bad"}, TypeError),
        ({"checkpoint_chunks": 1}, TypeError),
        ({"checkpoint_chunks": "bad"}, ArgumentTypeError),
        ({"typo": 1}, TypeError),
        (False, TypeError),
    ],
)
def test_logprob_config_rejects_invalid_options(values, error):
    with pytest.raises(error):
        EngineConfig(logprob_computation=values)
