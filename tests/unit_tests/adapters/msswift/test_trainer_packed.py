"""Packed DPO forward, label and segment helpers."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import torch  # noqa: E402

from torchturbo.adapters.msswift import engine_registry  # noqa: E402
from torchturbo.adapters.msswift.inputs_bridging import (  # noqa: E402
    bridge_packed_inputs,
)
from torchturbo.adapters.msswift.trainer import TurboDPOTrainerMixin  # noqa: E402
from torchturbo.models.base import TurboModel  # noqa: E402
from torchturbo.models.modules.attention.ulysses import (  # noqa: E402
    patch_forward_for_sp_input_slicing,
    ulysses_pad_inputs,
)

# (prompt, chosen response, rejected response) for each preference pair.
_PAIRS = [
    ([1, 2], [3, 4], [9]),
    ([5], [6, 7, 8], [10, 3]),
]


def _rows():
    """Packed row order: every chosen sample, then every rejected sample."""
    chosen = [(prompt, response) for prompt, response, _ in _PAIRS]
    rejected = [(prompt, response) for prompt, _, response in _PAIRS]
    return chosen + rejected


def _logits(table, input_ids, position_ids):
    # Each logit depends only on its own token and position, so a packed
    # forward must match one forward per sample.
    return table[input_ids] + 0.1 * position_ids.unsqueeze(-1)


class _TokenModel(TurboModel, torch.nn.Module):
    def __init__(self, vocab_size=11):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = torch.nn.Parameter(
            torch.randn(vocab_size, vocab_size, generator=generator)
        )
        self.calls = []

    def forward(self, input_ids, position_ids, logits_to_keep=0, **kwargs):
        self.calls.append(dict(kwargs, logits_to_keep=logits_to_keep))
        logits = _logits(self.table, input_ids, position_ids)
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits)


class _ExpandedTokenModel(_TokenModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        extra_logits = torch.zeros_like(outputs.logits[:, :1])
        outputs.logits = torch.cat((extra_logits, outputs.logits), dim=1)
        return outputs


class _DSparkTokenModel(_TokenModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs.dspark_loss_fn = lambda labels: torch.tensor(0.0)
        return outputs


class _Engine:
    def __init__(self):
        self.backend = SimpleNamespace(prepare_inputs=lambda model, inputs: inputs)
        self.sp_info = {"sp_size": 1, "sp_group": None, "sp_rank": 0}
        self.recorded = []
        self.moe_steps = []

    def get_sp_info(self):
        return self.sp_info

    def _record_micro(self, inputs):
        self.recorded.append(tuple(inputs["input_ids"].shape))

    def _prepare_moe_runtime(self, model, inputs, gradient_accumulation_steps):
        self.moe_steps.append(gradient_accumulation_steps)


def _packed_batch():
    input_ids, labels, position_ids = [], [], []
    for prompt, response in _rows():
        input_ids += prompt + response
        labels += [-100] * len(prompt) + response
        position_ids += range(len(prompt) + len(response))
    batch = bridge_packed_inputs(
        {
            "input_ids": torch.tensor([input_ids]),
            "labels": torch.tensor([labels]),
            "position_ids": torch.tensor([position_ids]),
        },
        sequence_lengths=[len(prompt) + len(response) for prompt, response in _rows()],
    )
    batch["text_position_ids"] = batch["position_ids"].clone()
    batch["loss_scale"] = torch.ones_like(batch["labels"], dtype=torch.float32)
    return batch


def _trainer(monkeypatch, loss_type="sigmoid", ld_alpha=None, use_logits_to_keep=False):
    engine = _Engine()
    monkeypatch.setattr(engine_registry, "get_engine", lambda: engine)
    trainer = object.__new__(TurboDPOTrainerMixin)
    trainer.label_pad_token_id = -100
    trainer.loss_type = [loss_type]
    trainer.args = SimpleNamespace(ld_alpha=ld_alpha)
    trainer.aux_loss_enabled = False
    trainer.current_gradient_accumulation_steps = 4
    trainer.get_use_logits_to_keep = lambda: use_logits_to_keep
    return trainer, engine


def _reference(model, loss_type, ld_alpha=None):
    """Per-sample DPO outputs, one forward per sample."""
    rows = _rows()
    num_pairs = len(_PAIRS)
    sample_logps, chosen_logps, chosen_logits, rejected_logits = [], [], [], []
    for index, (prompt, response) in enumerate(rows):
        input_ids = torch.tensor([prompt + response])
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        # Position t predicts token t + 1, so the response is predicted from
        # the last prompt token through the second-to-last token.
        logits = _logits(model.table, input_ids, position_ids)[0, len(prompt) - 1:-1]
        token_logps = logits.log_softmax(-1).gather(
            -1, torch.tensor(response).unsqueeze(-1)
        ).squeeze(-1)
        weights = torch.ones(len(response))
        if loss_type == "ipo":
            weights = weights / len(response)
        if ld_alpha is not None:
            pair = index % num_pairs
            public_length = min(len(rows[pair][1]), len(rows[pair + num_pairs][1]))
            weights[public_length:] *= ld_alpha
        sample_logps.append((token_logps * weights).sum())
        if index < num_pairs:
            chosen_logps.append(token_logps)
            chosen_logits.append(logits.mean(-1))
        else:
            rejected_logits.append(logits.mean(-1))
    sample_logps = torch.stack(sample_logps)
    return {
        "chosen_logps": sample_logps[:num_pairs],
        "rejected_logps": sample_logps[num_pairs:],
        "mean_chosen_logits": torch.cat(chosen_logits).mean(),
        "mean_rejected_logits": torch.cat(rejected_logits).mean(),
        "nll_loss": -torch.cat(chosen_logps).mean(),
    }


def _assert_outputs_close(output, expected):
    assert output.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(output[key], value, msg=key)


@pytest.mark.parametrize("use_logits_to_keep", [False, True])
@pytest.mark.parametrize("loss_type", ["sigmoid", "ipo"])
def test_dpo_forward_matches_per_sample_reference(
    monkeypatch, loss_type, use_logits_to_keep
):
    trainer, engine = _trainer(
        monkeypatch, loss_type=loss_type, use_logits_to_keep=use_logits_to_keep
    )
    model = _TokenModel()

    output = trainer.concatenated_forward(model, _packed_batch())

    _assert_outputs_close(output, _reference(model, loss_type))
    (call,) = model.calls
    # Boundaries reach the model only as cu/max metadata.
    assert call["cu_seq_lens_q"].tolist() == [0, 4, 8, 11, 14]
    assert call["max_length_q"] == 4
    assert not {"attention_mask", "labels", "loss_scale", "text_position_ids"} & call.keys()
    keep = call["logits_to_keep"]
    assert isinstance(keep, torch.Tensor) is use_logits_to_keep
    if use_logits_to_keep:
        # One logit per supervised response token.
        assert int(keep.sum()) == sum(len(response) for _, response in _rows())
    assert engine.recorded == [(1, 14)]
    assert engine.moe_steps == [4]


def test_dpo_forward_applies_ld_alpha_to_the_policy_only(monkeypatch):
    trainer, engine = _trainer(monkeypatch, ld_alpha=0.5)
    model = _TokenModel()

    policy = trainer.concatenated_forward(model, _packed_batch())
    _assert_outputs_close(policy, _reference(model, "sigmoid", ld_alpha=0.5))

    model.eval()
    reference = trainer.concatenated_forward(model, _packed_batch(), is_ref_model=True)
    expected = _reference(model, "sigmoid")
    del expected["nll_loss"]
    _assert_outputs_close(reference, expected)
    # Reference forwards do not count toward MFU or the aux-loss scale.
    assert engine.recorded == [(1, 14)]
    assert engine.moe_steps == [4, 1]


@pytest.mark.parametrize("use_logits_to_keep", [False, True])
def test_sp_dpo_forward_matches_per_sample_reference(monkeypatch, use_logits_to_keep):
    """Every SP rank feeds the global row to a model that slices its inputs."""
    # The 14-token row pads to 16, so the last shard also holds the SP tail.
    sp_size = 4
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: sp_size)
    trainer, engine = _trainer(
        monkeypatch, loss_type="ipo", ld_alpha=0.5, use_logits_to_keep=use_logits_to_keep
    )
    model = _TokenModel()
    spec = model.get_sequence_parallel_input_slicing_spec()
    engine.backend.prepare_inputs = lambda model, inputs: ulysses_pad_inputs(
        inputs, spec, sp_size
    )

    def forward_on_every_rank(all_reduce):
        monkeypatch.setattr("torch.distributed.nn.functional.all_reduce", all_reduce)
        outputs = []
        for sp_rank in range(sp_size):
            engine.sp_info = {"sp_size": sp_size, "sp_group": None, "sp_rank": sp_rank}
            patch_forward_for_sp_input_slicing(spec, None, sp_size, sp_rank)
            outputs.append(trainer.concatenated_forward(model, _packed_batch()))
        return outputs

    # Collect each rank's partial sums, then replay every rank with their
    # total as the all-reduce result.
    partial_stats = []

    def record_partial_stats(stats, group):
        partial_stats.append(stats)
        return stats

    forward_on_every_rank(record_partial_stats)
    total_stats = torch.stack(partial_stats).sum(0)
    expected = _reference(model, "ipo", ld_alpha=0.5)
    for output in forward_on_every_rank(lambda stats, group: total_stats):
        _assert_outputs_close(output, expected)


@pytest.mark.parametrize(
    ("use_logits_to_keep", "expected_shapes"),
    [
        (False, "logits (1, 15) and labels (1, 14)"),
        (True, "logits (1, 9) and labels (1, 8)"),
    ],
)
def test_dpo_forward_rejects_model_expanded_sequence(
    monkeypatch, use_logits_to_keep, expected_shapes
):
    trainer, _ = _trainer(monkeypatch, use_logits_to_keep=use_logits_to_keep)

    with pytest.raises(
        ValueError,
        match="logits batch and sequence dimensions to match labels",
    ) as exc_info:
        trainer.concatenated_forward(_ExpandedTokenModel(), _packed_batch())
    assert expected_shapes in str(exc_info.value)


def test_dpo_forward_applies_the_model_input_hook(monkeypatch):
    trainer, _ = _trainer(monkeypatch)
    batch = _packed_batch()
    # The base hook rejects mask-delimited batches before SP padding.
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])

    with pytest.raises(ValueError, match="not attention_mask"):
        trainer.concatenated_forward(_TokenModel(), batch)


def test_dpo_forward_rejects_dspark_outputs(monkeypatch):
    trainer, _ = _trainer(monkeypatch)

    with pytest.raises(
        ValueError,
        match="DSpark training is not implemented for Swift DPO",
    ):
        trainer.concatenated_forward(_DSparkTokenModel(), _packed_batch())


@pytest.mark.parametrize("use_logits_to_keep", [False, True])
def test_dpo_forward_matches_swift_padding_free_forward(monkeypatch, use_logits_to_keep):
    pytest.importorskip("swift", minversion="4.0")
    from swift.rlhf_trainers import DPOTrainer

    trainer, _ = _trainer(monkeypatch, use_logits_to_keep=use_logits_to_keep)
    model = _TokenModel()
    output = trainer.concatenated_forward(model, _packed_batch())

    swift_trainer = object.__new__(DPOTrainer)
    swift_trainer.template = SimpleNamespace(padding_free=True, sequence_parallel_size=1)
    swift_trainer.args = SimpleNamespace(ld_alpha=None)
    swift_trainer.loss_type = ["sigmoid"]
    swift_trainer.label_pad_token_id = -100
    swift_trainer.aux_loss_enabled = False
    swift_trainer.is_encoder_decoder = False
    swift_trainer.get_use_logits_to_keep = lambda default_value: use_logits_to_keep
    with torch.no_grad():
        expected = DPOTrainer.concatenated_forward(swift_trainer, model, _packed_batch())

    _assert_outputs_close(output, expected)


def test_shift_packed_labels_does_not_cross_segments():
    labels = torch.tensor([[10, 11, 20, 21, 22, 30, 31, 99]])
    segment_ids = torch.tensor([[1, 1, 2, 2, 2, 3, 3, 0]])
    shifted = TurboDPOTrainerMixin._shift_packed_labels(labels, segment_ids, -100)
    assert shifted.tolist() == [[11, -100, 21, 22, -100, 31, -100, -100]]


def test_dpo_logps_use_full_log_softmax_and_mask_padding():
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]], dtype=torch.bfloat16
    )
    labels = torch.tensor([[2, -100]])

    logps, mean_logits, loss_mask = TurboDPOTrainerMixin.get_per_token_logps(
        None, logits, labels, label_pad_token_id=-100
    )
    expected_logps = logits.float().log_softmax(-1)[..., 2]

    assert torch.allclose(logps, torch.tensor([[expected_logps[0, 0], 0.0]]))
    assert torch.allclose(mean_logits, torch.tensor([[2.0, 0.0]]))
    assert loss_mask.tolist() == [[True, False]]
