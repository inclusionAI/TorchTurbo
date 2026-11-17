from __future__ import annotations

import copy
import math

import pytest
import torch

from torchturbo.engine.config import MuonOptimizerConfig
from torchturbo.engine.core import TorchTurboEngine
from torchturbo.engine.optimizers import MuonPlan, MuonRecipe, OptimizersContainer
from torchturbo.engine.optimizers.lr_scheduler import create_lr_scheduler


def _make_optimizer(learning_rates=(2.5e-5, 1e-5)):
    model = torch.nn.Linear(2, 2)
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(
            lr=learning_rates[0],
            adamw_lr=learning_rates[1],
            weight_decay=0.0,
        ),
    )
    groups = plan.get_param_groups_by_optimizer_name()
    # Scheduling only needs the real group dictionaries; no gradients are set.
    optimizers = {
        "Muon": torch.optim.SGD(
            [group.get_optimizer_param_group() for group in groups["Muon"]],
            lr=learning_rates[0],
        ),
        "AdamW": torch.optim.AdamW(
            [group.get_optimizer_param_group() for group in groups["AdamW"]],
            lr=learning_rates[1],
        ),
    }
    return OptimizersContainer(plan, optimizers)


def _make_engine(optimizer):
    engine = object.__new__(TorchTurboEngine)
    engine.optimizer = optimizer
    engine.lr_scheduler = None
    return engine


def _make_scheduler(
    optimizer, *, name="cosine_with_min_lr", warmup_steps=2, **scheduler_kwargs
):
    return create_lr_scheduler(
        name,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=10,
        scheduler_specific_kwargs={"min_lr": 1e-7, **scheduler_kwargs},
    )


def _child_learning_rates(optimizer):
    return [
        group["lr"]
        for child in optimizer.optimizers
        for group in child.param_groups
    ]


def _megatron_cosine_lr(step, peak_lr, min_lr, warmup_steps, num_steps):
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (num_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (peak_lr - min_lr)


@pytest.mark.parametrize(
    ("learning_rates", "warmup_steps"),
    [
        ((2.5e-5, 1e-5), 2),
        ((1e-5, 2.5e-5), 2),
        ((1e-5, 1e-5), 2),
        ((2.5e-5, 1e-5), 0),
    ],
)
def test_absolute_minimum_matches_megatron_group_curves(learning_rates, warmup_steps):
    optimizer = _make_optimizer(learning_rates)
    scheduler = _make_scheduler(optimizer, warmup_steps=warmup_steps)

    assert scheduler.base_lrs == list(learning_rates)
    rates_by_step = []
    for step in range(11):
        expected = [
            _megatron_cosine_lr(step, lr, 1e-7, warmup_steps, 10)
            for lr in learning_rates
        ]
        assert scheduler.get_last_lr() == pytest.approx(expected)
        assert _child_learning_rates(optimizer) == pytest.approx(expected)
        rates_by_step.append(scheduler.get_last_lr())
        if step < 10:
            optimizer.step()
            scheduler.step()

    midpoint = warmup_steps + (10 - warmup_steps) // 2
    assert rates_by_step[warmup_steps] == pytest.approx(learning_rates)
    assert rates_by_step[midpoint] == pytest.approx(
        [(lr + 1e-7) / 2 for lr in learning_rates]
    )
    assert rates_by_step[-1] == pytest.approx([1e-7, 1e-7])


@pytest.mark.parametrize("warmup_lr_rate", [None, 0.1])
def test_warmup_cosine_minimum_matches_hugging_face_for_each_group(warmup_lr_rate):
    from transformers.optimization import get_scheduler

    learning_rates = (2.5e-5, 1e-5)
    optimizer = _make_optimizer(learning_rates)
    scheduler = _make_scheduler(
        optimizer, name="cosine_warmup_with_min_lr", warmup_lr_rate=warmup_lr_rate
    )
    # With one parameter group, Hugging Face's minimum LR is absolute.
    reference_optimizers = [
        torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=lr)
        for lr in learning_rates
    ]
    reference_schedulers = [
        get_scheduler(
            "cosine_warmup_with_min_lr",
            reference_optimizer,
            num_warmup_steps=2,
            num_training_steps=10,
            scheduler_specific_kwargs={
                "min_lr": 1e-7,
                "warmup_lr_rate": warmup_lr_rate,
            },
        )
        for reference_optimizer in reference_optimizers
    ]

    rates_by_step = []
    for _ in range(10):
        rates_by_step.append(_child_learning_rates(optimizer))
        expected = [
            reference_optimizer.param_groups[0]["lr"]
            for reference_optimizer in reference_optimizers
        ]
        assert rates_by_step[-1] == pytest.approx(expected)
        optimizer.step()
        scheduler.step()
        for reference_optimizer, reference_scheduler in zip(
            reference_optimizers, reference_schedulers, strict=True
        ):
            reference_optimizer.step()
            reference_scheduler.step()

    # This schedule trains its last step at the minimum LR.
    assert rates_by_step[-1] == pytest.approx([1e-7, 1e-7])


def test_resumed_scheduler_uses_initial_learning_rates():
    learning_rates = (2.5e-5, 1e-5)
    optimizer = _make_optimizer(learning_rates)
    for group, initial_lr in zip(optimizer.param_groups, learning_rates, strict=True):
        group["initial_lr"] = initial_lr
        group["lr"] = initial_lr / 4
    scheduler = _make_scheduler(optimizer, last_epoch=4)

    assert scheduler.last_epoch == 5
    assert scheduler.base_lrs == list(learning_rates)
    assert _child_learning_rates(optimizer) == pytest.approx(
        [_megatron_cosine_lr(5, lr, 1e-7, 2, 10) for lr in learning_rates]
    )


def test_scheduler_preserves_explicit_num_cycles():
    learning_rates = (2.5e-5, 1e-5)
    optimizer = _make_optimizer(learning_rates)
    scheduler = _make_scheduler(optimizer, num_cycles=0.25)
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    assert _child_learning_rates(optimizer) == pytest.approx(
        [(lr + 1e-7) / 2 for lr in learning_rates]
    )


@pytest.mark.parametrize(
    ("name", "scheduler_kwargs"),
    [
        ("cosine_with_min_lr", {"min_lr_rate": 0.01}),
        ("linear", {}),
        ("cosine", None),
    ],
)
def test_other_engine_named_schedules_match_hugging_face(name, scheduler_kwargs):
    from transformers.optimization import get_scheduler

    optimizer = _make_optimizer()
    engine = _make_engine(optimizer)
    expected_optimizer = _make_optimizer()
    original_kwargs = copy.deepcopy(scheduler_kwargs)
    scheduler = engine.prepare_lr_scheduler(
        name,
        num_warmup_steps=2,
        num_training_steps=10,
        scheduler_specific_kwargs=scheduler_kwargs,
    )
    expected_scheduler = get_scheduler(
        name,
        expected_optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        scheduler_specific_kwargs=scheduler_kwargs,
    )
    assert engine.lr_scheduler is scheduler
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        expected_optimizer.step()
        expected_scheduler.step()
        assert _child_learning_rates(optimizer) == pytest.approx(
            _child_learning_rates(expected_optimizer)
        )
    assert scheduler_kwargs == original_kwargs
    if name == "cosine_with_min_lr":
        assert _child_learning_rates(optimizer) == pytest.approx([2.5e-7, 1e-7])


def test_polynomial_schedule_rejects_different_group_learning_rates():
    # The container has no single default LR when its children use different
    # LRs, so a schedule that needs one fails instead of borrowing a child's.
    with pytest.raises(KeyError, match="lr"):
        create_lr_scheduler(
            "polynomial",
            _make_optimizer(),
            num_warmup_steps=2,
            num_training_steps=10,
        )


def test_single_optimizer_groups_reach_the_same_absolute_minimum():
    optimizer = torch.optim.AdamW(
        [
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 2.5e-5},
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 1e-5},
        ],
        lr=0.1,
    )
    engine = _make_engine(optimizer)
    scheduler = engine.prepare_lr_scheduler(
        "cosine_with_min_lr",
        num_warmup_steps=2,
        num_training_steps=10,
        scheduler_specific_kwargs={"min_lr": 1e-7},
    )
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-7, 1e-7])


def test_engine_preserves_class_scheduler_keyword_and_optimizer_override():
    from accelerate import Accelerator

    raw_optimizer = _make_optimizer()
    optimizer = Accelerator(cpu=True).prepare_optimizer(raw_optimizer, device_placement=False)
    engine = _make_engine(raw_optimizer)
    scheduler = engine.prepare_lr_scheduler(
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        optimizer=optimizer,
        step_size=1,
        gamma=0.5,
    )

    assert engine.lr_scheduler is scheduler
    assert scheduler.optimizer is optimizer
    optimizer.step()
    scheduler.step()
    assert _child_learning_rates(raw_optimizer) == pytest.approx([1.25e-5, 5e-6])


@pytest.mark.parametrize("name", ["cosine_with_min_lr", "cosine_warmup_with_min_lr"])
@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        {"min_lr": -1e-7},
        {"min_lr": 2e-5},
        {"min_lr": 1e-7, "min_lr_rate": 0.01},
    ],
)
def test_invalid_absolute_minimum_is_rejected(name, scheduler_kwargs):
    optimizer = _make_optimizer()

    with pytest.raises(ValueError, match="min_lr"):
        _make_scheduler(optimizer, name=name, **scheduler_kwargs)

    assert _child_learning_rates(optimizer) == [2.5e-5, 1e-5]


def test_zero_learning_rate_group_with_zero_minimum_stays_zero():
    optimizer = _make_optimizer((0.0, 1e-5))
    scheduler = _make_scheduler(optimizer, min_lr=0.0)

    for _ in range(10):
        optimizer.step()
        scheduler.step()
        assert _child_learning_rates(optimizer)[0] == 0.0
    assert _child_learning_rates(optimizer) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("name", ["cosine_with_min_lr", "cosine_warmup_with_min_lr"])
def test_unknown_scheduler_kwargs_are_not_ignored(name):
    with pytest.raises(TypeError, match="unsupported_option"):
        _make_scheduler(_make_optimizer(), name=name, unsupported_option=True)
