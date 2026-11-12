"""Packed MTP and DSpark training contracts on the CPU reference model."""

import importlib
import json
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchturbo.models.deepseek_v4 import model as turbo_model
from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
from torchturbo.models.deepseek_v4 import mtp as dspark_mtp
from torchturbo.models.deepseek_v4.mtp import DeepseekV4MultiTokenPredictor
from torchturbo.models.deepseek_v4.parallel_plan import _replace_moe_only
from torchturbo.engine.config import EngineConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime

from .test_packed_inputs import (
    _make_model,
    _parallelize,
    restore_attention_forwards,
)


def _make_mtp_model(block_size=0, num_stages=2):
    model = _make_model(1)
    config = model.config
    config.num_nextn_predict_layers = num_stages
    config.dspark_num_layers = num_stages if block_size else 0
    config.compress_ratios = [1] * (1 + num_stages)
    config.sliding_window = 4
    config.dspark_block_size = block_size
    config.dspark_target_layer_ids = [0]
    config.dspark_markov_rank = 4
    config.dspark_enabled = True
    config.dspark_num_anchors = 32
    config.dspark_loss_decay_gamma = 4.0
    config.dspark_ce_loss_alpha = 0.1
    config.dspark_l1_loss_alpha = 0.9
    config.dspark_confidence_loss_alpha = 1.0
    model.mtp = DeepseekV4MultiTokenPredictor(config)
    model.train()
    _parallelize(model)
    return model


@pytest.mark.parametrize("padding", [0, 3])
def test_packed_mtp_logits_and_gradients_match_independent_samples(padding):
    model = _make_mtp_model()
    samples = [torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[6, 7, 8, 9, 10, 11]])]
    input_ids = torch.cat([*samples, torch.zeros(1, padding, dtype=torch.long)], dim=1)
    position_ids = torch.cat(
        [torch.arange(sample.size(1)) for sample in samples]
        + [torch.zeros(padding, dtype=torch.long)]
    ).unsqueeze(0)
    cu = torch.tensor([0, 5, 11], dtype=torch.int32)
    packed = model(
        input_ids=input_ids,
        position_ids=position_ids,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=6,
        max_length_k=6,
    )
    independent = [model(input_ids=sample) for sample in samples]
    packed_loss = 0
    independent_loss = 0
    for stage, packed_logits in enumerate(packed.mtp_logits):
        start = 0
        for sample, outputs in zip(samples, independent):
            num_predictions = sample.size(1) - stage - 2
            actual = packed_logits[:, start : start + num_predictions]
            expected = outputs.mtp_logits[stage][:, :num_predictions]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            packed_loss = packed_loss + actual.square().sum()
            independent_loss = independent_loss + expected.square().sum()
            start += sample.size(1)

    parameters = (
        model.embed.weight,
        model.layers[0].attn.wq_a.weight,
        model.mtp[0].h_proj.weight,
        model.mtp[1].attn.wkv.weight,
        model.head.weight,
    )
    packed_grads = torch.autograd.grad(packed_loss, parameters)
    independent_grads = torch.autograd.grad(independent_loss, parameters)
    for actual, expected in zip(packed_grads, independent_grads):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_dspark_context_keeps_first_token_and_respects_packed_boundaries():
    model = _make_mtp_model(block_size=2)
    input_ids = torch.arange(1, 11).unsqueeze(0)
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 0]])
    captured = []
    hook = model.mtp[0].attn.register_forward_pre_hook(
        lambda _module, _args, kwargs: captured.append(kwargs), with_kwargs=True
    )
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=4,
        max_length_k=4,
    )
    # Keep every label supervised to ensure cu, rather than label masking,
    # excludes sample boundaries and the alignment tail.
    loss = outputs.dspark_loss_fn(input_ids)
    hook.remove()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.mtp[0].main_proj.weight.grad.abs().sum() > 0
    attention = captured[0]
    window = model.config.sliding_window
    anchors = attention["context_anchors"][0] - window
    assert attention["block_mask"][0].all()
    assert anchors.tolist() == [0, 1, 2, 4, 5, 6]
    for anchor, context_mask in zip(anchors, attention["context_mask"][0]):
        sample_start = 0 if anchor < 4 else 4
        # Slot k holds position anchor - 1 - k.
        expected = torch.arange(window) < anchor - sample_start
        torch.testing.assert_close(context_mask, expected)


@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("block_size", [2, 4])
def test_dspark_packed_logits_match_independent_samples(block_size, num_stages):
    model = _make_mtp_model(block_size=block_size, num_stages=num_stages)

    assert len(model.mtp) == num_stages

    samples = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]])]
    input_ids = torch.cat(samples, dim=1)
    captured = []
    hook = model.mtp[-1].register_forward_hook(
        lambda _module, _args, output: captured.append(output[0])
    )
    outputs = model(
        input_ids=input_ids,
        position_ids=torch.arange(4).repeat(2).unsqueeze(0),
        cu_seq_lens_q=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    outputs.dspark_loss_fn(input_ids)
    packed_logits = captured.pop().view(1, -1, block_size, model.config.vocab_size)
    for sample_index, sample in enumerate(samples):
        outputs = model(input_ids=sample)
        outputs.dspark_loss_fn(sample)
        logits = captured.pop().view(1, -1, block_size, model.config.vocab_size)
        for anchor in range(3):
            num_predictions = min(block_size, 3 - anchor)
            actual = packed_logits[:, sample_index * 3 + anchor, :num_predictions]
            expected = logits[:, anchor, :num_predictions]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    hook.remove()


def test_dspark_num_layers_is_inferred_from_checkpoint_index(tmp_path):
    config = _make_model(1).config
    config.dspark_block_size = 2
    config._name_or_path = str(tmp_path)
    config.dspark_num_layers = 1
    weight_map = {
        f"mtp.{stage}.weight": "model-00001-of-00001.safetensors"
        for stage in range(3)
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    config.dspark_num_layers = (
        turbo_model.TurboDeepseekV4ForCausalLM._resolve_dspark_num_layers(config)
    )

    assert config.dspark_num_layers == 3


def test_dspark_num_layers_falls_back_to_persisted_config_without_checkpoint_index(
    tmp_path,
):
    config = _make_model(1).config
    config.dspark_block_size = 2
    config._name_or_path = str(tmp_path)
    config.dspark_num_layers = 2

    assert (
        turbo_model.TurboDeepseekV4ForCausalLM._resolve_dspark_num_layers(config)
        == 2
    )


def test_dspark_draft_loss_matches_autograd_reference(monkeypatch):
    # Several chunks, including a partial one, with masked rows in between.
    monkeypatch.setattr(dspark_mtp, "_DSPARK_LOSS_CHUNK_SIZE", 4)
    torch.manual_seed(0)
    rows, vocab = 10, 7
    logits = torch.randn(rows, vocab, requires_grad=True)
    target_logits = torch.randn(rows, vocab, requires_grad=True)
    targets = torch.randint(vocab, (rows,))
    weights = torch.rand(rows) * (torch.arange(rows) % 3 != 0)

    loss, ce_sum, l1_sum, accept_rate = dspark_mtp._DSparkDraftLoss.apply(
        logits, targets, weights, lambda start, end: target_logits[start:end], 0.1, 0.9
    )
    (loss * 3.0).backward()

    reference_logits = logits.detach().clone().requires_grad_()
    ce = F.cross_entropy(reference_logits, targets, reduction="none")
    l1 = (reference_logits.softmax(-1) - target_logits.detach().softmax(-1)).abs().sum(-1)
    reference_loss = 0.1 * (ce * weights).sum() + 0.9 * (l1 * weights).sum()
    (reference_loss * 3.0).backward()

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(ce_sum, (ce * weights).sum().detach())
    torch.testing.assert_close(l1_sum, (l1 * weights).sum().detach())
    torch.testing.assert_close(accept_rate, (1 - 0.5 * l1.detach()).clamp(0, 1))
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    assert target_logits.grad is None
    assert not accept_rate.requires_grad


def test_dspark_trains_backbone_and_draft_and_records_metrics():
    model = _make_mtp_model(block_size=2)
    input_ids = torch.arange(1, 11).unsqueeze(0)
    outputs = model(
        input_ids=input_ids, cu_seq_lens_q=torch.tensor([0, 10], dtype=torch.int32)
    )
    loss = outputs.dspark_loss_fn(input_ids)
    loss.backward()

    for parameter in (
        model.embed.weight,
        model.head.weight,
        model.layers[0].attn.wkv.weight,
        model.mtp[0].main_proj.weight,
        model.mtp[-1].markov_head.markov_w2.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    # These parameters are used only to construct the reference distribution.
    for parameter in (
        model.norm.weight,
        model.hc_head_fn,
        model.hc_head_scale,
        model.hc_head_base,
    ):
        assert parameter.grad is None

    metrics = model.mtp.pop_metrics()
    config = model.config
    torch.testing.assert_close(metrics["dspark/loss"], loss.detach())
    torch.testing.assert_close(
        config.dspark_ce_loss_alpha * metrics["dspark/ce_loss"]
        + config.dspark_l1_loss_alpha * metrics["dspark/l1_loss"]
        + config.dspark_confidence_loss_alpha * metrics["dspark/confidence_loss"],
        loss.detach(),
    )
    assert 0 <= metrics["dspark/accept_rate"] <= 1
    assert model.mtp.pop_metrics() == {}


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("num_stages", [1, 2])
def test_dspark_joint_training_checkpoint_gradients_match(use_reentrant, num_stages):
    input_ids = torch.arange(1, 11).unsqueeze(0)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]])
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)

    def run_backward(model):
        torch.manual_seed(12)
        outputs = model(
            input_ids=input_ids, position_ids=position_ids, cu_seq_lens_q=cu
        )
        labels = input_ids[:, 1:].clone()
        labels[:, 3] = -100
        main_loss = F.cross_entropy(
            outputs.logits[:, :-1].flatten(0, 1), labels.flatten()
        )
        loss = main_loss + outputs.dspark_loss_fn(input_ids)
        loss.backward()
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), gradients

    reference_loss, reference_grads = run_backward(
        _make_mtp_model(block_size=4, num_stages=num_stages)
    )
    model = _make_mtp_model(block_size=4, num_stages=num_stages)
    for layer in [*model.layers, *model.mtp]:
        # Match the backend's wrapper, which captures keyword arguments.
        def checkpointed_forward(*args, _forward=layer.forward, **kwargs):
            return checkpoint(
                partial(_forward, **kwargs), *args, use_reentrant=use_reentrant
            )

        layer.forward = checkpointed_forward
    loss, gradients = run_backward(model)

    torch.testing.assert_close(loss, reference_loss)
    assert gradients.keys() == reference_grads.keys()
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, reference_grads[name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("target_layer_ids", [[], [0, 0], [1, 0], [-1], [2]])
def test_dspark_rejects_invalid_target_layers(target_layer_ids):
    config = _make_model(1).config
    config.num_hidden_layers = 2
    config.num_nextn_predict_layers = 1
    config.compress_ratios = [1, 1, 1]
    config.dspark_block_size = 2
    config.dspark_num_layers = 1
    config.dspark_target_layer_ids = target_layer_ids
    config.dspark_markov_rank = 4

    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        DeepseekV4MultiTokenPredictor(config)


def test_dspark_rejects_uninferable_layer_count():
    config = _make_model(1).config
    config.dspark_block_size = 2
    config.dspark_target_layer_ids = [0]
    config.dspark_markov_rank = 4

    with pytest.raises(ValueError, match="DSpark layer count"):
        DeepseekV4MultiTokenPredictor(config)


@pytest.mark.parametrize("length", [1, 10])
def test_dspark_without_anchors_trains_one_masked_block(length):
    # Fully unsupervised rows and one-token rows offer no anchor.
    model = _make_mtp_model(block_size=2)
    input_ids = torch.arange(1, length + 1).unsqueeze(0)
    captured = []
    hook = model.mtp[0].attn.register_forward_pre_hook(
        lambda _module, _args, kwargs: captured.append(kwargs), with_kwargs=True
    )
    outputs = model(
        input_ids=input_ids,
        cu_seq_lens_q=torch.tensor([0, length], dtype=torch.int32),
    )
    loss = outputs.dspark_loss_fn(torch.full_like(input_ids, -100))
    hook.remove()
    loss.backward()

    assert captured[0]["block_mask"].shape == (1, 1)
    assert not captured[0]["block_mask"].any()
    assert loss == 0
    gradients = {
        name: parameter.grad
        for name, parameter in model.mtp.named_parameters()
        if parameter.grad is not None
    }
    assert "0.main_proj.weight" in gradients
    for name, gradient in gradients.items():
        assert not gradient.any(), name
    assert all(torch.isfinite(value) for value in model.mtp.pop_metrics().values())


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("aux_coeff,z_coeff", [(0.2, 0.0), (0.0, 0.3), (0.2, 0.3)])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
def test_dspark_placeholder_blocks_do_not_train_router(
    monkeypatch, tmp_path, aux_coeff, z_coeff, checkpoint_mode,
):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(MoEAuxLossAutoScaler, "loss_scale", 1.0)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'dspark-router'}", rank=0, world_size=1,
    )
    try:
        model = _make_mtp_model(block_size=2, num_stages=1)
        model.config.num_hash_layers = 0
        model.config.routed_scaling_factor = model.config.route_scale
        monkeypatch.setattr(MoELayer, "forward", MoELayer.forward)
        monkeypatch.setattr(
            MoELayer, "_torchturbo_deepseek_v4_origin_forward", MoELayer.forward,
            raising=False,
        )
        monkeypatch.setattr(dsv4.DeepseekV4RMSNorm, "forward", dsv4.DeepseekV4RMSNorm.forward)
        turbo_model.TurboDeepseekV4ForCausalLM._pre_init_patches(model.config)
        model.mtp[0].ffn = _replace_moe_only(
            model.mtp[0].ffn, model.config,
            {"sp_group": None}, {"ep_group": dist.group.WORLD, "ep_size": 1},
            MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
            EngineConfig.from_dict({"moe": {
                "aux_loss_coeff": aux_coeff, "z_loss_coeff": z_coeff,
            }}),
        )
        moe = model.mtp[0].ffn
        torch.nn.init.normal_(moe.experts.weight1, std=0.02)
        torch.nn.init.normal_(moe.experts.weight2, std=0.02)
        if checkpoint_mode is not None:
            forward = model.mtp[0].forward

            def checkpointed_forward(*args, **kwargs):
                return checkpoint(partial(forward, **kwargs), *args, use_reentrant=checkpoint_mode)

            model.mtp[0].forward = checkpointed_forward
        input_ids = torch.arange(1, 11).unsqueeze(0)
        outputs = model(
            input_ids=input_ids,
            cu_seq_lens_q=torch.tensor([0, 10], dtype=torch.int32),
        )
        loss = outputs.dspark_loss_fn(torch.full_like(input_ids, -100))
        loss.backward()

        assert loss == 0
        assert moe.router.weight.grad is not None
        torch.testing.assert_close(moe.router.weight.grad, torch.zeros_like(moe.router.weight))
        assert model.mtp[0].main_proj.weight.grad is not None
        for name, parameter in model.mtp.named_parameters():
            if parameter.grad is not None:
                assert not parameter.grad.any(), name
        metrics = moe.router.pop_metrics()
        assert all(torch.isfinite(value) and value == 0 for value in metrics.values())
        assert ("aux_loss" in metrics) == (aux_coeff > 0)
        assert ("z_loss" in metrics) == (z_coeff > 0)
    finally:
        dist.destroy_process_group()


def _dspark_sp_worker(rank, init_file, block_size, window):
    turbo_model.is_npu_available = lambda: False
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        model = _make_mtp_model(block_size=block_size)
        model.config.sliding_window = window
        lengths = [6, 10]
        total = sum(lengths)
        input_ids = torch.arange(total).unsqueeze(0) % 31 + 1
        position_ids = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
        cu = torch.tensor([0, lengths[0], total], dtype=torch.int32)
        parameters = [
            model.embed.weight,
            model.head.weight,
            model.layers[0].attn.wkv.weight,
            model.mtp[0].main_proj.weight,
            model.mtp[-1].attn.wkv.weight,
            model.mtp[-1].markov_head.markov_w2.weight,
        ]

        torch.manual_seed(123)
        reference = model(
            input_ids=input_ids, position_ids=position_ids, cu_seq_lens_q=cu
        ).dspark_loss_fn(input_ids)
        reference_grads = torch.autograd.grad(reference, parameters)

        _parallelize(model, sp_size=2, sp_rank=rank, sp_group=dist.group.WORLD)
        local = slice(rank * total // 2, (rank + 1) * total // 2)
        # Rank 0 samples the anchors; the same seed reproduces the reference.
        torch.manual_seed(123)
        loss = model(
            input_ids=input_ids[:, local],
            position_ids=position_ids[:, local],
            cu_seq_lens_q=cu,
        ).dspark_loss_fn(input_ids)
        grads = torch.autograd.grad(loss, parameters)

        torch.testing.assert_close(loss, reference, atol=1e-6, rtol=1e-5)
        for grad, reference_grad in zip(grads, reference_grads):
            # The loss all-reduce hands every rank the full upstream gradient;
            # FSDP averages the SP replicas afterwards.
            grad = grad.clone()
            dist.all_reduce(grad)
            torch.testing.assert_close(grad / 2, reference_grad, atol=1e-6, rtol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize(
    "block_size, window",
    # The second case needs halos longer than one 8-token shard.
    [(2, 4), (10, 10)],
)
def test_sp_dspark_loss_and_gradients_match_single_rank(tmp_path, block_size, window):
    mp.spawn(
        _dspark_sp_worker,
        args=(str(tmp_path / "gloo_init"), block_size, window),
        nprocs=2,
        join=True,
    )
