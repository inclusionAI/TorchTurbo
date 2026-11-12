"""Native loss-mask output-head selection, including model and engine contracts."""

import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from . import sparse_test_support as h
from torchturbo.adapters.rl.sparse import build_sparse_plan
from torchturbo.engine.config import EngineConfig, SparseLogitsConfig, build_from_args
from torchturbo.models.base import TurboModel


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(31)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def setup(*, sparse=True, model=None, **kwargs):
    mb, info = h.make_mb(h.packed, h.sp, **kwargs)
    return h.engine(
        h.core, model or h.SmallModel(mix_context=True), info, sparse=sparse
    ), mb


def run(eng, mb, **kwargs):
    return h.run_batch(eng, mb, h.MaskedPolicyLoss, **kwargs)


@pytest.mark.parametrize(
    "lengths,active,pad",
    [
        ((31,), (2, 7, 29), 0),
        ((7, 1, 13, 5), (0, 5, 6, 17, 18, 21), 0),
        ((7,), (), 0),
        ((7,), tuple(range(6)), 0),
        ((19,), (0, 17), 5),
    ],
)
@pytest.mark.parametrize("reduction", ["token", "sample"])
def test_dense_sparse_loss_and_all_parameter_gradients(lengths, active, pad, reduction):
    dense, mb = setup(sparse=False, lengths=lengths, active=active, pad=pad)
    sparse, _ = setup(model=copy.deepcopy(dense.model))
    a = run(dense, copy.deepcopy(mb), reduction=reduction)
    b = run(sparse, copy.deepcopy(mb), reduction=reduction)
    h.check_equal(a, b, mb["orig_mb"]["ppo_loss_mask"])
    assert sparse.model.seen == [(sum(lengths) + pad, max(1, len(active)))]
    if active:
        assert b["grads"]["embed.weight"].abs().sum() > 0


@pytest.mark.parametrize("return_dict", [True, False])
def test_real_qwen_forward_mtp_weights_hooks_and_two_optimizer_steps(return_dict):
    model = h.small_qwen_model()
    model.config.return_dict = return_dict
    dense, mb = setup(sparse=False, model=model)
    sparse, _ = setup(model=copy.deepcopy(model))
    before = {k: v.clone() for k, v in sparse.model.state_dict().items()}
    ids = {k: id(v) for k, v in sparse.model.named_parameters()}
    mtp = sparse.model.mtp
    hooks = []
    sparse.model.register_forward_pre_hook(lambda *a: hooks.append(True))
    a = run(dense, copy.deepcopy(mb))
    b = run(sparse, copy.deepcopy(mb))
    h.check_equal(a, b, mb["orig_mb"]["ppo_loss_mask"])
    assert model.mtp.calls == 1 and len(model.seen) == 2
    assert mtp.calls == 0 and sparse.model.seen == [(31, 3)]
    assert hooks == [True]
    assert ids == {k: id(v) for k, v in sparse.model.named_parameters()}
    assert sparse.model.mtp is mtp
    for k, v in sparse.model.state_dict().items():
        torch.testing.assert_close(before[k], v)
    opt = torch.optim.SGD(sparse.model.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad(set_to_none=True)
    run(sparse, copy.deepcopy(mb))
    opt.step()
    assert mtp.calls == 0 and len(sparse.model.seen) == 2
    assert sparse.model.mtp.proj.weight.grad is None
    torch.testing.assert_close(before["mtp.proj.weight"], sparse.model.mtp.proj.weight)
    # No global mode changes: the ordinary decorated forward remains intact.
    output = sparse.model(input_ids=mb["padded_mb"]["input_ids"])
    assert (isinstance(output, tuple)) is (not return_dict)
    assert mtp.calls == 1


def test_forward_only_and_compute_logprobs_are_dense():
    eng, mb = setup(model=h.small_qwen_model())
    assert eng.compute_logprobs([copy.deepcopy(mb)]).shape == (30,)
    eng.forward_backward_batch(
        [mb], None, lambda **kw: kw["logprobs"].sum(), lambda mb: 1, forward_only=True
    )
    assert eng.model.seen == [(31, 31)] * 4
    assert eng.model.mtp.calls == 2
    assert all(p.grad is None for p in eng.model.parameters())


def test_per_engine_opt_in_and_no_context_to_leak_after_failure():
    sparse, mb = setup(model=h.small_qwen_model())
    dense, _ = setup(sparse=False)
    with mock.patch.object(
        sparse.model.lm_head, "forward", side_effect=ValueError("failure")
    ):
        with pytest.raises(ValueError, match="failure"):
            run(sparse, copy.deepcopy(mb))
    run(dense, copy.deepcopy(mb))
    assert dense.model.seen == [(31, 31)]
    sparse.compute_logprobs([mb])
    assert sparse.model.mtp.calls == 1


def test_mask_zero_fill_avoids_ppo_overflow():
    eng, mb = setup()
    old = mb["orig_mb"]["old_logp"]
    mask = mb["orig_mb"]["ppo_loss_mask"]
    old[~mask] = -10000
    out = run(eng, mb)
    assert torch.equal(out["lp"][~mask], old[~mask])
    assert torch.equal(out["ent"][~mask], torch.zeros_like(out["ent"][~mask]))
    assert all(torch.isfinite(g).all() for g in out["grads"].values())


@pytest.mark.parametrize(
    "bad,match",
    [
        ("assertion", "loss_uses_ppo_mask"),
        ("raw", "raw-logit"),
        ("aux", "auxiliary"),
        ("mtp", "mtp.enabled"),
        ("eval", "training"),
        ("no_grad", "grad-enabled"),
        ("ep", "ep_size"),
        ("sp", "SP1"),
        ("unsupported", "sparse"),
    ],
)
def test_engine_guards_fail_before_head(bad, match):
    eng, mb = setup()
    kwargs = dict(
        process_fn=None,
        loss_fn=lambda **k: None,
        loss_weight_fn=lambda mb: 1,
        loss_uses_ppo_mask=True,
    )
    if bad == "assertion":
        kwargs.pop("loss_uses_ppo_mask")
    if bad == "raw":
        kwargs["process_fn"] = lambda: None
    if bad == "aux":
        kwargs["auxiliary_loss_fn"] = lambda: None
    if bad == "mtp":
        eng.config.mtp.enabled = True
    if bad == "eval":
        eng.model.eval()
    if bad == "ep":
        eng.config.parallel_plan = SimpleNamespace(ep_size=2)
    if bad == "sp":
        eng.get_sp_info = lambda: dict(sp_size=2)
    if bad == "unsupported":
        eng.model.forward_sparse_logits = TurboModel.forward_sparse_logits.__get__(
            eng.model
        )
    with torch.set_grad_enabled(bad != "no_grad"):
        with pytest.raises((ValueError, NotImplementedError), match=match):
            eng.forward_backward_batch([mb], **kwargs)
    assert eng.model.seen == []


@pytest.mark.parametrize(
    "bad,match",
    [
        ("mask", "loss_mask"),
        ("old", "old_logp"),
        ("cu", "cu_seqlens"),
        ("mm_orig", "text-only"),
        ("mm_padded", "text-only"),
        ("labels", "labels"),
        ("selected", "logits_to_keep"),
        ("singleton", "singleton"),
    ],
)
def test_plan_guards_fail_before_head(bad, match):
    eng, mb = setup()
    if bad == "mask":
        mb["orig_mb"]["ppo_loss_mask"] = torch.ones(31)
    if bad == "old":
        mb["orig_mb"]["old_logp"][0] = float("nan")
    if bad == "cu":
        mb["orig_mb"]["cu_seqlens"][-1] += 1
    if bad == "mm_orig":
        mb["orig_mb"]["pixel_values"] = torch.ones(1)
    if bad == "mm_padded":
        mb["padded_mb"]["image_grid_thw"] = torch.ones(1)
    if bad == "labels":
        mb["padded_mb"]["labels"] = torch.zeros(1)
    if bad == "selected":
        mb["padded_mb"]["logits_to_keep"] = torch.zeros(1)
    if bad == "singleton":
        mb["orig_mb"]["cu_seqlens"] = torch.arange(32)
    with pytest.raises(ValueError, match=match):
        run(eng, mb)
    assert eng.model.seen == []


def test_sp_predictor_boundary_alignment():
    _, mb = setup(lengths=(5, 1, 10), active=(0, 3, 4, 12))
    _, ctx = h.packed.prepare_packed_mb_inputs(mb)
    ids = ctx["mb_input"]["input_ids"]
    ctx["mb_input"]["_input_ids_rolled"] = ids[torch.tensor([7, 8])]
    plan = build_sparse_plan(ctx, dict(sp_size=8, sp_rank=3, sp_group=object()))
    assert plan.rows.tolist() == [0] and plan.labels.tolist() == [int(ids[7])]
    ctx["mb_input"]["_input_ids_rolled"] = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="misaligned"):
        build_sparse_plan(ctx, dict(sp_size=8, sp_rank=3, sp_group=object()))


def test_sparse_output_shape_and_temperature_contract():
    eng, mb = setup()
    _, ctx = h.packed.prepare_packed_mb_inputs(mb)
    plan = build_sparse_plan(ctx, dict(sp_size=1))
    with pytest.raises(ValueError, match="predictor rows"):
        eng._compute_logrobs_and_entropy(
            torch.zeros(31, 41), ctx, 1.0, True, sparse_plan=plan
        )
    with pytest.raises(ValueError, match="temperature"):
        eng._compute_logrobs_and_entropy(
            torch.zeros(3, 41), ctx, 0, True, sparse_plan=plan
        )


def test_native_config_roundtrip_and_args(tmp_path):
    from torchturbo import SparseLogitsConfig as Exported

    assert Exported is SparseLogitsConfig
    assert EngineConfig().sparse_logits.enabled is False
    values = dict(enabled=True)
    cfg = EngineConfig.from_dict(dict(sparse_logits=values))
    assert cfg.sparse_logits == SparseLogitsConfig(**values)
    serialized = cfg.to_dict()["sparse_logits"]
    assert serialized == values
    assert (
        EngineConfig.from_dict({"sparse_logits": serialized}).sparse_logits
        == cfg.sparse_logits
    )
    from torchturbo.engine.args import TorchTurboArgs

    path = tmp_path / "config.yaml"
    path.write_text(
        "engine_config:\n  sparse_logits:\n    enabled: true\n"
    )
    assert (
        build_from_args(TorchTurboArgs(config_path=str(path))).sparse_logits
        == cfg.sparse_logits
    )
    assert EngineConfig(sparse_logits=None).sparse_logits.enabled is False


@pytest.mark.parametrize(
    "values",
    [dict(enabled="false"), dict(max_rows=4096), dict(max_rows=0), dict(typo=1)],
)
def test_native_config_rejects_invalid_values(values):
    with pytest.raises((ValueError, TypeError)):
        EngineConfig(sparse_logits=values)


@pytest.mark.parametrize("num_rows", [4097, 5117, 8192])
@pytest.mark.parametrize("reduction", ["token", "sample"])
def test_large_masks_retain_all_rows_and_dense_gradients(num_rows, reduction):
    dense, mb = setup(
        sparse=False, lengths=(num_rows + 17,), active=tuple(range(1, num_rows + 1))
    )
    sparse, _ = setup(model=copy.deepcopy(dense.model))
    a = run(dense, copy.deepcopy(mb), reduction=reduction)
    b = run(sparse, copy.deepcopy(mb), reduction=reduction)
    h.check_equal(a, b, mb["orig_mb"]["ppo_loss_mask"])
    assert sparse.model.seen == [(num_rows + 17, num_rows)]


@pytest.mark.parametrize("rank", range(8))
def test_sp8_large_uneven_masks_preserve_rows_and_empty_shards(rank):
    local_length = 8192
    active = tuple(range(5117)) + tuple(
        range(7 * local_length, 7 * local_length + 4097)
    )
    _, mb = setup(lengths=(8 * local_length,), active=active)
    _, ctx = h.packed.prepare_packed_mb_inputs(mb)
    ids = ctx["mb_input"]["input_ids"]
    ctx["mb_input"]["_input_ids_rolled"] = torch.roll(ids, shifts=-1)[
        rank * local_length : (rank + 1) * local_length
    ]
    plan = build_sparse_plan(ctx, dict(sp_size=8, sp_rank=rank, sp_group=object()))
    expected = 5117 if rank == 0 else 4097 if rank == 7 else 0
    assert plan.rows.tolist() == list(range(expected))
    assert plan.model_rows.numel() == max(1, expected)
    assert plan.mask.count_nonzero().item() == len(active)
    if expected:
        torch.testing.assert_close(
            plan.labels, ids[rank * local_length + plan.rows + 1]
        )
    else:
        assert plan.model_rows.tolist() == [0]


def test_sparse_preserves_current_packed_input_contract():
    model = h.small_qwen_model()
    mb, info = h.make_mb(
        h.packed,
        h.sp,
        lengths=(7, 1, 13),
        active=(0, 6, 17),
        model=model,
    )
    inputs = mb["padded_mb"]
    assert inputs["input_ids"].shape == (1, 21)
    assert "cu_seqlens" not in inputs
    assert inputs["cu_seq_lens_q"] is mb["orig_mb"]["cu_seqlens"]
    assert inputs["cu_seq_lens_k"] is inputs["cu_seq_lens_q"]
    assert inputs["max_length_q"] == inputs["max_length_k"] == 13
    assert inputs["position_ids"].tolist() == [list(range(7)) + [0] + list(range(13))]
    dense = h.engine(h.core, model, info)
    sparse = h.engine(h.core, copy.deepcopy(model), info, sparse=True)
    a = run(dense, copy.deepcopy(mb))
    b = run(sparse, copy.deepcopy(mb))
    h.check_equal(a, b, mb["orig_mb"]["ppo_loss_mask"])


def test_sparse_uses_shared_logprob_helper():
    eng, mb = setup()
    with mock.patch.object(
        h.core, "gather_logprobs_entropy", wraps=h.packed.gather_logprobs_entropy
    ) as compute:
        run(eng, mb)
    compute.assert_called_once()
    assert compute.call_args.args[0].shape == (3, 41)
    assert compute.call_args.args[1].shape == (3,)
