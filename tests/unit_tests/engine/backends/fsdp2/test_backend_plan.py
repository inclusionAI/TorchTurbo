"""Unit tests for FSDP2 backend plan wrapping semantics."""

import os
import sys
import types
from contextlib import nullcontext

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch
import torch.nn as nn

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode
from torchturbo.engine.model_fqn import build_model_fqn_map
from torchturbo.models.bailing_moe_v3_vl.vision_balance import BailingVisionBalanceIO
from torchturbo.models.modules.cpu_resident import NodeSharedCPUResidentParameter
from torchturbo.parallel import (
    FSDPParallelPlan,
    FSDPPlanSpec,
    VisionBalanceMode,
    VisionBalanceSpec,
)


class TaggedModule(nn.Module):
    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag

    def get_norm(self):
        base_prefix = getattr(self, "base_model_prefix", "")
        base = getattr(self, base_prefix, None) if base_prefix else None
        norm = getattr(base, "norm", None) if isinstance(base, nn.Module) else None
        return norm if isinstance(norm, nn.Module) else None

    def get_lm_head(self):
        lm_head = getattr(self, "lm_head", None)
        return lm_head if isinstance(lm_head, nn.Module) else None


class FinalNorm(TaggedModule):
    """Class name contains 'Norm' so _find_final_norm detects it as the tail norm."""

    pass


def module_tags(module):
    modules = module if isinstance(module, list) else [module]
    tags = [m.tag for m in modules]
    return tags[0] if len(tags) == 1 else tags


def make_backend(*, mtp_enabled=False):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = types.SimpleNamespace(
        mtp=types.SimpleNamespace(enabled=mtp_enabled),
        loss_chunk=types.SimpleNamespace(enabled=False),
    )
    return backend


def make_preparing_backend(order, sp_info):
    backend = make_backend()
    backend.prefetch_manager = types.SimpleNamespace(
        setup=lambda model: order.append("prefetch")
    )
    backend.get_ep_info = lambda: {"ep_size": 1}
    backend.get_sp_info = lambda: sp_info
    backend._init_mesh = lambda plan: order.append("init_mesh")
    backend._replace_modules = lambda model, plan: order.append("replace") or model
    backend._apply_activation_checkpointing = (
        lambda model, plan: order.append("activation_checkpointing") or model
    )
    backend._shard_model = lambda model, plan: order.append("shard") or model
    return backend


def fsdp_config(*, full_offload=False):
    cpu_offload = types.SimpleNamespace(
        enabled=full_offload, mode=CPUOffloadMode.FULL
    )
    return types.SimpleNamespace(cpu_offload=cpu_offload)


def resolve_plan(model, plan=None):
    if plan is None:
        plan = FSDPParallelPlan()
    return FSDP2Backend._compile_parallel_plan(
        plan,
        build_model_fqn_map(model),
    )


def test_model_fqn_map_is_unavailable_before_prepare_model():
    backend = make_backend()
    backend._model_fqn_map = None

    with pytest.raises(RuntimeError, match="prepare_model"):
        backend.get_model_fqn_map()


def test_compile_parallel_plan_warns_for_unmatched_patterns(monkeypatch):
    from torchturbo.engine.backends.fsdp2 import backend as backend_module

    warnings = []
    monkeypatch.setattr(
        backend_module,
        "logger",
        types.SimpleNamespace(
            warning_rank0=lambda message, *args: warnings.append(message % args)
        ),
    )
    model = nn.Linear(2, 2)

    resolved = resolve_plan(
        model,
        FSDPParallelPlan(plan={"missing.*": FSDPPlanSpec(no_shard=True)}),
    )

    assert dict(resolved.modules) == {}
    assert dict(resolved.parameters) == {}
    assert warnings == [
        "FSDPParallelPlan patterns matched no model FQNs: ['missing.*']"
    ]


def test_cpu_offload_grad_norm_counts_replicated_grad_once(monkeypatch):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._device_runtime = types.SimpleNamespace(
        device=torch.device("cpu"), synchronize=lambda: None
    )
    sharded_param = types.SimpleNamespace(
        grad=types.SimpleNamespace(_local_tensor=torch.tensor([3.0]))
    )
    replicated_param = nn.Parameter(torch.zeros(1))
    replicated_param.grad = torch.tensor([4.0])

    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, **kwargs: tensor.mul_(4),
    )

    total_norm = backend._clip_grad_norm_cpu_offload(
        [], [sharded_param], [replicated_param], max_norm=1000.0, norm_type=2.0
    )

    torch.testing.assert_close(total_norm, torch.sqrt(torch.tensor(52.0)))


@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
@pytest.mark.parametrize("max_norm", [1.0, 1000.0])
def test_cpu_offload_norm_scales_experts_before_clipping(monkeypatch, norm_type, max_norm):
    backend = make_backend()
    backend.gradient_scaling_factor = 0.125
    backend._device_runtime = types.SimpleNamespace(
        device=torch.device("cpu"), synchronize=lambda: None
    )
    expert_grad = torch.tensor([24.0, 32.0])
    dense_grad = torch.tensor([12.0])
    expert_alias = expert_grad.view_as(expert_grad)
    expert = types.SimpleNamespace(grad=types.SimpleNamespace(_local_tensor=expert_grad))
    empty = types.SimpleNamespace(grad=types.SimpleNamespace(_local_tensor=torch.empty(0)))
    dense = types.SimpleNamespace(grad=types.SimpleNamespace(_local_tensor=dense_grad))
    replicated = nn.Parameter(torch.zeros(1))
    replicated.grad = torch.tensor([5.0])

    def reduce_shards(tensor, op, group):
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(4)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_shards)
    expected_norm = torch.tensor(12.0 if norm_type == float("inf") else (4 * 169 + 25) ** 0.5)
    factor = min(1.0, max_norm / (expected_norm.item() + 1e-6))
    norm = backend._clip_grad_norm_cpu_offload(
        [expert, empty], [dense], [replicated], max_norm, norm_type
    )

    torch.testing.assert_close(norm, expected_norm)
    torch.testing.assert_close(expert_alias, torch.tensor([3.0, 4.0]) * factor)
    torch.testing.assert_close(dense_grad, torch.tensor([12.0]) * factor)
    torch.testing.assert_close(replicated.grad, torch.tensor([5.0]) * factor)


@pytest.mark.parametrize("full_offload", [False, True])
@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
@pytest.mark.parametrize("max_norm", [1.0, float("inf")])
def test_grad_norm_matches_pytorch(monkeypatch, full_offload, norm_type, max_norm):
    backend = make_backend()
    backend.fsdp_config = fsdp_config(full_offload=full_offload)
    backend._mesh = None
    backend._device_runtime = types.SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    reference = nn.Parameter(parameter.detach().clone())
    reference.grad = parameter.grad.clone()

    norm = backend.clip_grad_norm_([parameter], max_norm, norm_type)
    expected = nn.utils.clip_grad_norm_([reference], max_norm, norm_type)

    torch.testing.assert_close(norm, expected)
    torch.testing.assert_close(parameter.grad, reference.grad)
    assert backend.clip_grad_norm_([], max_norm, norm_type).item() == 0.0


class CheckpointBlock(TaggedModule):
    def __init__(self, tag: str) -> None:
        super().__init__(tag)
        self.inner = TaggedModule(tag + ".inner")
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x, use_cache=False, past_key_values=None):
        if use_cache or past_key_values is not None:
            raise AssertionError("cache kwargs should be cleared")
        return x * self.weight


def test_standalone_world_shard_skips_default_descendant_wrapping(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), mesh))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.language_model = TaggedModule("language_model")
    model.language_model.layers = nn.ModuleList(
        [TaggedModule("L0"), TaggedModule("L1")]
    )

    plan = FSDPParallelPlan(
        plan={"language_model": FSDPPlanSpec(standalone_world_shard=True)}
    )

    backend._shard_model(model, resolve_plan(model, plan))

    assert calls == [
        ("language_model", "world"),
        ("root", "world"),
    ]


@pytest.mark.parametrize("device", ["meta", "cpu"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_cpu_resident_parameters_must_be_frozen(tmp_path, device, requires_grad):
    backend = make_backend()
    model = nn.Module()
    model.table = NodeSharedCPUResidentParameter(
        torch.ones(2, device=device), requires_grad=requires_grad, directory=tmp_path
    )
    model.weight = nn.Parameter(torch.ones(2, device=device))
    plan = resolve_plan(
        model, FSDPParallelPlan(plan={"weight": FSDPPlanSpec(no_shard=True)})
    )

    if requires_grad:
        with pytest.raises(AssertionError, match="CPU resident.*frozen.*table"):
            backend._collect_no_shard(model, plan)
    else:
        ignored, modules = backend._collect_no_shard(model, plan)
        assert ignored == {model.table, model.weight}
        assert modules == []


def test_no_shard_matches_parameter_names(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.hc_head_fn = nn.Parameter(torch.ones(1))
    model.layer = TaggedModule("layer")
    model.layer.attn_sink = nn.Parameter(torch.ones(1))
    model.layer.weight = nn.Parameter(torch.ones(1))

    plan = FSDPParallelPlan(
        plan={
            "hc_head_fn": FSDPPlanSpec(no_shard=True),
            "**.attn_sink": FSDPPlanSpec(no_shard=True),
        }
    )

    backend._shard_model(model, resolve_plan(model, plan))

    assert len(calls) == 1
    assert calls[0][0] == "root"
    assert calls[0][1]["ignored_params"] == {
        model.hc_head_fn,
        model.layer.attn_sink,
    }


def test_resolved_plan_targets_only_active_modules_to_save_copy():
    other = pytest.importorskip("peft.utils.other")
    model = TaggedModule("root")
    model.projection = other.ModulesToSaveWrapper(
        nn.Linear(2, 2),
        "default",
    )
    spec = FSDPPlanSpec(no_shard=True)

    resolved = resolve_plan(
        model,
        FSDPParallelPlan(
            plan={
                "projection": spec,
                "projection.weight": spec,
            }
        ),
    )

    assert dict(resolved.modules) == {
        "projection.modules_to_save.default": spec,
    }
    assert dict(resolved.parameters) == {
        "projection.modules_to_save.default.weight": spec,
    }


def test_resolved_plan_maps_trainable_tokens_to_base_embedding():
    other = pytest.importorskip("peft.utils.other")

    model = nn.Module()
    model.embed_tokens = other.TrainableTokensWrapper(
        nn.Embedding(8, 4),
        "default",
        token_indices=[0, 1],
    )
    module_spec = FSDPPlanSpec(standalone_world_shard=True)
    parameter_spec = FSDPPlanSpec(no_shard=True)

    resolved = resolve_plan(
        model,
        FSDPParallelPlan(
            plan={
                "embed_tokens": module_spec,
                "embed_tokens.weight": parameter_spec,
            }
        ),
    )

    assert dict(resolved.modules) == {"embed_tokens": module_spec}
    assert dict(resolved.parameters) == {
        "embed_tokens.token_adapter.base_layer.weight": parameter_spec,
    }


def test_inactive_modules_to_save_copy_gets_its_own_fsdp_unit(monkeypatch):
    other = pytest.importorskip("peft.utils.other")
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module, kwargs.get("ignored_params", set())))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    wrapper = other.ModulesToSaveWrapper(nn.Linear(2, 2), "default")
    model.layers = nn.ModuleList([wrapper])
    active = wrapper.modules_to_save["default"]
    inactive = wrapper.original_module

    plan = FSDPParallelPlan(
        plan={"layers": FSDPPlanSpec(no_shard=True)}
    )
    backend._shard_model(model, resolve_plan(model, plan))

    assert any(inactive in modules for modules, _ in calls)
    ignored_params = calls[0][1]
    assert set(active.parameters()) <= ignored_params
    assert set(inactive.parameters()).isdisjoint(ignored_params)


def test_no_shard_module_skips_descendant_wrapping(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.visual = TaggedModule("visual")
    model.visual.proj = nn.Linear(2, 2)
    model.visual.blocks = nn.ModuleList(
        [TaggedModule("V0"), TaggedModule("V1")]
    )
    model.visual.blocks[0].weight = nn.Parameter(torch.ones(1))
    model.visual.blocks[1].weight = nn.Parameter(torch.ones(1))
    model.layers = nn.ModuleList([TaggedModule("L0")])

    plan = FSDPParallelPlan(plan={"visual": FSDPPlanSpec(no_shard=True)})

    backend._shard_model(model, resolve_plan(model, plan))

    assert [call[0] for call in calls] == ["L0", "root"]
    assert calls[-1][1]["ignored_params"] == set(model.visual.parameters())


def test_default_policy_wraps_module_list_children(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), mesh))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.layers = nn.ModuleList([TaggedModule("L0"), TaggedModule("L1")])

    backend._shard_model(model, resolve_plan(model))

    assert calls == [
        ("L1", "world"),
        ("L0", "world"),
        ("root", "world"),
    ]


def test_global_cpu_offload_applies_to_all_fsdp_units(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config(full_offload=True)
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.layers = nn.ModuleList([TaggedModule("L0"), TaggedModule("L1")])

    backend._shard_model(model, resolve_plan(model))

    assert [call[0] for call in calls] == ["L1", "L0", "root"]
    assert all("offload_policy" in kwargs for _, kwargs in calls)


def test_carves_tail_unit_with_reshard_false_and_root(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    embed = TaggedModule("embed")
    lm_head = TaggedModule("lm_head")
    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = lm_head
    model.tok = embed
    model.get_input_embeddings = lambda: embed
    model.get_output_embeddings = lambda: lm_head

    backend._shard_model(model, resolve_plan(model))

    # Untied input embedding is carved into its own unit with global reshard.
    assert ("embed", True) in calls
    # final_norm + lm_head are grouped into a single tail unit, reshard=False.
    assert (["norm", "lm_head"], False) in calls
    # root also uses reshard_after_forward=False.
    assert ("root", False) in calls


def test_tail_unit_uses_input_embedding_when_tied(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    shared = nn.Parameter(torch.ones(2, 2))
    embed = TaggedModule("embed")
    embed.weight = shared
    lm_head = TaggedModule("lm_head")
    lm_head.weight = shared  # tied to input embedding
    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = lm_head
    model.tok = embed
    model.get_input_embeddings = lambda: embed
    model.get_output_embeddings = lambda: lm_head

    backend._shard_model(model, resolve_plan(model))

    # Tied: use the input embedding in the tail unit and skip lm_head.
    assert (["norm", "embed"], False) in calls
    assert not any(
        isinstance(tags, list) and "lm_head" in tags for tags, _ in calls
    )


def test_chunked_loss_rejects_tied_embeddings(monkeypatch):
    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", lambda module, mesh, **kwargs: None)

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.config = types.SimpleNamespace(
        mtp=types.SimpleNamespace(enabled=False),
        loss_chunk=types.SimpleNamespace(enabled=True)
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    shared = nn.Parameter(torch.ones(2, 2))
    embed = TaggedModule("embed")
    embed.weight = shared
    lm_head = TaggedModule("lm_head")
    lm_head.weight = shared  # tied to input embedding
    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = lm_head
    model.tok = embed
    model.get_input_embeddings = lambda: embed
    model.get_output_embeddings = lambda: lm_head

    with pytest.raises(ValueError, match="tied input/output"):
        backend._shard_model(model, resolve_plan(model))


def test_tail_unit_does_not_guess_lm_head_attr(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = TaggedModule("lm_head")

    backend._shard_model(model, resolve_plan(model))

    assert ("norm", False) in calls
    assert not any(
        isinstance(tags, list) and "lm_head" in tags for tags, _ in calls
    )


class VLMOuter(TaggedModule):
    """Outer VLM wrapper exposing a nested final norm via ``get_norm``, mirroring
    the qwen3.5 / kimi turbo wrappers that delegate to the inner language model.
    The norm is invisible to the direct-child class-name scan (it lives under
    ``model.language_model``), so discovery must come from ``get_norm``;
    otherwise it falls into the auto root unit and breaks CPU offload.
    """

    base_model_prefix = "model"

    def get_norm(self):
        return self.model.language_model.norm

    def get_output_embeddings(self):
        return self.lm_head


def test_vlm_nested_norm_via_get_norm_joins_tail_unit(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(
        cpu_offload=types.SimpleNamespace(
            enabled=False, mode=CPUOffloadMode.FULL
        )
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = VLMOuter("root")
    inner = TaggedModule("model")
    language_model = TaggedModule("language_model")
    language_model.norm = FinalNorm("norm")
    language_model.embed_tokens = TaggedModule("embed")
    inner.language_model = language_model
    inner.visual = TaggedModule("visual")
    model.model = inner
    lm_head = TaggedModule("lm_head")
    model.lm_head = lm_head
    model.get_input_embeddings = lambda: language_model.embed_tokens

    backend._shard_model(model, resolve_plan(model))

    # The get_norm-declared nested norm is paired with lm_head in a resident
    # tail unit instead of falling through to the auto root unit.
    assert (["norm", "lm_head"], False) in calls
    # Root unit is still created (resident) but must not own the norm.
    assert ("root", False) in calls
    # norm must be covered by the tail unit, not carved as a standalone unit.
    assert not any(tags == "norm" for tags, _ in calls)


def test_mtp_keeps_vlm_norm_separate_from_reused_lm_head(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend(mtp_enabled=True)
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = types.SimpleNamespace(
        cpu_offload=types.SimpleNamespace(
            enabled=False, mode=CPUOffloadMode.FULL
        ),
        pin_memory=True,
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = VLMOuter("root")
    model.model = TaggedModule("model")
    model.model.language_model = TaggedModule("language_model")
    model.model.language_model.norm = FinalNorm("norm")
    model.model.language_model.embed_tokens = TaggedModule("embed")
    model.model.visual = TaggedModule("visual")
    model.lm_head = TaggedModule("lm_head")
    model.get_input_embeddings = lambda: model.model.language_model.embed_tokens

    plan = FSDPParallelPlan(
        plan={"lm_head": FSDPPlanSpec(standalone_world_shard=True)}
    )
    backend._shard_model(model, resolve_plan(model, plan))

    assert ("norm", False) in calls
    assert ("lm_head", False) in calls
    assert (["norm", "lm_head"], False) not in calls


def test_tail_unit_skips_modules_inside_explicit_parent(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.language_model = TaggedModule("language_model")
    model.language_model.norm = FinalNorm("norm")
    model.language_model.lm_head = TaggedModule("lm_head")
    model.get_output_embeddings = lambda: model.language_model.lm_head

    plan = FSDPParallelPlan(
        plan={"language_model": FSDPPlanSpec(standalone_world_shard=True)}
    )
    backend._shard_model(model, resolve_plan(model, plan))

    assert ("language_model", True) in calls
    assert not any(
        isinstance(tags, list) and ("norm" in tags or "lm_head" in tags)
        for tags, _ in calls
    )


def test_explicit_output_embedding_merges_with_tail_norm(monkeypatch):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = make_backend()
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = TaggedModule("lm_head")
    model.get_output_embeddings = lambda: model.lm_head

    plan = FSDPParallelPlan(
        plan={"lm_head": FSDPPlanSpec(standalone_world_shard=True)}
    )
    backend._shard_model(model, resolve_plan(model, plan))

    assert (["norm", "lm_head"], False) in calls
    assert ("norm", False) not in calls
    assert ("lm_head", False) not in calls


@pytest.mark.parametrize("explicit_output_unit", [False, True])
def test_chunked_loss_keeps_output_embedding_separate_from_tail_norm(
    monkeypatch, explicit_output_unit
):
    calls = []

    def fake_fully_shard(module, mesh, **kwargs):
        calls.append((module_tags(module), kwargs.get("reshard_after_forward")))

    import torch.distributed.fsdp as fsdp

    monkeypatch.setattr(fsdp, "fully_shard", fake_fully_shard)

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend._mesh = object()
    backend._world_mesh = "world"
    backend.mixed_precision = types.SimpleNamespace(value="no")
    backend.reshard_after_forward = True
    backend.fsdp_config = fsdp_config()
    backend.config = types.SimpleNamespace(
        mtp=types.SimpleNamespace(enabled=False),
        loss_chunk=types.SimpleNamespace(enabled=True)
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)

    model = TaggedModule("root")
    model.base_model_prefix = "decoder"
    model.decoder = TaggedModule("decoder")
    model.decoder.norm = FinalNorm("norm")
    model.lm_head = TaggedModule("lm_head")
    model.get_output_embeddings = lambda: model.lm_head

    plan = FSDPParallelPlan(
        plan=(
            {"lm_head": FSDPPlanSpec(standalone_world_shard=True)}
            if explicit_output_unit
            else {}
        )
    )
    backend._shard_model(model, resolve_plan(model, plan))

    assert ("norm", False) in calls
    assert ("lm_head", False) in calls
    assert (["norm", "lm_head"], False) not in calls


@pytest.mark.parametrize("sp_size", [1, 2])
def test_prepare_model_applies_input_slicing_before_checkpointing_and_sharding(sp_size):
    from torchturbo.models.modules.attention.ulysses import SequenceParallelInputSlicingSpec

    order = []
    model = TaggedModule("root")
    sp_info = {"sp_size": sp_size, "sp_group": None, "sp_rank": 0}

    def get_input_slicing_spec():
        order.append("input_slicing")
        return SequenceParallelInputSlicingSpec(model, TaggedModule.forward)

    model.get_sequence_parallel_input_slicing_spec = get_input_slicing_spec

    backend = make_preparing_backend(order, sp_info)

    assert backend.prepare_model(model, FSDPParallelPlan()) is model

    assert order == [
        "init_mesh",
        "replace",
        *(["input_slicing"] if sp_size > 1 else []),
        "activation_checkpointing",
        "shard",
        "prefetch",
    ]
    assert dict(backend.get_model_fqn_map().modules) == {"": ""}


@pytest.mark.parametrize(
    "mode", [None, VisionBalanceMode.PATCH, VisionBalanceMode.PATCH_SQ]
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_prepare_model_registers_vision_balance_on_prepared_boundary(
    monkeypatch, mode, wrapped
):
    from torchturbo.engine.backends.fsdp2 import backend as backend_module
    from torchturbo.models.modules.attention.ulysses import SequenceParallelInputSlicingSpec

    order = []
    model = TaggedModule("root")
    model.encoder = nn.Linear(2, 2)
    model.unused_visual = nn.Linear(2, 2)
    model.vision_tower_module_names = ("unused_visual",)
    model.get_sequence_parallel_input_slicing_spec = lambda: (
        SequenceParallelInputSlicingSpec(model, TaggedModule.forward)
    )
    boundary = model.encoder
    if wrapped:
        other = pytest.importorskip("peft.utils.other")
        model.encoder = other.ModulesToSaveWrapper(model.encoder, "default")
        boundary = model.encoder.modules_to_save["default"]

    sp_info = {"sp_size": 2, "sp_group": None, "sp_rank": 1}
    backend = make_preparing_backend(order, sp_info)
    registrations = []

    def register(module, **kwargs):
        order.append("vision_balance")
        registrations.append((module, kwargs))

    monkeypatch.setattr(backend_module, "register_vision_balance", register)
    vision_spec = VisionBalanceSpec(mode=mode, io=BailingVisionBalanceIO())
    plan = FSDPParallelPlan(
        plan={"encoder": FSDPPlanSpec(vision_balance=vision_spec)}
    )

    assert backend.prepare_model(model, plan) is model

    assert order == [
        "init_mesh",
        "replace",
        "activation_checkpointing",
        "shard",
        "prefetch",
        *(["vision_balance"] if mode is not None else []),
    ]
    if mode is None:
        assert registrations == []
    else:
        assert registrations == [
            (boundary, {"spec": vision_spec, "sp_size": 2, "sp_rank": 1})
        ]
        assert registrations[0][1]["spec"] is vision_spec


def test_prepare_model_uses_per_boundary_vision_balance_modes(monkeypatch):
    from torchturbo.engine.backends.fsdp2 import backend as backend_module

    order = []
    model = TaggedModule("root")
    model.image_encoder = nn.Linear(2, 2)
    model.video_encoder = nn.Linear(2, 2)
    backend = make_preparing_backend(
        order, {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    )
    registrations = []
    monkeypatch.setattr(
        backend_module,
        "register_vision_balance",
        lambda module, **kwargs: registrations.append((module, kwargs["spec"].mode)),
    )
    plan = FSDPParallelPlan(
        plan={
            "image_encoder": FSDPPlanSpec(
                vision_balance=VisionBalanceSpec(
                    mode=VisionBalanceMode.PATCH, io=BailingVisionBalanceIO()
                )
            ),
            "video_encoder": FSDPPlanSpec(
                vision_balance=VisionBalanceSpec(
                    mode=VisionBalanceMode.PATCH_SQ, io=BailingVisionBalanceIO()
                )
            ),
        }
    )

    backend.prepare_model(model, plan)

    assert registrations == [
        (model.image_encoder, VisionBalanceMode.PATCH),
        (model.video_encoder, VisionBalanceMode.PATCH_SQ),
    ]


@pytest.mark.parametrize(
    "pattern, shadowed",
    [("missing", False), ("visual.weight", False), ("visual", True)],
)
def test_prepare_model_rejects_vision_balance_without_active_module(pattern, shadowed):
    order = []
    model = TaggedModule("root")
    model.visual = nn.Linear(2, 2)
    model.vision_tower_module_names = ("visual",)
    backend = make_preparing_backend(
        order, {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    )
    entries = {"*": FSDPPlanSpec()} if shadowed else {}
    entries[pattern] = FSDPPlanSpec(
        vision_balance=VisionBalanceSpec(
            mode=VisionBalanceMode.PATCH, io=BailingVisionBalanceIO()
        )
    )

    with pytest.raises(ValueError, match="does not resolve to a module"):
        backend.prepare_model(model, FSDPParallelPlan(plan=entries))

    assert order == ["init_mesh", "replace"]


def test_prepare_model_rejects_vision_balance_without_io_before_sharding():
    order = []
    model = TaggedModule("root")
    model.visual = nn.Linear(2, 2)
    backend = make_preparing_backend(
        order, {"sp_size": 1, "sp_group": None, "sp_rank": 0}
    )
    plan = FSDPParallelPlan(
        plan={
            "visual": FSDPPlanSpec(
                vision_balance=VisionBalanceSpec(mode=VisionBalanceMode.PATCH)
            )
        }
    )

    with pytest.raises(ValueError, match="requires a model-provided input/output adapter"):
        backend.prepare_model(model, plan)

    assert order == ["init_mesh", "replace"]


def test_activation_checkpointing_patches_plan_targets(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=True)
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    suspended_models = []
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: (
            suspended_models.append(model) or nullcontext()
        )
    )

    model = TaggedModule("root")
    model.config = types.SimpleNamespace(use_cache=True)
    model.layers = nn.ModuleList([CheckpointBlock("L0"), CheckpointBlock("L1")])
    model.layers[0].config = types.SimpleNamespace(use_cache=True)
    model.layers[0].gradient_checkpointing = True
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )
    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)

    backend._apply_activation_checkpointing(model, resolve_plan(model, plan))

    assert hasattr(model.layers[0], "_tt_original_forward")
    assert hasattr(model.layers[1], "_tt_original_forward")
    assert not hasattr(model.layers[0], "_checkpoint_wrapped_module")
    assert model.config.use_cache is False
    assert model.layers[0].config.use_cache is False
    assert model.layers[0].gradient_checkpointing is False

    x = torch.ones(1, requires_grad=False)
    out = model.layers[0](x, use_cache=True, past_key_values=object())

    assert torch.equal(out, torch.ones(1))
    assert x.requires_grad is False
    assert checkpoint_calls[-1]["use_reentrant"] is False
    assert "context_fn" in checkpoint_calls[-1]
    _, recompute_context = checkpoint_calls[-1]["context_fn"]()
    with recompute_context:
        pass
    assert suspended_models == [model]


def test_reentrant_checkpointing_suspends_prefetch_for_its_model(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True, use_reentrant=True
        )
    )
    suspended_models = []
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: (
            suspended_models.append(model) or nullcontext()
        )
    )

    model = TaggedModule("root")
    model.layers = nn.ModuleList([CheckpointBlock("L0")])
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )

    def fake_checkpoint(function, *args, **kwargs):
        assert kwargs["use_reentrant"] is True
        return function(*args)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)

    backend._apply_activation_checkpointing(model, resolve_plan(model, plan))
    model.layers[0](torch.ones(1))

    assert suspended_models == [model]


@pytest.mark.parametrize("keyword_input", [False, True])
@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "optional"])
def test_reentrant_checkpointing_recomputes_with_frozen_inputs(keyword_input, output_kind):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    torch.manual_seed(43)
    model = nn.Linear(4, 3)
    reference = nn.Linear(4, 3)
    reference.load_state_dict(model.state_dict())
    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True, use_reentrant=True
        )
    )
    suspended_models = []
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: (
            suspended_models.append(model) or nullcontext()
        )
    )
    forward_modes = []
    original_forward = model.forward

    def observed_forward(input):
        forward_modes.append(torch.is_grad_enabled())
        assert not input.requires_grad
        output = original_forward(input)
        if output_kind == "tuple":
            return (output,)
        if output_kind == "optional":
            return output, None
        return output

    model.forward = observed_forward
    plan = FSDPParallelPlan(plan={"": FSDPPlanSpec(activation_checkpoint=True)})
    backend._apply_activation_checkpointing(model, resolve_plan(model, plan))
    inputs = [torch.randn(2, 4), torch.randn(3, 4)]
    actual = [model(input=x) if keyword_input else model(x) for x in inputs]
    if output_kind != "tensor":
        for output in actual:
            assert isinstance(output, tuple)
            assert len(output) == (1 if output_kind == "tuple" else 2)
            if output_kind == "optional":
                assert output[1] is None
        actual = [output[0] for output in actual]
    expected = [reference(x) for x in inputs]
    for output, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, ref)
        assert output.requires_grad
    sum(output.square().sum() for output in actual).backward()
    sum(output.square().sum() for output in expected).backward()

    assert forward_modes == [False, False, True, True]
    assert suspended_models == [model, model]
    for parameter, ref in zip(model.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, ref.grad)
    assert all(not x.requires_grad and x.grad is None for x in inputs)

    before_step = model.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    assert not torch.equal(model.weight, before_step)
    torch.testing.assert_close(model.weight, reference.weight)


@pytest.mark.parametrize("freeze_parameters", [False, True])
def test_reentrant_checkpointing_preserves_input_gradients(freeze_parameters):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    torch.manual_seed(44)
    model = nn.Linear(4, 3).requires_grad_(not freeze_parameters)
    reference = nn.Linear(4, 3).requires_grad_(not freeze_parameters)
    reference.load_state_dict(model.state_dict())
    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True, use_reentrant=True
        )
    )
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: nullcontext()
    )
    plan = FSDPParallelPlan(plan={"": FSDPPlanSpec(activation_checkpoint=True)})
    backend._apply_activation_checkpointing(model, resolve_plan(model, plan))
    x = torch.randn(2, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = reference(ref_x)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for parameter, ref in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref.grad)


@pytest.mark.parametrize("mode", ["no_grad", "frozen_parameters"])
def test_reentrant_checkpointing_keeps_frozen_execution_without_gradients(mode):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    model = nn.Linear(4, 3).requires_grad_(mode != "frozen_parameters")
    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True, use_reentrant=True
        )
    )
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: nullcontext()
    )
    plan = FSDPParallelPlan(plan={"": FSDPPlanSpec(activation_checkpoint=True)})
    backend._apply_activation_checkpointing(model, resolve_plan(model, plan))
    with torch.set_grad_enabled(mode != "no_grad"):
        output = model(torch.randn(2, 4))
    assert not output.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


def test_activation_checkpointing_eval_uses_original_forward(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig

    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=True)
    )
    backend.rank = 0
    backend.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    backend.prefetch_manager = types.SimpleNamespace(
        suspend_expert_forward_prefetch=lambda model: nullcontext()
    )

    model = TaggedModule("root")
    model.layers = nn.ModuleList([CheckpointBlock("L0")])
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )
    checkpoint_calls = []

    def fake_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr("torch.utils.checkpoint.checkpoint", fake_checkpoint)

    backend._apply_activation_checkpointing(model, resolve_plan(model, plan))
    model.layers[0].eval()

    x = torch.ones(1, requires_grad=False)
    out = model.layers[0](x)

    assert torch.equal(out, torch.ones(1))
    assert checkpoint_calls == []
    assert x.requires_grad is False


def test_activation_checkpointing_disabled_warns_and_skips_plan_targets(monkeypatch):
    from torchturbo.engine.config import ActivationCheckpointingConfig
    from torchturbo.engine.backends.fsdp2 import backend as backend_module

    warnings = []
    backend = make_backend()
    backend.config = types.SimpleNamespace(
        activation_checkpointing=ActivationCheckpointingConfig(enabled=False)
    )
    backend.rank = 0
    monkeypatch.setattr(
        backend_module,
        "logger",
        types.SimpleNamespace(
            warning_rank0=lambda *args, **kwargs: warnings.append(args),
        ),
    )

    model = TaggedModule("root")
    model.layers = nn.ModuleList([CheckpointBlock("L0")])
    plan = FSDPParallelPlan(
        plan={"layers.*": FSDPPlanSpec(activation_checkpoint=True)}
    )

    assert backend._apply_activation_checkpointing(
        model, resolve_plan(model, plan)
    ) is model

    assert len(warnings) == 1
    assert "skipping activation checkpointing" in warnings[0][0]
    assert not hasattr(model.layers[0], "_tt_original_forward")
