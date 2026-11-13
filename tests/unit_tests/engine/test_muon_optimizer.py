from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import Shard

from torchturbo.engine.optimizers import (
    AttentionPerHeadComputeView,
    BatchedMatrixComputeView,
    ExpertDownComputeView,
    ExpertUpComputeView,
    ExcludeFromMuon,
    MuonParameterSpec,
    MuonRecipe,
    MuonPlan,
    MuonSpecProvider,
    OptimizerClassPlan,
    OptimizerRecipe,
    OptimizerPlan,
    OptimizersContainer,
    Owned,
    ParamGroup,
    collect_module_muon_specs,
)
from torchturbo.engine.backends.fsdp2.config import CPUOffloadConfig
from torchturbo.engine.config import (
    MuonOptimizerConfig,
    OptimizerBucketPolicy,
    QKClipConfig,
)
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.utils.torch_utils import torch_version


def test_batched_matrix_compute_view_round_trip_for_fused_columns():
    storage = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    view = BatchedMatrixComputeView(batch_size=2, column_groups=2).resolve(
        torch.Size((6, 8))
    )

    logical = view.logical_view(storage)

    assert logical.shape == (4, 3, 4)
    torch.testing.assert_close(logical[0], storage[0, :, :4])
    torch.testing.assert_close(logical[1], storage[0, :, 4:])
    torch.testing.assert_close(logical[2], storage[1, :, :4])
    torch.testing.assert_close(logical[3], storage[1, :, 4:])

    restored = torch.empty_like(storage)
    view.restore_storage_order(logical, restored)
    torch.testing.assert_close(restored, storage)


@pytest.mark.parametrize(
    ("storage_shape", "batch_size", "column_groups"),
    [((5, 8), 2, 1), ((6, 7), 2, 2), ((2, 3, 4), 2, 1)],
)
def test_batched_matrix_compute_view_rejects_invalid_storage(
    storage_shape, batch_size, column_groups
):
    with pytest.raises(ValueError, match="cannot be interpreted"):
        BatchedMatrixComputeView(batch_size, column_groups).resolve(
            torch.Size(storage_shape)
        )


def test_attention_view_is_one_matrix_per_head():
    view = AttentionPerHeadComputeView(num_heads=4).resolve(torch.Size((12, 5)))

    assert view.global_compute_shape == torch.Size((4, 3, 5))
    assert view.matrix_shape == (3, 5)


@pytest.mark.parametrize(
    ("compute_view", "storage_shape", "matrix_shape", "column_groups"),
    [
        (ExpertUpComputeView(2), (6, 4), (3, 4), 1),
        (ExpertUpComputeView(2, fused_gate_up=True), (6, 8), (3, 4), 2),
        (ExpertDownComputeView(2), (8, 3), (4, 3), 1),
    ],
)
def test_expert_views_preserve_logical_matrices_and_layout_signature(
    compute_view, storage_shape, matrix_shape, column_groups
):
    storage = torch.arange(storage_shape[0] * storage_shape[1]).reshape(storage_shape)
    resolved = compute_view.resolve(torch.Size(storage_shape))
    previous = BatchedMatrixComputeView(2, column_groups).resolve(storage.shape)
    rows, columns = matrix_shape
    expected = torch.stack(
        [
            storage[row : row + rows, column : column + columns]
            for row in range(0, storage.shape[0], rows)
            for column in range(0, storage.shape[1], columns)
        ]
    )

    assert resolved.matrix_shape == matrix_shape
    assert resolved.signature == previous.signature
    torch.testing.assert_close(
        resolved.logical_view(storage.view(resolved.global_compute_shape)), expected
    )


@pytest.mark.parametrize("view_type", (ExpertUpComputeView, ExpertDownComputeView))
@pytest.mark.parametrize("num_experts", (0, -1, True, 1.5))
def test_expert_views_require_a_positive_integer_expert_count(view_type, num_experts):
    with pytest.raises(ValueError, match="num_experts must be a positive integer"):
        view_type(num_experts=num_experts)


def test_expert_up_view_requires_boolean_gate_fusion():
    with pytest.raises(TypeError, match="fused_gate_up must be a bool"):
        ExpertUpComputeView(num_experts=2, fused_gate_up="true")


def test_sharded_muon_parameter_requires_compute_view():
    with pytest.raises(ValueError, match="requires a compute view"):
        MuonParameterSpec(compute_placement=Shard(0))


def test_fsdp_storage_layout_rejects_invalid_shard_dimension():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _FSDPStorageLayout

    tensor = SimpleNamespace(
        device_mesh=SimpleNamespace(ndim=1, mesh_dim_names=("dp",)),
        placements=(Shard(2),),
        ndim=2,
    )

    with pytest.raises(NotImplementedError, match=r"requires Shard\(0\) storage"):
        _FSDPStorageLayout("weight", tensor)


def test_local_storage_identity_distinguishes_empty_views_of_backing_allocations():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _LocalStorageIdentity

    backing = torch.empty(3, 4)
    replacement = torch.empty_like(backing)
    empty = backing[1:1]
    identity = _LocalStorageIdentity(empty)

    assert empty.numel() == replacement[1:1].numel() == 0
    assert empty.data_ptr() == replacement[1:1].data_ptr() == 0
    assert identity.matches(backing[1:1])
    assert not identity.matches(replacement[1:1])
    assert not identity.matches(backing[2:2])


@pytest.mark.skipif(
    torch_version() < (2, 9, 0),
    reason="requires torch.optim.Muon (PyTorch 2.9 or newer)",
)
def test_fsdp_muon_direction_matches_pytorch_muon_for_one_matrix():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )
    from torch.optim._muon import _zeropower_via_newtonschulz

    update = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -0.25, 3.0], [2.0, 0.5, 1.0], [1.5, -2.0, 0.0]]
    )
    coefficients = (3.4445, -4.7750, 2.0315)

    expected = _zeropower_via_newtonschulz(update, coefficients, 5, 1e-7)
    actual = update.clone()
    optimizer = SimpleNamespace(
        _group=lambda _item: {
            "ns_coefficients": coefficients,
            "ns_steps": 5,
            "eps": 1e-7,
        },
    )
    optimizer._ns_step_coefficients = (
        FSDPDistributedMuon._ns_step_coefficients.__get__(optimizer)
    )
    item = SimpleNamespace(resolved_compute_view=None)

    FSDPDistributedMuon._compute_update(optimizer, item, actual)

    torch.testing.assert_close(actual, expected.to(dtype=actual.dtype), rtol=0, atol=0)


@pytest.mark.skipif(
    torch_version() < (2, 9, 0),
    reason="requires torch.optim.Muon (PyTorch 2.9 or newer)",
)
@pytest.mark.parametrize("column_groups", (1, 2))
@pytest.mark.parametrize("matrix_shape", ((7, 3), (3, 7)))
def test_batched_muon_consumes_independent_logical_matrices(
    matrix_shape, column_groups
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon
    from torch.optim._muon import _zeropower_via_newtonschulz

    rows, columns = matrix_shape
    batch_size = 2
    storage_columns = columns * column_groups
    update = torch.randn(
        batch_size,
        rows,
        storage_columns,
        generator=torch.Generator().manual_seed(31),
    )
    view = BatchedMatrixComputeView(batch_size, column_groups).resolve(
        torch.Size((batch_size * rows, storage_columns))
    )
    coefficients = (3.4445, -4.7750, 2.0315)
    expected = torch.empty_like(update)
    for batch_index in range(batch_size):
        for column_group in range(column_groups):
            column_slice = slice(column_group * columns, (column_group + 1) * columns)
            matrix = update[batch_index, :, column_slice]
            direction = _zeropower_via_newtonschulz(matrix, coefficients, 5, 1e-7)
            expected[batch_index, :, column_slice].copy_(direction)

    optimizer = SimpleNamespace(
        _group=lambda _item: {
            "ns_coefficients": coefficients,
            "ns_steps": 5,
            "eps": 1e-7,
        },
    )
    optimizer._ns_step_coefficients = (
        FSDPDistributedMuon._ns_step_coefficients.__get__(optimizer)
    )
    item = SimpleNamespace(resolved_compute_view=view)
    actual = torch.stack(
        [
            update[batch_index, :, group * columns : (group + 1) * columns]
            for batch_index in range(batch_size)
            for group in range(column_groups)
        ]
    )
    FSDPDistributedMuon._compute_update(optimizer, item, actual)
    expected_matrices = torch.stack(
        [
            expected[batch_index, :, group * columns : (group + 1) * columns]
            for batch_index in range(batch_size)
            for group in range(column_groups)
        ]
    )

    # Independent and batched BF16 kernels may use different reduction orders.
    torch.testing.assert_close(actual, expected_matrices, rtol=1e-2, atol=1e-2)


def test_optimizer_class_plan_has_construction_inputs():
    model = nn.Linear(2, 2)
    plan = OptimizerClassPlan(
        model=model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0.2},
    )

    assert isinstance(plan, OptimizerPlan)
    assert plan.optimizer_cls is torch.optim.AdamW
    assert plan.optimizer_kwargs["lr"] == 0.1
    assert plan.optimizer_kwargs["weight_decay"] == 0.2
    groups_by_optimizer_name = plan.get_param_groups_by_optimizer_name()
    assert tuple(groups_by_optimizer_name) == ("AdamW",)
    groups = groups_by_optimizer_name["AdamW"]
    assert len(groups) == 2
    assert "weight_decay" not in groups[0].optimizer_kwargs
    assert groups[1].optimizer_kwargs["weight_decay"] == 0.0
    assert [
        tuple(group.params_by_fqn) for group in groups_by_optimizer_name["AdamW"]
    ] == [("weight",), ("bias",)]
    assert plan.get_param_groups() == groups
    plan_repr = repr(plan)
    assert "optimizer_cls=torch.optim.adamw.AdamW" in plan_repr
    assert "parameter_groups=" not in plan_repr
    assert "Parameter containing:" not in plan_repr


def test_muon_recipe_implements_optimizer_recipe_contract():
    recipe = MuonRecipe({})

    assert isinstance(recipe, OptimizerRecipe)


def test_muon_recipe_matches_patterns_in_insertion_order():
    per_head = MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=AttentionPerHeadComputeView(2),
    )
    recipe = MuonRecipe(
        {
            "model.layers.*.projection.weight": ExcludeFromMuon(),
            "model.layers.0.projection.weight": per_head,
        }
    )

    assert recipe.match("model.layers.0.projection.weight") == ExcludeFromMuon()
    assert recipe.match("model.layers.0.other.weight") is None


def test_optimizer_class_plan_validates_required_kwargs():
    with pytest.raises(ValueError, match="requires explicit values"):
        OptimizerClassPlan(
            model=nn.Linear(2, 2),
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 0.1},
        )


def test_optimizer_class_plan_rejects_non_mapping_kwargs():
    with pytest.raises(TypeError, match="must be a mapping"):
        OptimizerClassPlan(
            model=nn.Linear(2, 2),
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs=[],
        )


def test_muon_plan_accepts_resolved_recipe_config_and_classifies_parameters():
    recipe = MuonRecipe({})
    muon_config = MuonOptimizerConfig(lr=0.1, weight_decay=0.02)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2))
            self.bias = nn.Parameter(torch.empty(2))

        def get_optimizer_recipe(self, optimizer_name, prepared_model):
            raise AssertionError("MuonPlan must not resolve the model recipe")

    model = Model()
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=muon_config,
    )

    assert isinstance(plan, OptimizerPlan)
    assert not hasattr(plan, "recipe")
    assert not hasattr(plan, "config")
    assert plan.muon_fqns == ("weight",)
    assert plan.adamw_decay_fqns == ()
    assert plan.adamw_no_decay_fqns == ("bias",)
    assert tuple(bucket.parameter_fqns for bucket in plan.buckets) == (("weight",),)
    assert plan.overlap_comm_compute is True
    assert plan.adamw_kwargs["lr"] == 0.1
    assert plan.adamw_kwargs["weight_decay"] == 0.02
    assert tuple(plan.get_param_groups_by_optimizer_name()) == ("Muon", "AdamW")

    # The plan snapshots its inputs: later config mutation must not reach it.
    muon_config.lr = 9.0
    muon_config.weight_decay = 8.0
    muon_config.overlap_comm_compute = False

    assert dict(plan.parameter_specs) == {
        "weight": MuonParameterSpec(compute_placement=Owned())
    }
    assert plan.adamw_kwargs["lr"] == 0.1
    assert plan.adamw_kwargs["weight_decay"] == 0.02
    assert plan.overlap_comm_compute is True
    with pytest.raises(TypeError):
        plan.adamw_kwargs["lr"] = 1.0
    with pytest.raises(TypeError):
        recipe.parameter_overrides["weight"] = ExcludeFromMuon()

    plan_repr = repr(plan)
    assert "muon_parameters=1" in plan_repr
    assert "adamw_decay_parameters=0" in plan_repr
    assert "adamw_no_decay_parameters=1" in plan_repr
    assert "Parameter containing:" not in plan_repr


def test_muon_plan_validates_its_config_and_recipe():
    model = nn.Linear(2, 2)
    recipe = MuonRecipe({})
    config = MuonOptimizerConfig(lr=0.1, weight_decay=0.02)

    with pytest.raises(ValueError, match="configuration is missing"):
        MuonPlan(model=model, recipe=recipe, config=None)
    with pytest.raises(ValueError, match="does not provide a Muon optimizer recipe"):
        MuonPlan(model=model, recipe=None, config=config)


def _routing_model():
    from transformers import PretrainedConfig, PreTrainedModel

    class Model(PreTrainedModel):
        def __init__(self):
            super().__init__(PretrainedConfig())
            self.excluded_matrix = nn.Parameter(torch.empty(2, 2))
            self.cube = nn.Parameter(torch.empty(2, 2, 2))
            self.frozen = nn.Parameter(torch.empty(2, 2), requires_grad=False)
            self.projection = nn.Linear(2, 2)
            self.embedding = nn.Embedding(4, 2)
            self.output = nn.Linear(2, 4, bias=False)

        def get_output_embeddings(self):
            return self.output

    return Model()


def test_muon_defaults_exclude_embeddings_output_and_non_matrices():
    model = _routing_model()
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({"excluded_matrix": ExcludeFromMuon()}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == ("projection.weight",)
    assert plan.parameter_specs["projection.weight"] == MuonParameterSpec(
        compute_placement=Owned()
    )
    assert set(plan.adamw_decay_fqns) == {
        "excluded_matrix", "cube", "embedding.weight", "output.weight"
    }
    assert plan.adamw_no_decay_fqns == ("projection.bias",)
    assert "frozen" not in plan.parameter_specs
    assert "frozen" not in plan.adamw_decay_fqns


def test_muon_overrides_beat_defaults_and_keep_model_parameter_order():
    model = _routing_model()
    whole_matrix = MuonParameterSpec(compute_placement=Owned())
    plan = MuonPlan(
        model,
        recipe=MuonRecipe(
            {
                # A spec is itself the statement that the parameter uses Muon,
                # so it overrides the embedding and output-embedding defaults.
                "output.weight": whole_matrix,
                "embedding.weight": whole_matrix,
                "excluded_matrix": whole_matrix,
                "projection.weight": ExcludeFromMuon(),
            }
        ),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == (
        "excluded_matrix", "embedding.weight", "output.weight"
    )
    assert set(plan.adamw_decay_fqns) == {"cube", "projection.weight"}
    assert tuple(plan.parameter_specs) == plan.muon_fqns


def test_muon_plan_warns_and_ignores_unmatched_recipe_patterns(monkeypatch):
    from torchturbo.engine.optimizers import muon as muon_module

    warnings = []
    monkeypatch.setattr(
        muon_module,
        "logger",
        SimpleNamespace(
            warning_rank0=lambda message, *args: warnings.append(message % args)
        ),
    )
    model = _routing_model()

    plan = MuonPlan(
        model,
        recipe=MuonRecipe({"missing.*": ExcludeFromMuon()}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == ("excluded_matrix", "projection.weight")
    assert warnings == [
        "MuonRecipe patterns matched no model parameter FQNs: ['missing.*']"
    ]


def test_muon_plan_ignores_overrides_for_frozen_parameters():
    model = _routing_model()
    whole_matrix = MuonParameterSpec(compute_placement=Owned())

    # Freezing is a training-time decision, so a layout declared for a frozen
    # parameter is inert rather than a recipe error.
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({"frozen": whole_matrix}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert "frozen" not in plan.parameter_specs
    assert "frozen" not in plan.adamw_decay_fqns


def test_muon_default_rule_keeps_peft_adapters_out_of_muon():
    model = nn.Module()
    model.dense = nn.Linear(4, 4, bias=False)
    model.wrapped = _wrap_with_lora(nn.Linear(4, 4, bias=False))

    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    # Adapter matrices are low-rank factors, not the linear map Muon assumes.
    assert plan.muon_fqns == ("dense.weight", "wrapped.base_layer.weight")
    assert plan.adamw_decay_fqns == (
        "wrapped.lora_A.default.weight",
        "wrapped.lora_B.default.weight",
    )


def test_muon_default_rule_accepts_modeling_parameter_named_like_adapter():
    model = nn.Module()
    model.lora_A = nn.Linear(4, 4, bias=False)

    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == ("lora_A.weight",)


def test_muon_plan_warns_when_a_recipe_excludes_every_candidate(monkeypatch):
    from torchturbo.engine.optimizers import muon as muon_module

    warnings: list[str] = []
    monkeypatch.setattr(
        muon_module,
        "logger",
        SimpleNamespace(warning_rank0=lambda message: warnings.append(message)),
    )
    model = nn.Sequential(nn.Linear(2, 2, bias=False))

    plan = MuonPlan(
        model,
        recipe=MuonRecipe({"0.weight": ExcludeFromMuon()}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == ()
    assert plan.adamw_decay_fqns == ("0.weight",)
    assert len(warnings) == 1
    assert "falls back to AdamW" in warnings[0]


def test_muon_recipe_rejects_malformed_overrides():
    with pytest.raises(TypeError, match="must be a mapping"):
        MuonRecipe([("weight", ExcludeFromMuon())])
    with pytest.raises(TypeError, match="keys must be strings"):
        MuonRecipe({0: ExcludeFromMuon()})
    with pytest.raises(TypeError, match="must be a MuonParameterSpec"):
        MuonRecipe({"weight": Owned()})


class _FusedMatrixModule(nn.Module):
    """A module that packs several Muon matrices into one storage tensor."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 3))
        self.calls = 0

    def get_muon_parameter_specs(self) -> dict[str, MuonParameterSpec]:
        self.calls += 1
        return {
            "weight": MuonParameterSpec(
                compute_placement=Shard(0),
                compute_view=BatchedMatrixComputeView(batch_size=2),
            )
        }


@pytest.mark.parametrize("whole_matrix", [False, True])
def test_module_declared_spec_replaces_default_and_yields_to_model(whole_matrix):
    model = nn.Module()
    model.matrices = _FusedMatrixModule()
    assert isinstance(model.matrices, MuonSpecProvider)
    whole_spec = MuonParameterSpec(compute_placement=Owned())
    overrides = {}
    if whole_matrix:
        # The recipe is the strongest tier, so it beats the module's own layout.
        overrides["matrices.weight"] = whole_spec
    plan = MuonPlan(
        model,
        recipe=MuonRecipe(overrides),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    spec = plan.parameter_specs["matrices.weight"]
    assert model.matrices.calls == 1
    if whole_matrix:
        assert spec == whole_spec
    else:
        assert spec.compute_placement == Shard(0)
        assert spec.compute_view == BatchedMatrixComputeView(batch_size=2)


def test_module_spec_provider_inside_checkpoint_wrapper_runs_once():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    provider = _FusedMatrixModule()
    model = nn.Module()
    model.matrices = checkpoint_wrapper(provider)
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    fqn = "matrices._checkpoint_wrapped_module.weight"
    assert plan.muon_fqns == (fqn,)
    assert plan.parameter_specs[fqn] == MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=BatchedMatrixComputeView(batch_size=2),
    )
    assert provider.calls == 1


@pytest.mark.parametrize("exclusion", ["override", "frozen"])
def test_muon_compute_spec_does_not_select_excluded_parameters(exclusion):
    model = nn.Module()
    model.matrices = _FusedMatrixModule()
    model.projection = nn.Linear(3, 3, bias=False)
    overrides = {}
    if exclusion == "override":
        overrides["matrices.weight"] = ExcludeFromMuon()
    elif exclusion == "frozen":
        model.matrices.weight.requires_grad_(False)
    plan = MuonPlan(
        model,
        recipe=MuonRecipe(overrides),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == ("projection.weight",)
    assert ("matrices.weight" in plan.adamw_decay_fqns) == (exclusion != "frozen")


def test_muon_explicit_selection_rejects_non_matrix_parameter():
    model = _routing_model()
    with pytest.raises(ValueError, match="two-dimensional"):
        MuonPlan(
            model,
            recipe=MuonRecipe(
                {"cube": MuonParameterSpec(compute_placement=Owned())}
            ),
            config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
        )


def test_muon_buckets_follow_selected_parameter_order():
    model = nn.Module()
    model.first = nn.Linear(2, 2)
    model.excluded = nn.Linear(2, 2, bias=False)
    model.last = nn.Linear(2, 2, bias=False)
    plan = MuonPlan(
        model,
        recipe=MuonRecipe({"excluded.weight": ExcludeFromMuon()}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == ("first.weight", "last.weight")
    assert tuple(bucket.parameter_fqns for bucket in plan.buckets) == (
        ("first.weight",), ("last.weight",)
    )
    assert tuple(bucket.name for bucket in plan.buckets) == ("bucket.0", "bucket.1")


def _muon_plan(model, recipe, config):
    return MuonPlan(
        model,
        recipe=recipe,
        config=config,
    )


def _bucket_test_plan(bucket_policy):
    class Layer(nn.Module):
        def __init__(self, *, include_b=False):
            super().__init__()
            self.a = nn.Parameter(torch.empty(1, 1))
            if include_b:
                self.b = nn.Parameter(torch.empty(1, 1))

    model = nn.Module()
    model.layers = nn.ModuleList([Layer(include_b=True), Layer(), Layer()])
    return MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(
            lr=0.1,
            weight_decay=0.0,
            bucket=bucket_policy,
        ),
    )


def test_muon_plan_builds_parameter_buckets_in_model_order():
    plan = _bucket_test_plan(OptimizerBucketPolicy())

    assert tuple(bucket.name for bucket in plan.buckets) == (
        "bucket.0",
        "bucket.1",
        "bucket.2",
        "bucket.3",
    )
    assert (
        tuple(fqn for bucket in plan.buckets for fqn in bucket.parameter_fqns)
        == plan.muon_fqns
    )


def test_muon_plan_applies_engine_bucket_policies():
    by_size = _bucket_test_plan(
        OptimizerBucketPolicy(
            strategy="size",
            bucket_cap_mb=8 / (1024 * 1024),
        )
    )
    single = _bucket_test_plan(OptimizerBucketPolicy(strategy="single"))

    assert tuple(bucket.parameter_fqns for bucket in by_size.buckets) == (
        ("layers.0.a", "layers.0.b"),
        ("layers.1.a", "layers.2.a"),
    )
    assert tuple(bucket.parameter_fqns for bucket in single.buckets) == (
        ("layers.0.a", "layers.0.b", "layers.1.a", "layers.2.a"),
    )


def _bucket_test_parameter(
    name,
    tensor,
    group,
    *,
    batch_size=None,
    owner_group_rank=None,
    owner_group_ranks=None,
    column_groups=1,
):
    if torch_version() < (2, 10, 0):
        pytest.skip("Muon layout planning requires PyTorch 2.10 or newer")
    from torchturbo.engine.backends.fsdp2.optimizers.muon_layout import (
        _MuonLayoutPlanner,
    )

    mesh_index = group.mesh_indices_in_group_order[group.current_group_rank]
    num_ranks = len(group.mesh_indices_in_group_order)
    num_rows, row_offset = Shard.local_shard_size_and_offset(
        tensor.shape[0], num_ranks, mesh_index
    )
    local_tensor = tensor[row_offset : row_offset + num_rows].clone()
    view = (
        BatchedMatrixComputeView(batch_size, column_groups).resolve(tensor.shape)
        if batch_size is not None
        else None
    )
    mesh_ranks = [0] * num_ranks
    for group_rank, index in enumerate(group.mesh_indices_in_group_order):
        mesh_ranks[index] = group.global_ranks_in_group_order[group_rank]
    item = SimpleNamespace(
        fqn=name,
        param=SimpleNamespace(
            shape=tensor.shape,
            dtype=tensor.dtype,
            stride=tensor.stride,
            numel=tensor.numel,
            element_size=tensor.element_size,
            to_local=lambda: local_tensor,
        ),
        group_index=0,
        is_owned=view is None,
        compute_placement=Owned() if view is None else Shard(0),
        global_compute_shape=(
            tensor.shape if view is None else view.global_compute_shape
        ),
        resolved_compute_view=view,
        compute_view_key=("identity",) if view is None else view.signature,
        storage_layout=SimpleNamespace(
            global_shape=tensor.shape,
            global_stride=tensor.stride(),
            local_shape=local_tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            placements=(Shard(0),),
            global_ranks_in_mesh_order=tuple(mesh_ranks),
        ),
    )
    planner = _MuonLayoutPlanner(
        group.mesh_indices_in_group_order, group.current_group_rank
    )
    layout = planner.resolve_parameter(
        fqn=name,
        global_storage_shape=tensor.shape,
        local_storage_shape=local_tensor.shape,
        compute_view=view,
    )
    if owner_group_ranks is None:
        if view is None:
            owner_group_ranks = (owner_group_rank,)
        else:
            # Schedule tests fix ownership explicitly; optimizer tests exercise
            # the load balancer independently of these transport fixtures.
            owners = [0] * (batch_size * column_groups)
            for group_rank, mesh_index in enumerate(group.mesh_indices_in_group_order):
                count, offset = Shard.local_shard_size_and_offset(
                    batch_size, num_ranks, mesh_index
                )
                for index in range(offset, offset + count):
                    for column_group in range(column_groups):
                        owners[index * column_groups + column_group] = group_rank
            owner_group_ranks = tuple(owners)
    plan = planner.build_plan(layout, owner_group_ranks)
    return item, plan


@pytest.mark.skipif(
    torch_version() < (2, 10, 0),
    reason="Muon layout planning requires PyTorch 2.10 or newer",
)
def test_owned_schedule_uses_group_rank_order_for_splits_and_mesh_order_for_offsets(
    monkeypatch,
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        _BoundMuonParameter,
        _CompiledMuonBucket,
        _FSDPStorageLayout,
        _LocalStorageIdentity,
        _MuonCommGroup,
    )
    from torchturbo.engine.backends.fsdp2.optimizers.muon_layout import (
        _MuonLayoutPlanner,
    )

    process_group = object()

    class FakeParameter:
        def __init__(self, shape, local_shape):
            self._global = torch.empty(shape)
            self.shape = self._global.shape
            self.ndim = self._global.ndim
            self._local = torch.empty(local_shape)
            self.placements = (Shard(0),)
            self.device_mesh = SimpleNamespace(
                ndim=1,
                mesh_dim_names=("dp",),
                mesh=torch.tensor((2, 0, 1)),
                get_group=lambda: process_group,
            )

        def to_local(self):
            return self._local

        def stride(self):
            return self._global.stride()

    def make_item(name, shape, local_shape):
        param = FakeParameter(shape, local_shape)
        local = param.to_local()
        return _BoundMuonParameter(
            fqn=name,
            param=param,
            group_index=0,
            compute_placement=Owned(),
            global_compute_shape=torch.Size(shape),
            resolved_compute_view=None,
            compute_view_key=("identity",),
            storage_layout=_FSDPStorageLayout(name, param),
            local_storage_identity=_LocalStorageIdentity(local),
        )

    expected = (
        ((6, 0, 4), (6, 3, 6), (2, 3), (1, 4)),
        ((3, 0, 0), (0, 0, 0), (1, 3), (0, 4)),
        ((6, 0, 4), (4, 0, 4), (2, 3), (1, 4)),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda _group: 3)
    monkeypatch.setattr(
        dist,
        "get_global_rank",
        lambda _group, group_rank: group_rank,
    )
    for current_group_rank, (send, receive, first_local, second_local) in enumerate(
        expected
    ):
        monkeypatch.setattr(
            dist,
            "get_rank",
            lambda _group, rank=current_group_rank: rank,
        )
        items = (
            make_item("first", (5, 3), first_local),
            make_item("second", (2, 4), second_local),
        )
        group = _MuonCommGroup(items[0])
        planner = _MuonLayoutPlanner(
            group.mesh_indices_in_group_order, current_group_rank
        )
        plans = tuple(
            planner.build_plan(
                planner.resolve_parameter(
                    fqn=item.fqn,
                    global_storage_shape=item.param.shape,
                    local_storage_shape=item.param.to_local().shape,
                ),
                (owner,),
            )
            for item, owner in zip(items, (0, 2), strict=True)
        )
        compiled_bucket = _CompiledMuonBucket.compile(
            name="test",
            items=items,
            parameter_plans=plans,
            parameter_groups=(group, group),
        )
        schedule = compiled_bucket.forward
        assert schedule is not None

        assert schedule.input_splits == send
        assert schedule.output_splits == receive
        assert tuple(
            plan.storage_partition(current_group_rank).tensor_shape
            for plan in compiled_bucket.redistributed_plans
        ) == (
            torch.Size(first_local),
            torch.Size(second_local),
        )
        assert tuple(
            plan.compute_partition(current_group_rank).tensor_shape
            for plan in compiled_bucket.redistributed_plans
        ) == (
            torch.Size((5, 3)) if current_group_rank == 0 else torch.Size((0, 3)),
            torch.Size((2, 4)) if current_group_rank == 2 else torch.Size((0, 4)),
        )
        assert compiled_bucket.reverse.input_splits == receive
        assert compiled_bucket.reverse.output_splits == send

    assert schedule.input_spans_by_parameter[1][0].numel == 4


@pytest.mark.parametrize("local_first", (False, True))
def test_mixed_redistribution_round_trip_preserves_parameter_and_matrix_order(
    monkeypatch, local_first
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _CompiledMuonBucket,
        _MuonWorkspace,
    )
    from torchturbo.runtime.cpu import CpuRuntime

    tensors = {
        "local": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "batch": torch.arange(48, dtype=torch.float32).reshape(12, 4) + 100,
        "owned": torch.arange(6, dtype=torch.float32).reshape(2, 3) + 200,
        "fused": torch.arange(36, dtype=torch.float32).reshape(6, 6) + 300,
    }
    order = (
        ("local", "batch", "owned", "fused")
        if local_first
        else ("batch", "local", "owned", "fused")
    )
    process_group = object()
    buckets = []
    buffers = []
    optimizers = []
    computed = []
    for rank in range(3):
        group = SimpleNamespace(
            global_ranks_in_group_order=(0, 1, 2),
            mesh_indices_in_group_order=(2, 0, 1),
            current_group_rank=rank,
            process_group=process_group,
        )
        entries = {
            "local": _bucket_test_parameter(
                "local", tensors["local"], group, batch_size=3
            ),
            "batch": _bucket_test_parameter(
                "batch", tensors["batch"], group, batch_size=2
            ),
            "owned": _bucket_test_parameter(
                "owned", tensors["owned"], group, owner_group_rank=1
            ),
            "fused": _bucket_test_parameter(
                "fused", tensors["fused"], group,
                batch_size=3, column_groups=2,
                owner_group_ranks=(2, 0, 1, 2, 0, 1),
            ),
        }
        bucket = _CompiledMuonBucket.compile(
            name="mixed",
            items=tuple(entries[name][0] for name in order),
            parameter_plans=tuple(entries[name][1] for name in order),
            parameter_groups=(group,) * len(order),
        )
        workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=False)
        workspace.configure((bucket,))
        buffers.append(workspace.slot_for_bucket(0).buffers)
        buckets.append(bucket)

        def compute_update(item, compute, *, rank=rank, entries=entries):
            partition = entries[item.fqn][1].compute_partition(rank)
            original = tensors[item.fqn]
            if item.resolved_compute_view is None:
                expected = original
            else:
                view = item.resolved_compute_view
                matrices = tuple(
                    matrix
                    for block in original.chunk(view.batch_size, dim=0)
                    for matrix in block.chunk(view.column_groups, dim=1)
                )
                expected = torch.stack(
                    [matrices[index] for index in partition.matrix_indices]
                )
            torch.testing.assert_close(compute, expected, rtol=0, atol=0)
            computed.append((rank, item.fqn, tuple(compute.shape)))
            compute.mul_(2).add_(7)

        optimizer = object.__new__(FSDPDistributedMuon)
        optimizer._prepare_local = lambda item, prepared: prepared.copy_(
            item.param.to_local().view_as(prepared)
        )
        optimizer._compute_update = compute_update
        optimizer._apply_update = lambda item, direction: item.param.to_local().copy_(
            direction.view_as(item.param.to_local())
        )
        optimizers.append(optimizer)
        assert tuple(item.fqn for item in bucket.redistributed_items) == (
            "batch", "owned", "fused"
        )
        assert len(bucket.forward.input_spans_by_parameter) == 3
        assert bucket.forward.input_numel == bucket.reverse.output_numel
        assert bucket.forward.output_numel == bucket.reverse.input_numel
        storage_buffer, compute_buffer = buffers[-1].exchange_buffers(bucket)
        assert storage_buffer.numel() == bucket.reverse.output_numel
        assert compute_buffer.numel() == bucket.reverse.input_numel

    # The middle storage shard intersects two complete-matrix destinations.
    assert len(buckets[2].forward.input_spans_by_parameter[0]) == 2
    assert buckets[0].forward.output_numel == 12
    fused_partition = buckets[0].redistributed_plans[2].compute_partition(0)
    assert fused_partition.matrix_indices == (1, 4)
    pending_transfers = []
    collective_calls = []

    def all_to_all(output, source, *, output_split_sizes, input_split_sizes, group):
        assert group is process_group
        collective_calls.append(group)
        pending_transfers.append(
            (output, source, output_split_sizes, input_split_sizes)
        )
        if len(pending_transfers) != 3:
            return
        for destination_rank, (output, _, receive_splits, _) in enumerate(
            pending_transfers
        ):
            output_offset = 0
            for source_rank, (_, source, _, send_splits) in enumerate(
                pending_transfers
            ):
                numel = receive_splits[source_rank]
                assert numel == send_splits[destination_rank]
                input_offset = sum(send_splits[:destination_rank])
                output[output_offset : output_offset + numel].copy_(
                    source[input_offset : input_offset + numel]
                )
                output_offset += numel
            assert output_offset == output.numel()
        pending_transfers.clear()

    monkeypatch.setattr(dist, "all_to_all_single", all_to_all)
    for optimizer, bucket, buffer in zip(optimizers, buckets, buffers, strict=True):
        optimizer._prepare_redistributed(bucket, buffer)
    for optimizer, bucket, buffer in zip(optimizers, buckets, buffers, strict=True):
        optimizer._gather(bucket, buffer)
    for optimizer, bucket, buffer in zip(optimizers, buckets, buffers, strict=True):
        optimizer._compute_local_items(bucket, buffer)
        optimizer._compute_redistributed(bucket, buffer)
    for optimizer, bucket, buffer in zip(optimizers, buckets, buffers, strict=True):
        optimizer._scatter(bucket, buffer)
    for optimizer, bucket, buffer in zip(optimizers, buckets, buffers, strict=True):
        optimizer._finalize_redistributed(bucket, buffer)

    assert len(collective_calls) == 6
    assert pending_transfers == []
    assert [name for rank, name, _shape in computed if rank == 0] == ["local", "fused"]
    assert [shape for _rank, name, shape in computed if name == "batch"] == [
        (1, 6, 4), (1, 6, 4)
    ]
    for bucket in buckets:
        mesh_index = bucket.group.mesh_indices_in_group_order[
            bucket.group.current_group_rank
        ]
        for item in bucket.items:
            original = tensors[item.fqn]
            num_rows, offset = Shard.local_shard_size_and_offset(
                original.shape[0], 3, mesh_index
            )
            expected = original[offset : offset + num_rows] * 2 + 7
            torch.testing.assert_close(item.param.to_local(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", (None, 1))
def test_single_rank_owned_and_batched_matrices_compute_without_exchange(
    monkeypatch, batch_size
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _CompiledMuonBucket,
        _MuonWorkspace,
    )
    from torchturbo.runtime.cpu import CpuRuntime

    group = SimpleNamespace(
        global_ranks_in_group_order=(0,),
        mesh_indices_in_group_order=(0,),
        current_group_rank=0,
        process_group=object(),
    )
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    item, plan = _bucket_test_parameter(
        "weight", value, group, batch_size=batch_size, owner_group_rank=0
    )
    bucket = _CompiledMuonBucket.compile(
        name="local",
        items=(item,),
        parameter_plans=(plan,),
        parameter_groups=(group,),
    )
    assert not bucket.has_redistribution
    assert bucket.group is bucket.forward is bucket.reverse is None
    expected_shape = (3, 4) if batch_size is None else (1, 3, 4)
    assert bucket.local_item_compute_shapes == (torch.Size(expected_shape),)

    def reject_collective(*args, **kwargs):
        raise AssertionError("a complete local matrix needs no collective")

    monkeypatch.setattr(dist, "all_to_all_single", reject_collective)
    optimizer = object.__new__(FSDPDistributedMuon)
    optimizer._workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=False)
    optimizer._workspace.configure((bucket,))
    optimizer._compiled_buckets = (bucket,)
    optimizer._prepare_local = lambda item, prepared: prepared.copy_(
        item.param.to_local().view_as(prepared)
    )
    optimizer._compute_update = lambda item, compute: compute.mul_(2).add_(7)
    optimizer._apply_update = lambda item, direction: item.param.to_local().copy_(
        direction.view_as(item.param.to_local())
    )
    optimizer._run_synchronous()
    assert optimizer._workspace.slots == ()
    torch.testing.assert_close(item.param.to_local(), value * 2 + 7, rtol=0, atol=0)


@pytest.mark.parametrize("mismatch", ("group", "dtype", "parameter_count"))
def test_redistributed_bucket_rejects_incompatible_parameters(mismatch):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _CompiledMuonBucket

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1),
        mesh_indices_in_group_order=(0, 1),
        current_group_rank=0,
        process_group=object(),
    )
    second_group = (
        SimpleNamespace(
            global_ranks_in_group_order=(0, 2),
            mesh_indices_in_group_order=(0, 1),
            current_group_rank=0,
            process_group=object(),
        )
        if mismatch == "group"
        else group
    )
    first, first_plan = _bucket_test_parameter(
        "first", torch.empty(4, 3), group, owner_group_rank=0
    )
    second, second_plan = _bucket_test_parameter(
        "second",
        torch.empty(
            4, 3, dtype=torch.float64 if mismatch == "dtype" else torch.float32
        ),
        second_group,
        batch_size=1,
    )
    if mismatch == "group":
        error, message = NotImplementedError, "span more than one process group"
    elif mismatch == "dtype":
        error, message = ValueError, "mix dtype or device"
    else:
        error, message = ValueError, "one plan and group per parameter"
    with pytest.raises(error, match=message):
        _CompiledMuonBucket.compile(
            name="incompatible",
            items=(first, second),
            parameter_plans=(first_plan, second_plan),
            parameter_groups=(
                (group,) if mismatch == "parameter_count" else (group, second_group)
            ),
        )


def test_mixed_bucket_local_item_can_use_a_different_group_and_dtype():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _CompiledMuonBucket

    local_group = SimpleNamespace(
        global_ranks_in_group_order=(0,),
        mesh_indices_in_group_order=(0,),
        current_group_rank=0,
        process_group=object(),
    )
    transfer_group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1),
        mesh_indices_in_group_order=(0, 1),
        current_group_rank=0,
        process_group=object(),
    )
    local, local_plan = _bucket_test_parameter(
        "local", torch.empty(2, 3, dtype=torch.float64), local_group, batch_size=1
    )
    owned, owned_plan = _bucket_test_parameter(
        "owned", torch.empty(4, 3), transfer_group, owner_group_rank=1
    )
    bucket = _CompiledMuonBucket.compile(
        name="mixed",
        items=(local, owned),
        parameter_plans=(local_plan, owned_plan),
        parameter_groups=(local_group, transfer_group),
    )
    assert bucket.group is transfer_group
    assert bucket.local_items == (local,)
    assert bucket.local_item_compute_shapes == (torch.Size((1, 2, 3)),)
    assert bucket.redistributed_items == (owned,)
    assert bucket.dtype == torch.float32


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("group_size", "different group sizes"),
        ("parameter_count", "different parameter counts"),
        ("negative_split", "negative size"),
        ("buffer_gap", "do not form a packed buffer"),
        ("zero_span", "do not form a packed buffer"),
        ("zero_columns", "do not form a packed buffer"),
        ("overlapping_rows", "do not form a packed buffer"),
        ("negative_tensor_offset", "do not form a packed buffer"),
        ("span_total", "do not match split sizes"),
        ("local_capacity", "local Muon exchange has mismatched buffers"),
    ),
)
def test_packed_transfer_schedule_rejects_invalid_buffer_metadata(mismatch, message):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        _TransferSpan,
        _PackedTransferSchedule,
    )

    metadata = {
        "input_splits": (2, 0),
        "output_splits": (2, 0),
        "input_spans_by_parameter": ((_TransferSpan(0, 0, 1, 2, 2),),),
        "output_spans_by_parameter": ((_TransferSpan(0, 0, 1, 2, 2),),),
        "has_remote_transfers": False,
    }
    if mismatch == "group_size":
        metadata["output_splits"] = (2,)
    elif mismatch == "parameter_count":
        metadata["output_spans_by_parameter"] = ()
    elif mismatch == "negative_split":
        metadata["input_splits"] = (3, -1)
    elif mismatch == "buffer_gap":
        metadata["input_spans_by_parameter"] = ((_TransferSpan(1, 0, 1, 2, 2),),)
    elif mismatch == "zero_span":
        metadata["input_spans_by_parameter"] = ((_TransferSpan(0, 0, 0, 2, 2),),)
    elif mismatch == "zero_columns":
        metadata["input_spans_by_parameter"] = ((_TransferSpan(0, 0, 2, 0, 2),),)
    elif mismatch == "overlapping_rows":
        metadata["input_spans_by_parameter"] = ((_TransferSpan(0, 0, 1, 2, 1),),)
    elif mismatch == "negative_tensor_offset":
        metadata["input_spans_by_parameter"] = ((_TransferSpan(0, -1, 1, 2, 2),),)
    elif mismatch == "span_total":
        metadata["output_splits"] = (3, 0)
    else:
        metadata["output_splits"] = (3, 0)
        metadata["output_spans_by_parameter"] = ((_TransferSpan(0, 0, 1, 3, 3),),)
    with pytest.raises(RuntimeError, match=message):
        _PackedTransferSchedule(**metadata)


def test_redistributed_parameter_signature_uses_global_plan_metadata():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _CompiledMuonBucket,
    )

    signatures = []
    local_splits = []
    optimizer = SimpleNamespace(_group_signature=lambda _item: ())
    for rank in range(3):
        group = SimpleNamespace(
            global_ranks_in_group_order=(0, 1, 2),
            mesh_indices_in_group_order=(0, 1, 2),
            current_group_rank=rank,
            process_group=object(),
        )
        item, plan = _bucket_test_parameter(
            "matrix", torch.empty(12, 4), group, batch_size=2
        )
        bucket = _CompiledMuonBucket.compile(
            name="matrix",
            items=(item,),
            parameter_plans=(plan,),
            parameter_groups=(group,),
        )
        signatures.append(
            FSDPDistributedMuon._parameter_signature(optimizer, item, bucket)
        )
        local_splits.append(bucket.forward.output_splits)
    assert signatures[0] == signatures[1] == signatures[2]
    assert len(set(local_splits)) == 3


def _assignment_test_inputs(specs, group, *, dtype=torch.float32):
    from torchturbo.engine.backends.fsdp2.optimizers.muon_layout import (
        _MuonLayoutPlanner,
    )

    planner = _MuonLayoutPlanner(
        group.mesh_indices_in_group_order, group.current_group_rank
    )
    items = []
    layouts = []
    for name, (shape, batch_size, column_groups) in specs.items():
        tensor = torch.empty(shape, dtype=dtype)
        local_rows, _ = Shard.local_shard_size_and_offset(
            shape[0],
            len(group.global_ranks_in_group_order),
            group.mesh_indices_in_group_order[group.current_group_rank],
        )
        view = (
            BatchedMatrixComputeView(batch_size, column_groups).resolve(tensor.shape)
            if batch_size is not None
            else None
        )
        items.append(SimpleNamespace(fqn=name, param=tensor, group_index=0))
        layouts.append(
            planner.resolve_parameter(
                fqn=name,
                global_storage_shape=tensor.shape,
                local_storage_shape=torch.Size((local_rows, shape[1])),
                compute_view=view,
            )
        )
    return tuple(items), tuple(layouts), (group,) * len(items)


def _assignment_route_loads(inputs, assignments, *, ns_steps=5):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        _MuonRankLoad,
        _estimate_muon_compute_cost,
    )
    from torchturbo.engine.backends.fsdp2.optimizers.muon_layout import (
        _MuonLayoutPlanner,
    )

    loads_by_group = {}
    items, layouts, groups = inputs
    for item, layout, group, owners in zip(
        items, layouts, groups, assignments, strict=True
    ):
        ranks = group.global_ranks_in_group_order
        if ranks not in loads_by_group:
            loads_by_group[ranks] = [_MuonRankLoad() for _ in ranks]
        loads = loads_by_group[ranks]
        planner = _MuonLayoutPlanner(
            group.mesh_indices_in_group_order, group.current_group_rank
        )
        plan = planner.build_plan(layout, owners)
        for matrix, owner in zip(plan.matrices, owners, strict=True):
            loads[owner].compute += _estimate_muon_compute_cost(
                matrix.tensor_shape, ns_steps
            )
        for route in plan.storage_to_compute_routes:
            if route.source_group_rank == route.destination_group_rank:
                continue
            num_bytes = route.numel * item.param.element_size()
            loads[route.source_group_rank].send_bytes += num_bytes
            loads[route.destination_group_rank].receive_bytes += num_bytes
    return loads_by_group


@pytest.mark.parametrize(
    ("shape", "ns_steps", "expected_flops"),
    [
        ((3, 4), 2, 396),
        ((128, 128), 5, 62_914_560),
        ((128, 512), 5, 188_743_680),
        ((512, 128), 5, 188_743_680),
        ((128, 128), 0, 0),
    ],
)
def test_muon_compute_cost_counts_two_flops_per_multiply_add(
    shape, ns_steps, expected_flops
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        _estimate_muon_compute_cost,
    )

    assert _estimate_muon_compute_cost(torch.Size(shape), ns_steps) == expected_flops


@pytest.mark.parametrize("owned_columns", (128, 129))
@pytest.mark.parametrize("current_group_rank", (0, 15))
def test_full_bucket_assignment_avoids_shifting_seventeen_matrices_on_sixteen_ranks(
    owned_columns, current_group_rank
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    ranks = tuple(range(16))
    group = SimpleNamespace(
        global_ranks_in_group_order=ranks,
        mesh_indices_in_group_order=ranks,
        current_group_rank=current_group_rank,
    )
    inputs = _assignment_test_inputs(
        {
            "a.owned": ((128, owned_columns), None, 1),
            "b.heads": ((16 * 128, 128), 16, 1),
        },
        group,
    )
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 5})
    cumulative = {}

    owners = FSDPDistributedMuon._assign_matrix_owners(
        optimizer, *inputs, cumulative
    )

    # A second matrix is unavoidable on one rank. Tiny compute savings do not
    # justify transferring otherwise complete local heads.
    assert owners == ((0,), ranks)
    loads = _assignment_route_loads(inputs, owners)[ranks]
    assert cumulative[ranks] == loads
    unavoidable_owned_bytes = 128 * owned_columns * 4 * 15 // 16
    assert sum(load.send_bytes for load in loads) == unavoidable_owned_bytes
    assert sum(load.receive_bytes for load in loads) == unavoidable_owned_bytes


def test_full_bucket_assignment_moves_only_one_head_when_compute_savings_pay():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _COLLECTIVE_FLOPS_PER_BYTE,
    )

    ranks = tuple(range(16))
    group = SimpleNamespace(
        global_ranks_in_group_order=ranks,
        mesh_indices_in_group_order=ranks,
        current_group_rank=0,
    )
    inputs = _assignment_test_inputs(
        {
            "a.owned": ((128, 160), None, 1),
            "b.heads": ((16 * 128, 128), 16, 1),
        },
        group,
    )
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 5})
    owners = FSDPDistributedMuon._assign_matrix_owners(optimizer, *inputs, {})
    loads = _assignment_route_loads(inputs, owners)[ranks]
    local_loads = _assignment_route_loads(inputs, ((0,), ranks))[ranks]

    def bucket_cost(rank_loads):
        compute_peak = max(load.compute for load in rank_loads)
        traffic_peak = max(
            max(load.send_bytes, load.receive_bytes) for load in rank_loads
        )
        return compute_peak + 2 * _COLLECTIVE_FLOPS_PER_BYTE * traffic_peak

    # Move the head sharing the heavier Owned matrix; reserve other local heads.
    assert sum(owner != rank for rank, owner in enumerate(owners[1])) == 1
    assert bucket_cost(loads) < bucket_cost(local_loads)
    assert max(load.compute for load in loads) == 2 * 62_914_560
    unavoidable_owned_bytes = 128 * 160 * 4 * 15 // 16
    assert sum(load.send_bytes for load in loads) == (
        unavoidable_owned_bytes + 128 * 128 * 4
    )


def test_full_bucket_assignment_keeps_greedy_candidate_for_tied_bottlenecks():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    ranks = (0, 1, 2)
    group = SimpleNamespace(
        global_ranks_in_group_order=ranks,
        mesh_indices_in_group_order=ranks,
        current_group_rank=0,
    )
    inputs = _assignment_test_inputs(
        {f"owned{index}": ((128, 128), None, 1) for index in range(6)}, group
    )
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 5})
    cumulative = {}

    owners = FSDPDistributedMuon._assign_matrix_owners(
        optimizer, *inputs, cumulative
    )

    # Storage has 43/43/42 rows, so locality favors a 3/3/0 task split.
    # One move cannot lower either tied peak and adds traffic. The complete
    # greedy candidate reaches 2/2/2, where both peaks are lower.
    loads = _assignment_route_loads(inputs, owners)[ranks]
    assert [load.compute for load in loads] == [2 * 62_914_560] * 3
    assert max(max(load.send_bytes, load.receive_bytes) for load in loads) == 88_064
    assert cumulative[ranks] == loads


def test_matrix_assignment_commits_only_winning_routes_to_each_group_history():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _MuonRankLoad,
    )

    dense_group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1, 2, 3),
        mesh_indices_in_group_order=(0, 1, 2, 3),
        current_group_rank=0,
    )
    expert_group = SimpleNamespace(
        global_ranks_in_group_order=(0, 4),
        mesh_indices_in_group_order=(0, 1),
        current_group_rank=0,
    )
    dense_inputs = _assignment_test_inputs(
        {
            "dense.owned": ((128, 128), None, 1),
            "dense.heads": ((4 * 128, 128), 4, 1),
        },
        dense_group,
    )
    expert_inputs = _assignment_test_inputs(
        {"experts": ((12, 6), 3, 2)}, expert_group
    )
    inputs = tuple(
        dense + expert
        for dense, expert in zip(dense_inputs, expert_inputs, strict=True)
    )
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 5})
    cumulative = {}
    for _ in range(2):
        previous = copy.deepcopy(cumulative)
        owners = FSDPDistributedMuon._assign_matrix_owners(
            optimizer, *inputs, cumulative
        )
        route_loads = _assignment_route_loads(inputs, owners)
        assert cumulative.keys() == route_loads.keys()
        for ranks, loads in route_loads.items():
            prior_loads = previous.get(ranks, [_MuonRankLoad() for _ in ranks])
            assert cumulative[ranks] == [
                _MuonRankLoad(
                    compute=prior.compute + current.compute,
                    send_bytes=prior.send_bytes + current.send_bytes,
                    receive_bytes=prior.receive_bytes + current.receive_bytes,
                )
                for prior, current in zip(prior_loads, loads, strict=True)
            ]


@pytest.mark.parametrize("prior_compute", (0, 10**12))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_mixed_bucket_keeps_local_heads_when_projected_peaks_tie(
    prior_compute, dtype
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _MuonRankLoad,
    )

    ranks = tuple(range(8))
    group = SimpleNamespace(
        global_ranks_in_group_order=ranks,
        mesh_indices_in_group_order=ranks,
        current_group_rank=0,
    )
    specs = {
        "a.q": ((64, 64), 8, 1),
        "b.mlp": ((256, 256), None, 1),
        "c.q": ((64, 64), 8, 1),
        "d.mlp": ((256, 256), None, 1),
    }
    inputs = _assignment_test_inputs(specs, group, dtype=dtype)
    loads = [_MuonRankLoad() for _ in ranks]
    loads[3].compute = prior_compute
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 5})
    first_heads, first_mlp, second_heads, second_mlp = (
        FSDPDistributedMuon._assign_matrix_owners(
            optimizer, *inputs, {ranks: loads}
        )
    )

    assert (first_mlp, second_mlp) == ((0,), (1,))
    # The large matrices determine both peaks. Local heads on ranks 2-7
    # remain local even when a different owner has a lower cumulative load.
    assert first_heads[2:] == second_heads[2:] == tuple(range(2, 8))
    remote_head_elements = 8 * 64 * sum(
        owner != matrix_index
        for owners in (first_heads, second_heads)
        for matrix_index, owner in enumerate(owners)
    )
    assert remote_head_elements == 2048
    unavoidable_mlp_elements = 2 * 256 * 256 * 7 // 8
    element_size = inputs[0][0].param.element_size()
    assert sum(load.receive_bytes for load in loads) == (
        unavoidable_mlp_elements + remote_head_elements
    ) * element_size
    assert sum(load.send_bytes for load in loads) == sum(
        load.receive_bytes for load in loads
    )


@pytest.mark.parametrize(
    ("matrix_size", "dtype", "num_moved_heads"),
    [
        (128, torch.float32, 0),
        (128, torch.bfloat16, 1),
        (256, torch.float32, 1),
    ],
)
def test_matrix_migration_trades_compute_against_actual_storage_bytes(
    matrix_size, dtype, num_moved_heads
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    ranks = tuple(range(16))
    group = SimpleNamespace(
        global_ranks_in_group_order=ranks,
        mesh_indices_in_group_order=ranks,
        current_group_rank=0,
    )
    owned_columns = matrix_size * 135 // 128
    specs = {
        "a.owned": ((matrix_size, owned_columns), None, 1),
        "b.heads": ((16 * matrix_size, matrix_size), 16, 1),
    }
    inputs = _assignment_test_inputs(specs, group, dtype=dtype)
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 5})
    owners = FSDPDistributedMuon._assign_matrix_owners(optimizer, *inputs, {})
    # Halving transfer bytes or doubling matrix dimensions makes one move pay.
    assert sum(owner != rank for rank, owner in enumerate(owners[1])) == num_moved_heads
    loads = _assignment_route_loads(inputs, owners)[ranks]
    element_size = inputs[0][0].param.element_size()
    unavoidable_owned_bytes = matrix_size * owned_columns * element_size * 15 // 16
    assert sum(load.send_bytes for load in loads) == (
        unavoidable_owned_bytes + num_moved_heads * matrix_size**2 * element_size
    )


@pytest.mark.parametrize("separate_buckets", (False, True))
def test_matrix_ownership_balances_bucket_cost_before_cumulative_cost(
    separate_buckets,
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1, 2, 3),
        mesh_indices_in_group_order=(0, 1, 2, 3),
        current_group_rank=0,
    )
    specs = {"first": ((6, 4), 2, 1), "second": ((6, 4), 2, 1)}
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 2})
    cumulative = {}
    if separate_buckets:
        assignments = tuple(
            FSDPDistributedMuon._assign_matrix_owners(
                optimizer, *_assignment_test_inputs({name: spec}, group), cumulative
            )[0]
            for name, spec in specs.items()
        )
    else:
        assignments = FSDPDistributedMuon._assign_matrix_owners(
            optimizer, *_assignment_test_inputs(specs, group), cumulative
        )
    loads = cumulative[(0, 1, 2, 3)]
    if separate_buckets:
        # Each bucket saves a transfer by reusing the nearest storage owners.
        assert assignments == ((0, 2), (0, 2))
        assert [load.compute for load in loads] == [792, 0, 792, 0]
        assert [load.send_bytes for load in loads] == [0, 64, 0, 0]
        assert [load.receive_bytes for load in loads] == [32, 0, 32, 0]
    else:
        # An empty storage rank can compute when that lowers bucket cost.
        assert assignments == ((0, 2), (1, 3))
        assert [load.compute for load in loads] == [396] * 4
        assert [load.send_bytes for load in loads] == [32, 48, 32, 0]
        assert [load.receive_bytes for load in loads] == [16, 32, 16, 48]
    assert sum(load.send_bytes for load in loads) == sum(
        load.receive_bytes for load in loads
    )
    if not separate_buckets:
        reverse_specs = dict(reversed(tuple(specs.items())))
        reverse_assignments = FSDPDistributedMuon._assign_matrix_owners(
            optimizer, *_assignment_test_inputs(reverse_specs, group), {}
        )
        assert dict(zip(reverse_specs, reverse_assignments, strict=True)) == {
            "first": (0, 2),
            "second": (1, 3),
        }


def test_matrix_owner_avoids_extra_traffic_despite_lower_cumulative_compute():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _MuonRankLoad,
    )

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1, 2, 3),
        mesh_indices_in_group_order=(0, 1, 2, 3),
        current_group_rank=3,
    )
    cumulative = {
        (0, 1, 2, 3): [
            _MuonRankLoad(compute=100),
            _MuonRankLoad(compute=100),
            _MuonRankLoad(compute=100),
            _MuonRankLoad(),
        ]
    }
    inputs = _assignment_test_inputs({"matrix": ((2, 4), 1, 1)}, group)
    assert inputs[1][0].storage_partitions[3].numel == 0
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 2})
    assignments = FSDPDistributedMuon._assign_matrix_owners(
        optimizer, *inputs, cumulative
    )
    assert assignments == ((0,),)
    assert cumulative[(0, 1, 2, 3)][0].receive_bytes == 16
    assert cumulative[(0, 1, 2, 3)][3] == _MuonRankLoad()


def test_owned_matrix_stays_indivisible_while_fused_column_groups_are_tasks():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1, 2, 3),
        mesh_indices_in_group_order=(0, 1, 2, 3),
        current_group_rank=0,
    )
    inputs = _assignment_test_inputs(
        {"owned": ((6, 6), None, 1), "fused": ((6, 6), 2, 2)}, group
    )
    assert len(inputs[1][0].matrices) == 1
    assert len(inputs[1][1].matrices) == 4
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 2})
    owned, fused = FSDPDistributedMuon._assign_matrix_owners(optimizer, *inputs, {})
    assert len(owned) == 1
    assert fused == (0, 0, 2, 2)
    # Column groups are independent tasks even when they choose the same owner.
    fused_only = _assignment_test_inputs({"fused": ((6, 6), 2, 2)}, group)
    assert FSDPDistributedMuon._assign_matrix_owners(
        optimizer, *fused_only, {}
    ) == ((0, 1, 2, 3),)
    with pytest.raises(ValueError, match="does not support a batched view"):
        MuonParameterSpec(
            compute_placement=Owned(), compute_view=BatchedMatrixComputeView(2)
        )


def test_matrix_assignment_keeps_cumulative_loads_separate_by_group():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 2})
    cumulative = {}
    for ranks in ((0, 1), (0, 2)):
        group = SimpleNamespace(
            global_ranks_in_group_order=ranks,
            mesh_indices_in_group_order=(0, 1),
            current_group_rank=0,
        )
        owners = FSDPDistributedMuon._assign_matrix_owners(
            optimizer,
            *_assignment_test_inputs({"matrix": ((4, 3), None, 1)}, group),
            cumulative,
        )
        assert owners == ((0,),)
    assert cumulative[(0, 1)] == cumulative[(0, 2)]
    assert cumulative[(0, 1)] is not cumulative[(0, 2)]


def test_matrix_assignment_uses_cumulative_traffic_peak_before_rank_tie():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _MuonRankLoad,
    )

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1),
        mesh_indices_in_group_order=(0, 1),
        current_group_rank=0,
    )
    inputs = _assignment_test_inputs({"matrix": ((4, 4), None, 1)}, group)
    optimizer = SimpleNamespace(_group=lambda _item: {"ns_steps": 2})
    assert FSDPDistributedMuon._assign_matrix_owners(optimizer, *inputs, {}) == ((0,),)

    cumulative = {
        (0, 1): [
            _MuonRankLoad(receive_bytes=100),
            _MuonRankLoad(send_bytes=100),
        ]
    }
    # Both ranks hold 32 bytes. Owner 0 raises the traffic peak to 132 bytes;
    # owner 1 keeps it at 100 bytes with the same compute and transfer cost.
    owners = FSDPDistributedMuon._assign_matrix_owners(optimizer, *inputs, cumulative)
    assert owners == ((1,),)
    assert [load.send_bytes for load in cumulative[(0, 1)]] == [32, 100]
    assert [load.receive_bytes for load in cumulative[(0, 1)]] == [100, 32]


def test_transfer_span_uses_tensor_slice_offset_and_strided_columns():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _TransferSpan

    source = torch.arange(40, dtype=torch.float32)[7:31].view(4, 6)
    span = _TransferSpan(2, 1, 3, 2, 6)
    packed = torch.full((10,), -1.0)
    span.buffer_view(packed).copy_(span.tensor_view(source))
    torch.testing.assert_close(packed[2:8].view(3, 2), source[:3, 1:3])
    assert torch.all(packed[:2] == -1)
    assert torch.all(packed[8:] == -1)
    backing = torch.full((40,), -1.0)
    destination = backing[11:35].view(4, 6)
    span.tensor_view(destination).copy_(span.buffer_view(packed))
    expected = torch.full((4, 6), -1.0)
    expected[:3, 1:3] = source[:3, 1:3]
    torch.testing.assert_close(destination, expected)
    assert torch.all(backing[:11] == -1)
    assert torch.all(backing[35:] == -1)


def test_compiled_bucket_validation_hashes_all_buckets_once_per_group(monkeypatch):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    first_process_group = object()
    second_process_group = object()
    first_group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1),
        mesh_indices_in_group_order=(0, 1),
        process_group=first_process_group,
    )
    second_group = SimpleNamespace(
        global_ranks_in_group_order=(0, 2),
        mesh_indices_in_group_order=(0, 1),
        process_group=second_process_group,
    )

    def make_bucket(name, group):
        return SimpleNamespace(
            name=name,
            items=(name,),
            has_redistribution=True,
            group=group,
            device=torch.device("cpu"),
        )

    collectives = []

    def all_gather(gathered, local_hash, *, group):
        collectives.append((group, local_hash.item()))
        for value in gathered:
            value.copy_(local_hash)

    monkeypatch.setattr(dist, "all_gather", all_gather)
    optimizer = SimpleNamespace(
        _parameter_signature=lambda item, bucket: (bucket.name, item)
    )
    buckets = (
        make_bucket("second.0", second_group),
        make_bucket("first.0", first_group),
        make_bucket("second.1", second_group),
    )

    FSDPDistributedMuon._validate_compiled_buckets_across_ranks(optimizer, buckets)

    first_hashes = dict(collectives)
    assert [group for group, _hash in collectives] == [
        first_process_group,
        second_process_group,
    ]

    collectives.clear()
    changed_buckets = (*buckets[:-1], make_bucket("second.changed", second_group))
    FSDPDistributedMuon._validate_compiled_buckets_across_ranks(
        optimizer,
        changed_buckets,
    )
    changed_hashes = dict(collectives)

    assert changed_hashes[first_process_group] == first_hashes[first_process_group]
    assert changed_hashes[second_process_group] != first_hashes[second_process_group]


def test_compiled_bucket_validation_rejects_rank_mismatch(monkeypatch):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1),
        mesh_indices_in_group_order=(0, 1),
        process_group=object(),
    )
    bucket = SimpleNamespace(
        name="bucket",
        items=("parameter",),
        has_redistribution=True,
        group=group,
        device=torch.device("cpu"),
    )

    def all_gather(gathered, local_hash, *, group):
        for value in gathered:
            value.copy_(local_hash)
        gathered[-1].add_(1)

    monkeypatch.setattr(dist, "all_gather", all_gather)
    optimizer = SimpleNamespace(
        _parameter_signature=lambda item, bucket: (bucket.name, item)
    )

    with pytest.raises(RuntimeError, match="compiled Muon buckets differ"):
        FSDPDistributedMuon._validate_compiled_buckets_across_ranks(
            optimizer,
            (bucket,),
        )


def test_mixed_bucket_hash_allows_local_experts_on_different_edp_subgroups(monkeypatch):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import FSDPDistributedMuon

    process_group = object()
    group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1, 2, 3),
        mesh_indices_in_group_order=(0, 1, 2, 3),
        process_group=process_group,
    )
    dense_plan = SimpleNamespace(
        matrices=((0, 0, 0, 4, 4),),
        storage_partitions=((0, 1), (1, 1), (2, 1), (3, 1)),
        compute_partitions=((0, 4), (4, 0), (4, 0), (4, 0)),
        storage_to_compute_routes=tuple((rank, 0) for rank in range(4)),
    )

    def make_item(name, global_shape, local_shape, mesh_ranks, *, owned):
        global_tensor = torch.empty(global_shape)
        local_tensor = torch.empty(local_shape)
        return SimpleNamespace(
            fqn=name,
            param=SimpleNamespace(
                shape=global_tensor.shape,
                stride=global_tensor.stride,
                dtype=global_tensor.dtype,
                to_local=lambda: local_tensor,
            ),
            compute_placement=Owned() if owned else Shard(0),
            compute_view_key=(
                ("identity",) if owned else ("batched_matrix", 2, 4, 4, 1)
            ),
            storage_layout=SimpleNamespace(
                placements=(Shard(0),),
                global_ranks_in_mesh_order=mesh_ranks,
            ),
        )

    hashes = []

    def all_gather(gathered, local_hash, *, group):
        assert group is process_group
        hashes.append(local_hash.item())
        for value in gathered:
            value.copy_(local_hash)

    monkeypatch.setattr(dist, "all_gather", all_gather)
    optimizer = SimpleNamespace(_group_signature=lambda _item: ())
    optimizer._parameter_signature = lambda item, bucket: (
        FSDPDistributedMuon._parameter_signature(optimizer, item, bucket)
    )
    # The columns of an (edp, ep) mesh [[0, 1], [2, 3]] own different experts.
    # Both EDP groups already hold complete matrices for their local Muon work.
    for rank in range(4):
        edp_ranks = (0, 2) if rank % 2 == 0 else (1, 3)
        expert = make_item(
            "experts.weight", (8, 4), (4, 4), edp_ranks, owned=False
        )
        dense = make_item(
            "dense.weight", (4, 4), (1, 4), (0, 1, 2, 3), owned=True
        )
        bucket = SimpleNamespace(
            name="mixed",
            items=(expert, dense),
            redistributed_items=(dense,),
            redistributed_plans=(dense_plan,),
            has_redistribution=True,
            group=group,
            device=torch.device("cpu"),
        )
        FSDPDistributedMuon._validate_compiled_buckets_across_ranks(
            optimizer, (bucket,)
        )

    assert len(hashes) == 4
    assert len(set(hashes)) == 1

    dense_signature = optimizer._parameter_signature(dense, bucket)
    dense.storage_layout.global_ranks_in_mesh_order = (0, 2, 1, 3)
    assert optimizer._parameter_signature(dense, bucket) != dense_signature


def test_muon_workspace_uses_synchronous_fallback_without_device_streams():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        _MuonWorkspace,
    )
    from torchturbo.runtime.cpu import CpuRuntime

    workspace = _MuonWorkspace(
        CpuRuntime(),
        overlap_comm_compute=True,
    )
    workspace.configure(())

    assert not workspace.async_enabled
    assert workspace.slots == ()


def test_muon_async_failure_joins_caller_and_transfer_streams():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    waits = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            waits.append((self.name, other.name))

    caller = FakeStream("caller")
    transfer = FakeStream("transfer")
    workspace = SimpleNamespace(
        device_runtime=SimpleNamespace(current_stream=lambda: caller),
        transfer_stream=transfer,
        overlap_comm_compute=True,
        record_compute_stream=lambda _stream: None,
        slot_for_bucket=lambda _index: object(),
    )
    optimizer = SimpleNamespace(
        _workspace=workspace,
        _compiled_buckets=(SimpleNamespace(has_redistribution=True),),
    )

    def fail_gather(_plan, _slot):
        raise RuntimeError("gather failed")

    optimizer._enqueue_gather = fail_gather

    with pytest.raises(RuntimeError, match="gather failed"):
        FSDPDistributedMuon._run_async(optimizer)

    assert waits == [
        ("transfer", "caller"),
        ("transfer", "caller"),
        ("caller", "transfer"),
    ]


@pytest.mark.parametrize("async_enabled", [False, True])
def test_muon_failed_update_rejects_retry_before_closure(async_enabled):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    calls = []

    def fail_update():
        calls.append("update")
        raise RuntimeError("partial update failed")

    optimizer = SimpleNamespace(
        _failed=False,
        _step_scoped_workspace=False,
        _acquire_workspace=lambda: None,
        _workspace=SimpleNamespace(async_enabled=async_enabled),
        _compiled_buckets=(SimpleNamespace(has_redistribution=True),),
        _preflight_step=lambda: calls.append("preflight"),
        _run_async=fail_update,
        _run_synchronous=fail_update,
    )

    with pytest.raises(RuntimeError, match="partial update failed"):
        FSDPDistributedMuon.step(optimizer)

    assert optimizer._failed
    with pytest.raises(RuntimeError, match="cannot retry a failed update"):
        FSDPDistributedMuon.step(optimizer, lambda: calls.append("closure"))
    assert calls == ["preflight", "update"]


def test_muon_preflight_failure_can_be_corrected_before_retry():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    calls = []

    def fail_preflight():
        raise RuntimeError("missing gradients")

    optimizer = SimpleNamespace(
        _failed=False,
        _step_scoped_workspace=False,
        _acquire_workspace=lambda: None,
        _workspace=SimpleNamespace(async_enabled=False),
        _preflight_step=fail_preflight,
        _run_synchronous=lambda: calls.append("update"),
    )
    with pytest.raises(RuntimeError, match="missing gradients"):
        FSDPDistributedMuon.step(optimizer)
    assert not optimizer._failed
    assert calls == []

    optimizer._preflight_step = lambda: None
    assert FSDPDistributedMuon.step(optimizer, lambda: 1.5) == 1.5
    assert FSDPDistributedMuon.step(optimizer) is None
    assert not optimizer._failed
    assert calls == ["update", "update"]


@pytest.mark.parametrize(
    ("overlap", "kinds", "expected_gathers"),
    [
        (True, ("d0", "d1", "d2"), ["d0:slot0", "d1:slot1", "d2:slot0"]),
        (True, ("d0", "local", "d2"), ["d0:slot0", "d2:slot1"]),
        (
            True,
            ("d0", "local", "d1", "d2", "d3"),
            ["d0:slot0", "d1:slot1", "d2:slot0", "d3:slot1"],
        ),
        (
            True,
            ("d0", "d1", "local", "d2"),
            ["d0:slot0", "d1:slot1", "d2:slot0"],
        ),
        (False, ("d0", "d1", "d2"), ["d0:slot0", "d1:slot0", "d2:slot0"]),
        (False, ("d0", "local", "d2"), ["d0:slot0", "d2:slot0"]),
        (True, ("d0", "local"), ["d0:slot0"]),
        (False, ("d0", "local"), ["d0:slot0"]),
    ],
)
def test_muon_async_pipeline_matches_torchtitan_ordering(
    overlap, kinds, expected_gathers
):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    trace = []

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            trace.append(f"record:{self.name}@{stream.name}")

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            trace.append(f"wait:{self.name}<-{other.name}")

        def wait_event(self, event):
            trace.append(f"wait:{self.name}<-{event.name}")

    class NoopStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

    caller = FakeStream("caller")
    transfer = FakeStream("transfer")
    num_slots = 2 if overlap else 1
    slots = tuple(
        SimpleNamespace(
            buffers=SimpleNamespace(name=f"slot{index}"),
            input_ready=FakeEvent(f"input{index}"),
            compute_done=FakeEvent(f"compute{index}"),
            finished=FakeEvent(f"finished{index}"),
            active_bucket=None,
        )
        for index in range(num_slots)
    )
    slot_indices = {}
    owned_bucket_index = 0
    for bucket_index, name in enumerate(kinds):
        if name != "local":
            slot_indices[bucket_index] = owned_bucket_index % num_slots
            owned_bucket_index += 1
    plans = tuple(
        SimpleNamespace(
            name=name,
            has_redistribution=name != "local",
            local_items=(name,),
        )
        for name in kinds
    )
    workspace = SimpleNamespace(
        device_runtime=SimpleNamespace(
            current_stream=lambda: caller,
            stream=lambda _stream: NoopStreamContext(),
        ),
        transfer_stream=transfer,
        overlap_comm_compute=overlap,
        local_buffers=SimpleNamespace(name="local"),
        record_compute_stream=lambda stream: trace.append(
            f"record_stream:{stream.name}"
        ),
        slot_for_bucket=lambda index: slots[slot_indices[index]],
    )
    optimizer = SimpleNamespace(
        _workspace=workspace,
        _compiled_buckets=plans,
    )

    optimizer._prepare_redistributed = lambda plan, buffers: trace.append(
        f"prepare:{plan.name}:{buffers.name}"
    )
    optimizer._gather = lambda plan, buffers: trace.append(
        f"gather:{plan.name}:{buffers.name}"
    )
    optimizer._compute_local_items = lambda bucket, buffers: trace.append(
        f"local:{bucket.local_items[0]}:{buffers.name}"
    )
    optimizer._compute_redistributed = lambda plan, buffers: trace.append(
        f"owned:{plan.name}:{buffers.name}"
    )
    optimizer._scatter = lambda plan, buffers: trace.append(
        f"scatter:{plan.name}:{buffers.name}"
    )
    optimizer._finalize_redistributed = lambda plan, buffers: trace.append(
        f"finalize:{plan.name}:{buffers.name}"
    )
    optimizer._enqueue_gather = lambda plan, slot: (
        FSDPDistributedMuon._enqueue_gather(optimizer, plan, slot)
    )
    optimizer._enqueue_scatter = lambda plan, slot: (
        FSDPDistributedMuon._enqueue_scatter(optimizer, plan, slot)
    )

    FSDPDistributedMuon._run_async(optimizer)

    assert all(slot.active_bucket is None for slot in slots)
    assert [
        entry.removeprefix("gather:") for entry in trace if entry.startswith("gather:")
    ] == expected_gathers
    assert [entry.split(":")[1] for entry in trace if entry.startswith("scatter:")] == [
        name for name in kinds if name != "local"
    ]

    def latest_before(entry: str, limit: int) -> int:
        """Return the latest position of ``entry`` before ``limit``."""
        for index in range(limit - 1, -1, -1):
            if trace[index] == entry:
                return index
        raise AssertionError(f"{entry!r} never happens before position {limit}")

    # Every redistributed bucket must join its gather before compute reads the
    # received buffer, and join its compute before the return transfer reuses
    # that buffer. Dropping either dependency races on a real device but still
    # yields correct values wherever streams execute inline, so only the trace
    # can prove both dependencies exist. Slots are reused across buckets, so
    # each bucket resolves its own events by scanning back from its own work.
    for bucket_index, name in enumerate(kinds):
        if name == "local":
            continue
        slot_ordinal = slot_indices[bucket_index]
        compute_index = trace.index(f"owned:{name}:slot{slot_ordinal}")
        input_wait = latest_before(
            f"wait:caller<-input{slot_ordinal}", compute_index
        )
        latest_before(f"record:input{slot_ordinal}@transfer", input_wait)

        scatter_index = trace.index(f"scatter:{name}:slot{slot_ordinal}")
        compute_wait = latest_before(
            f"wait:transfer<-compute{slot_ordinal}", scatter_index
        )
        latest_before(f"record:compute{slot_ordinal}@caller", compute_wait)

    transfer_waits = [
        index
        for index, entry in enumerate(trace)
        if entry == "wait:transfer<-caller"
    ]
    if overlap:
        # TorchTitan establishes this dependency once. Later gathers may be
        # submitted after a local bucket without re-joining the caller stream.
        assert len(transfer_waits) == 1
        for bucket_index, name in enumerate(kinds):
            if name != "local":
                continue
            local_index = trace.index("local:local:local", transfer_waits[0])
            owned_before = [
                item for item in kinds[:bucket_index] if item != "local"
            ]
            if owned_before:
                latest_ordinal = len(owned_before) - 1
                latest_finished = f"wait:caller<-finished{latest_ordinal % 2}"
                assert trace.index(latest_finished) > local_index
            if bucket_index + 1 < len(kinds):
                next_owned = next(
                    item
                    for item in kinds[bucket_index + 1 :]
                    if item != "local"
                )
                next_prepare = next(
                    index
                    for index, entry in enumerate(trace)
                    if entry.startswith(f"prepare:{next_owned}:")
                )
                assert next_prepare > local_index

        if kinds == ("d0", "d1", "d2"):
            # d2 reuses d0's slot, so d0's return must be submitted first.
            assert trace.index("scatter:d0:slot0") < trace.index(
                "gather:d2:slot0"
            )
            # d2's gather uses a different slot from d1 and is submitted first.
            assert trace.index("gather:d2:slot0") < trace.index(
                "scatter:d1:slot1"
            )
        if kinds == ("d0", "local", "d2"):
            # The next gather is submitted after local work, but no second
            # transfer<-caller dependency delays it until local completes.
            assert trace.index("local:local:local") < trace.index(
                "prepare:d2:slot1"
            )
            assert trace.index("gather:d2:slot1") < trace.index(
                "scatter:d0:slot0"
            )
        if kinds == ("d0", "d1", "local", "d2"):
            assert trace.index("local:local:local") < trace.index(
                "prepare:d2:slot0"
            )
            assert trace.index("scatter:d0:slot0") < trace.index(
                "gather:d2:slot0"
            )
            assert trace.index("gather:d2:slot0") < trace.index(
                "scatter:d1:slot1"
            )
    if not overlap:
        # Without overlap the caller joins the gather before any bucket work,
        # so local items of a redistributed bucket also run after that join.
        input_waits = [
            index for index, entry in enumerate(trace) if entry == "wait:caller<-input0"
        ]
        local_compute = [
            index for index, entry in enumerate(trace) if entry.startswith("local:d")
        ]
        assert len(input_waits) == len([name for name in kinds if name != "local"])
        assert len(local_compute) == len(input_waits)
        for wait, compute in zip(input_waits, local_compute, strict=True):
            assert wait < compute
        if "local" in kinds:
            assert trace.index("wait:caller<-finished0") < trace.index(
                "local:local:local"
            )

    first_trace = trace.copy()
    trace.clear()
    FSDPDistributedMuon._run_async(optimizer)
    assert trace == first_trace
    assert all(slot.active_bucket is None for slot in slots)


def test_muon_synchronous_path_records_current_stream_before_using_scratch():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
    )

    operations = []
    caller = object()
    local_buffers = object()
    workspace = SimpleNamespace(
        device_runtime=SimpleNamespace(current_stream=lambda: caller),
        local_buffers=local_buffers,
        record_compute_stream=lambda stream: operations.append(("record", stream)),
    )
    optimizer = SimpleNamespace(
        _workspace=workspace,
        _compiled_buckets=(
            SimpleNamespace(has_redistribution=False, local_items=("local",)),
        ),
        _compute_local_items=lambda bucket, buffers: operations.append(
            ("compute", bucket.local_items, buffers)
        ),
    )

    FSDPDistributedMuon._run_synchronous(optimizer)

    assert operations == [
        ("record", caller),
        ("compute", ("local",), local_buffers),
    ]


class _TwoMatrixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)


def test_param_group_merges_name_and_parameter_selection_in_model_order():
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
    )

    group = ParamGroup(
        model,
        optimizer_name="Muon",
        param_names=("0.bias", "1.bias"),
        params=(model[1].weight, model[0].weight),
        optimizer_kwargs={"lr": 0.1},
    )

    assert group.params_by_fqn == {
        "0.weight": model[0].weight,
        "0.bias": model[0].bias,
        "1.weight": model[1].weight,
        "1.bias": model[1].bias,
    }
    assert group.optimizer_name == "Muon"
    assert group.optimizer_kwargs == {"lr": 0.1}
    assert not hasattr(group, "pattern")
    assert not hasattr(group, "param_names")
    assert not hasattr(group, "params")

    optimizer_group = group.get_optimizer_param_group()
    assert set(optimizer_group) == {"params", "lr"}
    assert optimizer_group["params"] == tuple(group.params_by_fqn.values())
    assert optimizer_group["lr"] == 0.1


def test_param_group_rejects_missing_or_frozen_parameters():
    model = nn.Linear(2, 2)
    model.bias.requires_grad_(False)

    with pytest.raises(ValueError, match="is not in the model"):
        ParamGroup(
            model,
            optimizer_name="AdamW",
            param_names=("missing",),
        )
    with pytest.raises(ValueError, match="does not require gradients"):
        ParamGroup(
            model,
            optimizer_name="AdamW",
            params=(model.bias,),
        )
    with pytest.raises(ValueError, match="must belong to the prepared model"):
        ParamGroup(
            model,
            optimizer_name="AdamW",
            params=(nn.Parameter(torch.empty(2, 2)),),
        )


def test_param_group_requires_selection_and_reserves_torch_keys():
    model = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="requires param_names"):
        ParamGroup(model, optimizer_name="AdamW")
    with pytest.raises(ValueError, match="reserved keys"):
        ParamGroup(
            model,
            optimizer_name="AdamW",
            param_names=("weight",),
            optimizer_kwargs={"params": ()},
        )


class _TestOptimizerPlan(OptimizerPlan):
    __slots__ = ("_param_groups", "_param_groups_by_optimizer_name")

    def __init__(self, model, param_groups):
        self._param_groups = tuple(param_groups)
        self._validate_param_groups(model, self._param_groups)
        self._param_groups_by_optimizer_name = (
            self._group_param_groups_by_optimizer_name(self._param_groups)
        )

    def get_param_groups(self):
        return self._param_groups

    def get_param_groups_by_optimizer_name(self):
        return self._param_groups_by_optimizer_name


def test_optimizer_plan_groups_resolved_param_groups_by_optimizer_name():
    model = _TwoMatrixModel()
    first_group = ParamGroup(
        model,
        param_names=("first.weight",),
        optimizer_name="First",
        optimizer_kwargs={"lr": 0.2},
    )
    second_group = ParamGroup(
        model,
        param_names=("second.weight",),
        optimizer_name="Second",
        optimizer_kwargs={"lr": 0.1},
    )
    plan = _TestOptimizerPlan(
        model,
        (first_group, second_group),
    )

    groups_by_name = plan.get_param_groups_by_optimizer_name()
    assert plan.get_param_groups() == (first_group, second_group)
    assert tuple(groups_by_name) == ("First", "Second")
    assert groups_by_name["First"] == (first_group,)
    assert groups_by_name["Second"] == (second_group,)


def test_optimizer_plan_rejects_parameters_claimed_by_multiple_groups():
    model = _TwoMatrixModel()
    first_group = ParamGroup(
        model,
        param_names=("first.weight",),
        optimizer_name="First",
    )
    overlapping_group = ParamGroup(
        model,
        param_names=("first.weight", "second.weight"),
        optimizer_name="Second",
    )

    with pytest.raises(ValueError, match="assigned to multiple parameter groups"):
        _TestOptimizerPlan(
            model,
            (first_group, overlapping_group),
        )


def test_optimizer_plan_requires_complete_trainable_parameter_coverage():
    model = _TwoMatrixModel()
    group = ParamGroup(
        model,
        param_names=("first.weight",),
        optimizer_name="SGD",
    )

    with pytest.raises(ValueError, match="did not claim every trainable parameter"):
        _TestOptimizerPlan(model, (group,))


def test_optimizer_plan_requires_groups_from_the_prepared_model():
    model = _TwoMatrixModel()
    other_model = _TwoMatrixModel()
    group = ParamGroup(
        other_model,
        param_names=("first.weight", "second.weight"),
        optimizer_name="SGD",
    )

    with pytest.raises(ValueError, match="prepared model by identity"):
        _TestOptimizerPlan(model, (group,))


def _prepare_test_optimizers(plan, factories):
    return {
        name: factories[name]([group.get_optimizer_param_group() for group in groups])
        for name, groups in plan.get_param_groups_by_optimizer_name().items()
    }


def test_optimizers_container_shares_param_groups_with_scheduler():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.2},
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1},
            ),
        ],
    )
    container = OptimizersContainer(
        plan,
        _prepare_test_optimizers(
            plan,
            {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW},
        ),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(container, step_size=1, gamma=0.5)

    container.step()
    scheduler.step()

    assert container.param_groups[0] is container.optimizers[0].param_groups[0]
    assert container.param_groups[1] is container.optimizers[1].param_groups[0]
    assert [group["lr"] for group in container.param_groups] == [0.1, 0.05]


def _two_optimizer_container(*, sgd_kwargs, adamw_kwargs):
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.2},
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.2},
            ),
        ],
    )
    sgd_group, adamw_group = plan.get_param_groups()
    return OptimizersContainer(
        plan,
        {
            "SGD": torch.optim.SGD(
                [sgd_group.get_optimizer_param_group()], **sgd_kwargs
            ),
            "AdamW": torch.optim.AdamW(
                [adamw_group.get_optimizer_param_group()], **adamw_kwargs
            ),
        },
    )


def test_optimizers_container_defaults_keep_only_options_children_agree_on():
    container = _two_optimizer_container(
        sgd_kwargs={"lr": 0.3, "weight_decay": 0.01, "momentum": 0.9},
        adamw_kwargs={"lr": 0.3, "weight_decay": 0.01, "eps": 1e-6},
    )

    assert container.defaults["lr"] == 0.3
    assert container.defaults["weight_decay"] == 0.01
    # Only one child defines these.
    assert "momentum" not in container.defaults
    assert "betas" not in container.defaults


def test_optimizers_container_defaults_drop_options_children_disagree_on():
    container = _two_optimizer_container(
        sgd_kwargs={"lr": 0.3, "weight_decay": 0.01},
        adamw_kwargs={"lr": 0.3, "weight_decay": 0.02},
    )

    assert container.defaults["lr"] == 0.3
    assert "weight_decay" not in container.defaults


@pytest.mark.parametrize("scheduler_name", ["polynomial", "cosine_with_min_lr"])
def test_optimizers_container_supports_schedulers_reading_defaults(scheduler_name):
    from transformers.optimization import (
        get_cosine_with_min_lr_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
    )

    container = _two_optimizer_container(
        sgd_kwargs={"lr": 0.2},
        adamw_kwargs={"lr": 0.2},
    )

    if scheduler_name == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            container,
            1,
            10,
            lr_end=0.03,
        )
    else:
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            container, 1, 10, min_lr=0.03
        )

    assert scheduler.base_lrs == [0.2, 0.2]
    for _ in range(10):
        container.step()
        scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([0.03, 0.03])


def test_optimizers_container_rejects_new_param_groups_without_mutation():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight", "second.weight"),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.1},
            )
        ],
    )
    container = OptimizersContainer(
        plan,
        _prepare_test_optimizers(plan, {"SGD": torch.optim.SGD}),
    )
    child = container.optimizers[0]
    facade_groups = tuple(container.param_groups)
    child_groups = tuple(child.param_groups)

    with pytest.raises(RuntimeError, match=r"does not support add_param_group\(\)"):
        container.add_param_group({"params": [nn.Parameter(torch.ones(2, 2))]})

    assert len(container.param_groups) == len(facade_groups)
    assert all(
        actual is expected
        for actual, expected in zip(container.param_groups, facade_groups, strict=True)
    )
    assert len(child.param_groups) == len(child_groups)
    assert all(
        actual is expected
        for actual, expected in zip(child.param_groups, child_groups, strict=True)
    )


def test_optimizers_container_rejects_inconsistent_plan_group_views():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="SGD",
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="SGD",
            ),
        ],
    )
    plan._param_groups_by_optimizer_name = {
        "SGD": tuple(reversed(plan.get_param_groups()))
    }

    with pytest.raises(ValueError, match="group views are inconsistent"):
        OptimizersContainer(plan, {})


def test_optimizers_container_rejects_changed_param_group_options():
    model = nn.Linear(2, 2)
    plan = OptimizerClassPlan(
        model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0.2},
    )
    groups = [
        group.get_optimizer_param_group()
        for group in plan.get_param_groups_by_optimizer_name()["AdamW"]
    ]
    groups[1]["weight_decay"] = 0.2
    optimizer = torch.optim.AdamW(groups, **plan.optimizer_kwargs)

    with pytest.raises(ValueError, match="changed parameter-group option"):
        OptimizersContainer(plan, {"AdamW": optimizer})


def test_optimizers_container_state_dict_round_trip():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.2, "momentum": 0.9},
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1},
            ),
        ],
    )
    container = OptimizersContainer(
        plan,
        _prepare_test_optimizers(
            plan,
            {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW},
        ),
    )

    for param in model.parameters():
        param.grad = torch.ones_like(param)
    container.step()
    saved = copy.deepcopy(container.state_dict())

    assert saved["format_version"] == 1
    assert tuple(saved["optimizers"]) == ("SGD", "AdamW")

    for optimizer in container.optimizers:
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    container.load_state_dict(saved)

    live_groups = [
        group for optimizer in container.optimizers for group in optimizer.param_groups
    ]
    assert all(
        facade is child
        for facade, child in zip(container.param_groups, live_groups, strict=True)
    )
    assert [group["lr"] for group in container.param_groups] == [0.2, 0.1]
    assert all(optimizer.state for optimizer in container.optimizers)


def test_optimizers_container_rejects_state_for_different_children():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.2},
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1},
            ),
        ],
    )
    container = OptimizersContainer(
        plan,
        _prepare_test_optimizers(
            plan,
            {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW},
        ),
    )
    state_dict = container.state_dict()
    state_dict["optimizers"] = {
        "AdamW": state_dict["optimizers"]["AdamW"],
        "SGD": state_dict["optimizers"]["SGD"],
    }

    with pytest.raises(ValueError, match="state names do not match"):
        container.load_state_dict(state_dict)


def test_optimizers_container_refresh_param_groups_keeps_scheduler_effective():
    """A child load replaces its group dicts, orphaning the facade.

    Checkpoint restore loads into the children, so without a refresh an LR
    scheduler would keep writing learning rates the children never read.
    """
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.2},
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 0.1},
            ),
        ],
    )
    container = OptimizersContainer(
        plan,
        _prepare_test_optimizers(
            plan,
            {"SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW},
        ),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(container, step_size=1, gamma=0.5)
    container.step()

    def facade_is_live():
        live = [
            group for optimizer in container.optimizers for group in optimizer.param_groups
        ]
        return len(live) == len(container.param_groups) and all(
            facade is child
            for facade, child in zip(container.param_groups, live, strict=True)
        )

    assert facade_is_live()

    # This is what a checkpoint restore does to every child optimizer.
    for optimizer in container.optimizers:
        optimizer.load_state_dict(optimizer.state_dict())
    assert not facade_is_live()

    container.refresh_param_groups()
    assert facade_is_live()

    scheduler.step()
    assert [group["lr"] for group in container.param_groups] == [
        group["lr"]
        for optimizer in container.optimizers
        for group in optimizer.param_groups
    ]


def test_optimizers_container_rejects_backend_name_mismatch():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight", "second.weight"),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.1},
            )
        ],
    )
    groups = [
        group.get_optimizer_param_group()
        for group in plan.get_param_groups_by_optimizer_name()["SGD"]
    ]

    with pytest.raises(ValueError, match="names do not match"):
        OptimizersContainer(
            plan,
            {"AdamW": torch.optim.SGD(groups, lr=0.1)},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "exactly the assigned parameters"),
        ("duplicate", "duplicate parameters"),
        ("foreign", "outside the optimizer plan"),
    ],
)
def test_optimizers_container_validates_parameter_identity(mutation, message):
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight", "second.weight"),
                optimizer_name="SGD",
                optimizer_kwargs={"lr": 0.1},
            )
        ],
    )
    groups = [
        group.get_optimizer_param_group()
        for group in plan.get_param_groups_by_optimizer_name()["SGD"]
    ]
    optimizer = torch.optim.SGD(groups, lr=0.1)
    params = optimizer.param_groups[0]["params"]
    if mutation == "missing":
        params.pop()
    elif mutation == "duplicate":
        params.append(params[0])
    else:
        params[0] = nn.Parameter(params[0].detach().clone())

    with pytest.raises(ValueError, match=message):
        OptimizersContainer(plan, {"SGD": optimizer})


def test_optimizers_container_rejects_parameters_swapped_between_optimizers():
    model = _TwoMatrixModel()
    plan = _TestOptimizerPlan(
        model,
        [
            ParamGroup(
                model,
                param_names=("first.weight",),
                optimizer_name="First",
                optimizer_kwargs={"lr": 0.1},
            ),
            ParamGroup(
                model,
                param_names=("second.weight",),
                optimizer_name="Second",
                optimizer_kwargs={"lr": 0.1},
            ),
        ],
    )
    groups = plan.get_param_groups_by_optimizer_name()
    optimizers_by_name = {
        "First": torch.optim.SGD(
            [group.get_optimizer_param_group() for group in groups["Second"]]
        ),
        "Second": torch.optim.SGD(
            [group.get_optimizer_param_group() for group in groups["First"]]
        ),
    }

    with pytest.raises(ValueError, match="assigned to a different optimizer"):
        OptimizersContainer(plan, optimizers_by_name)


class _Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.q_a_proj = nn.Linear(hidden_size, 3, bias=False)
        self.q_b_proj = nn.Linear(3, num_heads * 2, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, 3, bias=False)
        self.kv_b_proj = nn.Linear(3, num_heads * 2, bias=False)
        self.o_proj = nn.Linear(num_heads * 2, hidden_size, bias=False)


class _DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)


class _MoEMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, num_experts: int):
        super().__init__()
        self.experts = GroupedMLP(
            num_local_experts=num_experts,
            use_te=False,
            config=MoEConfig(
                hidden_size=hidden_size,
                moe_ffn_hidden_size=ffn_size,
                gated_linear_unit=True,
            ),
        )
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.shared_experts = _DenseMLP(hidden_size, ffn_size)


class _Layer(nn.Module):
    def __init__(self, hidden_size, ffn_size, num_heads, num_experts=None):
        super().__init__()
        self.self_attn = _Attention(hidden_size, num_heads)
        self.mlp = (
            _DenseMLP(hidden_size, ffn_size)
            if num_experts is None
            else _MoEMLP(hidden_size, ffn_size, num_experts)
        )
        self.input_layernorm = nn.LayerNorm(hidden_size)


class _KimiLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, num_heads: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                _Layer(hidden_size, ffn_size, num_heads),
                _Layer(hidden_size, ffn_size, num_heads, num_experts=2),
                _Layer(hidden_size, ffn_size, num_heads, num_experts=2),
            ]
        )
        self.model.embed_tokens = nn.Embedding(8, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 8, bias=False)


def _kimi_recipe_model():
    """Stand-in Kimi K2.5 model that reports output embeddings like Transformers."""
    from transformers import PretrainedConfig, PreTrainedModel

    class Model(PreTrainedModel):
        def __init__(self):
            super().__init__(PretrainedConfig())
            hidden_size = 4
            ffn_size = 3
            num_heads = 2
            self.language_model = _KimiLanguageModel(
                hidden_size, ffn_size, num_heads
            )
            self.vision_tower = nn.Linear(hidden_size, hidden_size, bias=False)
            self.mm_projector = nn.Linear(hidden_size, hidden_size, bias=False)
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(
                    num_attention_heads=num_heads,
                    hidden_size=hidden_size,
                    moe_intermediate_size=ffn_size,
                    first_k_dense_replace=1,
                    q_lora_rank=hidden_size,
                    qk_nope_head_dim=1,
                    qk_rope_head_dim=1,
                    v_head_dim=1,
                ),
                vision_config=SimpleNamespace(mm_projector_type="patchmerger"),
            )

        def get_output_embeddings(self):
            return self.language_model.lm_head

    return Model()


def _wrap_with_lora(base_layer: nn.Linear) -> nn.Module:
    tuners_utils = pytest.importorskip("peft.tuners.tuners_utils")

    class LoRAWrappedLinear(nn.Module, tuners_utils.BaseTunerLayer):
        def __init__(self, wrapped: nn.Linear):
            super().__init__()
            self.base_layer = wrapped
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(wrapped.in_features, 2, bias=False)}
            )
            self.lora_B = nn.ModuleDict(
                {"default": nn.Linear(2, wrapped.out_features, bias=False)}
            )

    return LoRAWrappedLinear(base_layer)


def _wrap_with_modules_to_save(
    original_module: nn.Linear,
    adapter_name: str,
) -> nn.Module:
    other = pytest.importorskip("peft.utils.other")
    wrapper = other.ModulesToSaveWrapper(original_module, adapter_name)
    wrapper.modules_to_save[adapter_name].requires_grad_(True)
    return wrapper


def _wrap_with_peft_prefix(model: nn.Module) -> nn.Module:
    peft = pytest.importorskip("peft")
    tuners_utils = pytest.importorskip("peft.tuners.tuners_utils")

    class Tuner(tuners_utils.BaseTuner):
        def _create_and_replace(self, *args, **kwargs):
            raise AssertionError("not used")

    tuner = Tuner.__new__(Tuner)
    nn.Module.__init__(tuner)
    tuner.model = model

    wrapped = peft.PeftModel.__new__(peft.PeftModel)
    nn.Module.__init__(wrapped)
    wrapped.base_model = tuner
    return wrapped


@pytest.mark.parametrize("gated", [False, True])
def test_grouped_mlp_annotates_materialized_expert_shapes(gated):
    model = GroupedMLP(
        num_local_experts=4,
        use_te=False,
        config=MoEConfig(
            hidden_size=4,
            moe_ffn_hidden_size=3,
            gated_linear_unit=gated,
        ),
    )
    column_groups = 2 if gated else 1
    model.weight1 = nn.Parameter(torch.empty(2 * 4, column_groups * 3))
    model.weight2 = nn.Parameter(torch.empty(2 * 3, 4))

    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert isinstance(model, MuonSpecProvider)
    assert plan.muon_fqns == ("weight1", "weight2")
    assert plan.parameter_specs["weight1"] == MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=ExpertUpComputeView(num_experts=2, fused_gate_up=gated),
    )
    assert plan.parameter_specs["weight2"] == MuonParameterSpec(
        compute_placement=Shard(0),
        compute_view=ExpertDownComputeView(num_experts=2),
    )


def test_grouped_mlp_annotation_runs_after_parameter_replacement():
    model = GroupedMLP(
        num_local_experts=2,
        use_te=False,
        config=MoEConfig(
            hidden_size=4,
            moe_ffn_hidden_size=3,
            gated_linear_unit=True,
        ),
    )
    old_weight = model.weight1
    state_dict = {
        name: torch.empty_like(parameter)
        for name, parameter in model.named_parameters()
    }

    model.load_state_dict(state_dict, assign=True)
    specs = collect_module_muon_specs(model)

    assert model.weight1 is not old_weight
    assert isinstance(
        specs["weight1"].compute_view,
        ExpertUpComputeView,
    )


def _get_kimi_recipe(model):
    return _load_kimi_recipe_module().get_muon_recipe(model)


def test_kimi_recipe_declares_qk_projections_without_attention_implementation():
    model = _kimi_recipe_model()
    recipe = _get_kimi_recipe(model)

    config = MuonOptimizerConfig(lr=0.1, weight_decay=0.02)
    assert _muon_plan(model, recipe, config).qk_clip_plan is None

    config.qk_clip = QKClipConfig()
    plan = _muon_plan(model, recipe, config)
    assert len(plan.qk_clip_plan.layers) == len(model.language_model.model.layers)


def test_kimi_recipe_annotates_only_special_parameters():
    model = _kimi_recipe_model()
    recipe = _get_kimi_recipe(model)
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    q_b = plan.parameter_specs[
        "language_model.model.layers.0.self_attn.q_b_proj.weight"
    ]
    assert isinstance(q_b.compute_view, AttentionPerHeadComputeView)
    assert q_b.compute_placement == Shard(0)

    weight1 = plan.parameter_specs[
        "language_model.model.layers.1.mlp.experts.weight1"
    ]
    assert weight1.compute_view == ExpertUpComputeView(
        num_experts=2, fused_gate_up=True
    )
    assert weight1.compute_view.resolve(torch.Size((8, 6))).matrix_shape == (4, 3)

    router = plan.parameter_specs["language_model.model.layers.1.mlp.router.weight"]
    assert router.compute_placement == Owned()

    assert not any("layernorm" in fqn for fqn in plan.parameter_specs)
    assert set(plan.adamw_decay_fqns) == {
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "vision_tower.weight",
        "mm_projector.weight",
    }
    assert tuple(
        fqn for bucket in plan.buckets for fqn in bucket.parameter_fqns
    ) == plan.muon_fqns
    assert all(len(bucket.parameter_fqns) == 1 for bucket in plan.buckets)
    assert isinstance(recipe, MuonRecipe)

    assert recipe.match(
        "language_model.model.layers.1.mlp.router.weight"
    ) is None
    assert recipe.parameter_overrides["vision_tower.**"] == ExcludeFromMuon()
    assert recipe.parameter_overrides["mm_projector.**"] == ExcludeFromMuon()
    # The fixture sets q_lora_rank, so only the q_b_proj pattern is declared.
    assert tuple(recipe.parameter_overrides) == (
        "language_model.model.layers.*.self_attn.q_b_proj.weight",
        "language_model.model.layers.*.self_attn.kv_b_proj.weight",
        "vision_tower.**",
        "mm_projector.**",
    )


def _load_kimi_recipe_module():
    recipe_path = (
        Path(__file__).parents[2]
        / "src"
        / "torchturbo"
        / "models"
        / "kimi_k25"
        / "optimizer_recipe.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "torchturbo_kimi_optimizer_recipe_lora", recipe_path
    )
    assert module_spec is not None and module_spec.loader is not None
    recipe_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(recipe_module)
    return recipe_module


def test_kimi_recipe_declares_only_the_configured_query_projection():
    model = _kimi_recipe_model()
    model.config.text_config.q_lora_rank = None

    patterns = tuple(_get_kimi_recipe(model).parameter_overrides)

    assert "language_model.model.layers.*.self_attn.q_proj.weight" in patterns
    assert "language_model.model.layers.*.self_attn.q_b_proj.weight" not in patterns


def test_kimi_recipe_omits_a_parameterless_projector():
    model = _kimi_recipe_model()
    model.config.vision_config.mm_projector_type = "identity"

    assert "mm_projector.**" not in _get_kimi_recipe(model).parameter_overrides


def test_kimi_recipe_patterns_all_match_the_prepared_model(monkeypatch):
    from torchturbo.engine.optimizers import muon as muon_module

    warnings: list[str] = []
    monkeypatch.setattr(
        muon_module,
        "logger",
        SimpleNamespace(
            warning_rank0=lambda message, *args: warnings.append(message % args)
        ),
    )
    model = _kimi_recipe_model()

    MuonPlan(
        model,
        recipe=_get_kimi_recipe(model),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert warnings == []


def test_kimi_recipe_treats_standard_lora_as_adamw_fallback():
    model = _kimi_recipe_model()
    model.requires_grad_(False)
    projection = model.language_model.model.layers[0].self_attn.q_a_proj
    model.language_model.model.layers[0].self_attn.q_a_proj = _wrap_with_lora(
        projection
    )

    recipe = _get_kimi_recipe(model)
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    adapter_fqns = (
        "language_model.model.layers.0.self_attn.q_a_proj.lora_A.default.weight",
        "language_model.model.layers.0.self_attn.q_a_proj.lora_B.default.weight",
    )
    assert plan.muon_fqns == ()
    assert plan.adamw_decay_fqns == adapter_fqns
    # The engine rule keeps adapters off Muon, so the recipe says nothing here.
    assert not any(fqn in recipe.parameter_overrides for fqn in adapter_fqns)


def test_muon_default_rule_keeps_trainable_token_delta_out_of_muon():
    other = pytest.importorskip("peft.utils.other")

    model = nn.Module()
    model.embed_tokens = other.TrainableTokensWrapper(
        nn.Embedding(8, 4),
        "default",
        token_indices=[0, 1],
    )

    plan = MuonPlan(
        model,
        recipe=MuonRecipe({}),
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    delta_fqn = (
        "embed_tokens.token_adapter.trainable_tokens_delta.default"
    )
    assert plan.muon_fqns == ()
    assert delta_fqn in plan.adamw_decay_fqns


def test_kimi_recipe_resolves_trainable_peft_base_layer_names():
    model = _kimi_recipe_model()
    model.requires_grad_(False)
    projection = model.language_model.model.layers[0].self_attn.q_b_proj
    wrapped = _wrap_with_lora(projection)
    wrapped.base_layer.weight.requires_grad_(True)
    model.language_model.model.layers[0].self_attn.q_b_proj = wrapped

    recipe = _get_kimi_recipe(model)
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    assert plan.muon_fqns == (
        "language_model.model.layers.0.self_attn.q_b_proj.base_layer.weight",
    )
    spec = plan.parameter_specs[plan.muon_fqns[0]]
    assert isinstance(spec.compute_view, AttentionPerHeadComputeView)


def test_kimi_recipe_matches_modules_to_save_with_any_adapter_name():
    model = _kimi_recipe_model()
    model.requires_grad_(False)
    projection = model.language_model.model.layers[0].self_attn.q_b_proj
    wrapped = _wrap_with_modules_to_save(projection, "custom_adapter")
    model.language_model.model.layers[0].self_attn.q_b_proj = wrapped

    recipe = _get_kimi_recipe(model)
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    fqn = (
        "language_model.model.layers.0.self_attn.q_b_proj."
        "modules_to_save.custom_adapter.weight"
    )
    assert plan.muon_fqns == (fqn,)
    spec = plan.parameter_specs[fqn]
    assert isinstance(spec.compute_view, AttentionPerHeadComputeView)
    assert spec.compute_placement == Shard(0)


def test_kimi_recipe_reaches_parameters_through_outer_peft_prefix():
    model = _kimi_recipe_model()
    prepared_model = _wrap_with_peft_prefix(model)

    # The recipe remains in the model's namespace while the resolved plan uses
    # the exact parameter name from the prepared tree.
    recipe = _get_kimi_recipe(model)
    plan = MuonPlan(
        prepared_model,
        recipe=recipe,
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    fqn = (
        "base_model.model.language_model.model.layers.0."
        "self_attn.q_b_proj.weight"
    )
    assert isinstance(
        plan.parameter_specs[fqn].compute_view,
        AttentionPerHeadComputeView,
    )


def test_kimi_recipe_leaves_unknown_trainable_matrix_on_default_path():
    model = _kimi_recipe_model()
    model.language_model.model.layers[0].unknown_matrix = nn.Parameter(
        torch.empty(2, 2)
    )

    recipe = _get_kimi_recipe(model)
    plan = MuonPlan(
        model,
        recipe=recipe,
        config=MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    fqn = "language_model.model.layers.0.unknown_matrix"
    assert plan.parameter_specs[fqn] == MuonParameterSpec(compute_placement=Owned())
    assert fqn not in recipe.parameter_overrides


def test_fsdp2_muon_pure_fallback_builds_fixed_adamw_only():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = SimpleNamespace(device=torch.device("cpu"))
    backend.fsdp_config = SimpleNamespace(cpu_offload=CPUOffloadConfig())
    backend._sp_size = 1
    backend._mesh = {"edp": object()}
    model = nn.Linear(2, 2, bias=True)
    recipe = MuonRecipe({"weight": ExcludeFromMuon()})
    plan = _muon_plan(
        model,
        recipe,
        MuonOptimizerConfig(
            lr=0.1,
            weight_decay=0.02,
            adamw_betas=(0.8, 0.95),
            adamw_eps=1e-6,
            adamw_fused=False,
        ),
    )

    optimizer = backend.prepare_optimizer(plan)

    assert type(optimizer) is torch.optim.AdamW
    adamw_groups = optimizer.param_groups
    assert len(adamw_groups) == 2
    assert sorted(group["weight_decay"] for group in adamw_groups) == [0.0, 0.02]
    assert all(group["betas"] == (0.8, 0.95) for group in adamw_groups)
    assert all(group["eps"] == 1e-6 for group in adamw_groups)
    assert all(group["fused"] is False for group in adamw_groups)
    # Configured values must reach the defaults, not only the groups.
    assert optimizer.defaults["lr"] == 0.1
    assert optimizer.defaults["betas"] == (0.8, 0.95)


def test_fsdp2_builds_optimizer_class_plan_and_preserves_parameters():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = SimpleNamespace(device=torch.device("cpu"))
    backend.fsdp_config = SimpleNamespace(cpu_offload=CPUOffloadConfig())
    backend._mesh = {"edp": object()}
    model = nn.Linear(2, 2, bias=True)
    plan = OptimizerClassPlan(
        model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0.02},
    )

    optimizer = backend.prepare_optimizer(plan)

    assert type(optimizer) is torch.optim.AdamW
    adamw_groups = optimizer.param_groups
    assert [len(group["params"]) for group in adamw_groups] == [1, 1]
    assert adamw_groups[0]["params"][0] is model.weight
    assert adamw_groups[1]["params"][0] is model.bias
    assert all("param_names" not in group for group in adamw_groups)
    assert [group["weight_decay"] for group in adamw_groups] == [0.02, 0.0]


def test_fsdp2_optimizer_class_plan_swaps_in_adamw_swap_for_optimizer_offload(
    monkeypatch,
):
    import torchturbo.engine.backends.fsdp2.optimizers.adamw as adamw_module
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode

    # AdamWSwap pins host memory on construction, which is unavailable on some
    # platforms. Record the constructor inputs instead of building the real one.
    recorded = {}

    class RecordingAdamWSwap(torch.optim.AdamW):
        def __init__(self, param_groups, **kwargs):
            recorded["kwargs"] = kwargs
            super().__init__(
                param_groups,
                lr=kwargs["lr"],
                weight_decay=kwargs["weight_decay"],
            )

    monkeypatch.setattr(adamw_module, "AdamWSwap", RecordingAdamWSwap)

    device_runtime = SimpleNamespace(device=torch.device("cpu"))
    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = device_runtime
    backend.fsdp_config = SimpleNamespace(
        cpu_offload=CPUOffloadConfig(
            enabled=True,
            mode=CPUOffloadMode.OPTIMIZER,
            optimizer_memory_fraction=0.5,
        )
    )
    backend._mesh = {"edp": object()}
    model = nn.Linear(2, 2, bias=True)
    plan = OptimizerClassPlan(
        model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0.02, "fused": True},
    )

    optimizer = backend.prepare_optimizer(plan)

    assert type(optimizer) is RecordingAdamWSwap
    # AdamWSwap hardcodes fused=True and rejects an explicit fused kwarg.
    assert "fused" not in recorded["kwargs"]
    assert recorded["kwargs"]["device_runtime"] is device_runtime
    assert recorded["kwargs"]["optimizer_memory_fraction"] == 0.5


def test_fsdp2_optimizer_class_plan_rejects_non_adamw_for_optimizer_offload():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.backends.fsdp2.config import CPUOffloadMode

    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = SimpleNamespace(device=torch.device("cpu"))
    backend.fsdp_config = SimpleNamespace(
        cpu_offload=CPUOffloadConfig(enabled=True, mode=CPUOffloadMode.OPTIMIZER)
    )
    backend._mesh = {"edp": object()}
    model = nn.Linear(2, 2, bias=True)
    plan = OptimizerClassPlan(
        model,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.1, "weight_decay": 0.02},
    )

    with pytest.raises(ValueError, match="only supports AdamW"):
        backend.prepare_optimizer(plan)


@pytest.mark.parametrize("bias", [False, True])
def test_fsdp2_muon_allows_sp_and_returns_final_optimizer(monkeypatch, bias):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
    from torchturbo.engine.backends.fsdp2.optimizers import FSDPDistributedMuon

    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = SimpleNamespace(device=torch.device("cuda"))
    backend.fsdp_config = SimpleNamespace(cpu_offload=CPUOffloadConfig())
    backend._sp_size = 8
    backend._mesh = {"edp": object()}
    model = nn.Linear(2, 2, bias=bias)
    recipe = MuonRecipe({})

    def fake_muon_init(self, plan, **_kwargs):
        groups = tuple(
            group.get_optimizer_param_group()
            for group in plan.get_param_groups_by_optimizer_name()["Muon"]
        )
        self._param_groups_frozen = False
        torch.optim.Optimizer.__init__(self, groups, {})
        self._param_groups_frozen = True

    monkeypatch.setattr(FSDPDistributedMuon, "__init__", fake_muon_init)
    plan = _muon_plan(
        model,
        recipe,
        MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )

    optimizer = backend.prepare_optimizer(plan)

    if bias:
        assert type(optimizer) is OptimizersContainer
        assert tuple(optimizer.optimizers_by_name) == ("Muon", "AdamW")
        assert type(optimizer.optimizers_by_name["Muon"]) is FSDPDistributedMuon
        assert type(optimizer.optimizers_by_name["AdamW"]) is torch.optim.AdamW
    else:
        assert type(optimizer) is FSDPDistributedMuon


def test_fsdp2_muon_requires_pytorch_2_10(monkeypatch):
    import torchturbo.engine.backends.fsdp2.optimizers.muon as muon_module
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = SimpleNamespace(device=torch.device("cuda"))
    backend.fsdp_config = SimpleNamespace(cpu_offload=CPUOffloadConfig())
    backend._mesh = {"edp": object()}
    model = nn.Linear(2, 2, bias=False)
    plan = _muon_plan(
        model,
        MuonRecipe({}),
        MuonOptimizerConfig(lr=0.1, weight_decay=0.02),
    )
    monkeypatch.setattr(muon_module, "torch_version", lambda: (2, 9, 1))

    with pytest.raises(RuntimeError, match="requires PyTorch 2.10 or newer"):
        backend.prepare_optimizer(plan)


def test_optimizer_class_plan_rejects_non_class():
    model = nn.Linear(2, 2)

    with pytest.raises(TypeError, match="optimizer_cls must be a class"):
        OptimizerClassPlan(
            model=model,
            optimizer_cls=object(),
            optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0},
        )


def test_optimizer_class_plan_rejects_non_optimizer_class():
    with pytest.raises(TypeError, match="torch.optim.Optimizer subclass"):
        OptimizerClassPlan(
            model=nn.Linear(2, 2),
            optimizer_cls=nn.Linear,
            optimizer_kwargs={"lr": 0.1, "weight_decay": 0.0},
        )


def test_fsdp2_rejects_unsupported_optimizer_plan():
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = object.__new__(FSDP2Backend)

    class UnknownPlan(OptimizerPlan):
        def get_param_groups(self):
            return ()

        def get_param_groups_by_optimizer_name(self):
            return {}

    with pytest.raises(TypeError, match="does not support optimizer plan"):
        backend.prepare_optimizer(UnknownPlan())


def _workspace_test_bucket(name, *, owned_shapes=(), local_tensors=()):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _CompiledMuonBucket

    transfer_group = SimpleNamespace(
        global_ranks_in_group_order=(0, 1),
        mesh_indices_in_group_order=(0, 1),
        current_group_rank=0,
        process_group=object(),
    )
    local_group = SimpleNamespace(
        global_ranks_in_group_order=(0,),
        mesh_indices_in_group_order=(0,),
        current_group_rank=0,
        process_group=object(),
    )
    entries = tuple(
        _bucket_test_parameter(
            f"{name}.owned{index}",
            torch.empty(shape),
            transfer_group,
            owner_group_rank=0,
        )
        for index, shape in enumerate(owned_shapes)
    ) + tuple(
        _bucket_test_parameter(
            f"{name}.local{index}", tensor, local_group, batch_size=1
        )
        for index, tensor in enumerate(local_tensors)
    )
    return _CompiledMuonBucket.compile(
        name=name,
        items=tuple(item for item, _plan in entries),
        parameter_plans=tuple(plan for _item, plan in entries),
        parameter_groups=(transfer_group,) * len(owned_shapes)
        + (local_group,) * len(local_tensors),
    )


def _workspace_test_buckets():
    return (
        _workspace_test_bucket("first", owned_shapes=((2, 3), (2, 4))),
        _workspace_test_bucket(
            "local",
            local_tensors=(
                torch.empty(2, 4),
                torch.empty(3, 7, dtype=torch.float64),
            ),
        ),
        _workspace_test_bucket("second", owned_shapes=((3, 4),)),
        _workspace_test_bucket(
            "mixed",
            owned_shapes=((4, 5),),
            local_tensors=(
                torch.empty(5, 6),
                torch.empty(5, 7, dtype=torch.float64),
            ),
        ),
    )


def test_muon_workspace_configure_cpu_reserves_one_slot_for_all_owned_buckets():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _MuonWorkspace
    from torchturbo.runtime.cpu import CpuRuntime

    workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=True)
    buckets = _workspace_test_buckets()

    workspace.configure(buckets)

    assert workspace.slot_index_by_bucket == (0, None, 0, 0)
    assert len(workspace.slots) == 1
    assert workspace.transfer_stream is None
    slot = workspace.slots[0]
    assert slot.input_ready is slot.compute_done is slot.finished is None
    assert slot.active_bucket is None
    assert all(workspace.slot_for_bucket(index) is slot for index in (0, 2, 3))
    with pytest.raises(RuntimeError, match="does not have a pipeline slot"):
        workspace.slot_for_bucket(1)

    key = (torch.device("cpu"), torch.float32)
    double_key = (torch.device("cpu"), torch.float64)
    assert slot.buffers.storage_exchange[key].numel() == 10
    assert slot.buffers.compute_exchange[key].numel() == 20
    assert slot.buffers.storage_scratch[key].numel() == 10
    assert slot.buffers.compute_scratch[key].numel() == 30
    assert slot.buffers.compute_scratch[double_key].numel() == 35
    assert workspace.local_buffers.compute_scratch[key].numel() == 8
    assert workspace.local_buffers.compute_scratch[double_key].numel() == 21
    assert workspace.local_buffers.storage_exchange == {}
    assert workspace.local_buffers.compute_exchange == {}
    assert workspace.local_buffers.storage_scratch == {}

    first_storage, first_compute = slot.buffers.exchange_buffers(buckets[0])
    last_storage, last_compute = slot.buffers.exchange_buffers(buckets[3])
    assert first_storage.shape == (7,)
    assert first_compute.shape == (14,)
    assert last_storage.shape == (10,)
    assert last_compute.shape == (20,)
    first_storage.fill_(7)
    first_compute.fill_(11)
    torch.testing.assert_close(last_storage[:7], torch.full((7,), 7.0))
    torch.testing.assert_close(last_compute[:14], torch.full((14,), 11.0))
    assert first_storage.data_ptr() != first_compute.data_ptr()


def test_muon_workspace_configure_grows_then_reuses_buffers_and_accepts_local_only():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _MuonWorkspace
    from torchturbo.runtime.cpu import CpuRuntime

    workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=True)
    small_buckets = _workspace_test_buckets()
    workspace.configure(small_buckets)
    slot = workspace.slots[0]
    key = (torch.device("cpu"), torch.float32)
    pools = (
        slot.buffers.storage_exchange,
        slot.buffers.compute_exchange,
        slot.buffers.storage_scratch,
        slot.buffers.compute_scratch,
        workspace.local_buffers.compute_scratch,
    )
    original_tensors = tuple(pool[key] for pool in pools)
    large_buckets = (
        _workspace_test_bucket(
            "large",
            owned_shapes=((8, 9), (3, 4)),
            local_tensors=(torch.empty(10, 10),),
        ),
        _workspace_test_bucket("large_local", local_tensors=(torch.empty(10, 11),)),
    )

    workspace.configure(large_buckets)

    assert workspace.slots == (slot,)
    assert workspace.slot_index_by_bucket == (0, None)
    grown_tensors = tuple(pool[key] for pool in pools)
    assert tuple(tensor.numel() for tensor in grown_tensors) == (44, 84, 36, 100, 110)
    assert all(
        grown is not original
        for original, grown in zip(original_tensors, grown_tensors, strict=True)
    )
    for index, tensor in enumerate(grown_tensors):
        tensor.fill_(index + 1)

    workspace.configure(small_buckets)

    assert workspace.slot_index_by_bucket == (0, None, 0, 0)
    for index, (pool, tensor) in enumerate(zip(pools, grown_tensors, strict=True)):
        assert pool[key] is tensor
        torch.testing.assert_close(tensor, torch.full_like(tensor, index + 1))

    workspace.configure((small_buckets[1],))

    assert workspace.slot_index_by_bucket == (None,)
    assert workspace.slots == (slot,)
    local = workspace.local_buffers.compute_tensor(
        torch.Size((2, 4)), dtype=torch.float32, device=torch.device("cpu")
    )
    assert local.data_ptr() == grown_tensors[-1].data_ptr()
    torch.testing.assert_close(local, torch.full((2, 4), 5.0))
    with pytest.raises(RuntimeError, match="does not have a pipeline slot"):
        workspace.slot_for_bucket(0)


@pytest.mark.parametrize("overlap", (True, False))
def test_muon_workspace_configure_async_creates_and_reuses_streams_slots_and_buffers(
    overlap,
):
    from contextlib import nullcontext

    from torchturbo.engine.backends.fsdp2.optimizers.muon import _MuonWorkspace
    from torchturbo.runtime.cpu import CpuRuntime

    class FakeStream:
        def __init__(self):
            self.waited_streams = []

        def wait_stream(self, stream):
            self.waited_streams.append(stream)

    class FakeEvent:
        def __init__(self):
            self.recorded_streams = []

        def record(self, stream):
            self.recorded_streams.append(stream)

    class AsyncCpuRuntime(CpuRuntime):
        def __init__(self):
            self.caller = FakeStream()
            self.created_streams = []
            self.created_events = []
            self.recorded_tensors = []

        @property
        def supports_async_streams(self):
            return True

        def current_stream(self):
            return self.caller

        def new_stream(self, priority=0):
            stream = FakeStream()
            self.created_streams.append(stream)
            return stream

        def new_event(self):
            event = FakeEvent()
            self.created_events.append(event)
            return event

        def stream(self, stream=None):
            assert stream is None or isinstance(stream, FakeStream)
            return nullcontext()

        def record_stream(self, tensor, stream):
            assert tensor.device.type == "cpu"
            self.recorded_tensors.append((tensor, stream))

    runtime = AsyncCpuRuntime()
    workspace = _MuonWorkspace(runtime, overlap_comm_compute=overlap)
    buckets = _workspace_test_buckets()
    workspace.configure((buckets[1],))
    assert workspace.slots == ()
    assert runtime.created_streams == runtime.created_events == []

    workspace.configure(buckets)

    expected_indices = (0, None, 1, 0) if overlap else (0, None, 0, 0)
    assert workspace.async_enabled
    assert workspace.slot_index_by_bucket == expected_indices
    slots = workspace.slots
    assert len(slots) == (2 if overlap else 1)
    assert runtime.created_streams == [workspace.transfer_stream]
    assert len(runtime.created_events) == 3 * len(slots)
    assert workspace.slot_for_bucket(0) is workspace.slot_for_bucket(3)
    key = (torch.device("cpu"), torch.float32)
    if overlap:
        assert workspace.slot_for_bucket(2) is slots[1]
        assert slots[0].buffers.storage_exchange[key].numel() == 10
        assert slots[1].buffers.storage_exchange[key].numel() == 8
        assert slots[0].buffers.compute_scratch[key].numel() == 30
        assert slots[1].buffers.compute_scratch[key].numel() == 12

    tensors = []
    for index, slot in enumerate(slots):
        assert slot.active_bucket is None
        assert slot.input_ready.recorded_streams == [workspace.transfer_stream]
        assert slot.compute_done.recorded_streams == [runtime.caller]
        assert slot.finished.recorded_streams == [workspace.transfer_stream]
        storage = slot.buffers.storage_exchange[key]
        storage.fill_(index + 1)
        tensors.append(storage)
    if overlap:
        assert tensors[0].data_ptr() != tensors[1].data_ptr()

    workspace.configure(buckets)

    assert workspace.slots is slots
    assert workspace.slot_index_by_bucket == expected_indices
    assert runtime.created_streams == [workspace.transfer_stream]
    assert len(runtime.created_events) == 3 * len(slots)
    for index, (slot, tensor) in enumerate(zip(slots, tensors, strict=True)):
        assert slot.buffers.storage_exchange[key] is tensor
        torch.testing.assert_close(tensor, torch.full_like(tensor, index + 1))
        assert slot.input_ready.recorded_streams == [workspace.transfer_stream] * 2
        assert slot.compute_done.recorded_streams == [runtime.caller] * 2
        assert slot.finished.recorded_streams == [workspace.transfer_stream] * 2
        assert any(
            recorded is tensor and stream is runtime.caller
            for recorded, stream in runtime.recorded_tensors
        )

    first_caller = runtime.caller
    registrations = len(runtime.recorded_tensors)
    workspace.record_compute_stream(first_caller)
    assert first_caller.waited_streams == []
    assert len(runtime.recorded_tensors) == registrations

    second_caller = FakeStream()
    workspace.record_compute_stream(second_caller)
    assert second_caller.waited_streams == [first_caller]
    for tensor in tensors:
        assert any(
            recorded is tensor and stream is second_caller
            for recorded, stream in runtime.recorded_tensors
        )
    registrations = len(runtime.recorded_tensors)
    workspace.record_compute_stream(second_caller)
    assert second_caller.waited_streams == [first_caller]
    assert len(runtime.recorded_tensors) == registrations

    workspace.configure(buckets)
    assert first_caller.waited_streams == [second_caller]
    workspace.configure(buckets)
    assert first_caller.waited_streams == [second_caller]
    assert second_caller.waited_streams == [first_caller]
    for index, (slot, tensor) in enumerate(zip(slots, tensors, strict=True)):
        assert slot.buffers.storage_exchange[key] is tensor
        torch.testing.assert_close(tensor, torch.full_like(tensor, index + 1))


def test_muon_workspace_required_bytes_counts_buffers_and_temporaries():
    from torchturbo.engine.backends.fsdp2.optimizers.muon import _MuonWorkspace
    from torchturbo.runtime.cpu import CpuRuntime

    workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=False)
    buckets = _workspace_test_buckets()
    required = workspace.required_bytes(buckets)

    workspace.configure(buckets)
    allocated = sum(
        tensor.numel() * tensor.element_size()
        for buffers in (
            workspace.local_buffers,
            *(slot.buffers for slot in workspace.slots),
        )
        for pool in (
            buffers.storage_exchange,
            buffers.compute_exchange,
            buffers.storage_scratch,
            buffers.compute_scratch,
        )
        for tensor in pool.values()
    )
    # The FP64 (1, 5, 7) local matrix dominates both temporaries: three BF16
    # copies of its values and Gram matrix plus an FP32 norm, and staging of
    # its gradient and momentum shards.
    newton_schulz_bytes = 6 * (5 * 7 + 5 * 5) + 4
    staging_bytes = 2 * 5 * 7 * 8
    assert required == allocated + newton_schulz_bytes + staging_bytes


def test_muon_offloaded_workspace_budget_counts_allocator_cache(monkeypatch):
    from torchturbo.engine.backends.fsdp2.optimizers.muon import (
        FSDPDistributedMuon,
        _MuonWorkspace,
    )
    from torchturbo.runtime.cpu import CpuRuntime

    class MemoryStatsRuntime(CpuRuntime):
        free = 0
        cached = 0

        def memory_info(self):
            return self.free, self.free + self.cached

        def memory_reserved(self):
            return self.cached

        def memory_allocated(self):
            return 0

    peer_ok = True

    def all_reduce(status, op):
        assert op == dist.ReduceOp.MIN
        if not peer_ok:
            status.zero_()

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    runtime = MemoryStatsRuntime()
    buckets = _workspace_test_buckets()
    optimizer = object.__new__(FSDPDistributedMuon)
    optimizer._offload = CPUOffloadConfig(
        enabled=True, mode="optimizer", optimizer_memory_fraction=0.5
    )
    optimizer._device_runtime = runtime
    optimizer._workspace = _MuonWorkspace(CpuRuntime(), overlap_comm_compute=False)
    optimizer._compiled_buckets = buckets
    optimizer._bound_parameters = tuple(
        item for bucket in buckets for item in bucket.items
    )
    optimizer._step_scoped_workspace = True
    required = optimizer._workspace.required_bytes(buckets)

    # Freed activations stay cached by the allocator after backward and can
    # serve the workspace even when the driver reports no free memory.
    runtime.free, runtime.cached = 0, 2 * required
    optimizer._acquire_workspace()
    assert optimizer._workspace.configured
    assert optimizer._workspace.slots
    optimizer.release_workspace()

    runtime.free, runtime.cached = required, 0
    with pytest.raises(RuntimeError, match="before any update") as failure:
        optimizer._acquire_workspace()
    assert "optimizer_memory_fraction=0.5" in str(failure.value.__cause__)
    assert not optimizer._workspace.configured
    assert not optimizer._workspace.slots

    # A failure on another rank stops this rank after its own allocation.
    runtime.free, runtime.cached = 2 * required, 0
    peer_ok = False
    with pytest.raises(RuntimeError, match="before any update") as failure:
        optimizer._acquire_workspace()
    assert failure.value.__cause__ is None
    assert not optimizer._workspace.configured
    assert not optimizer._workspace.slots
