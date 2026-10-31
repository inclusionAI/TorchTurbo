"""Tests for TorchTurboEngine one-stop-shop API."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import torchturbo.engine.core as engine_core
from torchturbo.engine import (
    BaseBackend,
    BaseCheckpointManager,
    CheckpointFormat,
    DistributedBackend,
    EngineConfig,
    FSDP2Backend,
    MixedPrecisionPolicy,
    MuonOptimizerConfig,
    MuonPlan,
    OptimizerClassPlan,
    OptimizerConfig,
    TorchTurboEngine,
    build_model_fqn_map,
    register_backend,
)
from torchturbo.engine.backend_registry import _BACKENDS
from torchturbo.engine.backends.fsdp2.config import FSDPConfig
from torchturbo.engine.checkpoint.coordinator import CheckpointCoordinator, CheckpointExportPluginConsumer
from torchturbo.engine.optimizers import (
    MuonParameterSpec,
    MuonRecipe,
    OptimizersContainer,
    Owned,
)
from torchturbo.models.base import TurboModel
from torchturbo.models.modules.moe.moe_layer import BaseMoELayer, MoERuntimeInputs
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.runtime.cpu import CpuRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def _minimal_packed_inputs(**extra):
    """Smallest batch satisfying the packed training input contract."""
    inputs = {
        "cu_seq_lens_q": torch.tensor([0, 1], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 1], dtype=torch.int32),
        "max_length_q": 1,
        "max_length_k": 1,
    }
    inputs.update(extra)
    return inputs


def test_turbo_model_requires_an_explicit_optimizer_recipe():
    # Engine-managed optimizers are opt-in per architecture, so the base class
    # never guesses a parameter split.
    for optimizer_name in ("muon", "unknown"):
        with pytest.raises(
            NotImplementedError,
            match=f"optimizer recipe for {optimizer_name!r}",
        ):
            TurboModel().get_optimizer_recipe(optimizer_name)


class PackedLossModel(TurboModel, nn.Module):
    """TinyModel plus the TurboModel contract for engine.compute_loss."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class TinyLogitsModel(nn.Module):
    """Small logits-producing model for the packed RL engine API tests."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids):
        logits = input_ids.float().unsqueeze(-1) * self.weight
        return SimpleNamespace(logits=logits)


class RecordingLayout:
    def native_model_dir(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, "model")

    def optimizer_dir(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, "optimizer")

    def native_model_status(self, checkpoint_dir):
        from torchturbo.engine.checkpoint import ComponentStatus

        return ComponentStatus.MISSING

    def optimizer_status(self, checkpoint_dir):
        from torchturbo.engine.checkpoint import ComponentStatus

        return ComponentStatus.MISSING


class RecordingCheckpointManager(RecordingLayout, BaseCheckpointManager):
    """Records the checkpoint calls the engine forwards to its backend."""

    def __init__(self):
        self.calls = []

    def load_hf(self, model, model_path, *, weight_converter, assign=False):
        self.calls.append(
            ("load_hf", model_path, weight_converter, assign)
        )

    def save_hf(
        self,
        model,
        output_dir,
        *,
        source_path,
        weight_converter,
    ):
        self.calls.append(("save_hf", output_dir, source_path, weight_converter))

    def save_native(self, model, output_dir):
        self.calls.append(("save_native", output_dir))

    def load_native(self, model, checkpoint_dir):
        self.calls.append(("load_native", checkpoint_dir))

    def save_optimizer(self, model, optimizer, output_dir):
        self.calls.append(("save_optimizer", optimizer, output_dir))

    def load_optimizer(self, model, optimizer, checkpoint_dir):
        self.calls.append(("load_optimizer", optimizer, checkpoint_dir))

    def materialize(self, model):
        self.calls.append(("materialize", model))
        return model


class DummyBackend(BaseBackend):
    name = "dummy"

    def __init__(self, config, device_runtime):
        super().__init__(config, device_runtime)
        self._checkpoint = RecordingCheckpointManager()

    def prepare_model(self, model, parallel_plan=None, peft_config=None):
        model._dummy_sharded = True
        return model

    def materialize(self, model):
        return model.to_empty(device=self.device)

    def create_checkpoint_manager(self):
        return self._checkpoint

    def prepare_optimizer(
        self,
        plan,
    ):
        name = plan.optimizer_name
        groups = [
            group.get_optimizer_param_group()
            for group in plan.get_param_groups_by_optimizer_name()[name]
        ]
        return plan.optimizer_cls(groups, **plan.optimizer_kwargs)

    def get_dp_info(self) -> dict[str, int | None]:
        return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

    def get_ep_info(self) -> dict[str, int | None]:
        return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

    def get_sp_info(self) -> dict[str, int | None]:
        return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

    def get_parallel_context(self):
        return {
            "dp": self.get_dp_info(),
            "ep": self.get_ep_info(),
            "sp": self.get_sp_info(),
        }

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        loss = torch.tensor(0.0)
        return (loss, None) if return_outputs else loss


class ProfilingBackend(DummyBackend):
    pass


class RecordingProfiler:
    def __init__(self):
        self.calls = []

    def start(self):
        self.calls.append("start")

    def step(self):
        self.calls.append("step")

    def stop(self):
        self.calls.append("stop")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_engine_state(monkeypatch):
    saved = dict(_BACKENDS)
    ops_module = importlib.import_module("torchturbo.ops.ops")
    monkeypatch.setattr(ops_module, "_OPS", None)
    monkeypatch.setattr(engine_core, "get_device_runtime", CpuRuntime)
    yield
    _BACKENDS.clear()
    _BACKENDS.update(saved)


@pytest.fixture
def dummy_engine():
    # EngineConfig selects a supported backend; the registry supplies its class.
    register_backend("fsdp2", DummyBackend)
    return TorchTurboEngine()


# ---------------------------------------------------------------------------
# Tests: Config
# ---------------------------------------------------------------------------


class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.backend == DistributedBackend.FSDP2
        assert isinstance(cfg.backend_config, FSDPConfig)

    def test_from_dict(self):
        cfg = EngineConfig.from_dict(
            {"backend": "fsdp2", "mixed_precision": "fp32"}
        )
        assert cfg.mixed_precision == MixedPrecisionPolicy.FP32

    @pytest.mark.parametrize("policy", ["fp16", "fp8"])
    def test_from_dict_rejects_unsupported_mixed_precision(self, policy):
        with pytest.raises(
            ValueError, match=f"does not support mixed_precision='{policy}'"
        ):
            EngineConfig.from_dict({"backend": "fsdp2", "mixed_precision": policy})

    @pytest.mark.parametrize("policy", ["fp16", "fp8"])
    def test_constructor_rejects_unsupported_mixed_precision(self, policy):
        with pytest.raises(
            ValueError, match=f"does not support mixed_precision='{policy}'"
        ):
            EngineConfig(mixed_precision=policy)


# ---------------------------------------------------------------------------
# Tests: Engine core functionality
# ---------------------------------------------------------------------------


class TestPackedInputContract:
    def test_compute_loss_rejects_attention_mask(self):
        engine = object.__new__(TorchTurboEngine)
        engine.backend = SimpleNamespace(
            prepare_inputs=lambda model, inputs: inputs,
            compute_loss=lambda model, inputs, **kwargs: torch.tensor(0.0),
        )
        engine.metric_tracker = engine_core.MetricTracker()
        engine.config = SimpleNamespace(mfu=SimpleNamespace(enabled=False))

        with pytest.raises(ValueError, match="not attention_mask"):
            engine.compute_loss(
                PackedLossModel(), _minimal_packed_inputs(attention_mask=torch.ones(1, 2))
            )

    def test_compute_loss_requires_cu_metadata(self):
        engine = object.__new__(TorchTurboEngine)
        engine.backend = SimpleNamespace(
            prepare_inputs=lambda model, inputs: inputs,
            compute_loss=lambda model, inputs, **kwargs: torch.tensor(0.0),
        )
        engine.metric_tracker = engine_core.MetricTracker()
        engine.config = SimpleNamespace(mfu=SimpleNamespace(enabled=False))

        with pytest.raises(ValueError, match="cu_seq_lens_q/k"):
            engine.compute_loss(PackedLossModel(), {"input_ids": torch.ones(1, 4)})


class TestEngineSetup:
    @pytest.mark.parametrize("entrypoint", ["build_model", "prepare_model"])
    def test_model_build_patches_hf_before_parallelized_hook(self, monkeypatch, entrypoint):
        from transformers import PretrainedConfig, PreTrainedModel
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        import transformers.integrations.flash_attention as flash_integration
        import transformers.modeling_flash_attention_utils as flash_utils

        from torchturbo.models.base import TurboModel
        from torchturbo.models.modules.attention.interface import attention_forward
        from torchturbo.models.modules.attention.ulysses import TURBO_ULYSSES_ATTENTION
        from torchturbo.models.modules.attention.ulysses import (
            ulysses_attention_forward,
        )

        monkeypatch.setattr(
            type(ALL_MASK_ATTENTION_FUNCTIONS),
            "_global_mapping",
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.copy(),
        )
        order = []
        sp_binding = []
        parallel_context = {"sp": {"sp_group": object(), "sp_size": 2}}
        converter = object()
        original_binding = Mock()
        monkeypatch.setattr(flash_integration, "_flash_attention_forward", original_binding)
        monkeypatch.setattr(flash_utils, "_get_unpad_data", flash_utils._get_unpad_data)

        class LifecycleModel(TurboModel, PreTrainedModel):
            config_class = PretrainedConfig

            def __init__(self, config: PretrainedConfig) -> None:
                # This lifecycle fixture has no attention or flash-attn dependency.
                config._attn_implementation = "eager"
                super().__init__(config)
                self.projection = nn.Linear(2, 2, bias=False)

            def on_parallelized(self, context: dict) -> None:
                order.append("on_parallelized")
                assert context is parallel_context
                assert flash_integration._flash_attention_forward is attention_forward
                self.register_sp_attention_interface(context["sp"])
                sp_binding.append(ALL_ATTENTION_FUNCTIONS.get(TURBO_ULYSSES_ATTENTION))
                # The registry path must leave the shared FA helper alone.
                assert flash_integration._flash_attention_forward is attention_forward

            def on_materialized(self) -> None:
                order.append("on_materialized")
                assert self.projection.weight.device.type == "cpu"
                assert flash_integration._flash_attention_forward is attention_forward
                assert ALL_ATTENTION_FUNCTIONS[TURBO_ULYSSES_ATTENTION] is sp_binding[0]

            def get_weight_converter(self, context, model_path):
                order.append("weight_converter")
                assert context is parallel_context
                assert model_path == "local_model"
                assert torch.empty(0).device.type == "cpu"
                return converter

        def backend_prepare(model, parallel_plan, peft_config):
            order.append("backend_prepare")
            assert isinstance(model, LifecycleModel)
            assert model.projection.weight.is_meta
            assert flash_integration._flash_attention_forward is original_binding
            assert parallel_plan is engine.parallel_plan
            assert peft_config is None
            return model

        def backend_context():
            order.append("parallel_context")
            assert flash_integration._flash_attention_forward is attention_forward
            return parallel_context

        def backend_materialize(model):
            order.append("backend_load")
            assert flash_integration._flash_attention_forward is attention_forward
            return model.to_empty(device="cpu")

        def checkpoint_load(model, model_path, *, weight_converter, assign):
            assert model_path == "local_model"
            assert weight_converter is converter
            assert assign is True
            return backend_materialize(model)

        engine = object.__new__(TorchTurboEngine)
        engine.config = EngineConfig()
        engine.model = previous_model = TinyModel()
        engine.model_name_or_path = "previous_model"
        engine.plugin_manager = Mock()
        engine.parallel_plan = None
        engine._checkpoint = None
        engine.backend = Mock(spec=BaseBackend)
        checkpoint_manager = Mock(spec=BaseCheckpointManager)
        checkpoint_manager.load_hf.side_effect = checkpoint_load
        engine.backend.prepare_model.side_effect = backend_prepare
        engine.backend.create_checkpoint_manager.return_value = checkpoint_manager
        engine.backend.get_parallel_context.side_effect = backend_context
        engine.backend.materialize = backend_materialize
        engine.backend.finalize_model_load.side_effect = lambda model: model
        freeze_calls = []
        monkeypatch.setattr(
            engine,
            "_apply_freeze_policy",
            lambda model, policy: freeze_calls.append((model, policy)),
        )
        monkeypatch.setattr(engine, "_log_info", lambda: None)
        monkeypatch.setattr(engine_core.dist, "is_initialized", lambda: False)
        patch_transformers = engine._patch_transformers

        def record_patch():
            order.append("hf_patch")
            patch_transformers()

        monkeypatch.setattr(engine, "_patch_transformers", record_patch)
        arguments = dict(
            model_name_or_path="local_model",
            model_cls=LifecycleModel,
            init_kwargs={"config": PretrainedConfig()},
        )
        try:
            if entrypoint == "build_model":
                model = engine.build_model(**arguments, load_weights=False)
                assert engine.model is previous_model
                assert engine.model_name_or_path == "previous_model"
                assert freeze_calls == []
            else:
                model = engine.prepare_model(**arguments)
                assert engine.model is model
                assert engine.model_name_or_path == "local_model"
                assert freeze_calls == [(model, engine.config.freeze_policy)]
        finally:
            ALL_ATTENTION_FUNCTIONS.pop(TURBO_ULYSSES_ATTENTION, None)

        assert isinstance(model, LifecycleModel)
        expected = [
            "backend_prepare", "hf_patch", "parallel_context", "on_parallelized",
        ]
        if entrypoint != "build_model":
            expected.append("weight_converter")
        expected.extend(["backend_load", "on_materialized"])
        assert order == expected
        assert flash_integration._flash_attention_forward is attention_forward
        assert sp_binding[0].func is ulysses_attention_forward
        assert sp_binding[0].keywords["sp_size"] == 2
        engine.backend.prepare_model.assert_called_once()
        engine.backend.create_checkpoint_manager.assert_called_once()
        assert isinstance(engine.checkpoint, CheckpointCoordinator)
        assert engine.checkpoint._manager is checkpoint_manager

    def test_default_init(self, tmp_path):
        dist.init_process_group(
            "gloo",
            init_method=(tmp_path / "process_group").as_uri(),
            rank=0,
            world_size=1,
        )
        try:
            engine = TorchTurboEngine()
            assert isinstance(engine.backend, FSDP2Backend)
            assert engine.backend.rank == 0
            assert engine.backend.device == torch.device("cpu")
            assert engine.model is None
        finally:
            dist.destroy_process_group()

    def test_prepare_model_not_supported_type(self, dummy_engine):
        from transformers import PretrainedConfig

        with pytest.raises(TypeError, match="requires a TurboModel subclass"):
            dummy_engine.prepare_model(
                "local_model",
                model_cls=TinyModel,
                init_kwargs={"config": PretrainedConfig()},
            )
        assert dummy_engine.model is None

    def test_prepare_optimizer(self, dummy_engine):
        engine = dummy_engine

        # Inject model manually to avoid huggingface download logic
        model = TinyModel()
        engine.model = engine.backend.prepare_model(model)

        engine.prepare_optimizer(
            torch.optim.AdamW,
            lr=0.01,
            weight_decay=0.0,
        )
        assert isinstance(engine.optimizer, torch.optim.AdamW)
        assert engine.optimizer.param_groups[0]["params"] == [model.linear.weight]
        assert engine.optimizer.param_groups[0]["lr"] == 0.01
        assert engine.optimizer.param_groups[0]["weight_decay"] == 0.0

    def test_prepare_optimizer_without_model_raises(self, dummy_engine):
        engine = dummy_engine
        with pytest.raises(RuntimeError, match="prepare_model"):
            engine.prepare_optimizer(torch.optim.AdamW)

    def test_framework_optimizer_defaults_to_adamw(self, dummy_engine):
        engine = dummy_engine
        engine.model = engine.backend.prepare_model(TinyModel())

        optimizer = engine.prepare_optimizer(lr=0.01, weight_decay=0.0)

        assert type(optimizer) is torch.optim.AdamW

    def test_framework_optimizer_is_passed_to_backend_as_class_plan(self):
        prepared = []
        model = TinyModel()

        class Backend:
            def prepare_optimizer(self, plan):
                assert isinstance(plan, OptimizerClassPlan)
                assert plan.optimizer_cls is torch.optim.AdamW
                assert plan.optimizer_kwargs["lr"] == 0.01
                assert plan.optimizer_kwargs["weight_decay"] == 0.2
                groups_by_optimizer_name = plan.get_param_groups_by_optimizer_name()
                assert tuple(groups_by_optimizer_name) == ("AdamW",)
                optimizer = torch.optim.AdamW(
                    [
                        group.get_optimizer_param_group()
                        for group in groups_by_optimizer_name["AdamW"]
                    ],
                    **plan.optimizer_kwargs,
                )
                prepared.append(optimizer)
                return optimizer

        engine = object.__new__(TorchTurboEngine)
        engine.model = model
        engine.backend = Backend()
        engine.config = EngineConfig()

        result = engine.prepare_optimizer(
            torch.optim.AdamW,
            lr=0.01,
            weight_decay=0.2,
        )

        assert result is prepared[0]
        assert engine.optimizer is result

    def test_engine_builds_muon_plan_for_backend(self):
        prepared = []
        map_requests = []
        model = TinyModel()

        class Backend:
            name = "test"

            def get_model_fqn_map(self):
                map_requests.append(model)
                return build_model_fqn_map(model)

            def prepare_optimizer(self, plan):
                assert isinstance(plan, MuonPlan)
                assert dict(plan.parameter_specs) == {
                    "linear.weight": MuonParameterSpec(compute_placement=Owned())
                }
                assert plan.muon_kwargs["lr"] == engine.config.optimizer.muon.lr
                assert (
                    plan.overlap_comm_compute
                    is engine.config.optimizer.muon.overlap_comm_compute
                )
                groups = [
                    group.get_optimizer_param_group()
                    for group in plan.get_param_groups_by_optimizer_name()["Muon"]
                ]
                optimizer = torch.optim.Optimizer(groups, {})
                prepared.append(optimizer)
                return optimizer

        engine = object.__new__(TorchTurboEngine)
        engine.model = model
        expected_recipe = MuonRecipe({})

        def get_optimizer_recipe(optimizer_name):
            assert optimizer_name == "muon"
            return expected_recipe

        engine.model.get_optimizer_recipe = get_optimizer_recipe
        engine.backend = Backend()
        engine.config = EngineConfig(
            optimizer=OptimizerConfig(
                name="muon",
                muon=MuonOptimizerConfig(lr=0.1, weight_decay=0.01),
            )
        )

        result = engine.prepare_optimizer()

        assert result is prepared[0]
        assert engine.optimizer is result
        assert map_requests == [model]

    def test_engine_preserves_backend_optimizer_container(self):
        prepared = []

        class Model(nn.Linear):
            def __init__(self):
                super().__init__(4, 1)

            def get_optimizer_recipe(self, _optimizer_name):
                return MuonRecipe({})

        model = Model()

        class Backend:
            name = "test"

            def get_model_fqn_map(self):
                return build_model_fqn_map(model)

            def prepare_optimizer(self, plan):
                optimizers_by_name = {
                    name: torch.optim.Optimizer(
                        [group.get_optimizer_param_group() for group in groups],
                        {},
                    )
                    for name, groups in plan.get_param_groups_by_optimizer_name().items()
                }
                optimizer = OptimizersContainer(plan, optimizers_by_name)
                prepared.append(optimizer)
                return optimizer

        engine = object.__new__(TorchTurboEngine)
        engine.model = model
        engine.backend = Backend()
        engine.config = EngineConfig(
            optimizer=OptimizerConfig(
                name="muon",
                muon=MuonOptimizerConfig(lr=0.1, weight_decay=0.01),
            )
        )

        result = engine.prepare_optimizer()

        assert result is prepared[0]
        assert engine.optimizer is result
        assert isinstance(result, OptimizersContainer)
        assert tuple(result.optimizers_by_name) == ("Muon", "AdamW")

    @pytest.mark.parametrize(
        ("optimizer_cls", "optimizer_kwargs"),
        [
            (torch.optim.SGD, {}),
            (None, {"lr": 0.1, "weight_decay": 0.01}),
        ],
    )
    def test_engine_managed_optimizer_rejects_framework_inputs(
        self,
        optimizer_cls,
        optimizer_kwargs,
    ):
        engine = object.__new__(TorchTurboEngine)
        engine.model = TinyModel()
        engine.config = EngineConfig(optimizer=OptimizerConfig(name="muon"))

        with pytest.raises(ValueError, match="cannot be supplied"):
            engine.prepare_optimizer(optimizer_cls, **optimizer_kwargs)

    def test_engine_rejects_unsupported_engine_managed_optimizer(self):
        engine = object.__new__(TorchTurboEngine)
        engine.model = TinyModel()
        engine.config = EngineConfig(optimizer=OptimizerConfig(name="muon"))
        # OptimizerConfig rejects unknown names, so reach the engine's dispatch
        # backstop by mutating the validated config.
        engine.config.optimizer.name = "future_optimizer"

        with pytest.raises(
            NotImplementedError,
            match="cannot prepare optimizer 'future_optimizer'",
        ):
            engine.prepare_optimizer()

    def test_compute_loss_defers_eval_metrics_until_explicit_collect(
        self, dummy_engine
    ):
        engine = dummy_engine
        model = PackedLossModel().eval()
        model.pop_metrics = lambda: {"test_metric": torch.tensor(2.0)}

        engine.compute_loss(model, _minimal_packed_inputs())
        logs = {}
        engine.report_metrics(logs, lambda value: value.reshape(1))
        assert logs == {}

        engine.collect_metrics(model)
        engine.report_metrics(logs, lambda value: value.reshape(1))

        assert logs == {"test_metric": 2.0}

    def test_compute_loss_defers_training_metrics_until_explicit_collect(
        self, dummy_engine
    ):
        engine = dummy_engine
        model = PackedLossModel()
        model.pop_metrics = lambda: {"test_metric": torch.tensor(2.0)}

        engine.compute_loss(model, _minimal_packed_inputs())
        logs = {}
        engine.report_metrics(logs, lambda value: value.reshape(1))
        assert logs == {}

        engine.collect_metrics(model)
        engine.report_metrics(logs, lambda value: value.reshape(1))
        assert logs == {"test_metric": 2.0}

    def test_compute_loss_sets_moe_runtime_inputs(self):
        calls = []

        class RuntimeMoELayer(BaseMoELayer):
            def __init__(self):
                nn.Module.__init__(self)
                self._runtime_inputs = MoERuntimeInputs()

            def forward(self, hidden_states, input_ids=None):
                return self._runtime_inputs.padding_mask

        class RuntimeModel(TurboModel, nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                self.moe = RuntimeMoELayer()

            def get_moe_runtime_inputs(self, inputs, parallel_context):
                calls.append("moe_runtime")
                return MoERuntimeInputs(padding_mask=inputs["padding_mask"])

        class RuntimeBackend:
            def prepare_inputs(self, model, inputs):
                calls.append("prepare_inputs")
                return {**inputs, "padding_mask": prepared_padding_mask}

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                calls.append("compute_loss")
                assert inputs["padding_mask"] is prepared_padding_mask
                return torch.tensor(0.0)

            def get_dp_info(self):
                return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

            def get_sp_info(self):
                return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

            def get_ep_info(self):
                return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

            def get_parallel_context(self):
                return {
                    "dp": self.get_dp_info(),
                    "ep": self.get_ep_info(),
                    "sp": self.get_sp_info(),
                }

        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()
        engine.config = SimpleNamespace(mfu=SimpleNamespace(enabled=False))
        model = RuntimeModel()
        padding_mask = torch.tensor([[False, True]])
        prepared_padding_mask = torch.tensor([[False, True, True]])

        engine.compute_loss(model, _minimal_packed_inputs(padding_mask=padding_mask))

        assert model.moe._runtime_inputs.padding_mask is prepared_padding_mask
        assert calls == ["prepare_inputs", "moe_runtime", "compute_loss"]

    @pytest.mark.parametrize(
        ("training", "gradient_accumulation_steps", "expected_scale"),
        # Eval/reference forwards must preserve the pending policy aux scale.
        [(True, 4, 0.25), (False, 4, None)],
    )
    def test_compute_loss_sets_moe_aux_loss_scale(
        self,
        monkeypatch,
        training,
        gradient_accumulation_steps,
        expected_scale,
    ):
        class RuntimeBackend:
            def prepare_inputs(self, model, inputs):
                return inputs

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(0.0)

            def get_parallel_context(self):
                return {
                    "dp": {"dp_rank": 0, "dp_size": 1, "dp_group": None},
                    "ep": {"ep_rank": 0, "ep_size": 1, "ep_group": None},
                    "sp": {"sp_rank": 0, "sp_size": 1, "sp_group": None},
                }

            def get_dp_info(self):
                return {"dp_rank": 0, "dp_size": 1, "dp_group": None}

            def get_sp_info(self):
                return {"sp_rank": 0, "sp_size": 1, "sp_group": None}

            def get_ep_info(self):
                return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

            def get_parallel_context(self):
                return {
                    "dp": self.get_dp_info(),
                    "ep": self.get_ep_info(),
                    "sp": self.get_sp_info(),
                }

        class RuntimeMoELayer(BaseMoELayer):
            def __init__(self):
                nn.Module.__init__(self)

            def forward(self, hidden_states, input_ids=None):
                return hidden_states

        class RuntimeModel(TurboModel, nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = RuntimeMoELayer()

            def get_moe_runtime_inputs(self, inputs, parallel_context):
                return MoERuntimeInputs()

        scales = []
        monkeypatch.setattr(
            MoEAuxLossAutoScaler,
            "set_loss_scale",
            lambda scale: scales.append(scale),
        )
        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()
        engine.config = SimpleNamespace(mfu=SimpleNamespace(enabled=False))
        model = RuntimeModel().train(training)

        engine.compute_loss(
            model,
            _minimal_packed_inputs(),
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        assert scales == ([] if expected_scale is None else [expected_scale])

    def test_compute_loss_skips_moe_setup_for_dense(self, monkeypatch):
        class RuntimeBackend:
            def prepare_inputs(self, model, inputs):
                return inputs

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(0.0)

        scales = []
        monkeypatch.setattr(
            MoEAuxLossAutoScaler,
            "set_loss_scale",
            lambda scale: scales.append(scale),
        )
        engine = object.__new__(TorchTurboEngine)
        engine.backend = RuntimeBackend()
        engine.metric_tracker = engine_core.MetricTracker()
        engine.config = SimpleNamespace(mfu=SimpleNamespace(enabled=False))

        engine.compute_loss(PackedLossModel(), _minimal_packed_inputs())

        assert scales == []

    def test_forward_backward_batch_runs_forward_and_backward(self):
        engine = object.__new__(TorchTurboEngine)
        engine.model = TinyLogitsModel()
        engine.get_sp_info = lambda: {"sp_size": 1}
        engine.get_dp_info = lambda: {"dp_size": 1, "dp_group": None}

        mb_item = {
            "orig_mb": {"input_ids": torch.tensor([1, 2])},
            "padded_mb": {"input_ids": torch.tensor([[1, 2]])},
        }
        calls = []

        def process_fn(logits, ctx, loss_fn):
            calls.append((tuple(logits.shape), ctx["mb_input"]))
            return loss_fn(logits)

        engine.forward_backward_batch(
            [mb_item],
            process_fn=process_fn,
            loss_fn=lambda logits: logits.square().mean(),
            loss_weight_fn=lambda _: torch.tensor(1.0),
        )

        assert len(calls) == 1
        assert calls[0][0] == (2, 1)
        assert engine.model.weight.grad is not None
        assert torch.isfinite(engine.model.weight.grad)

    def test_forward_backward_batch_forward_only_skips_backward(self):
        engine = object.__new__(TorchTurboEngine)
        engine.model = TinyLogitsModel()
        engine.get_sp_info = lambda: {"sp_size": 1}
        engine.get_dp_info = lambda: {"dp_size": 1, "dp_group": None}

        mb_item = {
            "orig_mb": {"input_ids": torch.tensor([1, 2])},
            "padded_mb": {"input_ids": torch.tensor([[1, 2]])},
        }
        engine.forward_backward_batch(
            [mb_item],
            process_fn=lambda logits, _ctx, _loss_fn: logits.square().mean(),
            loss_fn=lambda logits: logits.square().mean(),
            loss_weight_fn=lambda _: torch.tensor(1.0),
            forward_only=True,
        )

        assert engine.model.weight.grad is None

    def test_fit_is_unsupported_compatibility_stub(self):
        engine = object.__new__(TorchTurboEngine)

        with pytest.raises(NotImplementedError, match="no longer supported"):
            engine.fit([])

    def test_prepare_lr_scheduler(self, dummy_engine):
        engine = dummy_engine

        engine.model = engine.backend.prepare_model(TinyModel())
        engine.prepare_optimizer(
            torch.optim.AdamW,
            lr=0.01,
            weight_decay=0.0,
        )

        engine.prepare_lr_scheduler(
            torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.1
        )
        assert isinstance(engine.lr_scheduler, torch.optim.lr_scheduler.StepLR)
        assert engine.lr_scheduler.optimizer is engine.optimizer
        for _ in range(2):
            engine.optimizer.step()
            engine.lr_scheduler.step()
        assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)

    def test_unknown_backend_raises(self):
        with pytest.raises(
            ValueError, match="'nonexistent' is not a valid DistributedBackend"
        ):
            TorchTurboEngine({"backend": "nonexistent"})

    def test_unregistered_backend_raises(self):
        _BACKENDS.pop("fsdp2")
        with pytest.raises(ValueError, match="Unknown distributed backend 'fsdp2'"):
            TorchTurboEngine()

    def test_failed_backend_init_does_not_publish_ops(self, monkeypatch):
        ops_module = importlib.import_module("torchturbo.ops.ops")
        monkeypatch.setattr(ops_module, "_OPS", None)

        class FailingBackend(DummyBackend):
            def __init__(self, config, device_runtime):
                raise RuntimeError("backend init failed")

        monkeypatch.setattr(engine_core, "get_device_runtime", CpuRuntime)
        monkeypatch.setattr(engine_core, "get_backend", lambda _: FailingBackend)

        with pytest.raises(RuntimeError, match="backend init failed"):
            TorchTurboEngine(EngineConfig())

        assert ops_module._OPS is None

    def test_profiling_uses_device_runtime_profiler(self, monkeypatch, tmp_path):
        runtime = CpuRuntime()
        profiler = RecordingProfiler()
        profiler_kwargs = {}

        def fake_profiler(
            *,
            output_dir: str,
            worker_name: str,
            profiling_enabled: bool,
            record_shapes: bool,
            aic_metrics: bool,
            with_stack: bool,
            profiling_step: int | None,
            memory_snapshot_steps: int | None,
        ):
            profiler_kwargs["output_dir"] = output_dir
            profiler_kwargs["worker_name"] = worker_name
            profiler_kwargs["profiling_enabled"] = profiling_enabled
            profiler_kwargs["record_shapes"] = record_shapes
            profiler_kwargs["aic_metrics"] = aic_metrics
            profiler_kwargs["with_stack"] = with_stack
            profiler_kwargs["profiling_step"] = profiling_step
            return profiler

        monkeypatch.setattr(runtime, "profiler", fake_profiler)
        monkeypatch.setattr(engine_core, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(
            engine_core,
            "get_backend",
            lambda backend_name: DummyBackend if backend_name == "fsdp2" else None,
        )

        engine = TorchTurboEngine(EngineConfig())
        engine.start_profiling(str(tmp_path), True, None)
        engine.step_profiling()
        engine.stop_profiling()

        assert profiler.calls == ["start", "step", "stop"]
        assert profiler_kwargs["output_dir"] == str(tmp_path / "profiling")
        assert profiler_kwargs["worker_name"].startswith("rank")
        assert profiler_kwargs["profiling_enabled"] is True
        assert profiler_kwargs["record_shapes"] is False
        assert profiler_kwargs["aic_metrics"] is False
        assert profiler_kwargs["with_stack"] is False
        assert profiler_kwargs["profiling_step"] is None
        assert (tmp_path / "profiling").is_dir()

    def test_profiling_config_knobs_reach_runtime_profiler(
        self, monkeypatch, tmp_path
    ):
        runtime = CpuRuntime()
        recorder = RecordingProfiler()
        profiler_kwargs = {}

        def fake_profiler(
            *,
            output_dir: str,
            worker_name: str,
            profiling_enabled: bool,
            record_shapes: bool,
            aic_metrics: bool,
            with_stack: bool,
            profiling_step: int | None,
            memory_snapshot_steps: int | None,
        ):
            profiler_kwargs["profiling_enabled"] = profiling_enabled
            profiler_kwargs["record_shapes"] = record_shapes
            profiler_kwargs["aic_metrics"] = aic_metrics
            profiler_kwargs["with_stack"] = with_stack
            profiler_kwargs["profiling_step"] = profiling_step
            return recorder

        monkeypatch.setattr(runtime, "profiler", fake_profiler)
        monkeypatch.setattr(engine_core, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(
            engine_core,
            "get_backend",
            lambda backend_name: DummyBackend if backend_name == "fsdp2" else None,
        )

        engine = TorchTurboEngine(EngineConfig())

        # detailed_kernel_info fans out to both shape and utilization knobs.
        engine.start_profiling(
            str(tmp_path), True, None, {"detailed_kernel_info": True}
        )
        assert profiler_kwargs["record_shapes"] is True
        assert profiler_kwargs["aic_metrics"] is True
        assert profiler_kwargs["with_stack"] is False
        assert profiler_kwargs["profiling_step"] is None

        engine.start_profiling(
            str(tmp_path), True, None, {"profiling_step": 20}
        )
        assert profiler_kwargs["profiling_step"] == 20

        engine.start_profiling(str(tmp_path), True, None, {"with_stack": True})
        assert profiler_kwargs["record_shapes"] is False
        assert profiler_kwargs["aic_metrics"] is False
        assert profiler_kwargs["with_stack"] is True

        # Knobs are ignored when profiling is disabled (snapshot-only mode).
        engine.start_profiling(str(tmp_path), False, 3, {"detailed_kernel_info": True})
        assert profiler_kwargs["record_shapes"] is False
        assert profiler_kwargs["aic_metrics"] is False
        assert profiler_kwargs["with_stack"] is False
        assert profiler_kwargs["profiling_step"] is None

    def test_profiling_start_failure_clears_previous_profiler(
        self, monkeypatch, tmp_path
    ):
        runtime = CpuRuntime()
        first_profiler = RecordingProfiler()

        class FailingProfiler(RecordingProfiler):
            def start(self):
                self.calls.append("start")
                raise RuntimeError("profiler start failed")

        second_profiler = FailingProfiler()
        profilers = iter((first_profiler, second_profiler))

        def fake_profiler(
            *,
            output_dir: str,
            worker_name: str,
            profiling_enabled: bool,
            record_shapes: bool,
            aic_metrics: bool,
            with_stack: bool,
            profiling_step: int | None,
            memory_snapshot_steps: int | None,
        ):
            # Failure-path test: kwargs are irrelevant.
            del (
                output_dir,
                worker_name,
                profiling_enabled,
                record_shapes,
                aic_metrics,
                with_stack,
                profiling_step,
                memory_snapshot_steps,
            )
            return next(profilers)

        monkeypatch.setattr(runtime, "profiler", fake_profiler)
        monkeypatch.setattr(engine_core, "get_device_runtime", lambda: runtime)
        monkeypatch.setattr(
            engine_core,
            "get_backend",
            lambda backend_name: DummyBackend if backend_name == "fsdp2" else None,
        )

        engine = TorchTurboEngine(EngineConfig())
        engine.start_profiling(str(tmp_path), True, None)

        with pytest.raises(RuntimeError, match="profiler start failed"):
            engine.start_profiling(str(tmp_path), True, None)

        assert first_profiler.calls == ["start", "stop"]
        assert second_profiler.calls == ["start"]
        with pytest.raises(RuntimeError, match="No active profiler"):
            engine.step_profiling()

    def test_registered_backend_override(self, dummy_engine):
        engine = dummy_engine
        assert isinstance(engine.backend, DummyBackend)
        assert engine.backend.config is engine.config
        assert engine.backend.device_runtime is engine.device_runtime

        model = TinyModel()
        engine.model = engine.backend.prepare_model(model)

        assert engine.model is model
        assert engine.model._dummy_sharded

    def test_repr(self, dummy_engine):
        engine = dummy_engine
        assert "not setup" in repr(engine)
        assert "DummyBackend" in repr(engine)

        engine.model = engine.backend.prepare_model(TinyModel())

        assert "ready" in repr(engine)
        assert "not setup" not in repr(engine)


# ---------------------------------------------------------------------------
# resolve_gradient_accumulation_steps (--tt_gbs vs legacy SP compensation)
# ---------------------------------------------------------------------------


def _make_engine_for_ga(sp_size, world_size, gbs=None):
    """Build a bare TorchTurboEngine with mocked sp/dp info + config.gbs.

    Patches the engine module's ``dist`` reference so
    ``is_initialized`` is True and ``get_world_size`` reports ``world_size``.
    Returns ``(engine, restore)`` — call ``restore()`` in the test's ``finally``.
    """
    engine = object.__new__(TorchTurboEngine)
    engine.get_sp_info = lambda: {"sp_size": sp_size}
    engine.config = SimpleNamespace(
        gbs=gbs,
        mfu=SimpleNamespace(enabled=False),
    )
    cd = engine_core.dist
    orig_init, orig_world = cd.is_initialized, cd.get_world_size
    cd.is_initialized = lambda: True
    cd.get_world_size = lambda group=None: world_size

    def restore():
        cd.is_initialized = orig_init
        cd.get_world_size = orig_world

    return engine, restore


def test_resolve_ga_tt_gbs_derives_from_formula():
    # world=64, gbs=32, sp=4 -> ga = 32*4/64 = 2 (tt_gbs requires framework ga=1)
    engine, restore = _make_engine_for_ga(sp_size=4, world_size=64, gbs=32)
    try:
        assert engine.resolve_gradient_accumulation_steps(1) == 2
    finally:
        restore()


def test_resolve_ga_tt_gbs_matches_legacy_when_divisible():
    # When gbs == framework_ga * world (i.e. gbs/world is integer),
    # the tt_gbs formula and the legacy framework_ga*sp give the same answer.
    # world=8, sp=2; framework_ga=1 -> legacy ga=2; gbs such that gbs*sp/world=2 -> gbs=8
    engine, restore = _make_engine_for_ga(sp_size=2, world_size=8, gbs=8)
    try:
        assert engine.resolve_gradient_accumulation_steps(1) == 2
    finally:
        restore()


def test_resolve_ga_legacy_when_gbs_unset():
    engine, restore = _make_engine_for_ga(sp_size=4, world_size=64, gbs=None)
    try:
        # legacy: framework_ga * sp_size
        assert engine.resolve_gradient_accumulation_steps(3) == 12
    finally:
        restore()


def test_resolve_ga_legacy_sp1_noop():
    engine, restore = _make_engine_for_ga(sp_size=1, world_size=8, gbs=None)
    try:
        assert engine.resolve_gradient_accumulation_steps(5) == 5
    finally:
        restore()


def test_resolve_ga_tt_gbs_indivisible_raises():
    # 31*4 = 124, 124 % 64 != 0
    engine, restore = _make_engine_for_ga(sp_size=4, world_size=64, gbs=31)
    try:
        with pytest.raises(ValueError, match="divisible"):
            engine.resolve_gradient_accumulation_steps(1)
    finally:
        restore()


def test_resolve_ga_tt_gbs_minimum_one():
    # world=64, gbs=1, sp=4 -> 1*4/64 = 0 remainder 4 -> raises (indivisible),
    # so use gbs=16, sp=4, world=64 -> 16*4/64 = 1
    engine, restore = _make_engine_for_ga(sp_size=4, world_size=64, gbs=16)
    try:
        assert engine.resolve_gradient_accumulation_steps(1) == 1
    finally:
        restore()


@pytest.mark.parametrize("framework_ga", [2, 4, 8])
def test_resolve_ga_tt_gbs_rejects_explicit_framework_ga(framework_ga):
    engine, restore = _make_engine_for_ga(sp_size=4, world_size=64, gbs=32)
    try:
        with pytest.raises(ValueError, match="Cannot set both tt_gbs"):
            engine.resolve_gradient_accumulation_steps(framework_ga)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Tests: Checkpoint I/O
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hf_export", "save_kwargs", "expected_formats"),
    [
        ("final", {}, [CheckpointFormat.NATIVE]),
        ("final", {"final": True}, [CheckpointFormat.NATIVE, CheckpointFormat.HF]),
        ("every", {}, [CheckpointFormat.NATIVE, CheckpointFormat.HF]),
        (
            "every",
            {"final": True},
            [CheckpointFormat.NATIVE, CheckpointFormat.HF],
        ),
        ("final", {"format": CheckpointFormat.HF}, [CheckpointFormat.HF]),
        (
            "every",
            {"format": CheckpointFormat.NATIVE},
            [CheckpointFormat.NATIVE],
        ),
    ],
)
def test_save_model_writes_dcp_state_and_exports_hf_by_policy(
    hf_export, save_kwargs, expected_formats, tmp_path
):
    class ModelConfig:
        model_type = "test"

        def to_dict(self):
            return {"model_type": self.model_type}

    class CheckpointModel(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.config = ModelConfig()

        def get_weight_converter(self, parallel_context, model_path):
            return (parallel_context, model_path)

    register_backend("fsdp2", DummyBackend)
    engine = TorchTurboEngine({"checkpoint": {"hf_export": hf_export}})
    engine._checkpoint = CheckpointCoordinator(
        engine.config,
        engine.backend.create_checkpoint_manager(),
        parallel_context=engine.backend.get_parallel_context(),
        export_plugin_consumer=CheckpointExportPluginConsumer(),
    )
    engine.model_name_or_path = "/source"
    output_dir = str(tmp_path / "checkpoint-1")

    engine.save_model(CheckpointModel(), output_dir, **save_kwargs)

    assert [
        CheckpointFormat(call[0].removeprefix("save_"))
        for call in engine.checkpoint._manager.calls
    ] == expected_formats
    assert engine.checkpoint._manager is engine.backend.create_checkpoint_manager()


def test_save_model_rejects_a_string_format_before_writing(tmp_path):
    register_backend("fsdp2", DummyBackend)
    engine = TorchTurboEngine()
    engine._checkpoint = CheckpointCoordinator(
        engine.config,
        engine.backend.create_checkpoint_manager(),
        parallel_context=engine.backend.get_parallel_context(),
        export_plugin_consumer=CheckpointExportPluginConsumer(),
    )
    engine.model_name_or_path = "/source"

    with pytest.raises(TypeError, match="CheckpointFormat"):
        engine.save_model(nn.Linear(2, 2), str(tmp_path), format="native")

    assert engine.checkpoint._manager.calls == []


def test_checkpoint_loads_and_optimizer_state_delegate_to_the_backend(dummy_engine):
    class RecordingCoordinator:
        def __init__(self):
            self.calls = []

        def load_model(self, model, checkpoint_dir, *, source_path):
            self.calls.append(("load_model", checkpoint_dir, source_path))

        def save_optimizer_and_scheduler(
            self, model, optimizer, lr_scheduler, output_dir
        ):
            self.calls.append(
                ("save_optimizer_and_scheduler", optimizer, lr_scheduler, output_dir)
            )

        def load_optimizer_and_scheduler(
            self, model, optimizer, lr_scheduler, checkpoint_dir
        ):
            self.calls.append(
                ("load_optimizer_and_scheduler", optimizer, lr_scheduler, checkpoint_dir)
            )

    dummy_engine._checkpoint = RecordingCoordinator()
    dummy_engine.model_name_or_path = "/source"
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    dummy_engine.load_from_checkpoint(model, "/out/checkpoint-1")
    dummy_engine.save_optimizer_and_scheduler(
        model, optimizer, scheduler, "/out/checkpoint-1"
    )
    dummy_engine.load_optimizer_and_scheduler(
        model, optimizer, scheduler, "/out/checkpoint-1"
    )

    assert dummy_engine.checkpoint.calls == [
        ("load_model", "/out/checkpoint-1", "/source"),
        ("save_optimizer_and_scheduler", optimizer, scheduler, "/out/checkpoint-1"),
        ("load_optimizer_and_scheduler", optimizer, scheduler, "/out/checkpoint-1"),
    ]
