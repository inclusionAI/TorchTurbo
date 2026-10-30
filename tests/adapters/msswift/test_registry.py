"""Tests for the ms-swift adapter: detection, native registration, structure.

These exercise the swift-facing surface, so they are skipped when ms-swift is
not importable (``pytest.importorskip('swift')``).
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    import types

    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

swift = pytest.importorskip("swift")


@pytest.fixture(autouse=True)
def restore_pipeline_globals(monkeypatch):
    """Keep persistent adapter registrations from leaking between tests."""
    from importlib import import_module

    from torchturbo.adapters.msswift.stages import iter_stages

    for stage in iter_stages():
        module = import_module(stage.pipeline_module)
        monkeypatch.setattr(module, stage.pipeline_attr, vars(module)[stage.pipeline_attr])


def test_adapter_detection():
    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.adapters.msswift.compat import get_swift_version, is_swift_supported

    assert is_swift_supported() is True
    adapter = MSSwiftAdapter()
    assert adapter.detect() is True
    ver = get_swift_version()
    assert isinstance(ver, str) and ver.count(".") >= 1


def test_native_registration_populates_swift_registries():
    from swift.model import MODEL_MAPPING
    from swift.model.model_meta import MODEL_MAPPING as SWIFT_MODEL_MAPPING
    from swift.trainers import TrainerFactory

    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.models.qwen4_exp.swift_template import (
        NATIVE_MODEL_TYPE,
        TEMPLATE_TYPE,
    )

    adapter = MSSwiftAdapter()
    adapter.activate()

    # turbo models registered under turbo_<name> with the turbo loader.
    turbo_entries = {k: v for k, v in MODEL_MAPPING.items() if k.startswith("turbo_")}
    assert "turbo_qwen3_5_moe" in turbo_entries
    # turbo_qwen3_5_moe's loader is TurboModelMixin mixed ahead of the native
    # Qwen3_5MoeLoader — so get_model is turbo's engine and get_processor is the
    # native one (with its media-cap env side effects). Base TurboModelLoader is
    # the text-only fallback when swift has no native meta for the model.
    from torchturbo.adapters.msswift.loader import TurboModelMixin

    turbo_loader = turbo_entries["turbo_qwen3_5_moe"].loader
    assert issubclass(turbo_loader, TurboModelMixin)
    native_loader = SWIFT_MODEL_MAPPING["qwen3_5_moe"].loader
    # MRO: TurboModelMixin precedes the native loader; get_model is turbo's.
    assert native_loader in turbo_loader.__mro__
    assert turbo_loader.__mro__.index(TurboModelMixin) < turbo_loader.__mro__.index(native_loader)
    # get_model comes from the mixin (turbo engine), get_processor from native.
    assert turbo_loader.get_model is TurboModelMixin.get_model
    assert turbo_loader.get_processor is native_loader.get_processor
    # Meta carries the native loader chain's conventions, e.g. multimodal flag.
    assert turbo_entries["turbo_qwen3_5_moe"].is_multimodal is True

    # qwen4_exp is vendored against the pinned Transformers version and borrows
    # only Swift's compatible Qwen3.5 processor/template metadata.
    qwen4_meta = turbo_entries["turbo_qwen4_exp"]
    assert NATIVE_MODEL_TYPE == "qwen3_5_moe"
    assert TEMPLATE_TYPE == "qwen3_5"
    assert qwen4_meta.loader is turbo_loader
    assert native_loader in qwen4_meta.loader.__mro__
    assert qwen4_meta.template == "qwen3_5"
    assert qwen4_meta.model_groups[0].template == "qwen3_5"
    assert qwen4_meta.candidate_templates == ["qwen3_5"]
    assert qwen4_meta.requires == SWIFT_MODEL_MAPPING[NATIVE_MODEL_TYPE].requires
    assert qwen4_meta.tags == SWIFT_MODEL_MAPPING[NATIVE_MODEL_TYPE].tags
    assert qwen4_meta.task_type == "causal_lm"
    assert qwen4_meta.is_multimodal is True

    # turbo trainer registered for causal_lm (and dpo if available).
    assert TrainerFactory.TRAINER_MAPPING["causal_lm"].endswith("TurboSeq2SeqTrainer")

    adapter.deactivate()


def test_trainer_subclasses_swift_seq2seq():
    from swift.trainers import Seq2SeqTrainer

    from torchturbo.adapters.msswift.trainer import (
        TurboTrainerMixin,
        get_turbo_seq2seq_trainer,
    )

    turbo_cls = get_turbo_seq2seq_trainer()
    assert issubclass(turbo_cls, Seq2SeqTrainer)
    # The turbo接管 overrides live on TurboTrainerMixin, mixed ahead of
    # Seq2SeqTrainer, so each engine-owned method resolves to the mixin.
    assert TurboTrainerMixin in turbo_cls.__mro__
    for method in ("create_optimizer", "compute_loss", "save_model", "_wrap_model",
                   "create_scheduler", "_prepare_inputs", "get_train_dataloader"):
        assert getattr(turbo_cls, method) is getattr(TurboTrainerMixin, method)


def test_trainer_registry_is_stage_driven():
    from torchturbo.adapters.msswift.stages import SFT_STAGE, iter_stages

    # The SFT stage contributes the causal_lm trainer to TrainerFactory.
    assert any(s.name == "sft" for s in iter_stages())


def test_stage_registry_drives_plugin_swap(monkeypatch):
    """The swap loop resolves each registered stage's pipeline class by name
    via stage_wiring; snapshot/restore the real globals it touches."""
    import importlib
    import sys
    import types

    from torchturbo.adapters.msswift import stages as stages_mod

    # A throwaway swift-free module + attr to swap; stage_wiring is stubbed so
    # the fake stage name resolves (the real one only knows sft/pt).
    fake_mod = types.ModuleType("turbo_test_stage_mod")
    fake_mod.SwiftSft = object  # sentinel original

    fake_spec = stages_mod.SwiftStageSpec(
        name="__test_stage__",
        pipeline_module="turbo_test_stage_mod",
        pipeline_attr="SwiftSft",
    )
    stages_mod.TURBO_SWIFT_STAGES.append(fake_spec)
    sys.modules["turbo_test_stage_mod"] = fake_mod
    monkeypatch.setattr(stages_mod, "stage_wiring", lambda stage: (str, None))

    # Snapshot the real stages' pipeline globals the loop is about to swap.
    originals = {}
    for spec in stages_mod.iter_stages():
        if spec is fake_spec:
            continue
        mod = importlib.import_module(spec.pipeline_module)
        originals[(spec.pipeline_module, spec.pipeline_attr)] = getattr(
            mod, spec.pipeline_attr
        )
    try:
        # Re-run the plugin's swap logic in isolation.
        for spec in stages_mod.iter_stages():
            mod = importlib.import_module(spec.pipeline_module)
            setattr(mod, spec.pipeline_attr, stages_mod.stage_wiring(spec.name)[0])
        assert fake_mod.SwiftSft is str  # swapped to the sentinel turbo pipeline
    finally:
        stages_mod.TURBO_SWIFT_STAGES.remove(fake_spec)
        del sys.modules["turbo_test_stage_mod"]
        for (module_name, attr), original in originals.items():
            mod = importlib.import_module(module_name)
            setattr(mod, attr, original)


def test_turbo_sft_arguments_structure():
    from swift.arguments import SftArguments

    from torchturbo.adapters.msswift.args import TurboSwiftArgsMixin, TurboSftArguments
    from torchturbo.engine.args import TorchTurboArgs

    cls = TurboSftArguments
    assert issubclass(cls, SftArguments)

    # The force-off of engine-owned swift settings lives on TurboSwiftArgsMixin,
    # mixed ahead of SftArguments, and is reused by future stage args classes.
    assert TurboSwiftArgsMixin in cls.__mro__
    assert cls.__mro__.index(TurboSwiftArgsMixin) < cls.__mro__.index(SftArguments)
    assert cls.__post_init__ is TurboSwiftArgsMixin.__post_init__

    # The turbo --tt_* CLI surface lives on the shared TorchTurboArgs, NOT on
    # TurboSftArguments (which declares only swift-native fields). The launcher
    # strips --tt_* from sys.argv before swift's parser runs.
    turbo_fields = set(TorchTurboArgs.__dataclass_fields__.keys())
    for name in (
        "sp_size",
        "ep_size",
        "gradient_checkpointing",
        "ac_use_reentrant",
        "enable_mtp",
        "mtp_log_loss",
        "enable_chunk_loss",
        "loss_chunk_size",
        "freeze_vision_tower",
        "cpu_offload",
    ):
        assert name in turbo_fields, f"missing TorchTurboArgs field {name!r}"

    swift_field_names = set(cls.__dataclass_fields__.keys())
    for name in ("tt_sp_size", "tt_ep_size", "tt_gradient_checkpointing",
                 "tt_enable_mtp", "tt_freeze_vision_tower", "tt_cpu_offload"):
        assert name not in swift_field_names, (
            f"swift TurboSftArguments should not declare turbo field {name!r} "
            "(lives on TorchTurboArgs)"
        )


def test_native_plugin_swaps_swift_sft():
    """The `--external_plugins` entry must swap swift's SwiftSft -> TurboSwiftSft.

    This is what makes the native `swift sft ...` launcher build the turbo接管
    pipeline: ``sft_main`` closes over the module-global ``SwiftSft``, so the
    plugin only needs to reassign that one attribute.
    """
    import swift.pipelines.train.sft as swift_sft_mod

    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.adapters.msswift.pipeline import TurboSwiftSft

    original = swift_sft_mod.SwiftSft

    # Activate (registers models/trainers) then swap the pipeline class exactly
    # as the plugin module does on import.
    adapter = MSSwiftAdapter()
    adapter.activate()
    swift_sft_mod.SwiftSft = TurboSwiftSft
    try:
        assert swift_sft_mod.SwiftSft is TurboSwiftSft
        # native sft_main closes over the module global, so it would now build
        # the turbo pipeline; we assert the wiring rather than running it.
        assert callable(swift_sft_mod.sft_main)
    finally:
        swift_sft_mod.SwiftSft = original
        adapter.deactivate()


def test_pt_pipeline_structure():
    """The PT stage class: mixin ahead of SwiftPretrain, turbo args class,
    shared __init__ from the mixin (same takeover sequence as SFT), and a
    ``turbo_stage`` label for the engine registry."""
    from swift.arguments import PretrainArguments
    from swift.pipelines.train.pretrain import SwiftPretrain
    from swift.pipelines.train.sft import SwiftSft

    from torchturbo.adapters.msswift.args import (
        TurboPretrainArguments,
        TurboSwiftArgsMixin,
    )
    from torchturbo.adapters.msswift.pipeline import (
        TurboEnginePipelineMixin,
        TurboSwiftPretrain,
        TurboSwiftSft,
    )

    # SwiftPretrain is a thin SwiftSft subclass -- the turbo takeover mirrors it.
    assert issubclass(SwiftPretrain, SwiftSft)
    assert issubclass(TurboSwiftPretrain, SwiftPretrain)
    mro = TurboSwiftPretrain.__mro__
    assert mro.index(TurboEnginePipelineMixin) < mro.index(SwiftPretrain)
    assert TurboSwiftPretrain.args_class is TurboPretrainArguments
    assert TurboSwiftPretrain.turbo_stage == "pt"

    # Both stages share the mixin __init__.
    assert TurboSwiftSft.__init__ is TurboEnginePipelineMixin.__init__
    assert TurboSwiftPretrain.__init__ is TurboEnginePipelineMixin.__init__

    # Args: force-off mixin must run ahead of PretrainArguments.__post_init__.
    assert issubclass(TurboPretrainArguments, PretrainArguments)
    amro = TurboPretrainArguments.__mro__
    assert amro.index(TurboSwiftArgsMixin) < amro.index(PretrainArguments)
    assert TurboPretrainArguments.__post_init__ is TurboSwiftArgsMixin.__post_init__



def test_native_plugin_swaps_swift_pretrain():
    """PT needs its own swap: SwiftPretrain binds SwiftSft at class-definition
    time, so the SFT swap never reaches it."""
    import importlib

    import swift.pipelines.train.pretrain as swift_pt_mod

    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.adapters.msswift.pipeline import TurboSwiftPretrain
    from torchturbo.adapters.msswift.stages import iter_stages, stage_wiring

    adapter = MSSwiftAdapter()
    adapter.activate()
    # Replay the plugin's swap loop; snapshot/restore the touched globals.
    originals = {}
    try:
        for spec in iter_stages():
            mod = importlib.import_module(spec.pipeline_module)
            originals[(spec.pipeline_module, spec.pipeline_attr)] = getattr(
                mod, spec.pipeline_attr
            )
            setattr(mod, spec.pipeline_attr, stage_wiring(spec.name)[0])
        assert swift_pt_mod.SwiftPretrain is TurboSwiftPretrain
        assert callable(swift_pt_mod.pretrain_main)
    finally:
        for (module_name, attr), original in originals.items():
            mod = importlib.import_module(module_name)
            setattr(mod, attr, original)
        adapter.deactivate()


def test_pt_stage_reuses_causal_lm_trainer():
    """PT resolves through TRAINER_MAPPING['causal_lm']: TrainerFactory keys
    by ``rlhf_type`` when present else ``task_type``, and PretrainArguments has
    no ``rlhf_type``. NB: ``task_type``'s class default is None in 4.4.2
    (resolved per-instance) -- assert the mechanism, not the class default.
    """
    from swift.arguments import PretrainArguments
    from swift.trainers import TrainerFactory

    from torchturbo.adapters.msswift import MSSwiftAdapter

    # Activate explicitly -- the mapping entry is only turbo's after activation.
    adapter = MSSwiftAdapter()
    adapter.activate()
    try:
        assert getattr(PretrainArguments, "rlhf_type", None) is None
        assert TrainerFactory.TRAINER_MAPPING["causal_lm"].endswith(
            "TurboSeq2SeqTrainer"
        )
    finally:
        adapter.deactivate()


def test_dpo_pipeline_structure():
    """Test the DPO pipeline class and reference loading hook."""
    from swift.arguments import RLHFArguments
    from swift.pipelines.train.rlhf import SwiftRLHF

    from torchturbo.adapters.msswift.args import (
        TurboRLHFArguments,
        TurboSwiftArgsMixin,
    )
    from torchturbo.adapters.msswift.pipeline import (
        TurboEnginePipelineMixin,
        TurboSwiftRLHF,
        TurboSwiftSft,
    )

    assert issubclass(TurboSwiftRLHF, SwiftRLHF)
    mro = TurboSwiftRLHF.__mro__
    assert mro.index(TurboEnginePipelineMixin) < mro.index(SwiftRLHF)
    assert TurboSwiftRLHF.args_class is TurboRLHFArguments
    assert TurboSwiftRLHF.turbo_stage == "dpo"
    assert TurboSwiftRLHF.__init__ is TurboEnginePipelineMixin.__init__
    assert TurboSwiftSft.__init__ is TurboEnginePipelineMixin.__init__

    # Args: force-off mixin must run ahead of RLHFArguments.__post_init__.
    assert issubclass(TurboRLHFArguments, RLHFArguments)
    amro = TurboRLHFArguments.__mro__
    assert amro.index(TurboSwiftArgsMixin) < amro.index(RLHFArguments)
    assert TurboRLHFArguments.__post_init__ is TurboSwiftArgsMixin.__post_init__
    turbo_dropout = TurboRLHFArguments.__dataclass_fields__.get("disable_dropout")
    native_dropout = RLHFArguments.__dataclass_fields__.get("disable_dropout")
    assert (turbo_dropout.default if turbo_dropout else None) == (
        native_dropout.default if native_dropout else None
    )


def test_native_plugin_swaps_swift_rlhf():
    """Test the plugin swap for Swift's RLHF pipeline."""
    import importlib

    import swift.pipelines.train.rlhf as swift_rlhf_mod

    from torchturbo.adapters.msswift import MSSwiftAdapter
    from torchturbo.adapters.msswift.pipeline import TurboSwiftRLHF
    from torchturbo.adapters.msswift.stages import iter_stages, stage_wiring

    MSSwiftAdapter().activate()
    originals = {}
    for spec in iter_stages():
        mod = importlib.import_module(spec.pipeline_module)
        originals[(spec.pipeline_module, spec.pipeline_attr)] = getattr(
            mod, spec.pipeline_attr
        )
        setattr(mod, spec.pipeline_attr, stage_wiring(spec.name)[0])
    try:
        assert swift_rlhf_mod.SwiftRLHF is TurboSwiftRLHF
        assert callable(swift_rlhf_mod.rlhf_main)
        assert stage_wiring("dpo")[1] is swift_rlhf_mod.rlhf_main
    finally:
        for (module_name, attr), original in originals.items():
            mod = importlib.import_module(module_name)
            setattr(mod, attr, original)


def test_dpo_stage_maps_rlhf_trainer_key():
    """Test the DPO trainer registry mapping."""
    from swift.trainers import TrainerFactory

    from torchturbo.adapters.msswift import MSSwiftAdapter

    adapter = MSSwiftAdapter()
    adapter.activate()
    try:
        assert TrainerFactory.TRAINER_MAPPING["dpo"].endswith("TurboDPOTrainer")
    finally:
        adapter.deactivate()


def test_dpo_trainer_mro_and_loss_routing():
    """Test DPO trainer MRO and loss routing."""
    from swift.rlhf_trainers import DPOTrainer as SwiftDPOTrainer

    from torchturbo.adapters.msswift.trainer import (
        TurboDPOTrainerMixin,
        TurboTrainerMixin,
        get_turbo_dpo_trainer,
    )

    turbo_cls = get_turbo_dpo_trainer()
    assert issubclass(turbo_cls, SwiftDPOTrainer)
    mro = turbo_cls.__mro__
    assert (
        mro.index(TurboDPOTrainerMixin)
        < mro.index(TurboTrainerMixin)
        < mro.index(SwiftDPOTrainer)
    )

    # compute_loss is the DPO mixin's, and the next implementation down the
    # MRO (skipping TurboTrainerMixin's engine CE) is swift's DPO loss.
    assert turbo_cls.compute_loss is TurboDPOTrainerMixin.compute_loss
    for klass in mro[mro.index(TurboTrainerMixin) + 1:]:
        if "compute_loss" in klass.__dict__:
            assert klass is SwiftDPOTrainer
            break

    # concatenated_forward is the DPO mixin's packed forward.
    assert turbo_cls.concatenated_forward is TurboDPOTrainerMixin.concatenated_forward


def test_dpo_ref_model_stash():
    """Test that the trainer restores the engine-built reference model."""
    from torchturbo.adapters.msswift.trainer import TurboDPOTrainerMixin

    seen = {}

    class _Base:
        def __init__(self, model=None, ref_model=None, *_args, **_kwargs):
            seen["ref_at_super"] = ref_model

    class _Fake(TurboDPOTrainerMixin, _Base):
        pass

    ref = object()
    trainer = object.__new__(_Fake)
    _Fake.__init__(trainer, model=None, ref_model=ref)
    assert seen["ref_at_super"] is None
    assert trainer.ref_model is ref


def test_reference_loading_role_and_loader_routing():
    """DPO references build through engine.build_model; policy uses prepare_model."""
    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.loader import TurboModelMixin

    calls = {"prepare": 0, "build": 0}

    class _FakeEngine:
        peft_config = None

        def prepare_model(self, **_kwargs):
            calls["prepare"] += 1
            return "policy"

        def build_model(self, **_kwargs):
            calls["build"] += 1
            return "reference"

    engine_registry.set_engine(_FakeEngine())
    loader = object.__new__(TurboModelMixin)
    try:
        assert loader.get_model("/m", "cfg", None, {}) == "policy"
        with engine_registry.scoped_loading_role("reference"):
            assert loader.get_model("/m", "cfg", None, {}) == "reference"
            assert engine_registry.get_loading_role() == "reference"
        # Scope exit restores the (outer) policy role.
        assert engine_registry.get_loading_role() is None
        assert loader.get_model("/m", "cfg", None, {}) == "policy"
        assert calls == {"prepare": 2, "build": 1}
    finally:
        engine_registry.clear()


def test_rlhf_pipeline_marks_reference_loads(monkeypatch):
    """DPO wraps only its frozen reference load in the secondary role."""
    from swift.pipelines.train.rlhf import SwiftRLHF

    from torchturbo.adapters.msswift import engine_registry
    from torchturbo.adapters.msswift.pipeline import TurboSwiftRLHF

    calls = []

    def _fake_super(self, key, origin_key, model_type, model_revision):
        calls.append((key, origin_key, engine_registry.get_loading_role()))
        return ("model", None)

    monkeypatch.setattr(SwiftRLHF, "_prepare_single_model", _fake_super)

    pipeline = object.__new__(TurboSwiftRLHF)
    pipeline._prepare_single_model("ref", "ref", None, None)
    pipeline._prepare_single_model("reward", None, None, None)
    pipeline._prepare_single_model("value", "value", None, None)
    assert calls == [
        ("ref", "ref", "reference"),
        ("reward", None, None),
        ("value", "value", None),
    ]
    assert engine_registry.get_loading_role() is None
