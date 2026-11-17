from __future__ import annotations

import pytest
import torch.nn as nn

from torchturbo.engine.model_fqn import build_model_fqn_map


def test_model_fqn_map_is_identity_for_an_unwrapped_model():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())

    fqn_map = build_model_fqn_map(model)

    assert dict(fqn_map.modules) == {"": "", "0": "0", "1": "1"}
    assert dict(fqn_map.parameters) == {
        "0.weight": "0.weight",
        "0.bias": "0.bias",
    }


def test_model_fqn_map_crosses_checkpoint_wrapper_explicitly():
    checkpoint = pytest.importorskip(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    )
    model = nn.Module()
    model.projection = checkpoint.checkpoint_wrapper(nn.Linear(2, 2))

    fqn_map = build_model_fqn_map(model)

    assert fqn_map.modules["projection"] == "projection"
    assert fqn_map.parameters["projection.weight"] == (
        "projection._checkpoint_wrapped_module.weight"
    )


def test_model_fqn_map_rejects_ambiguous_parameter_targets():
    checkpoint = pytest.importorskip(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    )

    class AmbiguousModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.direct = nn.Linear(2, 2)
            self.wrapped = checkpoint.checkpoint_wrapper(nn.Linear(2, 2))

        def named_children(self):
            yield "projection", self.direct
            yield "projection", self.wrapped

    with pytest.raises(ValueError, match="projection.weight.*resolves to both"):
        build_model_fqn_map(AmbiguousModule())


def test_model_fqn_map_crosses_peft_model_and_tuner_prefixes():
    peft = pytest.importorskip("peft")
    tuners_utils = pytest.importorskip("peft.tuners.tuners_utils")

    class Tuner(tuners_utils.BaseTuner):
        def _create_and_replace(self, *args, **kwargs):
            raise AssertionError("not used")

    tuner = Tuner.__new__(Tuner)
    nn.Module.__init__(tuner)
    tuner.model = nn.Linear(2, 2)

    model = peft.PeftModel.__new__(peft.PeftModel)
    nn.Module.__init__(model)
    model.base_model = tuner

    fqn_map = build_model_fqn_map(model)

    assert fqn_map.modules[""] == "base_model.model"
    assert fqn_map.parameters["weight"] == "base_model.model.weight"


def test_model_fqn_map_maps_tuner_layer_to_its_base_parameters():
    tuners_utils = pytest.importorskip("peft.tuners.tuners_utils")

    class TunerLayer(nn.Module, tuners_utils.BaseTunerLayer):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(2, 2)
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(2, 1, bias=False)}
            )

    model = nn.Module()
    model.projection = TunerLayer()

    fqn_map = build_model_fqn_map(model)

    assert fqn_map.modules["projection"] == "projection"
    assert fqn_map.parameters["projection.weight"] == (
        "projection.base_layer.weight"
    )
    assert not any("lora_A" in fqn for fqn in fqn_map.parameters.values())


def test_model_fqn_map_selects_only_active_modules_to_save_copy():
    other = pytest.importorskip("peft.utils.other")
    model = nn.Module()
    model.projection = other.ModulesToSaveWrapper(nn.Linear(2, 2), "default")

    fqn_map = build_model_fqn_map(model)

    assert fqn_map.modules["projection"] == (
        "projection.modules_to_save.default"
    )
    assert fqn_map.parameters["projection.weight"] == (
        "projection.modules_to_save.default.weight"
    )
    assert not any("original_module" in fqn for fqn in fqn_map.parameters.values())

    model.projection.enable_adapters(False)
    disabled_fqn_map = build_model_fqn_map(model)

    assert disabled_fqn_map.modules["projection"] == "projection.original_module"
    assert disabled_fqn_map.parameters["projection.weight"] == (
        "projection.original_module.weight"
    )


def test_model_fqn_map_crosses_trainable_tokens_wrapper():
    other = pytest.importorskip("peft.utils.other")

    model = nn.Module()
    model.embed_tokens = other.TrainableTokensWrapper(
        nn.Embedding(8, 4),
        "default",
        token_indices=[0, 1],
    )

    fqn_map = build_model_fqn_map(model)

    assert fqn_map.modules["embed_tokens"] == "embed_tokens"
    assert "embed_tokens.token_adapter" not in fqn_map.modules
    assert fqn_map.parameters["embed_tokens.weight"] == (
        "embed_tokens.token_adapter.base_layer.weight"
    )
    assert not any(
        "trainable_tokens_delta" in fqn
        for fqn in fqn_map.parameters.values()
    )


@pytest.mark.parametrize(
    ("checkpoint_outside_peft", "root_fqn", "prefix"),
    [
        (True, "", "_checkpoint_wrapped_module.base_model.model"),
        (False, "base_model.model", "base_model.model._checkpoint_wrapped_module"),
    ],
)
def test_model_fqn_map_composes_checkpoint_and_peft_wrappers(
    checkpoint_outside_peft,
    root_fqn,
    prefix,
):
    checkpoint = pytest.importorskip(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    )
    peft = pytest.importorskip("peft")
    tuners_utils = pytest.importorskip("peft.tuners.tuners_utils")
    other = pytest.importorskip("peft.utils.other")

    class Tuner(tuners_utils.BaseTuner):
        def _create_and_replace(self, *args, **kwargs):
            raise AssertionError("not used")

    class TunerLayer(nn.Module, tuners_utils.BaseTunerLayer):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Linear(2, 2, bias=False)
            self.lora_A = nn.ModuleDict(
                {"default": nn.Linear(2, 1, bias=False)}
            )

    modeling = nn.Module()
    modeling.projection = TunerLayer()
    modeling.head = other.ModulesToSaveWrapper(nn.Linear(2, 2), "default")

    tuner = Tuner.__new__(Tuner)
    nn.Module.__init__(tuner)
    tuner.model = (
        modeling
        if checkpoint_outside_peft
        else checkpoint.checkpoint_wrapper(modeling)
    )
    prepared = peft.PeftModel.__new__(peft.PeftModel)
    nn.Module.__init__(prepared)
    prepared.base_model = tuner
    if checkpoint_outside_peft:
        prepared = checkpoint.checkpoint_wrapper(prepared)

    fqn_map = build_model_fqn_map(prepared)

    assert fqn_map.modules[""] == root_fqn
    assert fqn_map.modules["projection"] == f"{prefix}.projection"
    assert fqn_map.parameters["projection.weight"] == (
        f"{prefix}.projection.base_layer.weight"
    )
    assert fqn_map.modules["head"] == (
        f"{prefix}.head.modules_to_save.default"
    )
    assert fqn_map.parameters["head.weight"] == (
        f"{prefix}.head.modules_to_save.default.weight"
    )
    assert len(set(fqn_map.modules.values())) == len(fqn_map.modules)
    assert len(set(fqn_map.parameters.values())) == len(fqn_map.parameters)
