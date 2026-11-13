"""Tests for the stage registry: SFT + PT specs, idempotent register.

Swift-free (the pipeline factories import lazily), matching the msswift test
suite's no-swift conventions.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    import types

    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg


def test_sft_and_pt_stages_registered():
    from torchturbo.adapters.msswift.stages import iter_stages

    names = [spec.name for spec in iter_stages()]
    assert "sft" in names
    assert "pt" in names
    assert "dpo" in names


def test_sft_stage_spec_fields():
    from torchturbo.adapters.msswift.stages import SFT_STAGE

    assert SFT_STAGE.pipeline_module == "swift.pipelines.train.sft"
    assert SFT_STAGE.pipeline_attr == "SwiftSft"
    # SFT owns the causal_lm trainer mapping (PT reuses it).


def test_pt_stage_spec_fields():
    from torchturbo.adapters.msswift.stages import PT_STAGE

    assert PT_STAGE.pipeline_module == "swift.pipelines.train.pretrain"
    assert PT_STAGE.pipeline_attr == "SwiftPretrain"


def test_dpo_stage_spec_fields():
    from torchturbo.adapters.msswift.stages import DPO_STAGE, TURBO_TRAINERS

    assert DPO_STAGE.pipeline_module == "swift.pipelines.train.rlhf"
    assert DPO_STAGE.pipeline_attr == "SwiftRLHF"
    # The DPO stage is keyed by swift's rlhf_type, not task_type.
    assert TURBO_TRAINERS["dpo"].endswith("TurboDPOTrainer")


def test_pt_stage_swaps_its_own_pipeline_global():
    """PT needs its own stage: ``SwiftPretrain`` binds ``SwiftSft`` at
    class-definition time, so the SFT swap never reaches it."""
    from torchturbo.adapters.msswift.stages import SFT_STAGE, PT_STAGE

    assert SFT_STAGE.pipeline_module != PT_STAGE.pipeline_module
    assert SFT_STAGE.pipeline_attr != PT_STAGE.pipeline_attr


def test_register_stage_is_idempotent_on_name():
    from torchturbo.adapters.msswift.stages import (
        PT_STAGE,
        SwiftStageSpec,
        iter_stages,
        register_stage,
    )

    before = len(iter_stages())
    register_stage(PT_STAGE)  # same name -> no-op
    # A same-named clone is rejected: first registration wins.
    clone = SwiftStageSpec(
        name=PT_STAGE.name,
        pipeline_module="some.other.module",
        pipeline_attr="Other",
    )
    register_stage(clone)
    assert len(iter_stages()) == before
    by_name = {s.name: s for s in iter_stages()}
    assert by_name["pt"].pipeline_module == PT_STAGE.pipeline_module
