"""Tests for the on-disk layout of FSDP2 checkpoints."""

import json
import os

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

import pytest
import torch

from torchturbo.engine.checkpoint import ComponentStatus
from torchturbo.engine.backends.fsdp2.checkpoint import layout as fsdp_layout
from torchturbo.engine.checkpoint import CheckpointFormat, layout


def _write_dcp(checkpoint_dir, name, *, complete, metadata=".metadata"):
    dcp_dir = checkpoint_dir / name
    dcp_dir.mkdir(parents=True)
    (dcp_dir / "__0_0.distcp").write_bytes(b"data")
    if complete:
        (dcp_dir / metadata).write_bytes(b"metadata")
    return dcp_dir


_TWO_SHARDS = (
    "model-00000-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
)


def _write_hf(checkpoint_dir, *, index=True, shards=_TWO_SHARDS):
    hf_dir = checkpoint_dir / layout.HF_MODEL_DIR
    hf_dir.mkdir(parents=True)
    for shard in shards:
        (hf_dir / shard).write_bytes(b"shard")
    if index:
        weight_map = {"a.weight": _TWO_SHARDS[0], "b.weight": _TWO_SHARDS[1]}
        (hf_dir / layout.HF_INDEX_FILE).write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    return hf_dir


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("missing", ComponentStatus.MISSING),
        ("no_metadata", ComponentStatus.INCOMPLETE),
        ("tmp_metadata", ComponentStatus.INCOMPLETE),
        ("global_metadata", ComponentStatus.COMPLETE),
        ("rank_metadata", ComponentStatus.COMPLETE),
    ],
)
def test_dcp_status_uses_the_metadata_file_as_commit_marker(tmp_path, setup, expected):
    dcp_dir = tmp_path / "model_distcp"
    if setup == "no_metadata":
        _write_dcp(tmp_path, "model_distcp", complete=False)
    elif setup == "tmp_metadata":
        _write_dcp(tmp_path, "model_distcp", complete=True, metadata=".metadata.tmp")
    elif setup == "global_metadata":
        _write_dcp(tmp_path, "model_distcp", complete=True)
    elif setup == "rank_metadata":
        _write_dcp(tmp_path, "model_distcp", complete=True, metadata="__0.metadata")

    assert fsdp_layout.dcp_status(str(dcp_dir)) is expected


def test_hf_status_requires_the_index_and_every_listed_shard(tmp_path):
    assert layout.hf_status(str(tmp_path / "missing")) is ComponentStatus.MISSING

    no_index = _write_hf(tmp_path / "no_index", index=False)
    assert layout.hf_status(str(no_index)) is ComponentStatus.INCOMPLETE

    missing_shard = _write_hf(tmp_path / "missing_shard", shards=_TWO_SHARDS[:1])
    assert layout.hf_status(str(missing_shard)) is ComponentStatus.INCOMPLETE

    complete = _write_hf(tmp_path / "complete")
    assert layout.hf_status(str(complete)) is ComponentStatus.COMPLETE

    corrupt = _write_hf(tmp_path / "corrupt")
    (corrupt / layout.HF_INDEX_FILE).write_text("{")
    assert layout.hf_status(str(corrupt)) is ComponentStatus.INCOMPLETE


@pytest.mark.parametrize(
    "index",
    [
        [],
        {},
        {"weight_map": []},
        {"weight_map": {}},
        {"weight_map": {"layer.weight": None}},
        {"weight_map": {"layer.weight": 1}},
    ],
)
def test_hf_status_rejects_invalid_index_schema(tmp_path, index):
    hf_dir = tmp_path / layout.HF_MODEL_DIR
    hf_dir.mkdir()
    (hf_dir / layout.HF_INDEX_FILE).write_text(json.dumps(index))

    assert layout.hf_status(str(hf_dir)) is ComponentStatus.INCOMPLETE


@pytest.mark.parametrize(
    ("dcp", "hf", "expected"),
    [
        ("complete", "missing", CheckpointFormat.NATIVE),
        ("complete", "complete", CheckpointFormat.NATIVE),
        ("missing", "complete", CheckpointFormat.HF),
    ],
)
def test_resolve_model_format_prefers_exact_dcp_state(tmp_path, dcp, hf, expected):
    if dcp == "complete":
        _write_dcp(tmp_path, fsdp_layout.MODEL_DCP_DIR, complete=True)
    if hf == "complete":
        _write_hf(tmp_path)

    assert layout.resolve_model_format(
        str(tmp_path),
        fsdp_layout.dcp_status(str(tmp_path / fsdp_layout.MODEL_DCP_DIR)),
        str(tmp_path / fsdp_layout.MODEL_DCP_DIR),
    ) == expected


def test_complete_dcp_ignores_an_interrupted_optional_hf_export(tmp_path):
    _write_dcp(tmp_path, fsdp_layout.MODEL_DCP_DIR, complete=True)
    _write_hf(tmp_path, index=False)

    assert layout.resolve_model_format(
        str(tmp_path),
        ComponentStatus.COMPLETE,
        str(tmp_path / fsdp_layout.MODEL_DCP_DIR),
    ) is CheckpointFormat.NATIVE


@pytest.mark.parametrize(
    ("dcp", "hf", "incomplete_dir"),
    [
        ("incomplete", "missing", fsdp_layout.MODEL_DCP_DIR),
        ("incomplete", "complete", fsdp_layout.MODEL_DCP_DIR),
        ("missing", "incomplete", layout.HF_MODEL_DIR),
    ],
)
def test_resolve_model_format_rejects_interrupted_saves(
    tmp_path, dcp, hf, incomplete_dir
):
    if dcp != "missing":
        _write_dcp(tmp_path, fsdp_layout.MODEL_DCP_DIR, complete=dcp == "complete")
    if hf == "complete":
        _write_hf(tmp_path)
    elif hf == "incomplete":
        _write_hf(tmp_path, index=False)

    with pytest.raises(RuntimeError, match=f"{incomplete_dir} are incomplete"):
        layout.resolve_model_format(
            str(tmp_path),
            fsdp_layout.dcp_status(str(tmp_path / fsdp_layout.MODEL_DCP_DIR)),
            str(tmp_path / incomplete_dir),
        )


def test_resolve_model_format_rejects_checkpoints_without_weights(tmp_path):
    _write_dcp(tmp_path, fsdp_layout.OPTIMIZER_DCP_DIR, complete=True)

    with pytest.raises(FileNotFoundError, match="No model weights"):
        layout.resolve_model_format(
            str(tmp_path), ComponentStatus.MISSING, str(tmp_path / "model")
        )


@pytest.mark.parametrize("output", ["source", "source/sft", "source_link/sft"])
def test_output_dir_in_the_source_model_is_rejected(tmp_path, output):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "source_link").symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be the source model directory"):
        layout.validate_output_dir(str(tmp_path / output), str(source))


@pytest.mark.parametrize("output", ["sibling", "."])
def test_output_dir_outside_the_source_model_is_accepted(tmp_path, output):
    source = tmp_path / "source"
    source.mkdir()

    layout.validate_output_dir(str(tmp_path / output), str(source))


def test_scheduler_state_round_trips_through_an_atomic_write(tmp_path):
    state = {"last_epoch": 4, "base_lrs": [0.1, 0.01]}
    checkpoint_dir = tmp_path / "checkpoint"

    layout.save_scheduler(state, str(checkpoint_dir))

    assert sorted(os.listdir(checkpoint_dir)) == [layout.SCHEDULER_FILE]
    assert layout.load_scheduler(str(checkpoint_dir)) == state


@pytest.mark.parametrize("contents", [None, b"", b"not a scheduler"])
def test_load_scheduler_rejects_missing_or_interrupted_state(tmp_path, contents):
    if contents is not None:
        (tmp_path / layout.SCHEDULER_FILE).write_bytes(contents)

    with pytest.raises(FileNotFoundError if contents is None else RuntimeError, match="missing|incomplete"):
        layout.load_scheduler(str(tmp_path))


def test_load_scheduler_rejects_a_truncated_legacy_save(tmp_path):
    scheduler_path = tmp_path / layout.SCHEDULER_FILE
    torch.save(
        {"last_epoch": 4},
        scheduler_path,
        _use_new_zipfile_serialization=False,
    )
    scheduler_path.write_bytes(scheduler_path.read_bytes()[:1])

    with pytest.raises(RuntimeError, match="incomplete"):
        layout.load_scheduler(str(tmp_path))


def test_load_scheduler_rejects_a_non_mapping_state(tmp_path):
    torch.save(["not", "a", "state dict"], tmp_path / layout.SCHEDULER_FILE)

    with pytest.raises(RuntimeError, match="expected a state dict"):
        layout.load_scheduler(str(tmp_path))
