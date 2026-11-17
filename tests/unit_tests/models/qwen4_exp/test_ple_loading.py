from __future__ import annotations

import json

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn

from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import (
    DistParamLoader,
)
from torchturbo.models.modules.cpu_resident import CPUResidentParameter
from torchturbo.models.qwen4_exp import ple as impl
from torchturbo.models.qwen4_exp.model import Qwen4ExpWeightConverter

from .test_ple import adapted, new_reference


def make_checkpoint(path, dtype=torch.float32, shard_sizes=None):
    reference = new_reference()
    rows = reference.ple_embedding.ngram_embedding.weight.detach().to(dtype)
    tensors = {}
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = rows.chunk(4) if shard_sizes is None else rows.split(shard_sizes)
    for index, shard in enumerate(shards):
        tensors[f"{prefix}.shard_{index}.weight"] = shard.clone()
    path.mkdir()
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": dict.fromkeys(tensors, "model.safetensors"),
            }
        )
    )
    layer = adapted(reference)
    layer.ple_embedding.ngram_embedding = impl.NGramEmbedding(
        torch.empty(rows.shape, device="meta", dtype=torch.float32)
    )
    layer.ple_embedding.ngram_embedding.weight.directory = path.parent
    decoder = nn.Module()
    decoder.ple = layer
    text = nn.Module()
    text.layers = nn.ModuleList([decoder])
    # Loading uses the original n-gram geometry, not a tensor being restored.
    from torchturbo.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    text.config = Qwen4ExpTextConfig(
        split_ngram_parts=4, make_ngram_vocab_size_divisible_by=8
    )
    return text, rows


def load_tables(text, path):
    from pathlib import Path

    from transformers import PretrainedConfig

    path = Path(path)
    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = text
    root.config = PretrainedConfig(text_config=text.config)
    root.all_tied_weights_keys = {}
    converter = Qwen4ExpWeightConverter(
        root,
        {"ep": {"ep_group": dist.group.WORLD, "ep_rank": 0, "ep_size": 1}},
        str(path),
    )
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    loader = object.__new__(DistParamLoader)
    loader.model = root
    loader.path = path
    loader.param_converter = converter
    loader.param_to_filename = weight_map
    loader._load_cpu_resident_weights()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_load_source_shards_into_unlinked_shared_table(
    tmp_path, node_group, dtype
):
    text, expected = make_checkpoint(tmp_path / "source", dtype)
    storage = tmp_path
    load_tables(text, tmp_path / "source")
    table = text.layers[0].ple.ple_embedding.ngram_embedding.weight
    assert table.dtype == dtype
    torch.testing.assert_close(table, expected)
    assert not list(storage.glob("torchturbo-cpu-resident-*"))
    embedding = text.layers[0].ple.ple_embedding.ngram_embedding
    parameters = dict(embedding.named_parameters())
    assert set(parameters) == {"weight"}
    assert all(
        isinstance(parameter, CPUResidentParameter) for parameter in parameters.values()
    )
    assert all(parameter.device.type == "cpu" for parameter in parameters.values())
    assert all(not parameter.requires_grad for parameter in parameters.values())
    torch.testing.assert_close(embedding.state_dict()["weight"], expected)


def test_load_source_shards_preserves_uneven_row_sizes(
    tmp_path, node_group
):
    num_rows = new_reference().ple_embedding.ngram_embedding.weight.shape[0]
    sizes = (1, 2, 3, num_rows - 6)
    text, expected = make_checkpoint(tmp_path / "source", shard_sizes=sizes)

    load_tables(text, tmp_path / "source")

    embedding = text.layers[0].ple.ple_embedding.ngram_embedding
    assert embedding.weight.shape[0] == sum(sizes)
    torch.testing.assert_close(embedding.weight, expected)
    assert not list(tmp_path.glob("torchturbo-cpu-resident-*"))


@pytest.mark.parametrize("invalid", ["row_count", "embedding_dim", "mixed_dtype"])
def test_invalid_source_shards_fail_before_mapping(
    tmp_path, node_group, invalid
):
    from safetensors.torch import load_file

    source = tmp_path / "source"
    text, _ = make_checkpoint(source)
    tensors = load_file(str(source / "model.safetensors"))
    name = next(iter(tensors))
    tensor = tensors[name]
    if invalid == "row_count":
        tensors[name] = tensor[:-1].clone()
    elif invalid == "embedding_dim":
        tensors[name] = tensor[:, :-1].clone()
    else:
        tensors[name] = tensor.double()
    save_file(tensors, str(source / "model.safetensors"))

    with pytest.raises(ValueError, match="PLE|shard|row"):
        load_tables(text, source)
    assert not list(tmp_path.glob("torchturbo-cpu-resident-*"))


def test_missing_source_shard_fails_without_leaking_file(
    tmp_path, node_group
):
    text, _ = make_checkpoint(tmp_path / "source")
    index_path = tmp_path / "source/model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop(next(iter(index["weight_map"])))
    index_path.write_text(json.dumps(index))
    with pytest.raises((ValueError, KeyError), match="shard_0"):
        load_tables(text, tmp_path / "source")
    assert not list(tmp_path.glob("torchturbo-cpu-resident-*"))


def test_source_free_materialization_allocates_registered_cpu_table(
    tmp_path, node_group
):
    text = nn.Module()
    decoder = nn.Module()
    decoder.ple = adapted(new_reference())
    rows = decoder.ple.ple_embedding.ngram_embedding.weight
    embedding = impl.NGramEmbedding(torch.empty_like(rows, device="meta"))
    decoder.ple.ple_embedding.ngram_embedding = embedding
    text.layers = nn.ModuleList([decoder])
    embedding.weight.directory = tmp_path

    embedding.weight.materialize_()

    assert embedding.weight.shape == rows.shape
    assert embedding.weight.dtype == rows.dtype
    assert all(parameter.device.type == "cpu" for parameter in embedding.parameters())
    assert all(not parameter.requires_grad for parameter in embedding.parameters())
    assert list(tmp_path.iterdir()) == []
