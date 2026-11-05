"""SP and packed-sample tests for the Qwen3.8 CANN LIv2 adapter."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextConfig,
    Qwen4ExpTextQSAIndexer,
)
from torchturbo.models.qwen4_exp import npu as qwen_npu
from torchturbo.models.qwen4_exp import qsa as qwen_qsa
from torchturbo.models.qwen4_exp.npu import qsa_indexer_forward_cann
from torchturbo.models.qwen4_exp.qsa import qsa_indexer_forward

try:
    import torch_npu  # noqa: F401
    import cann_ops_transformer_qwen38liv2  # noqa: F401
except ImportError:
    pytest.skip(
        "CANN LIv2 test requires torch_npu and the operator package",
        allow_module_level=True,
    )


def _config():
    return Qwen4ExpTextConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=16,
        indexer_compress_ratio=4,
    )


def _make_indexer(device, config=None, seed=42):
    torch.manual_seed(seed)
    indexer = Qwen4ExpTextQSAIndexer(config or _config(), 0)
    indexer = indexer.to(device=device, dtype=torch.bfloat16)
    for parameter in indexer.parameters():
        parameter.requires_grad_(False)
    return indexer.eval()


def _rope_tables(length, dim, device):
    positions = torch.arange(length, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = torch.outer(positions, inv_freq)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).unsqueeze(0)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).unsqueeze(0)
    return cos, sin


def _cu(lengths, device):
    offsets = torch.tensor(lengths).cumsum(0).tolist()
    return torch.tensor([0, *offsets], dtype=torch.int32, device=device)


def _full_inputs(lengths, device):
    hidden = torch.randn(1, sum(lengths), 64, dtype=torch.bfloat16, device=device)
    return hidden, _rope_tables(sum(lengths), 128, device), _cu(lengths, device)


def _check_rank(lengths, sp_size, sp_rank, hidden, rope, cu, device, indexer):
    local_length = hidden.shape[1] // sp_size
    start = sp_rank * local_length
    local_hidden = hidden[:, start : start + local_length].contiguous()
    local_rope = tuple(table[:, start : start + local_length].contiguous() for table in rope)
    cann = qsa_indexer_forward_cann(
        indexer, local_hidden, local_rope, cu, sp_size=sp_size, sp_rank=sp_rank
    )
    qwen_qsa._ENABLE_NPU_LI = False
    reference = qsa_indexer_forward(
        indexer, local_hidden, local_rope, cu, sp_size=sp_size, sp_rank=sp_rank
    )
    torch.npu.synchronize()
    ratio = indexer.compress_ratio
    for row in range(local_length):
        position = start + row
        segment_start = cu.tolist()[0]
        for stop in cu.tolist()[1:]:
            if stop > position:
                break
            segment_start = stop
        local_position = position - segment_start
        complete_blocks = (local_position + 1) // ratio
        tail_start = segment_start + complete_blocks * ratio
        expected_tail = list(range(tail_start, position + 1))
        num_blocks = min(complete_blocks, indexer.block_topk)
        cann_tokens = [token for token in cann[0, row].tolist() if token >= 0]
        reference_tokens = [token for token in reference[0, row].tolist() if token >= 0]
        assert len(cann_tokens) == num_blocks * ratio + len(expected_tail)
        assert cann_tokens[num_blocks * ratio :] == expected_tail
        for block in range(num_blocks):
            selected = cann_tokens[block * ratio : (block + 1) * ratio]
            block_start = segment_start + (selected[0] - segment_start) // ratio * ratio
            assert selected == list(range(block_start, block_start + ratio))
        assert set(cann_tokens) == set(reference_tokens), (
            f"selection mismatch at sp={sp_size} rank={sp_rank} "
            f"lengths={lengths}, row={row}"
        )


def _mock_full_gather(monkeypatch, indexer, hidden, rope):
    _, raw_keys = indexer.index_qk_proj(hidden).split(
        (indexer.index_n_heads * indexer.index_head_dim, indexer.index_head_dim), dim=-1
    )
    full = torch.cat((raw_keys, *rope), dim=-1)

    def gather(tensor, **kwargs):
        return full.movedim(1, 0).contiguous()

    monkeypatch.setattr(qwen_npu, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(qwen_qsa, "gather_from_sequence_parallel_region", gather)


@pytest.fixture(autouse=True)
def _force_torch_reference(monkeypatch):
    monkeypatch.setattr(qwen_qsa, "_ENABLE_NPU_LI", False)


@pytest.mark.parametrize("lengths", [[128], [128, 64], [128, 66, 64], [1, 2, 3, 122]])
@pytest.mark.parametrize(
    "sp_size,sp_rank",
    [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)],
)
def test_cann_matches_torch_for_every_rank(lengths, sp_size, sp_rank, monkeypatch):
    torch.npu.set_device(0)
    if sum(lengths) % sp_size:
        pytest.skip("The test shape is not divisible by this SP size")
    hidden, rope, cu = _full_inputs(lengths, "npu")
    if sp_size > 1:
        indexer = _make_indexer("npu")
        _mock_full_gather(monkeypatch, indexer, hidden, rope)
    else:
        indexer = _make_indexer("npu")
    _check_rank(lengths, sp_size, sp_rank, hidden, rope, cu, "npu", indexer)


def test_empty_rank_and_topk_boundary(monkeypatch):
    torch.npu.set_device(0)
    hidden, rope, cu = _full_inputs([128], "npu")
    empty_cu = torch.tensor([0, 64, 64], dtype=torch.int32, device="npu")
    monkeypatch.setattr(
        qwen_npu, "gather_from_sequence_parallel_region", lambda tensor, **kwargs: tensor
    )
    actual = qsa_indexer_forward_cann(
        _make_indexer("npu"), hidden[:, 64:], tuple(t[:, 64:] for t in rope),
        empty_cu, sp_size=2, sp_rank=1,
    )
    assert torch.equal(actual, torch.full_like(actual, -1))

    monkeypatch.setattr(
        qwen_npu,
        "gather_from_sequence_parallel_region",
        lambda tensor, **kwargs: tensor,
    )
    monkeypatch.setattr(
        qwen_qsa,
        "gather_from_sequence_parallel_region",
        lambda tensor, **kwargs: tensor,
    )


def test_production_shape_sp2(monkeypatch):
    torch.npu.set_device(0)
    config = _config()
    config.indexer_budget = 2048
    indexer = _make_indexer("npu", config)
    hidden, rope, cu = _full_inputs([256], "npu")
    _mock_full_gather(monkeypatch, indexer, hidden, rope)
    for rank in (0, 1):
        _check_rank([256], 2, rank, hidden, rope, cu, "npu", indexer)


def _distributed_worker(rank, world_size, port, lengths):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    try:
        torch.npu.set_device(rank)
        hidden, rope, cu = _full_inputs(lengths, "npu")
        _check_rank(
            lengths, world_size, rank, hidden, rope, cu, "npu", _make_indexer("npu")
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("lengths,sp_size", [([128, 64], 2), ([128, 66, 64], 2)])
def test_cann_matches_torch_with_real_sp_all_gather(lengths, sp_size):
    if dist.is_available() and dist.is_initialized():
        pytest.skip("Run the distributed test in a fresh process")
    torch.npu.set_device(0)
    port = 29631 + sum(lengths) % 1000
    mp.start_processes(
        _distributed_worker,
        args=(sp_size, port, lengths),
        nprocs=sp_size,
        join=True,
        start_method="spawn",
    )
