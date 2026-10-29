"""Real process-group PLE execution and shared checkpoint table loading."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from functools import partial
from itertools import pairwise
from pathlib import Path
from types import MethodType

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torchturbo.models.modules.attention.ulysses import (
    SequenceParallelInputSlicingSpec,
    patch_forward_for_sp_input_slicing,
)
from torchturbo.models.modules.cpu_resident import CPUResidentParameter
from torchturbo.models.qwen4_exp import ple as impl
from torchturbo.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from torchturbo.utils.flash_attn_utils import cu_seqlens_to_segment_ids
from torchturbo.utils.torch_utils import torch_version

from .test_ple import adapted, new_reference, row_ids
from .test_ple_loading import load_tables, make_checkpoint


def _init_process_group(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def _packed_reference(reference, hidden, ids, boundaries):
    outputs = [
        reference(hidden[:, start:stop], ids[:, start:stop], None)
        for start, stop in pairwise(boundaries.tolist())
    ]
    if boundaries[-1] < ids.shape[1]:
        outputs.append(hidden[:, boundaries[-1] :] * 0)
    return torch.cat(outputs, dim=1)


class _CheckpointText(nn.Module):
    """Route the production SP input wrapper through a checkpointed PLE."""

    def __init__(self, layer, reentrant: bool):
        super().__init__()
        decoder = nn.Module()
        decoder.ple = layer
        self.layers = nn.ModuleList([decoder])
        self.reentrant = reentrant

    def forward(self, inputs_embeds, ple_input_ids, **kwargs):
        fn = partial(self.layers[0].ple, ple_inputs=ple_input_ids, past_key_values=None)
        return checkpoint(fn, inputs_embeds, use_reentrant=self.reentrant)


def _check_sp(
    group: dist.ProcessGroup,
    group_id: int,
    length: int,
    boundaries: tuple[int, ...],
    reentrant: bool | None = None,
) -> None:
    reference = new_reference().double()
    layer = adapted(reference)
    assert torch.count_nonzero(layer.conv1d.weight) > 0
    sp_size = dist.get_world_size(group)
    sp_rank = dist.get_rank(group)
    total = length * sp_size
    local_slice = slice(sp_rank * length, (sp_rank + 1) * length)
    ids = (torch.arange(total)[None] % 7) + 3
    if total > 2:
        ids[0, 1] = 2
    if total > 8:
        ids[0, 7] = 2
    cu = torch.tensor(boundaries, dtype=torch.int32)
    torch.manual_seed(73 + group_id)
    hidden = torch.randn(1, total, 16, dtype=torch.double, requires_grad=True)
    probe = torch.randn_like(hidden)
    expected = _packed_reference(reference, hidden, ids, cu)
    expected.backward(probe)
    layer.forward = MethodType(
        partial(impl.ple_forward, sp_group=group, sp_size=sp_size, sp_rank=sp_rank),
        layer,
    )

    if reentrant is None:
        local_hidden = hidden[:, local_slice].detach().clone().requires_grad_()
        segments = cu_seqlens_to_segment_ids(cu, total)
        indices = row_ids(reference.ple_embedding, ids[0], segments[0])
        rows = layer.ple_embedding.ngram_embedding(indices[local_slice])
        batch = impl.PLEBatch((rows.flatten(-2)[None],), segments)
        actual = layer(local_hidden, batch, None)
    else:
        text = _CheckpointText(layer, reentrant)
        spec = SequenceParallelInputSlicingSpec(text, _CheckpointText.forward)
        patch_forward_for_sp_input_slicing(spec, group, sp_size, sp_rank)
        impl.install_ple_prefetch(text, sp_size=sp_size, sp_rank=sp_rank)
        local_hidden = hidden.detach().clone().requires_grad_()
        actual = text(inputs_embeds=local_hidden, ple_input_ids=ids, cu_seq_lens_q=cu)

    context = (
        f"group={group_id}, rank={sp_rank}, length={length}, "
        f"cu={boundaries}, checkpoint={reentrant}"
    )

    def message(detail: str) -> str:
        return f"{context}\n{detail}"

    torch.testing.assert_close(actual, expected[:, local_slice], msg=message)
    actual.backward(probe[:, local_slice])
    actual_hidden_grad = (
        local_hidden.grad if reentrant is None else local_hidden.grad[:, local_slice]
    )
    # The upstream RMSNorm accumulates in FP32 even with FP64 parameters.
    torch.testing.assert_close(
        actual_hidden_grad,
        hidden.grad[:, local_slice],
        atol=1e-6,
        rtol=1e-5,
        msg=message,
    )
    reference_params = dict(reference.named_parameters())
    for name, parameter in layer.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, (context, name)
            continue
        assert parameter.grad is not None, (context, name)
        dist.all_reduce(parameter.grad, group=group)
        torch.testing.assert_close(
            parameter.grad,
            reference_params[name].grad,
            atol=1e-6,
            rtol=1e-5,
            msg=message,
        )


def _run_sp(rank: int, init_file: str) -> None:
    _init_process_group(rank, 4, init_file)
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        group_id = rank % 2
        group = groups[group_id]
        # H=9: long shards, short shards, and a completely padded rank.
        cases = ((12, (0, 5, 23)), (4, (0, 3, 8)), (12, (0, 3, 10)))
        for length, boundaries in cases:
            _check_sp(group, group_id, length, boundaries)
        for reentrant in (False, True):
            _check_sp(group, group_id, 4, (0, 3, 7), reentrant)
        # Histories span more than one preceding rank when L < H.
        for length, boundaries in ((3, (0, 1, 11)), (1, (0, 4))):
            _check_sp(dist.group.WORLD, 2, length, boundaries)
    finally:
        dist.destroy_process_group()


def _new_loading_text():
    decoder = nn.Module()
    decoder.ple = adapted(new_reference())
    rows = decoder.ple.ple_embedding.ngram_embedding.weight
    decoder.ple.ple_embedding.ngram_embedding = impl.NGramEmbedding(
        torch.empty_like(rows, device="meta")
    )
    text = nn.Module()
    text.layers = nn.ModuleList([decoder])
    text.config = Qwen4ExpTextConfig(
        split_ngram_parts=4, make_ngram_vocab_size_divisible_by=8
    )
    return text


def _run_loading(rank: int, init_file: str, source: str, storage: str, fail: bool) -> None:
    _init_process_group(rank, 2, init_file)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    try:
        text = _new_loading_text()
        text.layers[0].ple.ple_embedding.ngram_embedding.weight.directory = Path(storage)
        if fail and rank == 1:

            from torchturbo.engine.backends.fsdp2.checkpoint import weight_loader

            def fail_write(*args, **kwargs):
                raise OSError("injected PLE shard write failure")

            weight_loader.safe_open = fail_write
        if fail:
            with pytest.raises(RuntimeError, match="injected PLE shard write failure"):
                load_tables(text, source)
        else:
            load_tables(text, source)
            table = text.layers[0].ple.ple_embedding.ngram_embedding.weight
            expected = new_reference().ple_embedding.ngram_embedding.weight.bfloat16()
            assert table.dtype == torch.bfloat16
            torch.testing.assert_close(table, expected)
            embedding = text.layers[0].ple.ple_embedding.ngram_embedding
            assert len(list(embedding.parameters())) == 1
            assert all(parameter.device.type == "cpu" for parameter in embedding.parameters())
            assert all(not parameter.requires_grad for parameter in embedding.parameters())
            torch.testing.assert_close(embedding.state_dict()["weight"], expected)
            dist.barrier()
            if rank == 0:
                table[0].fill_(13)
            dist.barrier()
            torch.testing.assert_close(table[0], torch.full_like(table[0], 13))
        dist.barrier()
        assert list(Path(storage).iterdir()) == []
    finally:
        dist.destroy_process_group()


def _short_conv_fp32(self, hidden_states, past_key_values):
    # CPU BF16 Conv1d rounds partial input gradients. Use the same FP32
    # accumulation oracle as the single-process convolution tests.
    output = F.conv1d(
        F.pad(hidden_states.transpose(1, 2).float(), (self.short_conv_state_len, 0)),
        self.conv1d.weight.float(),
        dilation=self.conv1d.dilation,
        groups=self.conv1d.groups,
    )
    return F.silu(output.to(hidden_states.dtype)).transpose(1, 2)


def _run_fsdp(rank: int, init_file: str) -> None:
    _init_process_group(rank, 2, init_file)
    try:
        reference = new_reference().bfloat16()
        reference._short_conv = MethodType(_short_conv_fp32, reference)
        layer = adapted(new_reference())
        embedding = layer.ple_embedding.ngram_embedding
        table = embedding.weight
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        fully_shard(
            layer,
            mesh=mesh,
            ignored_params=set(layer.ple_embedding.ngram_embedding.parameters()),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        assert table.device.type == "cpu"
        assert table.dtype == torch.float32
        assert all(
            isinstance(parameter, CPUResidentParameter)
            for parameter in embedding.parameters()
        )
        assert all(not parameter.requires_grad for parameter in embedding.parameters())
        assert set(embedding.state_dict()) == {"weight"}
        ids = torch.tensor([[3, 2, 5, 6, 4, 7, 2, 8, 3, 0]])
        cu = torch.tensor([0, 4, 9], dtype=torch.int32)
        segments = cu_seqlens_to_segment_ids(cu, ids.shape[1])
        indices = row_ids(layer.ple_embedding, ids[0], segments[0])
        rows = embedding.gather(indices.flatten()).view(1, ids.shape[1], -1)
        batch = impl.PLEBatch((rows,), segments)
        assert batch.rows[0].dtype == torch.float32
        torch.manual_seed(91 + rank)
        hidden = torch.randn(1, ids.shape[1], 16, requires_grad=True)
        reference_hidden = hidden.detach().bfloat16().requires_grad_()
        probe = torch.randn_like(reference_hidden)
        expected = _packed_reference(reference, reference_hidden, ids, cu)
        actual = layer(hidden, batch, None)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
        actual.backward(probe)
        expected.backward(probe)
        torch.testing.assert_close(
            hidden.grad, reference_hidden.grad.float(), atol=0.015, rtol=0.015
        )
        reference_params = dict(reference.named_parameters())
        for name, parameter in layer.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None, name
                continue
            reference_grad = reference_params[name].grad.float()
            dist.all_reduce(reference_grad)
            reference_grad /= 2
            torch.testing.assert_close(
                parameter.grad.full_tensor(), reference_grad, atol=0.03, rtol=0.03
            )
        assert table.dtype == torch.float32
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
def test_ple_outputs_and_gradients_across_real_sp_groups(tmp_path) -> None:
    mp.spawn(_run_sp, args=(str(tmp_path / "ple-sp"),), nprocs=4, join=True)


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
@pytest.mark.parametrize("fail", [False, True], ids=["shared-rows", "peer-write-error"])
def test_shared_loading_across_local_processes(tmp_path, fail: bool) -> None:
    source = tmp_path / "source"
    make_checkpoint(source, torch.bfloat16)
    storage = tmp_path / "shared"
    storage.mkdir()
    mp.spawn(
        _run_loading,
        args=(str(tmp_path / "ple-load"), str(source), str(storage), fail),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()) or torch_version() < (2, 10, 0),
    reason="requires Gloo and CPU FSDP support in PyTorch 2.10",
)
def test_fsdp_mixed_precision_keeps_cpu_table_and_averages_gradients(tmp_path) -> None:
    mp.spawn(_run_fsdp, args=(str(tmp_path / "ple-fsdp"),), nprocs=2, join=True)
