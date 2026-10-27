import pytest
import torch

from torchturbo.ops.packed_params import build_packed_params
from torchturbo.utils.functools import cache


def _gdn_module():
    """Import the fla_npu-backed provider; these tests need an NPU container."""
    pytest.importorskip("fla_npu")
    from torchturbo.ops.providers.npu import _fla_gdn as gdn

    return gdn


def test_packed_params_builds_reusable_offsets_and_chunk_indices():
    cu_seqlens = torch.tensor([0, 65, 128], dtype=torch.int32)

    packed = build_packed_params(cu_seqlens)

    assert packed.cu_seqlens.dtype == torch.int64
    assert packed.cu_seqlens_list == [0, 65, 128]
    assert tuple(packed.chunk_indices["16"].shape) == (9, 2)
    assert packed.chunk_indices_list["64"] == [0, 0, 0, 1, 1, 0]


def test_packed_params_cache_is_owned_by_the_caller():
    calls = []

    @cache
    def get_packed(value):
        calls.append(value)
        return build_packed_params(value)

    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int64)
    packed1 = get_packed(cu_seqlens)
    packed2 = get_packed(cu_seqlens)

    assert packed1 is packed2
    assert calls == [cu_seqlens]


def test_causal_conv_consumes_prevalidated_offsets_without_host_conversion():
    gdn = _gdn_module()
    packed = build_packed_params(torch.tensor([0, 3], dtype=torch.int32))

    offsets = gdn.ascendc_causal_conv1d_function._as_int_list(
        packed.cu_seqlens,
        validated_offsets=packed.cu_seqlens_list,
    )

    assert offsets is packed.cu_seqlens_list


def test_gdr_consumes_packed_params_directly(monkeypatch):
    gdn = _gdn_module()
    calls = []

    def fake_apply(*args):
        calls.append(args)
        return args[0].transpose(1, 2).contiguous(), None

    monkeypatch.setattr(gdn.ChunkGatedDeltaRuleFunction, "apply", fake_apply)
    q = torch.randn(1, 3, 1, 4)
    packed = build_packed_params(torch.tensor([0, 3], dtype=torch.int32))

    gdn.ascendc_chunk_gated_delta_rule(
        q,
        torch.randn_like(q),
        torch.randn_like(q),
        torch.randn(1, 3, 1),
        torch.randn(1, 3, 1),
        packed_params=packed,
    )

    args = calls[0]
    assert args[8] is packed.cu_seqlens
    assert args[9] is packed.cu_seqlens_list
    assert args[10] is packed.chunk_indices
    assert args[11] is packed.chunk_indices_list


def test_gdr_backward_intermediate_saving_is_opt_in(monkeypatch):
    gdn = _gdn_module()
    calls = []

    def fake_apply(*args):
        calls.append(args[-1])
        return args[0].transpose(1, 2).contiguous(), None

    monkeypatch.setattr(gdn.ChunkGatedDeltaRuleFunction, "apply", fake_apply)
    q = torch.randn(1, 3, 1, 4)
    inputs = (
        q,
        torch.randn_like(q),
        torch.randn_like(q),
        torch.randn(1, 3, 1),
        torch.randn(1, 3, 1),
    )

    gdn.ascendc_chunk_gated_delta_rule(*inputs)
    gdn.ascendc_chunk_gated_delta_rule(
        *inputs,
        save_backward_intermediates=True,
    )

    assert calls == [False, True]
