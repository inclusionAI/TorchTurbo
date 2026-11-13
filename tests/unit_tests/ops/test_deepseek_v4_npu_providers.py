from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

from torchturbo.ops import OpDispatcher
from torchturbo.ops.registry import registry
from torchturbo.runtime import TurboDevice


class _NpuRuntime:
    turbo_device = TurboDevice.NPU


def _install_fake_cann(monkeypatch):
    calls = []

    def lightning_indexer(*args, **kwargs):
        calls.append(("indexer", args, kwargs))
        return "indices", "scores"

    def lightning_indexer_metadata(*args, **kwargs):
        calls.append(("metadata", args, kwargs))
        return "metadata"

    cann_ops = SimpleNamespace(
        lightning_indexer=lightning_indexer,
        lightning_indexer_metadata=lightning_indexer_metadata,
    )
    cann = ModuleType("cann_ops_transformer")
    cann.ops = cann_ops
    cann.sparse_flash_mla = lambda *args, **kwargs: None
    cann.sparse_flash_mla_metadata = lambda *args, **kwargs: None
    cann.sparse_flash_mla_grad = lambda *args, **kwargs: None
    cann.sparse_flash_mla_grad_metadata = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", cann)

    importlib.import_module("torchturbo.ops.providers.npu.lightning_indexer")
    importlib.import_module("torchturbo.ops.providers.npu.sparse_flash_mla")
    return calls


def test_deepseek_v4_cann_ops_are_dispatcher_providers(monkeypatch):
    calls = _install_fake_cann(monkeypatch)
    dispatcher = OpDispatcher(device_runtime=_NpuRuntime(), registry=registry)

    assert (
        dispatcher._candidate_impl_cache[
            ("sparse_flash_mla", "npu", None)
        ].spec.impl_name
        == "npu_sparse_flash_mla"
    )
    assert (
        dispatcher._candidate_impl_cache[
            ("lightning_indexer", "npu", None)
        ].spec.impl_name
        == "npu_lightning_indexer"
    )
    assert (
        dispatcher._candidate_impl_cache[
            ("lightning_indexer_metadata", "npu", None)
        ].spec.impl_name
        == "npu_lightning_indexer_metadata"
    )

    assert dispatcher.lightning_indexer(1, 2, 3, 4, mode=2) == (
        "indices",
        "scores",
    )
    assert (
        dispatcher.lightning_indexer_metadata(
            num_heads_q=1,
            num_heads_k=2,
            head_dim=3,
            topk=4,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cmp_residual_k=None,
            batch_size=5,
            max_seqlen_q=6,
            max_seqlen_k=7,
            mode=4,
        )
        == "metadata"
    )
    assert [call[0] for call in calls] == ["indexer", "metadata"]
