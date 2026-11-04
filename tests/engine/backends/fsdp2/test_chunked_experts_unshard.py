"""Chunked MoE dispatch unshards FSDP expert weights once per layer."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from torchturbo.utils.torch_utils import torch_version


@pytest.fixture
def cpu_mesh(tmp_path, monkeypatch):
    if not dist.is_gloo_available():
        pytest.skip("CPU FSDP tests require Gloo")
    if torch_version() < (2, 10, 0):
        pytest.skip("CPU FSDP requires torch >= 2.10")
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        monkeypatch.setenv(
            "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo",
        )
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0,
            world_size=1,
        )
        yield init_device_mesh("cpu", (1,))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_fsdp_forward_hold_passthrough_plain_module():
    from torchturbo.models.modules.moe.token_dispatcher_chunked import (
        _fsdp_forward_hold,
    )

    module = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    expected = module(x)
    with _fsdp_forward_hold(module):
        torch.testing.assert_close(module(x), expected)


def _run_experts_unshard(rank: int, init_file: str) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
        FSDPParamGroup,
    )
    import torch.distributed.fsdp._fully_shard._fsdp_common as fsdp_common
    from torchturbo.models.modules.moe.token_dispatcher_chunked import (
        _fsdp_forward_hold,
        _fsdp_param_groups,
    )

    torch.set_num_threads(1)
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo",
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    original_unshard = FSDPParamGroup.unshard
    calls: list[bool] = []

    def spy(self, *args, **kwargs):
        if self._all_gather_result is None and not self.is_unsharded:
            calls.append(True)
        return original_unshard(self, *args, **kwargs)

    FSDPParamGroup.unshard = spy
    try:
        mesh = init_device_mesh("cpu", (2,))
        experts = nn.Linear(4, 4, bias=False)
        # Nested-unit semantics: reshard after every forward call, matching
        # expert containers under a checkpointed decoder layer (a root unit
        # would stay unsharded in auto mode and hide the per-call regather).
        fully_shard(experts, mesh=mesh, reshard_after_forward=True)
        torch.manual_seed(rank)
        x = torch.randn(2, 4)

        for _ in range(3):
            experts(x)
        assert len(calls) == 3, (
            f"rank {rank}: normal lifecycle unshards {len(calls)} != 3"
        )

        calls.clear()
        state = experts._get_fsdp_state()
        auto_before = state._auto_reshard_after_forward
        with _fsdp_forward_hold(experts):
            # The chunked dispatcher invokes FSDP-wrapped experts through
            # ``forward`` directly (hooks bypassed), once per chunk.
            outputs = [experts.forward(x) for _ in range(3)]
            state = experts._get_fsdp_state()
            assert all(
                group.is_unsharded for group in _fsdp_param_groups(state)
            )
        assert len(calls) == 1, (
            f"rank {rank}: chunk-loop unshards {len(calls)} != 1"
        )
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0])

        # Lifecycle restored: reshard policy back to the pre-context state
        # (with an explicit ``reshard_after_forward=True`` the state's auto
        # flag is False by design; both must come back unchanged) and the
        # weights resharded on exit.
        state = experts._get_fsdp_state()
        assert state._auto_reshard_after_forward == auto_before
        assert all(
            group._reshard_after_forward
            for group in _fsdp_param_groups(state)
        )
        assert not any(
            group.is_unsharded for group in _fsdp_param_groups(state)
        )
        experts(x)
        assert len(calls) == 2

        # Under activation recomputation the hold must not reshard on
        # exit: the chunked backward reuses the unsharded weights, and
        # its single post_backward performs the one reshard.
        if hasattr(fsdp_common, "is_bw"):
            saved_is_bw = fsdp_common.is_bw
            calls.clear()
            fsdp_common.is_bw = lambda: True
            try:
                with _fsdp_forward_hold(experts):
                    for _ in range(3):
                        experts.forward(x)
                state = experts._get_fsdp_state()
                assert all(
                    group.is_unsharded
                    for group in _fsdp_param_groups(state)
                )
            finally:
                fsdp_common.is_bw = saved_is_bw
            assert len(calls) == 1, (
                f"rank {rank}: recomputed forward unshards "
                f"{len(calls)} != 1"
            )
            experts.reshard()
        dist.barrier()
    finally:
        FSDPParamGroup.unshard = original_unshard
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="requires torch.distributed with Gloo support",
)
def test_fsdp_forward_hold_world2(tmp_path):
    if torch_version() < (2, 10, 0):
        pytest.skip("CPU FSDP requires torch >= 2.10")
    mp.spawn(
        _run_experts_unshard, args=(str(tmp_path / "experts-unshard"),),
        nprocs=2, join=True,
    )
