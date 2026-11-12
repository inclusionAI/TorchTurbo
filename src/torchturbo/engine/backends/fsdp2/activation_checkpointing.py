"""Runtime input restoration and FSDP hooks for activation recomputation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

from torch import nn

from torchturbo.utils.torch_utils import torch_version

if TYPE_CHECKING:
    from torchturbo.models.modules.moe.moe_layer import BaseMoELayer, MoERuntimeInputs

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
else:  # pragma: no cover - torch<2.6 compatibility
    from torch.distributed._composable.fsdp._fsdp_common import TrainingState
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule


@contextmanager
def restore_moe_runtime_inputs(
    snapshot: tuple[tuple[BaseMoELayer, MoERuntimeInputs], ...],
) -> Iterator[None]:
    """Use each graph's captured inputs and restore the caller's inputs on exit."""
    with ExitStack() as stack:
        for moe, inputs in snapshot:
            stack.callback(moe.set_runtime_inputs, moe._runtime_inputs)
            moe.set_runtime_inputs(inputs)
        yield


def _unshard_before_forward(submodule: FSDPModule, _args: tuple[Any, ...]) -> None:
    # FSDP before 2.11 skips unsharding units still marked PRE_BACKWARD,
    # even if an earlier graph already resharded their parameters.
    submodule.unshard()


def _restart_reentrant_forward(submodule: FSDPModule, _args: tuple[Any, ...]) -> None:
    # Reentrant recomputation builds a new autograd graph. FSDP must run
    # its forward hooks again to attach gradient reduction hooks, even
    # if another graph already used this unit in the same backward.
    # Reset only the module hook state; FSDP owns the parameter state.
    submodule._get_fsdp_state()._training_state = TrainingState.IDLE


@contextmanager
def fsdp_recompute_context(module: nn.Module, use_reentrant: bool) -> Iterator[None]:
    """Prepare descendant FSDP units only while recomputation is active.

    Non-reentrant early-stop can skip FSDP's post-forward cleanup, so it must
    unshard without restarting the forward lifecycle. Reentrant recomputation
    needs that lifecycle to attach gradient reduction hooks to its new graph.
    """
    if not use_reentrant and torch_version() >= (2, 11, 0):
        yield
        return
    hook = _restart_reentrant_forward if use_reentrant else _unshard_before_forward
    with ExitStack() as stack:
        # Prepare each nested unit when it runs, keeping normal backward
        # resharding and limiting live full parameters.
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                handle = child.register_forward_pre_hook(hook, prepend=use_reentrant)
                stack.callback(handle.remove)
        yield
