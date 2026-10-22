"""DCP (``torch.distributed.checkpoint``) state for the FSDP2 backend.

The DCP model state holds every ``model.state_dict()`` entry, including frozen
parameters. Freezing does not imply that a weight still equals its source.

Expert-parallel (EP) ranks hold different experts under the same FQN, and the
DTensor placement only covers the edp mesh dim. DCP keys shards by
``(fqn, global_offset)``, so every EP rank computes the same key for its own
experts and ``dedup_save_plans`` would keep the data of a single EP rank.
Entries sharded on the edp mesh get an ``::ep<rank>`` suffix before they reach
DCP. Loading passes the populated state through ``model.load_state_dict()`` so
model load hooks and extra state participate in restoration.
"""

from __future__ import annotations

import time

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchturbo.engine.optimizers import OptimizersContainer
from ....checkpoint.distributed import (
    collective_check,
    get_control_plane_group,
)
from torchturbo.logging import get_logger

logger = get_logger(__name__)

# Top-level keys of the saved state. DCP flattens nested keys as
# ``<top-level key>.<fqn>`` in its metadata.
_MODEL_KEY = "model"
_OPTIMIZER_KEY = "optimizer"


class DcpCheckpoint:
    """Per-rank DCP save/load of model and optimizer state.

    Optimizer state follows the model-aware DTensor layout exposed by each
    optimizer's ``state_dict()``; loading lets DCP fill local tensors in place
    before the state dict is written back through the optimizer.

    Args:
        ep_rank: Rank of this process in its expert-parallel group.
        ep_size: Number of ranks in the expert-parallel group.
        edp_mesh: Mesh that shards expert-parallel parameters.
    """

    _EP_SUFFIX = "::ep"

    def __init__(self, ep_rank: int, ep_size: int, edp_mesh: DeviceMesh) -> None:
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.edp_mesh = edp_mesh

    # ------------------------------------------------------------------ #
    # EP-unique keys (private).
    # ------------------------------------------------------------------ #
    def _ep_partitioned_fqns(self, model: nn.Module) -> frozenset[str]:
        """Return FQNs whose shards repeat the same offsets on every EP rank."""
        return frozenset(
            fqn
            for fqn, param in model.named_parameters()
            if isinstance(param, DTensor) and param.device_mesh == self.edp_mesh
        )

    def _stamp_fqn(self, fqn: str, ep_fqns: frozenset[str]) -> str:
        if fqn not in ep_fqns:
            return fqn
        return f"{fqn}{self._EP_SUFFIX}{self.ep_rank}"

    def _strip_fqn(self, fqn: str) -> str:
        idx = fqn.find(self._EP_SUFFIX)
        return fqn[:idx] if idx != -1 else fqn

    def _validate_ep_size(
        self,
        saved_keys: set[str],
        ep_fqns: frozenset[str],
        dcp_dir: str,
    ) -> None:
        """Validate EP layout only for entries consumed by the current model."""
        saved_ranks_by_fqn: dict[str, set[int]] = {}
        prefix = f"{_MODEL_KEY}."
        for key in saved_keys:
            fqn, separator, suffix = key.partition(self._EP_SUFFIX)
            if not separator or not fqn.startswith(prefix):
                continue
            model_fqn = fqn[len(prefix) :]
            if model_fqn not in ep_fqns:
                continue
            rank = suffix.split(".", 1)[0]
            if not rank.isdigit():
                continue
            saved_ranks_by_fqn.setdefault(model_fqn, set()).add(int(rank))
        expected_ranks = set(range(self.ep_size))
        for model_fqn, saved_ranks in saved_ranks_by_fqn.items():
            if saved_ranks == expected_ranks:
                continue
            raise ValueError(
                f"Checkpoint {dcp_dir} saved {model_fqn} on EP ranks "
                f"{sorted(saved_ranks)}, but the current run uses EP ranks "
                f"{sorted(expected_ranks)}."
            )

    def _stamp_optim_state(self, optim_state: dict, ep_fqns: frozenset[str]) -> dict:
        new_state = {
            self._stamp_fqn(fqn, ep_fqns): v
            for fqn, v in optim_state.get("state", {}).items()
        }
        new_pg = []
        for g in optim_state.get("param_groups", []):
            g = dict(g)
            g["params"] = [self._stamp_fqn(p, ep_fqns) for p in g.get("params", [])]
            new_pg.append(g)
        return {"state": new_state, "param_groups": new_pg}

    def _strip_optim_state(self, optim_state: dict) -> dict:
        new_state = {
            self._strip_fqn(fqn): v
            for fqn, v in optim_state.get("state", {}).items()
        }
        new_pg = []
        for g in optim_state.get("param_groups", []):
            g = dict(g)
            g["params"] = [self._strip_fqn(p) for p in g.get("params", [])]
            new_pg.append(g)
        return {"state": new_state, "param_groups": new_pg}

    # ------------------------------------------------------------------ #
    # Model state.
    # ------------------------------------------------------------------ #
    def save_model(self, model: nn.Module, dcp_dir: str) -> None:
        """Save all parameters, persistent buffers, and extra state to ``dcp_dir``."""
        ep_fqns = self._ep_partitioned_fqns(model)
        state_dict = {
            self._stamp_fqn(fqn, ep_fqns): value
            for fqn, value in model.state_dict().items()
        }
        start = time.perf_counter()
        dcp.save(
            {_MODEL_KEY: state_dict},
            checkpoint_id=dcp_dir,
            process_group=get_control_plane_group(),
        )
        logger.info_rank0(
            "[save] model DCP saved to %s in %.2fs (%d entries)",
            dcp_dir, time.perf_counter() - start, len(state_dict),
        )

    def load_model(self, model: nn.Module, dcp_dir: str) -> None:
        """Restore required model state, silently ignoring saved-only entries."""
        ep_fqns = self._ep_partitioned_fqns(model)
        state_dict = {
            self._stamp_fqn(fqn, ep_fqns): value
            for fqn, value in model.state_dict().items()
        }

        def preflight() -> None:
            metadata = FileSystemReader(dcp_dir).read_metadata()
            self._validate_ep_size(set(metadata.state_dict_metadata), ep_fqns, dcp_dir)
            # Default DCP planners retain the original nested paths. Compare
            # model entries, not flattened leaf keys of dict-valued extra state.
            saved = {
                self._strip_fqn(path[1])
                for path in metadata.planner_data.values()
                if path[0] == _MODEL_KEY
            }
            expected = {self._strip_fqn(key) for key in state_dict}
            missing = sorted(expected - saved)
            if missing:
                raise RuntimeError(
                    f"Model state in {dcp_dir} does not match the current model: "
                    f"missing entries {missing}. "
                    "The checkpoint must contain all state required by the "
                    "current model. Checkpoints that omitted frozen weights "
                    "cannot restore exact model state."
                )

        collective_check(preflight)
        start = time.perf_counter()
        dcp.load({_MODEL_KEY: state_dict}, checkpoint_id=dcp_dir)
        loaded_state_dict = {
            self._strip_fqn(fqn): value for fqn, value in state_dict.items()
        }
        model.load_state_dict(loaded_state_dict, strict=True)
        logger.info_rank0(
            "[load] model DCP loaded from %s in %.2fs (%d entries)",
            dcp_dir, time.perf_counter() - start, len(loaded_state_dict),
        )

    # ------------------------------------------------------------------ #
    # Optimizer state.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap_optimizer(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer | tuple[torch.optim.Optimizer, ...]:
        """Return the optimizer instances DCP should traverse.

        ``OptimizersContainer`` uses a named-child state format for generic
        serialization. DCP needs the standard state owned by each child so it
        can translate parameter identities to FQNs and reshard them
        independently.
        """
        if isinstance(optimizer, OptimizersContainer):
            return tuple(optimizer)
        return optimizer

    def save_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        dcp_dir: str,
    ) -> None:
        """Save the optimizer state of ``model`` to ``dcp_dir``."""
        options = StateDictOptions(full_state_dict=False)
        optimizers = self._unwrap_optimizer(optimizer)
        optim_state = get_optimizer_state_dict(model, optimizers, options=options)
        stamped = self._stamp_optim_state(
            optim_state, self._ep_partitioned_fqns(model)
        )
        start = time.perf_counter()
        dcp.save(
            {_OPTIMIZER_KEY: stamped},
            checkpoint_id=dcp_dir,
            process_group=get_control_plane_group(),
        )
        logger.info_rank0(
            "[save] optimizer state saved to %s in %.2fs",
            dcp_dir, time.perf_counter() - start,
        )

    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        dcp_dir: str,
    ) -> None:
        """Restore optimizer state saved by :meth:`save_optimizer`."""
        collective_check(lambda: FileSystemReader(dcp_dir).read_metadata())
        options = StateDictOptions(full_state_dict=False)
        optimizers = self._unwrap_optimizer(optimizer)

        optim_state = get_optimizer_state_dict(model, optimizers, options=options)
        stamped = self._stamp_optim_state(
            optim_state, self._ep_partitioned_fqns(model)
        )
        start = time.perf_counter()
        dcp.load({_OPTIMIZER_KEY: stamped}, checkpoint_id=dcp_dir)
        set_optimizer_state_dict(
            model, optimizers, self._strip_optim_state(stamped), options=options
        )
        logger.info_rank0(
            "[load] optimizer state loaded from %s in %.2fs",
            dcp_dir, time.perf_counter() - start,
        )
