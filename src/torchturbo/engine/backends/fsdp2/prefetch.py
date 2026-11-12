"""FSDP2 prefetch.

Discovers ordered decoder chains from an FSDP-wrapped model, keeping each
chain's full layer sequence (dense layers included) but only for chains that
contain at least one MoE layer; chains with no MoE are left to FSDP's native
prefetching. A chosen strategy then turns each chain into forward/backward
prefetch targets, applied through FSDP2's public
``set_modules_to_forward_prefetch`` / ``set_modules_to_backward_prefetch``
APIs. Importing this module does not touch any FSDP internals.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set

from torch import nn

from torchturbo.logging import get_logger
from torchturbo.utils.torch_utils import torch_version

from .config import FSDPConfig, FSDPPrefetchDirection, FSDPPrefetchStrategy

logger = get_logger(__name__)

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard import FSDPModule
else:
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule

try:
    from torchturbo.models.modules.moe.moe_layer import MoELayer
except Exception:  # pragma: no cover - MoE support optional
    MoELayer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PrefetchNode:
    """One node of a chain: an outer decoder block and its optional experts."""

    module: "FSDPModule"
    experts: Optional["FSDPModule"] = None


@dataclass
class PrefetchChain:
    """An ordered decoder-like chain plus the dense-to-MoE prefetch source."""

    name: str
    bootstrap: Optional["FSDPModule"]
    nodes: List[PrefetchNode] = field(default_factory=list)
    first_moe_index: int = 0


@dataclass
class _ModelPrefetchState:
    """Prefetch state owned by one prepared root model."""

    model: nn.Module
    chains: List[PrefetchChain]
    expert_forward_prefetch_targets: Dict[
        "FSDPModule", List["FSDPModule"]
    ]


# ---------------------------------------------------------------------------
# MoE detection
# ---------------------------------------------------------------------------
def _find_experts_container(layer: nn.Module) -> Optional["FSDPModule"]:
    """Return the FSDP-wrapped experts container inside ``layer``, or ``None``."""
    if MoELayer is None:
        return None
    for submodule in layer.modules():
        if isinstance(submodule, MoELayer):
            experts = getattr(submodule, "experts", None)
            if isinstance(experts, FSDPModule):
                return experts
    return None


# ---------------------------------------------------------------------------
# Chain discovery
# ---------------------------------------------------------------------------
def _collect_moe_internal_ids(model: nn.Module) -> Set[int]:
    """Ids of modules nested inside a ``MoELayer``.

    Used to skip experts-internal ``nn.ModuleList``s during discovery.
    """
    ids: Set[int] = set()
    if MoELayer is None:
        return ids
    for m in model.modules():
        if isinstance(m, MoELayer):
            for sub in m.modules():
                if sub is not m:
                    ids.add(id(sub))
    return ids


# ---------------------------------------------------------------------------
# Strategy layer: turn chains into forward/backward target maps
# ---------------------------------------------------------------------------
def _node_layer_modules(node: PrefetchNode) -> List["FSDPModule"]:
    """The FSDP modules of one whole logical layer (outer module + experts)."""
    mods: List["FSDPModule"] = [node.module]
    if node.experts is not None:
        mods.append(node.experts)
    return mods


def _add_target(
    target_map: Dict["FSDPModule", List["FSDPModule"]],
    source: Optional["FSDPModule"],
    targets: List["FSDPModule"],
) -> None:
    if source is None:
        return
    filtered = [t for t in targets if t is not None]
    if not filtered:
        return
    target_map.setdefault(source, []).extend(filtered)


def _dedup(mods: List["FSDPModule"]) -> List["FSDPModule"]:
    seen: Set[int] = set()
    out: List["FSDPModule"] = []
    for m in mods:
        if id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def _moe_indices(nodes: List[PrefetchNode]) -> List[int]:
    return [i for i, node in enumerate(nodes) if node.experts is not None]


def _source_for_node(
    chain: PrefetchChain, index: int
) -> Optional["FSDPModule"]:
    if index == 0 or index == chain.first_moe_index:
        return chain.bootstrap
    return chain.nodes[index - 1].module


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class FSDPPrefetchManager:
    """Discover MoE chains and manage prefetch targets for prepared models."""

    def __init__(self, config: FSDPConfig) -> None:
        self.config = config
        self.chains: List[PrefetchChain] = []
        self._model_states: Dict[int, _ModelPrefetchState] = {}
        self._expert_forward_prefetch_targets: Dict[
            "FSDPModule", List["FSDPModule"]
        ] = {}
        self._expert_forward_prefetch_suspend_depth: Dict["FSDPModule", int] = {}

    def setup(self, model: nn.Module) -> None:
        if model is None:
            raise ValueError("FSDPPrefetchManager.setup() requires a model")

        # set_modules_to_{forward,backward}_prefetch require torch >= 2.5.0.
        if torch_version() < (2, 5, 0):
            if self.config.prefetch_strategy != FSDPPrefetchStrategy.NATIVE:
                logger.warning_rank0(
                    "FSDP prefetch requires torch >= 2.5.0 for "
                    "set_modules_to_{forward,backward}_prefetch; got %s, "
                    "skipping prefetch.",
                    ".".join(map(str, torch_version())),
                )
            return

        self._remove_model_state(model)

        if self.config.prefetch_strategy == FSDPPrefetchStrategy.NATIVE:
            return

        enable_backward = (
            self.config.prefetch_direction
            == FSDPPrefetchDirection.FORWARD_BACKWARD
        )

        chains = self._discover_moe_chains(model)
        if not chains:
            if self.config.prefetch_strategy != FSDPPrefetchStrategy.AUTO:
                logger.warning_rank0(
                    "FSDP prefetch: no eligible MoE chain discovered; skipping"
                )
            return
        self._log_discovery(chains)

        forward_targets, backward_targets = self._build_targets(chains)

        expert_forward_targets = self._flush_forward_targets(
            forward_targets, chains
        )
        if enable_backward:
            self._flush_backward_targets(backward_targets)

        self._model_states[id(model)] = _ModelPrefetchState(
            model=model,
            chains=chains,
            expert_forward_prefetch_targets=expert_forward_targets,
        )
        self._sync_chains()

        logger.info_rank0(
            "FSDP prefetch configured: strategy=%s, forward=%s, backward=%s",
            self.config.prefetch_strategy.value,
            True,
            enable_backward,
        )

    def cleanup(self) -> None:
        self._clear_targets()
        self.chains = []
        self._model_states = {}
        self._expert_forward_prefetch_targets = {}
        self._expert_forward_prefetch_suspend_depth = {}

    def _remove_model_state(self, model: nn.Module) -> None:
        state = self._model_states.get(id(model))
        if state is None or state.model is not model:
            return

        del self._model_states[id(model)]
        self._clear_targets(state.chains)
        for source in state.expert_forward_prefetch_targets:
            self._expert_forward_prefetch_targets.pop(source, None)
            self._expert_forward_prefetch_suspend_depth.pop(source, None)
        self._sync_chains()

    def _sync_chains(self) -> None:
        self.chains = [
            chain
            for state in self._model_states.values()
            for chain in state.chains
        ]

    def _discover_moe_chains(self, model: nn.Module) -> List[PrefetchChain]:
        """Discover prefetch chains from an FSDP-wrapped model.

        Each ``nn.ModuleList`` keeps its full layer sequence (dense layers
        included) so prefetch edges follow real execution order and the
        dense-to-MoE boundary is handled naturally. A chain is dropped entirely
        when none of its layers is MoE — those are left to FSDP's native
        prefetching.
        """
        moe_internal_ids = _collect_moe_internal_ids(model)
        chains: List[PrefetchChain] = []
        seen_signatures: Set[tuple] = set()

        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            if id(module) in moe_internal_ids:
                continue

            nodes: List[PrefetchNode] = []
            for child in module:
                if not isinstance(child, FSDPModule):
                    continue
                nodes.append(
                    PrefetchNode(module=child, experts=_find_experts_container(child))
                )
            if not nodes:
                continue
            if not any(n.experts is not None for n in nodes):
                continue

            signature = tuple(id(n.module) for n in nodes)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            first_moe_idx = next(
                i for i, node in enumerate(nodes) if node.experts is not None
            )
            bootstrap = None
            if first_moe_idx > 0:
                bootstrap = nodes[first_moe_idx - 1].module
            chains.append(
                PrefetchChain(
                    name=name or "<root>",
                    bootstrap=bootstrap,
                    nodes=nodes,
                    first_moe_index=first_moe_idx,
                )
            )

        return chains

    def _build_targets(
        self, chains: Optional[List[PrefetchChain]] = None
    ) -> tuple[
        Dict["FSDPModule", List["FSDPModule"]],
        Dict["FSDPModule", List["FSDPModule"]],
    ]:
        forward_targets: Dict["FSDPModule", List["FSDPModule"]] = {}
        backward_targets: Dict["FSDPModule", List["FSDPModule"]] = {}
        target_chains = self.chains if chains is None else chains
        for chain in target_chains:
            if not chain.nodes:
                continue
            eff = self._resolve_strategy(chain)
            if eff == FSDPPrefetchStrategy.EXPERT_WITH_LAYER:
                self._build_expert_with_layer_targets(
                    chain, forward_targets, backward_targets
                )
            elif eff == FSDPPrefetchStrategy.EXPERT_LOOKAHEAD:
                self._build_expert_lookahead_targets(
                    chain, forward_targets, backward_targets
                )
        return forward_targets, backward_targets

    def _resolve_strategy(self, chain: PrefetchChain) -> FSDPPrefetchStrategy:
        if self.config.prefetch_strategy == FSDPPrefetchStrategy.AUTO:
            return FSDPPrefetchStrategy.EXPERT_LOOKAHEAD
        return self.config.prefetch_strategy

    def _build_expert_with_layer_targets(
        self,
        chain: PrefetchChain,
        forward_targets: Dict["FSDPModule", List["FSDPModule"]],
        backward_targets: Dict["FSDPModule", List["FSDPModule"]],
    ) -> None:
        nodes = chain.nodes
        n = len(nodes)
        fd = self.config.forward_prefetch_depth
        bd = self.config.backward_prefetch_depth
        forward_scheduled: Set[int] = set()
        backward_scheduled: Set[int] = set()

        for i in _moe_indices(nodes):
            source = _source_for_node(chain, i)
            if source is None:
                continue

            fwd: List["FSDPModule"] = []
            for k in range(i, min(i + fd, n)):
                for module in _node_layer_modules(nodes[k]):
                    if id(module) in forward_scheduled:
                        continue
                    forward_scheduled.add(id(module))
                    fwd.append(module)
            _add_target(forward_targets, source, fwd)

        for i in reversed(_moe_indices(nodes)):
            if i > 0:
                bwd: List["FSDPModule"] = []
                for k in range(i - 1, max(i - 1 - bd, -1), -1):
                    for module in _node_layer_modules(nodes[k]):
                        if id(module) in backward_scheduled:
                            continue
                        backward_scheduled.add(id(module))
                        bwd.append(module)
                _add_target(backward_targets, nodes[i].module, bwd)

    def _build_expert_lookahead_targets(
        self,
        chain: PrefetchChain,
        forward_targets: Dict["FSDPModule", List["FSDPModule"]],
        backward_targets: Dict["FSDPModule", List["FSDPModule"]],
    ) -> None:
        nodes = chain.nodes
        n = len(nodes)
        fd = self.config.forward_prefetch_depth
        bd = self.config.backward_prefetch_depth
        moe_indices = _moe_indices(nodes)
        forward_outer_scheduled: Set[int] = set()
        forward_experts_scheduled: Set[int] = set()
        backward_outer_scheduled: Set[int] = set()
        backward_experts_scheduled: Set[int] = set()

        # Forward: outer chain (predecessor -> upcoming outer layers) plus a
        # separate experts chain (each experts -> next contiguous experts).
        # Depth > 1 is a rolling horizon: once a module has been scheduled by
        # an earlier source, later sources only add the newly exposed horizon.
        for i in moe_indices:
            source = _source_for_node(chain, i)
            prev = i - 1
            # Doubles as the entry flag (is this the source that bootstraps a
            # contiguous MoE run?) and the run guard: it flips to False at the
            # first non-MoE node in the window, so the source never prefetches
            # experts across an intervening dense layer.
            prefetch_entry_experts = not (
                prev >= 0 and nodes[prev].experts is not None
            )
            source_targets: List["FSDPModule"] = []
            if source is not None:
                for k in range(i, min(i + fd, n)):
                    outer = nodes[k].module
                    if id(outer) not in forward_outer_scheduled:
                        forward_outer_scheduled.add(id(outer))
                        source_targets.append(outer)
                    if prefetch_entry_experts and nodes[k].experts is not None:
                        experts = nodes[k].experts
                        if id(experts) not in forward_experts_scheduled:
                            forward_experts_scheduled.add(id(experts))
                            source_targets.append(experts)
                    else:
                        prefetch_entry_experts = False
                _add_target(forward_targets, source, source_targets)
            expert_targets: List["FSDPModule"] = []
            for k in range(i + 1, min(i + 1 + fd, n)):
                if nodes[k].experts is None:
                    break
                experts = nodes[k].experts
                if id(experts) in forward_experts_scheduled:
                    continue
                forward_experts_scheduled.add(id(experts))
                expert_targets.append(experts)
            _add_target(forward_targets, nodes[i].experts, expert_targets)

        # Backward: outer chain (layer -> previous outer layers) plus experts
        # chain (experts -> previous contiguous experts). Sources fire from
        # high layer index to low layer index, so rolling state is tracked in
        # that order.
        for i in reversed(moe_indices):
            prev_idxs = range(i - 1, max(i - 1 - bd, -1), -1)  # i-1, ..., i-bd
            if i > 0:
                outer_targets: List["FSDPModule"] = []
                for k in prev_idxs:
                    outer = nodes[k].module
                    if id(outer) in backward_outer_scheduled:
                        continue
                    backward_outer_scheduled.add(id(outer))
                    outer_targets.append(outer)
                _add_target(
                    backward_targets,
                    nodes[i].module,
                    outer_targets,
                )
            expert_targets: List["FSDPModule"] = []
            for k in prev_idxs:
                if nodes[k].experts is None:
                    break
                experts = nodes[k].experts
                if id(experts) in backward_experts_scheduled:
                    continue
                backward_experts_scheduled.add(id(experts))
                expert_targets.append(experts)
            _add_target(backward_targets, nodes[i].experts, expert_targets)

    def _flush_forward_targets(
        self,
        forward_targets: Dict["FSDPModule", List["FSDPModule"]],
        chains: Optional[List[PrefetchChain]] = None,
    ) -> Dict["FSDPModule", List["FSDPModule"]]:
        target_chains = self.chains if chains is None else chains
        expert_source_ids = {
            id(node.experts)
            for chain in target_chains
            for node in chain.nodes
            if node.experts is not None
        }
        expert_forward_targets: Dict["FSDPModule", List["FSDPModule"]] = {}
        for source, targets in forward_targets.items():
            deduped_targets = _dedup(targets)
            source.set_modules_to_forward_prefetch(deduped_targets)
            if id(source) in expert_source_ids:
                expert_forward_targets[source] = deduped_targets
        self._expert_forward_prefetch_targets.update(expert_forward_targets)
        return expert_forward_targets

    @contextmanager
    def suspend_expert_forward_prefetch(
        self, model: Optional[nn.Module] = None
    ) -> Iterator[None]:
        """Temporarily disable expert-source forward prefetch.

        Used by activation-checkpoint recompute only. Normal forward still keeps
        expert lookahead; during backward recompute, nested expert FSDP units can
        still look like a normal forward and would otherwise prefetch later
        experts at the wrong time. When ``model`` is given, targets belonging to
        other prepared models remain active.
        """
        if model is None:
            sources = list(self._expert_forward_prefetch_targets)
        else:
            state = self._model_states.get(id(model))
            if state is None or state.model is not model:
                sources = []
            else:
                sources = list(state.expert_forward_prefetch_targets)
        if not sources:
            yield
            return

        try:
            for source in sources:
                depth = self._expert_forward_prefetch_suspend_depth.get(
                    source, 0
                )
                if depth == 0:
                    source.set_modules_to_forward_prefetch([])
                self._expert_forward_prefetch_suspend_depth[source] = depth + 1
            yield
        finally:
            for source in sources:
                depth = self._expert_forward_prefetch_suspend_depth.get(source)
                if depth is None:
                    continue
                if depth > 1:
                    self._expert_forward_prefetch_suspend_depth[source] = depth - 1
                    continue

                del self._expert_forward_prefetch_suspend_depth[source]
                targets = self._expert_forward_prefetch_targets.get(source)
                if targets is not None:
                    source.set_modules_to_forward_prefetch(list(targets))

    def _flush_backward_targets(
        self, backward_targets: Dict["FSDPModule", List["FSDPModule"]]
    ) -> None:
        for source, targets in backward_targets.items():
            source.set_modules_to_backward_prefetch(_dedup(targets))

    def _clear_targets(
        self, chains: Optional[List[PrefetchChain]] = None
    ) -> None:
        for module in self._collect_chain_modules(chains):
            module.set_modules_to_forward_prefetch([])
            module.set_modules_to_backward_prefetch([])

    def _collect_chain_modules(
        self, chains: Optional[List[PrefetchChain]] = None
    ) -> List["FSDPModule"]:
        modules: List["FSDPModule"] = []
        target_chains = self.chains if chains is None else chains
        for chain in target_chains:
            if chain.bootstrap is not None:
                modules.append(chain.bootstrap)
            for node in chain.nodes:
                modules.append(node.module)
                if node.experts is not None:
                    modules.append(node.experts)
        return _dedup(modules)

    def _log_discovery(
        self, chains: Optional[List[PrefetchChain]] = None
    ) -> None:
        target_chains = self.chains if chains is None else chains
        logger.info_rank0(
            "FSDP prefetch discovered %d chain(s):", len(target_chains)
        )
        for chain in target_chains:
            n_experts = sum(1 for n in chain.nodes if n.experts is not None)
            logger.info_rank0(
                "- %s: nodes=%d, experts=%d, bootstrap=%s",
                chain.name,
                len(chain.nodes),
                n_experts,
                "yes" if chain.bootstrap is not None else "no",
            )
