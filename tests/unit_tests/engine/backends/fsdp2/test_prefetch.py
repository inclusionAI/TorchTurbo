"""Unit tests for FSDP2 prefetch discovery and strategy.

These tests exercise the pure-Python logic (chain discovery + strategy target
maps + target flushing) with a fake module tree, so they run without GPU / real
FSDP wrapping. The optional CUDA ``grouped_gemm`` kernel is stubbed so
``torchturbo`` can import on CPU/mac.
"""

import sys
import types

if "grouped_gemm" not in sys.modules:  # pragma: no cover - env shim
    _gg = types.ModuleType("grouped_gemm")
    _gg.ops = types.SimpleNamespace()
    sys.modules["grouped_gemm"] = _gg

import pytest
import torch.nn as nn

import torchturbo.engine.backends.fsdp2.prefetch as pf
from torchturbo.engine.backends.fsdp2.config import (
    FSDPConfig,
    FSDPPrefetchDirection,
    FSDPPrefetchStrategy,
)


# ---------------------------------------------------------------------------
# Fakes standing in for FSDP-wrapped modules / MoELayer
# ---------------------------------------------------------------------------
class FakeFSDP(nn.Module):
    """Stand-in for an FSDP-wrapped module.

    Records the arguments passed to the public forward/backward prefetch APIs.
    """

    def __init__(self, tag: str = "") -> None:
        super().__init__()
        self.tag = tag
        self.forward_prefetch_arg = None
        self.backward_prefetch_arg = None

    def set_modules_to_forward_prefetch(self, mods):
        self.forward_prefetch_arg = list(mods)

    def set_modules_to_backward_prefetch(self, mods):
        self.backward_prefetch_arg = list(mods)


class FakeMoE(nn.Module):
    """Stand-in for the project's MoELayer."""

    def __init__(self, experts) -> None:
        super().__init__()
        self.experts = experts


def make_layer(tag: str, moe: bool = False) -> FakeFSDP:
    layer = FakeFSDP(tag)
    if moe:
        experts = FakeFSDP(tag + "E")
        layer.mlp = FakeMoE(experts)
        layer.experts_ref = experts  # test handle
    else:
        layer.mlp = nn.Linear(2, 2)
    return layer


def make_model(specs, container="layers", root_is_fsdp=True):
    """Build ``model.<container>`` = ModuleList(layers).

    ``container`` may be nested, e.g. ``"language_model.layers"``.
    """
    model = FakeFSDP("model") if root_is_fsdp else nn.Module()
    layers = nn.ModuleList([make_layer(f"L{i}", moe) for i, moe in enumerate(specs)])
    parts = container.split(".")
    holder = model
    for part in parts[:-1]:
        inner = FakeFSDP(part)
        setattr(holder, part, inner)
        holder = inner
    setattr(holder, parts[-1], layers)
    return model


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Patch isinstance targets to fakes."""
    monkeypatch.setattr(pf, "FSDPModule", FakeFSDP)
    monkeypatch.setattr(pf, "MoELayer", FakeMoE)


def _tagmap(target_map):
    return {src.tag: [t.tag for t in tgts] for src, tgts in target_map.items()}


def _discover_moe_chains(model):
    return pf.FSDPPrefetchManager(config=FSDPConfig())._discover_moe_chains(model)


def _build_targets(chain, strategy, fd=1, bd=1):
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(
            prefetch_strategy=strategy,
            forward_prefetch_depth=fd,
            backward_prefetch_depth=bd,
        )
    )
    manager.chains = [chain]
    return manager._build_targets()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_discover_dense_has_no_moe_chain():
    model = make_model([False, False, False])
    assert _discover_moe_chains(model) == []


def test_discover_moe_with_dense_prefix():
    model = make_model([False, False, True, True, True])
    (chain,) = _discover_moe_chains(model)
    # full layer sequence is kept, including the dense prefix
    assert [n.module.tag for n in chain.nodes] == ["L0", "L1", "L2", "L3", "L4"]
    assert [n.experts is None for n in chain.nodes] == [
        True,
        True,
        False,
        False,
        False,
    ]
    # experts container is the FSDP experts inside each MoE layer
    assert chain.nodes[2].experts is model.layers[2].experts_ref
    # bootstrap is the explicit source for the first MoE layer
    assert chain.bootstrap is model.layers[1]


def test_discover_includes_visual_moe_branch():
    model = make_model([True, True])
    # add a vision branch that is also a ModuleList of FSDP modules
    visual = FakeFSDP("visual")
    visual.blocks = nn.ModuleList([make_layer("v0", True), make_layer("v1", True)])
    model.visual = visual
    chains = _discover_moe_chains(model)
    by_name = {c.name: c for c in chains}
    assert "layers" in by_name
    assert "visual.blocks" in by_name
    assert by_name["visual.blocks"].bootstrap is None


def test_discover_includes_mtp_moe_branch():
    model = make_model([True, True])
    mtp = FakeFSDP("mtp")
    mtp.layers = nn.ModuleList([make_layer("m0", True)])
    model.mtp = mtp
    chains = _discover_moe_chains(model)
    by_name = {c.name: c for c in chains}
    assert "mtp.layers" in by_name
    assert by_name["mtp.layers"].bootstrap is None


def test_discover_skips_dense_only_branch():
    # language layers are MoE; a dense-only vision branch is left to native
    model = make_model([True, True])
    visual = FakeFSDP("visual")
    visual.blocks = nn.ModuleList([make_layer("v0"), make_layer("v1")])  # dense
    model.visual = visual
    chains = _discover_moe_chains(model)
    names = {c.name for c in chains}
    assert "layers" in names
    assert "visual.blocks" not in names


def test_discover_nested_root_language_model():
    model = make_model([True, True], container="language_model.layers")
    (chain,) = _discover_moe_chains(model)
    # First-layer MoE chains do not bootstrap from root/ancestor to avoid
    # prefetching the first layer too early.
    assert chain.bootstrap is None


def test_discover_first_moe_has_no_root_fallback():
    model = FakeFSDP("model")
    inner = nn.Module()
    inner.layers = nn.ModuleList([make_layer("L0", True)])
    model.language_model = inner
    (chain,) = _discover_moe_chains(model)
    assert chain.bootstrap is None


def test_discover_skips_experts_internal_modulelist():
    # A MoE layer whose experts is a raw ModuleList must not be seen as a chain.
    model = FakeFSDP("model")
    layer = FakeFSDP("L0")
    layer.mlp = FakeMoE(nn.ModuleList([FakeFSDP("e0"), FakeFSDP("e1")]))
    model.layers = nn.ModuleList([layer])
    # experts (raw ModuleList) is not a single FSDPModule -> no MoE chain
    assert _discover_moe_chains(model) == []


def test_discover_ignores_modulelist_without_fsdp_children():
    model = FakeFSDP("model")
    model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    assert _discover_moe_chains(model) == []


# ---------------------------------------------------------------------------
# Strategy target maps (verified against docs/design/fsdp-prefetch.md examples)
# ---------------------------------------------------------------------------
def _chain(specs, container="layers"):
    model = make_model(specs, container=container)
    (chain,) = _discover_moe_chains(model)
    return chain


def test_expert_with_layer_strategy_targets():
    chain = _chain([False, False, True, True, True])
    ft, bt = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_WITH_LAYER)
    assert _tagmap(ft) == {
        "L1": ["L2", "L2E"],
        "L2": ["L3", "L3E"],
        "L3": ["L4", "L4E"],
    }
    assert _tagmap(bt) == {
        "L2": ["L1"],
        "L3": ["L2", "L2E"],
        "L4": ["L3", "L3E"],
    }


def test_expert_lookahead_strategy_targets():
    chain = _chain([False, False, True, True, True])
    ft, bt = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    assert _tagmap(ft) == {
        "L1": ["L2", "L2E"],
        "L2": ["L3"],
        "L3": ["L4"],
        "L2E": ["L3E"],
        "L3E": ["L4E"],
    }
    assert _tagmap(bt) == {
        "L2": ["L1"],
        "L3": ["L2"],
        "L4": ["L3"],
        "L3E": ["L2E"],
        "L4E": ["L3E"],
    }


def test_first_moe_uses_chain_bootstrap_source():
    chain = _chain([False, False, True, True])
    chain.bootstrap = FakeFSDP("bootstrap")
    ft, _ = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    tm = _tagmap(ft)
    assert tm["bootstrap"] == ["L2", "L2E"]
    assert "L1" not in tm


def test_expert_lookahead_first_layer_is_moe():
    # k=0 boundary: first stage is MoE, so L0/L0E stay native; lookahead starts
    # from L0 -> L1 and L0E -> L1E.
    chain = _chain([True, True, True])
    ft, _ = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    assert "model" not in _tagmap(ft)
    assert _tagmap(ft)["L0"] == ["L1"]
    assert _tagmap(ft)["L0E"] == ["L1E"]


def test_depth2_first_layer_moe_keeps_rolling_horizon():
    chain = _chain([True, True, True])
    ft, _ = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD, fd=2)
    tm = _tagmap(ft)
    assert "model" not in tm
    assert tm["L0"] == ["L1", "L2"]
    assert tm["L0E"] == ["L1E", "L2E"]
    assert "L1" not in tm
    assert "L1E" not in tm


def test_expert_with_layer_depth2_first_layer_moe_has_no_root_source():
    chain = _chain([True, True, True])
    ft, _ = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_WITH_LAYER, fd=2)
    tm = _tagmap(ft)
    assert "model" not in tm
    assert tm["L0"] == ["L1", "L1E", "L2", "L2E"]
    assert "L1" not in tm


def test_interleaved_dense_moe_keeps_contiguous_layer_chain():
    # dense/MoE interleaved: full sequence kept; layer chain does not skip dense
    chain = _chain([False, True, False, True])
    assert [n.module.tag for n in chain.nodes] == ["L0", "L1", "L2", "L3"]
    ft, bt = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    tm = _tagmap(ft)
    # L3 is prefetched by its real predecessor L2, not skipped over
    assert tm["L2"] == ["L3", "L3E"]
    # experts do not look ahead across an intervening dense layer
    assert "L1E" not in tm

    backward_tm = _tagmap(bt)
    # Do not attach explicit backward targets to an intervening dense layer:
    # that would replace its native backward prefetch instead of adding to it.
    assert "L2" not in backward_tm


def test_auto_dense_uses_native():
    model = make_model([False, False, False])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(prefetch_strategy=FSDPPrefetchStrategy.AUTO)
    )
    manager.setup(model)
    assert manager.chains == []
    assert model.forward_prefetch_arg is None
    assert model.backward_prefetch_arg is None


def test_auto_moe_uses_expert_lookahead():
    chain = _chain([False, True, True])
    ft_auto, _ = _build_targets(chain, FSDPPrefetchStrategy.AUTO)
    ft_el, _ = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    assert _tagmap(ft_auto) == _tagmap(ft_el)


# ---------------------------------------------------------------------------
# Prefetch depth (multi-layer prefetch windows)
# ---------------------------------------------------------------------------
def test_expert_with_layer_depth2_targets():
    chain = _chain([False, False, True, True, True])
    ft, bt = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_WITH_LAYER, fd=2, bd=2)
    assert _tagmap(ft) == {
        "L1": ["L2", "L2E", "L3", "L3E"],
        "L2": ["L4", "L4E"],
    }
    assert _tagmap(bt) == {
        "L4": ["L3", "L3E", "L2", "L2E"],
        "L3": ["L1"],
        "L2": ["L0"],
    }


def test_expert_lookahead_depth2_targets():
    chain = _chain([False, False, True, True, True])
    ft, bt = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD, fd=2, bd=2)
    assert _tagmap(ft) == {
        "L1": ["L2", "L2E", "L3", "L3E"],
        "L2E": ["L4E"],
        "L2": ["L4"],
    }
    assert _tagmap(bt) == {
        "L4": ["L3", "L2"],
        "L4E": ["L3E", "L2E"],
        "L3": ["L1"],
        "L2": ["L0"],
    }


def test_depth2_experts_do_not_cross_dense():
    # [dense, MoE, dense, MoE]: the experts chain must stop at the intervening
    # dense layer even at depth 2, while the outer window may include it.
    chain = _chain([False, True, False, True])
    ft, bt = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_LOOKAHEAD, fd=2, bd=2)
    ftm = _tagmap(ft)
    assert ftm["L0"] == ["L1", "L1E", "L2"]
    assert ftm["L2"] == ["L3", "L3E"]
    assert "L1E" not in ftm  # experts must not span dense L2 forward
    btm = _tagmap(bt)
    assert btm["L3"] == ["L2", "L1"]
    assert "L3E" not in btm  # experts must not span dense L2 backward
    assert "L2" not in btm  # dense layer is never an explicit backward source


def test_depth_clamps_at_chain_boundaries():
    # depth larger than the chain must clamp, not raise or over-index
    chain = _chain([False, True, True])
    ft, _ = _build_targets(chain, FSDPPrefetchStrategy.EXPERT_WITH_LAYER, fd=5, bd=5)
    tm = _tagmap(ft)
    assert tm["L0"] == ["L1", "L1E", "L2", "L2E"]
    assert "L1" not in tm


# ---------------------------------------------------------------------------
# Flush forward / backward (public set_modules_to_*_prefetch APIs)
# ---------------------------------------------------------------------------
def test_flush_forward_dedup():
    chain = _chain([False, True, True])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    )
    manager.chains = [chain]
    ft, _ = manager._build_targets()
    manager._flush_forward_targets(ft)
    # L0 (dense predecessor of first MoE) bootstraps: layer edge L1 + expert boot L1E
    l0 = chain.nodes[0].module
    assert [m.tag for m in l0.forward_prefetch_arg] == ["L1", "L1E"]


def test_suspend_expert_forward_prefetch_only_clears_expert_sources():
    chain = _chain([False, True, True, True])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    )
    manager.chains = [chain]
    ft, _ = manager._build_targets()
    manager._flush_forward_targets(ft)

    l0 = chain.nodes[0].module
    l1 = chain.nodes[1].module
    l1e = chain.nodes[1].experts
    l2e = chain.nodes[2].experts
    l3e = chain.nodes[3].experts

    assert [m.tag for m in l0.forward_prefetch_arg] == ["L1", "L1E"]
    assert [m.tag for m in l1.forward_prefetch_arg] == ["L2"]
    assert [m.tag for m in l1e.forward_prefetch_arg] == ["L2E"]
    assert [m.tag for m in l2e.forward_prefetch_arg] == ["L3E"]

    with manager.suspend_expert_forward_prefetch():
        assert [m.tag for m in l0.forward_prefetch_arg] == ["L1", "L1E"]
        assert [m.tag for m in l1.forward_prefetch_arg] == ["L2"]
        assert l1e.forward_prefetch_arg == []
        assert l2e.forward_prefetch_arg == []

    assert [m.tag for m in l1e.forward_prefetch_arg] == ["L2E"]
    assert [m.tag for m in l2e.forward_prefetch_arg] == ["L3E"]
    assert l3e.forward_prefetch_arg is None


def test_flush_backward_sets_targets():
    chain = _chain([False, True, True])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    )
    manager.chains = [chain]
    _, bt = manager._build_targets()
    manager._flush_backward_targets(bt)
    l1 = chain.nodes[1].module
    # backward prefetch of L1 targets its predecessor L0 (module, not state)
    assert l1.backward_prefetch_arg == [chain.nodes[0].module]


# ---------------------------------------------------------------------------
# Manager wiring
# ---------------------------------------------------------------------------
def test_manager_native_is_noop():
    model = make_model([False, True, True])
    cfg = FSDPConfig(prefetch_strategy=FSDPPrefetchStrategy.NATIVE)
    pf.FSDPPrefetchManager(config=cfg).setup(model)
    assert model.forward_prefetch_arg is None
    assert model.backward_prefetch_arg is None


def test_manager_forward_only_no_backward():
    model = make_model([False, True, True])
    cfg = FSDPConfig(
        prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD,
        prefetch_direction=FSDPPrefetchDirection.FORWARD,
    )
    pf.FSDPPrefetchManager(config=cfg).setup(model)
    # forward configured, but no backward target on any module
    assert model.forward_prefetch_arg is None
    assert model.layers[0].forward_prefetch_arg is not None
    for layer in model.layers:
        assert layer.backward_prefetch_arg is None


def test_manager_setup_configures_backward_targets():
    model = make_model([False, True, True])
    cfg = FSDPConfig(
        prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD,
        prefetch_direction=FSDPPrefetchDirection.FORWARD_BACKWARD,
    )
    pf.FSDPPrefetchManager(config=cfg).setup(model)
    # forward bootstraps from the dense predecessor; backward on downstream layer -> previous layer
    assert model.forward_prefetch_arg is None
    assert model.layers[0].forward_prefetch_arg is not None
    assert model.layers[2].backward_prefetch_arg == [model.layers[1]]


def test_manager_cleanup_clears_targets():
    model = make_model([False, True, True])
    cfg = FSDPConfig(
        prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD,
        prefetch_direction=FSDPPrefetchDirection.FORWARD_BACKWARD,
    )
    manager = pf.FSDPPrefetchManager(config=cfg)
    manager.setup(model)
    assert model.layers[0].forward_prefetch_arg is not None
    assert model.layers[2].backward_prefetch_arg == [model.layers[1]]
    manager.cleanup()
    assert model.layers[0].forward_prefetch_arg == []
    assert model.layers[2].backward_prefetch_arg == []
    with manager.suspend_expert_forward_prefetch():
        pass
    assert model.layers[1].experts_ref.forward_prefetch_arg == []


def test_manager_setup_keeps_other_model_targets():
    policy = make_model([False, True, True])
    reference = make_model([False, True, True])
    cfg = FSDPConfig(
        prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD,
        prefetch_direction=FSDPPrefetchDirection.FORWARD_BACKWARD,
    )
    manager = pf.FSDPPrefetchManager(config=cfg)

    manager.setup(reference)
    manager.setup(policy)

    assert policy.layers[0].forward_prefetch_arg == [
        policy.layers[1],
        policy.layers[1].experts_ref,
    ]
    assert policy.layers[1].experts_ref.forward_prefetch_arg == [
        policy.layers[2].experts_ref
    ]
    assert policy.layers[2].backward_prefetch_arg == [policy.layers[1]]
    assert reference.layers[0].forward_prefetch_arg == [
        reference.layers[1],
        reference.layers[1].experts_ref,
    ]
    assert reference.layers[1].experts_ref.forward_prefetch_arg == [
        reference.layers[2].experts_ref
    ]
    assert len(manager.chains) == 2


def test_suspend_expert_forward_prefetch_can_target_one_model():
    policy = make_model([False, True, True])
    reference = make_model([False, True, True])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(
            prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD
        )
    )
    manager.setup(reference)
    manager.setup(policy)

    policy_experts = policy.layers[1].experts_ref
    reference_experts = reference.layers[1].experts_ref

    with manager.suspend_expert_forward_prefetch(policy):
        assert policy_experts.forward_prefetch_arg == []
        assert reference_experts.forward_prefetch_arg == [
            reference.layers[2].experts_ref
        ]

    assert policy_experts.forward_prefetch_arg == [
        policy.layers[2].experts_ref
    ]

    with manager.suspend_expert_forward_prefetch():
        assert policy_experts.forward_prefetch_arg == []
        assert reference_experts.forward_prefetch_arg == []

    assert policy_experts.forward_prefetch_arg == [
        policy.layers[2].experts_ref
    ]
    assert reference_experts.forward_prefetch_arg == [
        reference.layers[2].experts_ref
    ]


def test_suspend_expert_forward_prefetch_supports_same_model_nesting():
    model = make_model([False, True, True])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(
            prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD
        )
    )
    manager.setup(model)
    experts = model.layers[1].experts_ref
    targets = [model.layers[2].experts_ref]

    with manager.suspend_expert_forward_prefetch(model):
        assert experts.forward_prefetch_arg == []
        with manager.suspend_expert_forward_prefetch(model):
            assert experts.forward_prefetch_arg == []
        assert experts.forward_prefetch_arg == []

    assert experts.forward_prefetch_arg == targets

    with pytest.raises(RuntimeError, match="recompute failed"):
        with manager.suspend_expert_forward_prefetch(model):
            with manager.suspend_expert_forward_prefetch(model):
                raise RuntimeError("recompute failed")

    assert experts.forward_prefetch_arg == targets


def test_global_and_targeted_prefetch_suspensions_can_overlap():
    reference = make_model([False, True, True])
    policy = make_model([False, True, True])
    manager = pf.FSDPPrefetchManager(
        config=FSDPConfig(
            prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD
        )
    )
    manager.setup(reference)
    manager.setup(policy)

    policy_experts = policy.layers[1].experts_ref
    policy_targets = [policy.layers[2].experts_ref]
    reference_experts = reference.layers[1].experts_ref
    reference_targets = [reference.layers[2].experts_ref]

    with manager.suspend_expert_forward_prefetch(policy):
        with manager.suspend_expert_forward_prefetch():
            assert policy_experts.forward_prefetch_arg == []
            assert reference_experts.forward_prefetch_arg == []
        assert policy_experts.forward_prefetch_arg == []
        assert reference_experts.forward_prefetch_arg == reference_targets

    assert policy_experts.forward_prefetch_arg == policy_targets
    assert reference_experts.forward_prefetch_arg == reference_targets

    with manager.suspend_expert_forward_prefetch():
        with manager.suspend_expert_forward_prefetch(policy):
            assert policy_experts.forward_prefetch_arg == []
            assert reference_experts.forward_prefetch_arg == []
        assert policy_experts.forward_prefetch_arg == []
        assert reference_experts.forward_prefetch_arg == []

    assert policy_experts.forward_prefetch_arg == policy_targets
    assert reference_experts.forward_prefetch_arg == reference_targets


def test_manager_setup_clears_previous_targets():
    model = make_model([False, True, True])
    cfg = FSDPConfig(
        prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD,
        prefetch_direction=FSDPPrefetchDirection.FORWARD_BACKWARD,
    )
    manager = pf.FSDPPrefetchManager(config=cfg)
    manager.setup(model)
    assert model.layers[2].backward_prefetch_arg == [model.layers[1]]
    assert model.layers[0].forward_prefetch_arg is not None
    # switching to NATIVE and re-running setup clears prior targets
    cfg.prefetch_strategy = FSDPPrefetchStrategy.NATIVE
    manager.setup(model)
    assert model.layers[0].forward_prefetch_arg == []
    assert model.layers[2].backward_prefetch_arg == []


def test_manager_native_setup_only_clears_selected_model():
    reference = make_model([False, True, True])
    policy = make_model([False, True, True])
    cfg = FSDPConfig(
        prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD,
        prefetch_direction=FSDPPrefetchDirection.FORWARD_BACKWARD,
    )
    manager = pf.FSDPPrefetchManager(config=cfg)
    manager.setup(reference)
    manager.setup(policy)

    cfg.prefetch_strategy = FSDPPrefetchStrategy.NATIVE
    manager.setup(policy)

    assert policy.layers[0].forward_prefetch_arg == []
    assert policy.layers[2].backward_prefetch_arg == []
    assert reference.layers[0].forward_prefetch_arg == [
        reference.layers[1],
        reference.layers[1].experts_ref,
    ]
    assert reference.layers[2].backward_prefetch_arg == [reference.layers[1]]
    assert len(manager.chains) == 1


def test_manager_skips_when_torch_below_2_5(monkeypatch):
    # torch < 2.5 has no set_modules_to_*_prefetch: skip without touching modules
    monkeypatch.setattr(pf, "torch_version", lambda: (2, 4, 0))
    model = make_model([False, True, True])
    cfg = FSDPConfig(prefetch_strategy=FSDPPrefetchStrategy.EXPERT_LOOKAHEAD)
    pf.FSDPPrefetchManager(config=cfg).setup(model)
    assert model.forward_prefetch_arg is None
    assert model.layers[2].backward_prefetch_arg is None


# ---------------------------------------------------------------------------
# Config plumbing (dict / CLI overrides / LF adapter / env)
# ---------------------------------------------------------------------------
def test_fsdpconfig_defaults():
    cfg = FSDPConfig()
    assert cfg.prefetch_strategy == FSDPPrefetchStrategy.AUTO
    assert cfg.prefetch_direction == FSDPPrefetchDirection.FORWARD_BACKWARD
    assert cfg.forward_prefetch_depth == 1
    assert cfg.backward_prefetch_depth == 1


def test_fsdpconfig_prefetch_depth_rejects_below_one():
    with pytest.raises(ValueError):
        FSDPConfig(forward_prefetch_depth=0)
    with pytest.raises(ValueError):
        FSDPConfig(backward_prefetch_depth=0)


def test_build_from_args_reads_prefetch_depth():
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    tt = TorchTurboArgs(forward_prefetch_depth=3, backward_prefetch_depth=2)
    cfg = build_from_args(tt).backend_config
    assert cfg.forward_prefetch_depth == 3
    assert cfg.backward_prefetch_depth == 2


def test_build_from_args_rejects_bad_prefetch_depth():
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    with pytest.raises(ValueError):
        build_from_args(TorchTurboArgs(forward_prefetch_depth=0))


def test_engineconfig_from_dict_coerces_enums():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict(
        {
            "backend": "fsdp2",
            "prefetch_strategy": "expert_lookahead",
            "prefetch_direction": "forward",
            "forward_prefetch_depth": 3,
            "backward_prefetch_depth": 1,
        }
    )
    assert cfg.backend_config.prefetch_strategy is FSDPPrefetchStrategy.EXPERT_LOOKAHEAD
    assert cfg.backend_config.prefetch_direction is FSDPPrefetchDirection.FORWARD
    assert cfg.backend_config.forward_prefetch_depth == 3
    assert cfg.backend_config.backward_prefetch_depth == 1


def test_prefetch_direction_rejects_backward_only():
    with pytest.raises(ValueError):
        FSDPPrefetchDirection("backward")
    with pytest.raises(ValueError):
        FSDPConfig(prefetch_direction="backward")


def test_forward_only_prefetch_rejects_backward_depth():
    with pytest.raises(ValueError):
        FSDPConfig(prefetch_direction="forward", backward_prefetch_depth=2)


def test_build_from_args_defaults_auto():
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    cfg = build_from_args(TorchTurboArgs()).backend_config
    assert cfg.prefetch_strategy is FSDPPrefetchStrategy.AUTO
    assert cfg.prefetch_direction is FSDPPrefetchDirection.FORWARD_BACKWARD


def test_env_injects_prefetch_strategy(monkeypatch):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    # Use a non-default value so this asserts env injection, not the default.
    monkeypatch.setenv("TORCH_TURBO_PREFETCH_STRATEGY", "expert_with_layer")
    cfg = build_from_args(TorchTurboArgs())
    assert cfg.backend_config.prefetch_strategy == FSDPPrefetchStrategy.EXPERT_WITH_LAYER


def test_env_injects_prefetch_depth(monkeypatch):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    monkeypatch.setenv("TORCH_TURBO_FORWARD_PREFETCH_DEPTH", "3")
    monkeypatch.setenv("TORCH_TURBO_BACKWARD_PREFETCH_DEPTH", "2")
    cfg = build_from_args(TorchTurboArgs())
    assert cfg.backend_config.forward_prefetch_depth == 3
    assert cfg.backend_config.backward_prefetch_depth == 2


def test_env_rejects_bad_prefetch_depth(monkeypatch):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    monkeypatch.setenv("TORCH_TURBO_BACKWARD_PREFETCH_DEPTH", "0")
    with pytest.raises(ValueError):
        build_from_args(TorchTurboArgs())


def test_env_rejects_backward_only(monkeypatch):
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    monkeypatch.setenv("TORCH_TURBO_PREFETCH_DIRECTION", "backward")
    with pytest.raises(ValueError):
        build_from_args(TorchTurboArgs())


def test_engineconfig_nested_backend_config_rejects_backward_only():
    from torchturbo.engine.config import EngineConfig

    with pytest.raises(ValueError):
        EngineConfig.from_dict(
            {
                "backend": "fsdp2",
                "backend_config": {"prefetch_direction": "backward"},
            }
        )


def test_engineconfig_keeps_removed_prefetch_keys_in_extra():
    from torchturbo.engine.config import EngineConfig

    cfg = EngineConfig.from_dict({"backend": "fsdp2", "backward_prefetch": True})
    assert cfg.extra["backward_prefetch"] is True
