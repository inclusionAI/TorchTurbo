"""DeepSeek-V4 parameter semantics for TorchTurbo's distributed Muon."""

from __future__ import annotations

from torch import nn

from torchturbo.engine.optimizers import ExcludeFromMuon, MuonRecipe


def get_muon_recipe(model: nn.Module) -> MuonRecipe:
    """Keep replicated hyper-connection parameters on AdamW.

    Embeddings and the shared output head use the engine's AdamW defaults.
    GroupedMLP declares the independent expert matrices. Other sharded
    projection weights use the default whole-matrix Muon view.
    """
    excluded = ExcludeFromMuon()
    parameter_overrides = {
        "hc_head_fn": excluded,
        "**.hc_attn_fn": excluded,
        "**.hc_ffn_fn": excluded,
    }
    if model.mtp is not None:
        parameter_overrides["mtp.*.hc_head_fn"] = excluded
    return MuonRecipe(parameter_overrides=parameter_overrides)
