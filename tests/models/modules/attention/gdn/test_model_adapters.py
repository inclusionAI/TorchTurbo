"""GDN adapter binding covers backbone and MTP modules at every SP size."""

from types import SimpleNamespace

import pytest
from torch import nn

from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from torchturbo.models.qwen3_5_moe.model import TurboQwen3_5MoeForConditionalGeneration
from torchturbo.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
from torchturbo.models.qwen3_5_moe_text.model import TurboQwen3_5MoeTextForCausalLM


_ADAPTERS = [
    (TurboQwen3_5ForConditionalGeneration, Qwen3_5GatedDeltaNet),
    (TurboQwen3_5MoeForConditionalGeneration, Qwen3_5MoeGatedDeltaNet),
    (TurboQwen3_5MoeTextForCausalLM, Qwen3_5MoeGatedDeltaNet),
]


@pytest.mark.parametrize("adapter_cls,gdn_cls", _ADAPTERS)
@pytest.mark.parametrize("sp_size", [1, 2])
def test_gdn_class_forward_covers_backbone_and_mtp(
    adapter_cls, gdn_cls, sp_size, monkeypatch
):
    config = SimpleNamespace(
        hidden_size=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=None,
    )
    monkeypatch.setattr(f"{gdn_cls.__module__}.FusedRMSNormGated", None)
    owner = nn.Module()
    owner.model = nn.Sequential(gdn_cls(config, layer_idx=0))
    owner.mtp = nn.Sequential(gdn_cls(config, layer_idx=1))
    original_forward = gdn_cls.forward
    monkeypatch.setattr(gdn_cls, "forward", original_forward)
    # Keep GDN installation real without configuring unrelated attention
    # interfaces or MTP input slicing on this minimal model.
    owner.register_sp_attention_interface = lambda sp_info: None
    owner.patch_mtp_for_sp = lambda mtp_cls, sp_group, sp_size, sp_rank: None

    adapter_cls.on_parallelized(
        owner, {"sp": {"sp_group": None, "sp_size": sp_size, "sp_rank": 0}}
    )

    assert gdn_cls.forward is not original_forward
    assert owner.model[0].forward.__func__ is gdn_cls.forward
    assert owner.mtp[0].forward.__func__ is gdn_cls.forward
