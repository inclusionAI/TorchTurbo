from types import SimpleNamespace

import pytest
import torch

from torchturbo.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel


@pytest.mark.parametrize(("sp_size", "local_length"), [(1, 5), (1, 6), (2, 3), (3, 2)])
def test_indexer_receives_global_packed_boundaries(sp_size, local_length):
    captured = []

    def decoder(hidden_states, **kwargs):
        captured.append(kwargs["attention_mask"])
        return (hidden_states,)

    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=False, use_return_dict=False),
        layers=[decoder],
        _use_flash_attention_2=True,
        _sp_size=sp_size,
        rotary_emb=lambda *args, **kwargs: None,
        gradient_checkpointing=False,
        training=True,
        norm=lambda x: x,
    )
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    GlmMoeDsaModel.forward(
        model,
        inputs_embeds=torch.ones(1, local_length, 4),
        cu_seq_lens=cu,
    )

    expected = [1, 1, 2, 2, 2] + [0] * (local_length * sp_size - 5)
    assert captured[0].tolist() == [expected]
    assert cu.tolist() == [0, 2, 5]
