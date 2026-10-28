import torch

from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.router import TopKRouter, update_router_expert_bias


def make_router():
    config = MoEConfig(
        num_moe_experts=2,
        hidden_size=1,
        perform_initialization=False,
        moe_router_enable_expert_bias=True,
    )
    model = torch.nn.Module()
    model.router = TopKRouter(config)
    model.router.register_buffer(
        "local_tokens_per_expert", torch.zeros(2), persistent=False
    )
    return model


def test_router_bias_update_uses_global_load_and_resets_stats(monkeypatch):
    model = make_router()
    model.router.local_tokens_per_expert.copy_(torch.tensor([0.0, 10.0]))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    updated = update_router_expert_bias(model, gamma=0.001, clamp=0.05)

    assert updated == 1
    torch.testing.assert_close(model.router.expert_bias, torch.tensor([0.001, -0.001]))
    torch.testing.assert_close(
        model.router.local_tokens_per_expert, torch.zeros(2)
    )


def test_router_bias_respects_clamp_and_zero_gamma(monkeypatch):
    model = make_router()
    model.router.expert_bias.fill_(0.049)
    model.router.local_tokens_per_expert.copy_(torch.tensor([0.0, 2.0]))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    update_router_expert_bias(model, gamma=0.01, clamp=0.05)
    torch.testing.assert_close(model.router.expert_bias, torch.tensor([0.05, 0.039]))
    assert update_router_expert_bias(model, gamma=0.0) == 0
