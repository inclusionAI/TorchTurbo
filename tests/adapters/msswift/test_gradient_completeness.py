import torch

from torchturbo.adapters.msswift.trainer import gradient_completeness


def test_gradient_completeness_groups_experts_router_and_backbone():
    model = torch.nn.Module()
    model.experts = torch.nn.Linear(2, 2)
    model.router = torch.nn.Linear(2, 2)
    model.fc = torch.nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    stats = gradient_completeness(model)

    assert stats == {
        "experts": {"required": 2, "present": 2, "missing": 0},
        "router": {"required": 2, "present": 2, "missing": 0},
        "backbone": {"required": 2, "present": 2, "missing": 0},
    }
    model.router.weight.grad = None
    model.fc.bias.requires_grad_(False)
    stats = gradient_completeness(model)
    assert stats["router"] == {"required": 2, "present": 1, "missing": 1}
    assert stats["backbone"] == {"required": 1, "present": 1, "missing": 0}
