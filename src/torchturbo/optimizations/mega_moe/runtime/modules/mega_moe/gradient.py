"""Clear gradients only for experts with no routed rows."""
import torch

from ...moonep import load_allocator


def zero_empty_expert_gradients(
    grad1: torch.Tensor, grad2: torch.Tensor, group_list: torch.Tensor,
) -> None:
    """Preserve active experts and clear skipped experts in one AscendC launch."""
    load_allocator().zero_empty(grad1, grad2, group_list)
