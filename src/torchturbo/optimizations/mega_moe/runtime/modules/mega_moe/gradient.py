"""Initialize the standalone grouped-GEMM gradient outputs."""
import torch


def zero_empty_expert_gradients(grad1: torch.Tensor, grad2: torch.Tensor,
                               group_list: torch.Tensor) -> None:
    grad1.zero_()
    grad2.zero_()
