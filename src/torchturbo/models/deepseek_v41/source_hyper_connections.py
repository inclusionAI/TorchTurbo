"""Original hyper-connection helpers used by the V4.1 reference model."""
import torch
from torch import nn
from torch.nn import functional as F


class DeepseekV4HCSplitSinkhorn(nn.Module):
    """HC Split Sinkhorn for hyper-connections."""

    def __init__(self, hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

    def forward(self, mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Forward pass for HC split sinkhorn."""
        hc_mult = self.hc_mult
        eps = self.hc_eps

        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = (
            F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0))
            + eps
        )
        post = 2 * F.sigmoid(
            post * hc_scale[1]
            + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0)
        )
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(
            hc_mult, hc_mult
        ).unsqueeze(0).unsqueeze(0)

        comb = comb.softmax(-1) + eps
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)

        for _ in range(self.hc_sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            comb = comb / (row_sum + eps)
            col_sum = comb.sum(-2, keepdim=True)
            comb = comb / (col_sum + eps)

        return pre, post, comb

class HcPost(nn.Module):
    """HC post-processing: expand 1 -> hc copies.

    Kept as separate module for easy fusion operator substitution.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        # y_k = post_k * x + sum_n comb[n, k] * residual_n. Evaluated as two
        # batched matmuls: the broadcast form materializes a [b, s, n, n, d]
        # fp32 product (2.7 GB at the production shape) that dominates the
        # activation memory budget.
        b, s, n, d = residual.shape
        y = torch.bmm(
            comb.reshape(b * s, n, n).transpose(-1, -2),
            residual.reshape(b * s, n, d).float(),
        )
        y += post.reshape(b * s, n, 1) * x.reshape(b * s, 1, d).float()
        return y.reshape(b, s, n, d).to(x.dtype)
