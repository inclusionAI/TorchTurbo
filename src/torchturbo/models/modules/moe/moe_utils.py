import torch


def mask_padding_tokens(
    values: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Clear padding rows from a flattened token-by-expert tensor."""
    if padding_mask is None:
        return values
    valid_mask = (~padding_mask.reshape(-1).bool()).unsqueeze(-1)
    return values * valid_mask


def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor,
    topk: int,
    score_function: str,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the bias-free AUX routing map; True in padding_mask is excluded."""
    logits = logits.reshape(-1, logits.size(-1))
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    top_indices = torch.topk(scores, k=topk, dim=-1).indices
    routing_map = torch.zeros_like(logits, dtype=torch.int32).scatter_(
        1, top_indices, 1
    ).bool()

    if padding_mask is not None:
        scores = mask_padding_tokens(scores, padding_mask)
        routing_map = mask_padding_tokens(routing_map, padding_mask).bool()

    return routing_map, scores


def z_loss_func(
    logits: torch.Tensor,
    z_loss_coeff: float,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ST-MoE z-loss over local valid tokens.

    Args:
        logits: Router logits of shape ``[num_tokens, num_experts]``.
        z_loss_coeff: Z-loss coefficient.
        padding_mask: True entries are excluded from the mean.

    Returns:
        Scalar z-loss including ``z_loss_coeff``.
    """
    logsum = torch.logsumexp(logits, dim=-1)
    z_loss_values = torch.square(logsum)

    if padding_mask is not None:
        valid_mask = ~padding_mask.reshape(-1).bool()
        z_loss_values = z_loss_values * valid_mask
        num_valid_tokens = valid_mask.sum()
        z_loss = z_loss_values.sum() / torch.clamp(num_valid_tokens, min=1)
    else:
        z_loss = z_loss_values.sum() if logits.shape[0] == 0 else z_loss_values.mean()

    return z_loss * z_loss_coeff


class MoEAuxLossAutoScaler(torch.autograd.Function):
    loss_scale = 1.0

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss) * MoEAuxLossAutoScaler.loss_scale

    @classmethod
    def set_loss_scale(cls, scale: float) -> None:
        cls.loss_scale = scale
