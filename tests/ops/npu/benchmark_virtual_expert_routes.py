"""Reproducible route distributions for the MoE benchmark."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingCase:
    routing: str
    num_hot_experts: int = 0
    hot_fraction: float = 0.0
    hot_layout: str = "packed"
    seed: int = 0

    def hot_ids(self, ep_size: int, *, num_experts: int) -> list[int]:
        """Place hot experts consecutively or across their owning ranks."""
        if self.hot_layout == "packed":
            return list(range(self.num_hot_experts))
        local_experts = num_experts // ep_size
        return [(i % ep_size) * local_experts + i // ep_size for i in range(self.num_hot_experts)]

    def route_ids(
        self, tokens: int, ep_size: int, rank: int, *, num_experts: int, topk: int,
    ) -> torch.Tensor:
        """Generate distinct experts per token with the requested hot route share."""
        if not 1 <= topk <= num_experts or num_experts % ep_size:
            raise ValueError("topk must fit the experts, and experts must divide evenly across EP ranks.")
        if self.routing == "balanced":
            return torch.arange(tokens * topk).reshape(tokens, topk) % num_experts
        generator = torch.Generator().manual_seed(self.seed + rank)
        if self.routing == "random":
            return torch.rand(tokens, num_experts, generator=generator).topk(topk, dim=1).indices
        hot_ids = self.hot_ids(ep_size, num_experts=num_experts)
        cold_ids = [expert for expert in range(num_experts) if expert not in hot_ids]
        if not max(0, topk - len(cold_ids)) <= self.hot_fraction * topk <= min(topk, len(hot_ids)):
            raise ValueError("The hot share cannot be represented with distinct experts per token.")
        num_routes = tokens * topk
        num_hot_routes = round(num_routes * self.hot_fraction)
        base, remainder = divmod(num_hot_routes, tokens)
        hot_per_token = base + (torch.arange(tokens) < remainder)
        hot_mask = torch.arange(topk)[None, :] < hot_per_token[:, None]
        routes = torch.empty((tokens, topk), dtype=torch.int64)
        routes[hot_mask] = torch.tensor(hot_ids)[(torch.arange(num_hot_routes) + rank) % len(hot_ids)]
        routes[~hot_mask] = torch.tensor(cold_ids)[
            (torch.arange(num_routes - num_hot_routes) + rank) % len(cold_ids)
        ]
        return routes[torch.randperm(tokens, generator=generator)]

    def metadata(self, ep_size: int, *, num_experts: int, topk: int) -> dict:
        """Return enough information to distinguish every measured distribution."""
        return {
            "routing": self.routing,
            "route_seed": self.seed if self.routing != "balanced" else None,
            "num_hot_experts": self.num_hot_experts,
            "hot_fraction": self.hot_fraction,
            "hot_multiplier": self.hot_fraction * num_experts / self.num_hot_experts if self.num_hot_experts else None,
            "hot_layout": self.hot_layout if self.num_hot_experts else None,
            "hot_expert_ids": self.hot_ids(ep_size, num_experts=num_experts),
            "num_experts": num_experts,
            "local_experts": num_experts // ep_size,
            "topk": topk,
        }


def routing_cases(
    ep_size: int, distributions: list[str], seeds: list[int],
    hot_experts: list[int], hot_fractions: list[float], hot_layouts: list[str],
    *, num_experts: int, topk: int, hot_multipliers: list[float] | None = None,
) -> list[RoutingCase]:
    """Omit full-expert groups, non-hot shares, and duplicate one-expert layouts."""
    cases = []
    for distribution in distributions:
        if distribution == "balanced":
            cases.append(RoutingCase("balanced"))
        elif distribution == "random":
            cases.extend(RoutingCase("random", seed=seed) for seed in seeds)
        else:
            for count in hot_experts:
                if count >= num_experts:
                    continue
                fractions = [factor * count / num_experts for factor in hot_multipliers] if hot_multipliers else hot_fractions
                for fraction in fractions:
                    if fraction <= count / num_experts:
                        continue
                    if not max(0, topk - (num_experts - count)) <= fraction * topk <= min(topk, count):
                        continue
                    for layout in (hot_layouts[:1] if count == 1 else hot_layouts):
                        cases.append(RoutingCase("hotspot", count, fraction, layout, seeds[0]))
    return cases
