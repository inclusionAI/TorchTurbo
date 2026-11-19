"""Interface-only bridges to the original router and token dispatcher."""

from .moe_source_router import TopKRouter
from .moe_source_token_dispatcher import MoEAlltoAllTokenDispatcher


class SourceRouter(TopKRouter):
    def forward(self, hidden_states, input_ids=None, padding_mask=None, accumulate_tokens_per_expert=True):
        probs, routing_map, logits, top_indices = super().forward(hidden_states, input_ids, padding_mask)
        return probs, routing_map, top_indices, logits


class SourceTokenDispatcher(MoEAlltoAllTokenDispatcher):
    def dispatch_preprocess(self, hidden_states, routing_map, probs, *, expert_indices=None):
        self._source_probs = probs
        self._source_indices = expert_indices
        tokens, self._source_counts, probabilities = self.token_permutation(
            hidden_states, probs, routing_map, top_indices=expert_indices,
        )
        return tokens, probabilities

    def token_dispatch(self, tokens, probabilities):
        return tokens, probabilities

    def dispatch_postprocess(self, tokens, probabilities):
        return tokens, self._source_counts, probabilities

    def combine_preprocess(self, tokens):
        return tokens

    def token_combine(self, tokens):
        return tokens

    def combine_postprocess(self, tokens):
        probabilities = None
        if self.deferred_probs_to_unpermutation:
            probabilities = self._source_probs.gather(1, self._source_indices)
        output, _ = self.token_unpermutation(tokens, probs=probabilities)
        self._source_probs = self._source_indices = None
        return output
