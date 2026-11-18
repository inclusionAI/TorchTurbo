"""Native engine/model tests with a small causal backbone, not a full policy."""

from __future__ import annotations
import types
import torch
from torchturbo.adapters.rl import packed, sp
from torchturbo.engine import core
from torchturbo.engine.config import (
    EngineConfig,
    SparseLogitsConfig,
)
from torchturbo.models.qwen3_5.model import TurboQwen3_5ForConditionalGeneration
from torchturbo.models.base import TurboModel
from torchturbo.engine.backends.fsdp2 import FSDP2Backend
from torchturbo.models.modules.attention.ulysses import (
    SequenceParallelInputSlicingSpec,
    patch_forward_for_sp_input_slicing,
)


def load_runtime():
    return core, packed, sp, MaskedPolicyLoss


class MaskedPolicyLoss:
    """Minimal masked PPO objective; the framework tests its own GRPOLoss."""

    def __init__(self, params):
        self.params = params
        self.last_stats = {}

    def loss_weight(self, batch):
        return batch.data["ppo_loss_mask"].count_nonzero()

    def _loss_from_logprobs(self, logp, batch, entropy):
        data = batch.data
        mask = data["ppo_loss_mask"].bool()
        ratio = (logp - data["old_logp"]).exp()
        adv = data["advantages"]
        policy = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.28) * adv)
        if self.params["loss_reduction"] == "sample":
            pieces = policy.split([n[0] - 1 for n in batch.seqlens["packed_input_ids"]])
            masks = mask.split([len(p) for p in pieces])
            loss = torch.stack(
                [(p * m).sum() / m.sum().clamp(min=1) for p, m in zip(pieces, masks)]
            ).mean()
        else:
            loss = (policy * mask).sum() / mask.sum().clamp(min=1)
        ent = (entropy * mask).sum() / mask.sum().clamp(min=1)
        self.last_stats = {"entropy": ent.detach().item()}
        return loss - self.params["entropy_coeff"] * ent


class SmallModel(TurboModel, torch.nn.Module):
    def __init__(
        self,
        hidden=12,
        vocab=41,
        dtype=torch.float32,
        device="cpu",
        sp_rank=0,
        sp_size=1,
        post_embedding=False,
        mix_context=False,
    ):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden, dtype=dtype, device=device)
        self.lm_head = torch.nn.Linear(
            hidden, vocab, bias=False, dtype=dtype, device=device
        )
        self.config = types.SimpleNamespace(model_type="qwen3_5")
        self.mtp = None
        self.sp_rank, self.sp_size = sp_rank, sp_size
        self.post_embedding = post_embedding
        self.mix_context = mix_context
        self.seen = []

    def get_sequence_parallel_input_slicing_spec(self):
        if not self.post_embedding:
            return super().get_sequence_parallel_input_slicing_spec()
        return SequenceParallelInputSlicingSpec(
            self,
            SmallModel._forward_hidden,
            forward_name="_forward_hidden",
            sliced_inputs=(("inputs_embeds", 1),),
        )

    def _forward_hidden(self, inputs_embeds):
        return inputs_embeds

    def forward_sparse_logits(self, inputs, logits_to_keep):
        return self(**inputs, logits_to_keep=logits_to_keep, compute_mtp=False)

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        hidden = self.embed(input_ids)
        if self.mix_context:
            # Small causal surrogate makes earlier masked inputs affect selected
            # predictions, checking that row selection prunes ONLY the output head.
            hidden = hidden.cumsum(dim=1) / torch.arange(
                1, hidden.shape[1] + 1, device=hidden.device
            ).view(1, -1, 1).to(hidden.dtype)
        hidden = self._forward_hidden(inputs_embeds=hidden)
        ix = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        selected = hidden[:, ix, :]
        self.seen.append((input_ids.shape[1], selected.shape[1]))
        return types.SimpleNamespace(
            logits=self.lm_head(selected), metadata="preserved"
        )


def make_mb(
    packed,
    sp,
    *,
    lengths=(31,),
    active=(2, 7, 29),
    rank=0,
    world=1,
    group=None,
    post_embedding=False,
    device="cpu",
    vocab=41,
    pad=0,
    model=None,
):
    total = sum(lengths)
    ids = (torch.arange(total, device=device) * 13 + 1) % vocab
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    mask = torch.zeros(total - len(lengths), dtype=torch.bool, device=device)
    if active:
        mask[list(active)] = True
    old = torch.full((mask.numel(),), -4.0, device=device)
    data = dict(
        packed_input_ids=ids,
        ppo_loss_mask=mask,
        old_logp=old,
        advantages=torch.linspace(-1, 1, mask.numel(), device=device),
    )
    batch = types.SimpleNamespace(
        data=data, seqlens={"packed_input_ids": [[n] for n in lengths]}
    )
    orig = dict(input_ids=ids, cu_seqlens=cu, _packed_batch=batch, **data)
    info = dict(sp_size=world, sp_rank=rank, sp_group=group)
    prep_model = (
        model
        if model is not None
        else SmallModel(
            device=device,
            post_embedding=post_embedding,
            sp_rank=rank,
            sp_size=world,
        )
    )
    prep = core.TorchTurboEngine.__new__(core.TorchTurboEngine)
    prep.model = prep_model
    prep.get_sp_info = lambda: info
    prep.backend = FSDP2Backend.__new__(FSDP2Backend)
    prep.backend.get_sp_info = lambda: info
    model_inputs = prep.prepare_packed_micro_batch(orig, list(lengths))
    if pad:
        # Also exercise callers that append an explicit ignored tail at SP1.
        assert world == 1
        model_inputs["input_ids"] = torch.nn.functional.pad(
            model_inputs["input_ids"], (0, pad)
        )
        model_inputs["position_ids"] = torch.nn.functional.pad(
            model_inputs["position_ids"], (0, pad)
        )
        model_inputs["_pad_length"] = pad
    return dict(orig_mb=orig, padded_mb=model_inputs), info


def engine(core, model, info, *, sparse=False):
    obj = core.TorchTurboEngine.__new__(core.TorchTurboEngine)
    obj.model = model
    if info["sp_size"] > 1:
        patch_forward_for_sp_input_slicing(
            model.get_sequence_parallel_input_slicing_spec(),
            info["sp_group"],
            info["sp_size"],
            info["sp_rank"],
        )
    obj.get_sp_info = lambda: info
    obj.get_dp_info = lambda: dict(dp_size=1, dp_group=None)
    obj.config = EngineConfig(
        sparse_logits=SparseLogitsConfig(enabled=sparse),
    )
    return obj


def run_batch(eng, mb, Loss, *, entropy_coeff=0.03, reduction="token", temperature=0.7):
    loss = Loss(
        dict(
            loss_reduction=reduction,
            eps_clip=0.2,
            eps_clip_higher=0.28,
            c_clip=3.0,
            use_kl_in_loss=False,
            importance_sampling_level="token",
            entropy_coeff=entropy_coeff,
        )
    )
    captured = {}

    def loss_fn(logprobs, entropy, mb_input):
        value = loss._loss_from_logprobs(
            logprobs, mb_input["_packed_batch"], entropy=entropy
        )
        captured.update(
            lp=logprobs.detach().clone(),
            ent=entropy.detach().clone(),
            loss=value.detach().clone(),
            stats=dict(loss.last_stats),
        )
        return value

    def weight(mb_input):
        return (
            loss.loss_weight(mb_input["_packed_batch"]) * eng.get_sp_info()["sp_size"]
        )

    eng.forward_backward_batch(
        [mb],
        process_fn=None,
        loss_fn=loss_fn,
        loss_weight_fn=weight,
        temperature=temperature,
        device=eng.model.embed.weight.device,
        loss_uses_ppo_mask=True,
    )
    captured["grads"] = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in eng.model.named_parameters()
    }
    return captured


def check_equal(dense, sparse, mask, atol=2e-6, rtol=2e-5):
    for k in ("lp", "ent"):
        torch.testing.assert_close(
            dense[k][mask], sparse[k][mask], atol=atol, rtol=rtol
        )
    torch.testing.assert_close(dense["loss"], sparse["loss"], atol=atol, rtol=rtol)
    assert dense["grads"].keys() == sparse["grads"].keys()
    for key, value in dense["grads"].items():
        if value is None:
            assert sparse["grads"][key] is None, key
            continue
        assert torch.isfinite(sparse["grads"][key]).all(), key
        torch.testing.assert_close(value, sparse["grads"][key], atol=atol, rtol=rtol)
    for key, value in dense["stats"].items():
        assert abs(value - sparse["stats"][key]) < max(atol, abs(value) * rtol), key


def small_qwen_model(**kwargs):
    """Real native Qwen outer forward/API; only backbone and MTP math are small."""

    class Output(types.SimpleNamespace):
        def __getitem__(self, i):
            assert i == 0
            return self.hidden_states

    class Backbone(torch.nn.Module):
        def __init__(self, embed, rank, size, post):
            super().__init__()
            self.embed = embed
            self.language_model = types.SimpleNamespace(
                embed_tokens=embed, rotary_emb=lambda *a: None
            )
            self.rank, self.size, self.post = rank, size, post
            self.input_len = None

        def forward(self, input_ids, inputs_embeds=None, **kw):
            hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
            self.input_len = hidden.shape[1]
            hidden = hidden.cumsum(dim=1) / torch.arange(
                1, hidden.shape[1] + 1, device=hidden.device
            ).view(1, -1, 1).to(hidden.dtype)
            hidden = self._forward_hidden(inputs_embeds=hidden)
            return Output(
                hidden_states=hidden,
                past_key_values=None,
                attentions=None,
                rope_deltas=None,
            )

        def _forward_hidden(self, inputs_embeds):
            return inputs_embeds

    class MTP(torch.nn.Module):
        def __init__(self, embed):
            super().__init__()
            self.proj = torch.nn.Linear(
                embed.embedding_dim,
                embed.embedding_dim,
                bias=False,
                device=embed.weight.device,
                dtype=embed.weight.dtype,
            )
            self.calls = 0

        def forward(self, hidden_states, **kw):
            self.calls += 1
            return [self.proj(hidden_states)]

    class Model(TurboQwen3_5ForConditionalGeneration):
        def __init__(self):
            # Construct the real wrapper without loading a policy or vision tower.
            torch.nn.Module.__init__(self)
            small = SmallModel(**kwargs)
            self.embed, self.lm_head = small.embed, small.lm_head
            self.config, self.seen = small.config, []
            self.sp_rank, self.sp_size = small.sp_rank, small.sp_size
            self.post_embedding = small.post_embedding
            self.config.return_dict = True
            self.config.text_config = types.SimpleNamespace(
                pad_token_id=0, mtp_num_hidden_layers=1
            )
            self.model = Backbone(
                self.embed, self.sp_rank, self.sp_size, self.post_embedding
            )
            self.model.config = types.SimpleNamespace(
                image_token_id=1000, video_token_id=1001
            )
            self.mtp = MTP(self.embed)
            # Record both the main and auxiliary projection calls.
            self.lm_head.register_forward_pre_hook(self.record_head)

        def get_sequence_parallel_input_slicing_spec(self):
            if not self.post_embedding:
                return TurboModel.get_sequence_parallel_input_slicing_spec(self)
            return SequenceParallelInputSlicingSpec(
                self.model,
                Backbone._forward_hidden,
                forward_name="_forward_hidden",
                sliced_inputs=(("inputs_embeds", 1),),
            )

        def record_head(self, module, args):
            self.seen.append((self.model.input_len, args[0].shape[1]))

    return Model()
