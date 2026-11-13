"""Standard MTP masks shifted padding before router losses and statistics."""

import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.config import EngineConfig
from torchturbo.models.deepseek_v4 import mtp
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.moe_utils import MoEAuxLossAutoScaler
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.parallel import FSDPPlanSpec
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture(autouse=True)
def cpu_dispatch(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime,
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(MoEAuxLossAutoScaler, "loss_scale", 1.0)
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)


class RouterMTPStage(torch.nn.Module):
    """Isolate router losses while the real MTP driver shifts stage inputs."""

    def __init__(self, config, checkpoint_mode):
        super().__init__()
        self.ffn = MoELayer(
            config, object(),
            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
        )
        self.checkpoint_mode = checkpoint_mode

    def forward(self, hidden_states, inputs_embeds, **kwargs):
        if self.checkpoint_mode is None:
            outputs, _ = self.ffn(inputs_embeds)
        else:
            outputs, _ = checkpoint(
                self.ffn, inputs_embeds, use_reentrant=self.checkpoint_mode,
            )
        return hidden_states, outputs


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("checkpoint_target", ["0", "0.ffn"])
@pytest.mark.parametrize("aux_coeff,z_coeff", [(0.2, 0.0), (0.0, 0.3), (0.2, 0.3)])
def test_checkpoint_preserves_mtp_padding_for_each_forward(
    use_reentrant, checkpoint_target, aux_coeff, z_coeff,
):
    torch.manual_seed(17)
    config = MoEConfig(
        hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=3,
        moe_router_topk=2, moe_router_dtype="fp32", moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_aux_loss_coeff=aux_coeff, moe_z_loss_coeff=z_coeff,
        gated_linear_unit=True, activation_func=F.silu,
    )
    reference = SimpleNamespace(
        mtp=torch.nn.ModuleList([RouterMTPStage(config, None)]),
        embed=torch.nn.Embedding(8, 4), head=torch.nn.Identity(), padding_idx=0,
    )
    with torch.no_grad():
        for parameter in reference.mtp.parameters():
            parameter.normal_(std=0.2)
    candidate = copy.deepcopy(reference)
    backend = SimpleNamespace(
        config=EngineConfig.from_dict({
            "activation_checkpointing": {"enabled": True, "use_reentrant": use_reentrant},
        }),
        prefetch_manager=SimpleNamespace(suspend_expert_forward_prefetch=lambda model: nullcontext()),
    )
    FSDP2Backend._apply_activation_checkpointing(
        backend, candidate.mtp,
        SimpleNamespace(modules={checkpoint_target: FSDPPlanSpec(activation_checkpoint=True)}),
    )
    forwarder = mtp.DeepseekV4MTPForwarder()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    outputs, expected = [], []
    for boundaries in ([0, 3, 6], [0, 2, 6]):
        cu = torch.tensor(boundaries, dtype=torch.int32)
        outputs += forwarder._mtp_logits(candidate, torch.zeros(1, 6, 4), ids, None, None, cu)
        expected += forwarder._mtp_logits(reference, torch.zeros(1, 6, 4), ids, None, None, cu)
    latest_inputs = candidate.mtp[0].ffn._runtime_inputs
    for actual, ref in zip(outputs, expected):
        torch.testing.assert_close(actual, ref)
    sum(output.square().sum() for output in outputs).backward()
    sum(output.square().sum() for output in expected).backward()
    for actual, ref in zip(candidate.mtp.parameters(), reference.mtp.parameters()):
        torch.testing.assert_close(actual.grad, ref.grad)
    torch.testing.assert_close(candidate.embed.weight.grad, reference.embed.weight.grad)
    assert candidate.mtp[0].ffn._runtime_inputs is latest_inputs
    # Each forward contributes its unpadded assignments once, with or without replay.
    torch.testing.assert_close(
        candidate.mtp[0].ffn.router.local_tokens_per_expert,
        reference.mtp[0].ffn.router.local_tokens_per_expert,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sp_size,sp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("aux_coeff,z_coeff", [(0.2, 0.0), (0.0, 0.3), (0.2, 0.3)])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
def test_mtp_stage_masks_and_router_gradients(
    monkeypatch, packed, sp_size, sp_rank, aux_coeff, z_coeff, checkpoint_mode,
):
    torch.manual_seed(41)
    if packed:
        global_ids = torch.tensor([[1, 2, 0, 4, 5, 6, 0, 0]])
        cu = torch.tensor([0, 3, 6], dtype=torch.int32)
        valid_positions = [
            [True, True, False, True, True, False, False, False],
            [True, False, False, True, False, False, False, False],
            [False] * 8,
            [False] * 8,
        ]
    else:
        global_ids = torch.tensor([[1, 2, 3, 4], [4, 0, 2, 1]])
        cu = None
        valid_positions = [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
            [False] * 4,
        ]
    local_length = global_ids.size(1) // sp_size
    start = sp_rank * local_length
    local_ids = global_ids[:, start:start + local_length]

    def gather_ids(tensor, *, dim, sp_group, sp_size):
        torch.testing.assert_close(tensor, local_ids)
        return global_ids

    # Only substitute communication; use the production stage slicing and MoE.
    monkeypatch.setattr(mtp, "all_gather_sp_with_grad_reduce", gather_ids)
    config = MoEConfig(
        hidden_size=4, moe_ffn_hidden_size=3, num_moe_experts=3,
        moe_router_topk=2, moe_router_score_function="sigmoid",
        moe_router_dtype="fp32", moe_router_enable_expert_bias=True,
        moe_aux_loss_coeff=aux_coeff, moe_z_loss_coeff=z_coeff,
        gated_linear_unit=True, activation_func=F.silu,
    )
    model = SimpleNamespace(
        mtp=torch.nn.ModuleList(RouterMTPStage(config, checkpoint_mode) for _ in range(4)),
        embed=torch.nn.Embedding(8, 4), head=torch.nn.Identity(), padding_idx=0,
    )
    with torch.no_grad():
        for parameter in model.mtp.parameters():
            parameter.normal_(std=0.2)
    embeddings = []

    def capture_embedding(_module, args, output):
        output.retain_grad()
        embeddings.append((args[0].detach().clone(), output))

    handle = model.embed.register_forward_hook(capture_embedding)
    forwarder = mtp.DeepseekV4MTPForwarder(sp_size=sp_size, sp_rank=sp_rank)
    try:
        outputs = forwarder._mtp_logits(
            model, torch.zeros(*local_ids.shape, 4), local_ids,
            None, None, cu,
        )
    finally:
        handle.remove()
    sum(output.sum() * 0 for output in outputs).backward()

    for stage_index, (stage, (shifted_ids, embedded)) in enumerate(
        zip(model.mtp, embeddings, strict=True), start=1,
    ):
        expected_ids = torch.tensor([
            [row[pos] if pos < len(row) else 0
             for pos in range(start + stage_index, start + stage_index + local_length)]
            for row in global_ids.tolist()
        ])
        torch.testing.assert_close(shifted_ids, expected_ids)
        valid = torch.tensor(valid_positions[stage_index - 1])[start:start + local_length]
        valid = valid.unsqueeze(0).expand_as(local_ids)
        actual_mask = stage.ffn._runtime_inputs.padding_mask
        torch.testing.assert_close(actual_mask.expand_as(valid), ~valid)

        ref_inputs = embedded.detach().clone().requires_grad_()
        ref_weight = stage.ffn.router.weight.detach().clone().requires_grad_()
        logits = F.linear(ref_inputs, ref_weight)[valid]
        if logits.size(0):
            scores = logits.sigmoid()
            indices = scores.topk(2, dim=-1).indices
            counts = torch.bincount(indices.flatten(), minlength=3)
            scores = scores / scores.sum(-1, keepdim=True)
            aux = 3 * (scores.sum(0) * counts).sum() / (2 * logits.size(0) ** 2)
            z_loss = logits.logsumexp(-1).square().mean()
        else:
            aux = z_loss = logits.sum() * 0
            counts = torch.zeros(3, dtype=torch.long)
        (aux_coeff * aux + z_coeff * z_loss).backward()
        torch.testing.assert_close(stage.ffn.router.weight.grad, ref_weight.grad)
        torch.testing.assert_close(embedded.grad, ref_inputs.grad)
        assert not embedded.grad[~valid].any()
        # Non-reentrant recomputation can repeat bias-statistic collection.
        if checkpoint_mode is not False:
            torch.testing.assert_close(stage.ffn.router.local_tokens_per_expert, counts.float())
        metrics = stage.ffn.router.pop_metrics()
        if aux_coeff:
            torch.testing.assert_close(metrics["aux_loss_scaled"], aux.detach() * aux_coeff)
        if z_coeff:
            torch.testing.assert_close(metrics["z_loss_scaled"], z_loss.detach() * z_coeff)
