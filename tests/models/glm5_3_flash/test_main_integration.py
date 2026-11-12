"""Regression coverage for GLM integration with shared training interfaces."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from torchturbo.engine.backends.fsdp2.checkpoint.weight_loader import DistParamLoader
from torchturbo.engine.metrics.flops import ComputeProfile
from torchturbo.models.glm5_3_flash.configuration_glm5_3_flash import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from torchturbo.models.glm5_3_flash.mfu import Glm5NextFlops
from torchturbo.models.glm5_3_flash.model import (
    Glm5NextWeightConverter,
    TurboGlm5NextForConditionalGeneration,
)
from torchturbo.models.glm5_3_flash.modeling_glm5_3_flash import (
    Glm5NextTextMoE,
    Glm5NextTextModel,
    Glm5NextTextTopkRouter,
)
from torchturbo.models.glm5_3_flash.parallel_plan import (
    Glm5NextGroupedRouter,
    _replace_moe_only,
)
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.models.modules.moe.moe_layer import MoESubmodules
from torchturbo.models.modules.moe.config_converter import get_base_moe_config
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.optimizations.mega_moe.integration import RoutedExpertsBackend
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime


def _text_config(**kwargs):
    fields = dict(
        vocab_size=32, pad_token_id=0, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        q_lora_rank=4, kv_lora_rank=4, qk_nope_head_dim=4, v_head_dim=4,
        linear_num_heads=2, linear_head_dim=4, linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "indexed_attention"],
        indexer_types=["full", "full"], mlp_layer_types=["dense", "dense"],
        n_routed_experts=4, n_group=2, topk_group=1, num_experts_per_tok=2,
        moe_intermediate_size=8, index_topk=2, index_kpool=2,
        index_head_dim=4, index_n_heads=2, hc_mult=2, hc_sinkhorn_iters=3,
        kda_backend="torch", use_cache=False,
    )
    fields.update(kwargs)
    return Glm5NextTextConfig(**fields)


@pytest.mark.parametrize("topk", [1, 2])
def test_grouped_router_matches_checkpoint_and_shared_return_contract(topk):
    config = _text_config(num_experts_per_tok=topk)
    reference = Glm5NextTextTopkRouter(config)
    router = Glm5NextGroupedRouter(MoEConfig(**get_base_moe_config(
        config, 1, num_moe_experts=config.n_routed_experts,
        moe_router_num_groups=config.n_group, moe_router_group_topk=config.topk_group,
        moe_router_topk_scaling_factor=config.routed_scaling_factor,
        moe_router_score_function="sigmoid", moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
    )))
    torch.manual_seed(7)
    with torch.no_grad():
        reference.weight.normal_()
        reference.e_score_correction_bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        router.weight.copy_(reference.weight)
        router.expert_bias.copy_(reference.e_score_correction_bias)
    hidden = torch.randn(1, 3, 8, requires_grad=True)
    padding_mask = torch.tensor([[False, True, False]])
    ref_logits, ref_probs, ref_indices = reference(hidden)
    probs, routing_map, indices, logits = router(hidden, padding_mask=padding_mask)
    torch.testing.assert_close(logits.reshape_as(ref_logits), ref_logits)
    torch.testing.assert_close(indices, ref_indices)
    torch.testing.assert_close(probs.gather(1, indices), ref_probs)
    torch.testing.assert_close(routing_map.gather(1, indices), torch.ones_like(indices, dtype=torch.bool))
    expected_counts = routing_map[[0, 2]].sum(0).float()
    torch.testing.assert_close(router.local_tokens_per_expert, expected_counts)
    router(hidden, padding_mask=padding_mask, accumulate_tokens_per_expert=False)
    torch.testing.assert_close(router.local_tokens_per_expert, expected_counts)
    (probs * torch.arange(4)).sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_packed_boundaries_isolate_kda_and_sparse_attention_and_ignore_tail():
    torch.manual_seed(11)
    model = Glm5NextTextModel(_text_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])
    with torch.no_grad():
        packed = model(
            input_ids=ids, cu_seq_lens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        ).last_hidden_state
        first = model(input_ids=ids[:, :3]).last_hidden_state
        second = model(input_ids=ids[:, 3:5]).last_hidden_state
    torch.testing.assert_close(packed[:, :3], first)
    torch.testing.assert_close(packed[:, 3:5], second)


def test_shared_moe_replacement_preserves_glm_outputs_and_input_gradients(monkeypatch):
    runtime = CpuRuntime()
    monkeypatch.setattr(importlib.import_module("torchturbo.ops.ops"), "_OPS", None)
    monkeypatch.setattr(importlib.import_module("torchturbo.runtime.runtime"), "_SELECTED_RUNTIME", runtime)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.models.modules.moe.token_dispatcher"),
        "all_to_all", lambda group, tensor, *splits: tensor.clone(),
    )
    get_or_create_ops(device_runtime=runtime, default_policy=OpPolicy.NATIVE)
    config = _text_config(swiglu_limit=0.05)
    torch.manual_seed(23)
    reference = Glm5NextTextMoE(config, 0)
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    layer = _replace_moe_only(
        reference, SimpleNamespace(text_config=config), {"sp_group": None},
        {"ep_size": 1, "ep_group": object()},
        MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
    )
    with torch.no_grad():
        layer.router.weight.copy_(reference.gate.weight)
        layer.router.expert_bias.copy_(reference.gate.e_score_correction_bias)
        up = torch.stack([
            torch.cat((expert.gate_proj.weight, expert.up_proj.weight)).t()
            for expert in reference.experts
        ])
        down = torch.stack([expert.down_proj.weight.t() for expert in reference.experts])
        layer.experts.weight1.copy_(up.reshape_as(layer.experts.weight1))
        layer.experts.weight2.copy_(down.reshape_as(layer.experts.weight2))
    inputs = torch.randn(1, 5, 8, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    expected, _ = reference(ref_inputs)
    actual, _ = layer(inputs)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)


def test_vlm_keeps_global_metadata_until_its_internal_sp_boundary(tmp_path, monkeypatch):
    text = _text_config()
    vision = Glm5NextVisionConfig(
        depth=1, hidden_size=8, num_heads=2, intermediate_size=16,
        out_hidden_size=8, projection_intermediate_size=16, patch_size=2,
    )
    model = TurboGlm5NextForConditionalGeneration(Glm5NextConfig(
        text_config=text, vision_config=vision,
    ))
    spec = model.get_sequence_parallel_input_slicing_spec()
    assert spec.module is model.model.language_model
    assert spec.sliced_inputs == ()
    inputs = dict(
        input_ids=torch.tensor([[1, 2, 3]]), position_ids=torch.tensor([[0, 1, 2]]),
        cu_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
    )
    assert model.on_prepare_inputs(inputs) is inputs
    runtime = model.get_moe_runtime_inputs(inputs, {"sp": {"sp_size": 1}})
    assert not runtime.padding_mask.any()
    state = model.state_dict()
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in state},
    }))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    converter = model.get_weight_converter({
        "ep": {"ep_group": None, "ep_rank": 0, "ep_size": 1},
    }, str(tmp_path))
    shape, _ = converter.checkpoint_info.get_tensor_infos()["lm_head.weight"]
    assert shape == tuple(model.lm_head.weight.shape)


def test_mfu_uses_shared_compute_profile_and_component_contract():
    config = _text_config()
    profile = ComputeProfile(batch_size=1, seq_length=5, seq_segments=[3, 2])
    components = Glm5NextFlops(config, profile).components()
    counts = {component.name: component.fwd_flops for component in components}
    assert counts["lm_head"] == 2 * 5 * config.hidden_size * config.vocab_size
    assert counts["kda/state"] > 0
    assert counts["mla/sparse_scores"] > 0


def test_fp8_loader_reads_scales_from_another_shard(tmp_path):
    weight = torch.arange(16).reshape(4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file({"layer.weight": weight}, tmp_path / "weights.safetensors")
    save_file({"layer.weight_scale_inv": scale}, tmp_path / "scales.safetensors")
    converter = Glm5NextWeightConverter.__new__(Glm5NextWeightConverter)
    converter.dtype = torch.bfloat16
    converter.block_size = (2, 2)
    loader = DistParamLoader.__new__(DistParamLoader)
    loader.path = tmp_path
    loader.rank = 0
    loader.local_params = ["layer.weight"]
    loader.param_converter = converter
    loader.param_to_filename = {
        "layer.weight": "weights.safetensors",
        "layer.weight_scale_inv": "scales.safetensors",
    }
    decoded = loader._load_local_shards()["layer.weight"]
    expanded = scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
    torch.testing.assert_close(decoded, (weight.float() * expanded).bfloat16())


def test_megamoe_preserves_router_topk_order_and_zero_weight_routes():
    hidden = torch.arange(8).reshape(2, 4).float()
    probs = torch.tensor([[0.0, 0.0, 0.8], [0.7, 0.0, 0.3]])
    indices = torch.tensor([[2, 0], [0, 2]])
    seen = {}

    def experts(tokens, expert_ids, selected_probs):
        seen.update(ids=expert_ids, probs=selected_probs)
        return tokens, None

    backend = RoutedExpertsBackend(None, 128)
    result, _ = backend.route(experts, hidden, probs, expert_indices=indices)
    torch.testing.assert_close(result, hidden)
    torch.testing.assert_close(seen["ids"], indices)
    torch.testing.assert_close(seen["probs"], probs.gather(1, indices))
