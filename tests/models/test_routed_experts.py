# SPDX-License-Identifier: Apache-2.0
"""Framework ownership, independent feature switches and native dispatcher integration."""

from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchturbo.engine.args import TorchTurboArgs
from torchturbo.engine.config import EngineConfig, build_from_args
from torchturbo.engine.config import MoEConfig as EngineMoEConfig
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.optimizations import routed_experts as integration
from torchturbo.optimizations.mega_moe.backend import MegaMoeBackend
import sys
from torchturbo.models.modules.moe.utils import RoutingResult
from torchturbo.optimizations.routed_experts import RoutedExpertsBackend


def make_model() -> nn.Module:
    config = MoEConfig(
        hidden_size=4,
        moe_ffn_hidden_size=8,
        num_moe_experts=4,
        moe_router_topk=2,
        gated_linear_unit=True,
        activation_func=F.silu,
    )
    layers = []
    for _ in range(2):
        layer = MoELayer.__new__(MoELayer)
        nn.Module.__init__(layer)
        layer.config = config
        layer.experts = GroupedMLP(2, False, config)
        with torch.no_grad():
            layer.experts.weight1.fill_(0.1)
            layer.experts.weight2.fill_(0.2)
        layers.append(layer)
    return nn.Sequential(*layers)


def test_mixed_clip_positions_share_backend(monkeypatch):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    model = make_model()
    for layer, after in zip(model, (False, True)):
        layer.config = replace(
            layer.config, moe_swiglu_clamp_gate_after_activation=after
        )
        layer.experts.config = layer.config
        layer.experts.swiglu_limit = 4.0
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.moe.local_token_capacity = 128
    integration.bind_expert_backends(model, config, object())
    backend = model[0]._routed_experts_backend
    assert backend is model[1]._routed_experts_backend
    assert set(backend.modes) == {("pre_activation", 4.0), ("post_activation", 4.0)}


@pytest.mark.parametrize(
    "mega,moon", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("dropout", [None, 0.0, 0.25])
def test_lora_disables_accelerated_experts_before_replacement(
    monkeypatch, mega, moon, dropout
):
    from torchturbo.engine.backends.fsdp2 import backend as fsdp_backend

    backend = fsdp_backend.FSDP2Backend.__new__(fsdp_backend.FSDP2Backend)
    backend.config = EngineConfig(
        moe=EngineMoEConfig(enable_mega_moe=mega, enable_moonep=moon)
    )
    peft_config = (
        None if dropout is None else SimpleNamespace(r=4, lora_dropout=dropout)
    )
    messages = []
    monkeypatch.setattr(
        fsdp_backend,
        "logger",
        SimpleNamespace(warning_rank0=lambda message: messages.append(message)),
    )

    def stop_before_mesh(plan):
        assert backend.config.moe.enable_mega_moe is (mega and peft_config is None)
        assert backend.config.moe.enable_moonep is (moon and peft_config is None)
        raise RuntimeError("configuration checked before mesh and module replacement")

    monkeypatch.setattr(backend, "_init_mesh", stop_before_mesh)
    model = nn.Linear(4, 4)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="configuration checked before mesh"):
            backend.prepare_model(model, peft_config=peft_config)
    assert len(messages) == int((mega or moon) and peft_config is not None)
    if messages:
        assert "LoRA" in messages[0] and "native MoE" in messages[0]
        assert "enable_mega_moe" in messages[0] and "enable_moonep" in messages[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_configuration_roundtrip(tmp_path, enabled):
    value = {"moe": {"enable_mega_moe": enabled}}
    assert EngineConfig.from_dict(value).moe.enable_mega_moe is enabled
    path = tmp_path / "engine.yaml"
    path.write_text("engine_config:\n  moe:\n    enable_mega_moe: " + str(enabled).lower() + "\n")
    config = build_from_args(TorchTurboArgs(config_path=str(path)))
    assert config.moe.enable_mega_moe is enabled
    assert "enable_mega_moe" not in config.extra


def test_swift_lora_configuration_warns_and_skips_mega_moe_token_bounds(
    tmp_path, monkeypatch
):
    from torchturbo.adapters.msswift import utils

    path = tmp_path / "engine.yaml"
    path.write_text("engine_config:\n  moe:\n    enable_mega_moe: true\n")
    monkeypatch.setattr(
        utils, "get_torchturbo_args", lambda: TorchTurboArgs(config_path=str(path))
    )
    monkeypatch.setattr(utils, "_raise_if_unsupported", lambda args: None)
    monkeypatch.setattr(
        utils, "raise_if_evaluation_enabled", lambda *args, **kwargs: None
    )
    messages = []
    monkeypatch.setattr(utils, "logger", SimpleNamespace(warning_rank0=messages.append))
    args = SimpleNamespace(
        training_args=None, predict_with_generate=False, tuner_type="lora", bf16=True
    )
    config = utils.build_config_from_swift_args(args)
    assert config.moe.enable_mega_moe is False
    assert "mega_moe_local_tokens" not in config.extra
    assert len(messages) == 1 and "native MoE" in messages[0]


@pytest.mark.parametrize(
    "enabled,use_reentrant", [(False, False), (True, False), (True, True)]
)
def test_checkpoint_modes_are_independent_of_backend(
    monkeypatch, enabled, use_reentrant
):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.activation_checkpointing.enabled = enabled
    config.activation_checkpointing.use_reentrant = use_reentrant
    config.moe.local_token_capacity = 128
    model = make_model()
    integration.bind_expert_backends(model, config, object())
    assert model[0]._routed_experts_backend is model[1]._routed_experts_backend


def test_shared_backend_preserves_hooks_and_parameters(monkeypatch):
    from torchturbo.models.modules.moe.utils import RoutingResult

    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.moe.local_token_capacity = 128
    model = make_model()
    identities = {name: id(p) for name, p in model.named_parameters()}
    keys = set(model.state_dict())
    integration.bind_expert_backends(model, config, object())
    assert model[0]._routed_experts_backend is model[1]._routed_experts_backend
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert keys == set(model.state_dict())
    calls = []
    model[0].experts.register_forward_pre_hook(lambda *_: calls.append("pre"))
    model[0].experts.register_forward_hook(lambda *_: calls.append("post"))

    def execute(self, experts, hidden, indices, probs):
        calls.append("execute")
        assert experts.weight1 is model[0].experts.weight1
        assert indices.tolist() == [[1, 3]]
        assert probs.tolist() == [[0.0, 1.0]]
        return hidden * experts.weight1.mean()

    monkeypatch.setattr(MegaMoeBackend, "execute", execute)
    routing = RoutingResult(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[False, True, False, True]]),
        torch.tensor([[1, 3]]),
    )
    output = model[0]._routed_experts_backend.route(model[0], torch.ones(1, 4), routing)
    output.sum().backward()
    assert calls == ["pre", "execute", "post"]
    assert model[0].experts.weight1.grad is not None


@pytest.mark.parametrize("path", ["routed", "native"])
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_layer_routing_result_preserves_outputs_and_gradients(
    monkeypatch, path, score_function
):
    from torchturbo.models.modules.moe.utils import RoutingResult

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    probs = torch.tensor(
        [[0.0, 0.0, 0.0, 0.5], [0.1, 0.0, 0.2, 0.0]], requires_grad=True
    )
    indices = torch.tensor([[1, 3], [0, 2]])
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, indices, True)
    routing = RoutingResult(probs, mask, indices)
    logits = torch.zeros(2, 1, 4)
    padding = torch.tensor([False, True])

    class Router(nn.Module):
        def forward(self, inputs, input_ids, padding_mask, accumulate):
            assert padding_mask is padding and accumulate is True
            return routing, logits

    class Backend(integration.RoutedExpertsBackend):
        def execute(self, experts, x, ids, weights):
            assert ids is indices
            return x * weights.sum(-1, keepdim=True)

    layer = make_model()[0]
    layer.config = replace(layer.config, moe_router_score_function=score_function)
    layer._runtime_inputs = SimpleNamespace(
        padding_mask=padding, accumulate_tokens_per_expert=True
    )
    layer.shared_expert = layer.shared_experts = None
    layer.router = Router()
    layer._routed_experts_backend = Backend(object(), 128) if path == "routed" else None

    def native(x, result):
        assert result.probs is probs and result.routing_map is mask
        assert result.expert_indices is indices
        return x * result.selected_probs().sum(-1, keepdim=True)

    layer.compute_native_experts = native
    output, actual_logits = layer(hidden)
    expected = hidden * probs.gather(1, indices).sum(-1, keepdim=True)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(actual_logits, logits.reshape(2, 4))
    actual_grads = torch.autograd.grad(output.sum(), (hidden, probs), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, probs))
    for actual, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, want)
    assert routing.probs is probs


@pytest.mark.parametrize(
    "capacity,valid",
    [(128, True), (256, True), (127, False), (0, False), ("128", False)],
)
def test_adapter_validates_explicit_token_capacity(capacity, valid):
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.parallel_plan = SimpleNamespace(sp_size=1)
    config.moe.local_token_capacity = capacity
    args = SimpleNamespace(packing=False, max_length=100, per_device_train_batch_size=1)
    if valid:
        integration.configure_token_bound(config, args)
        assert config.moe.local_token_capacity == capacity
    else:
        with pytest.raises(ValueError, match="128-aligned and at least 128"):
            integration.configure_token_bound(config, args)


def test_router_ids_are_reused_without_sorting_or_reconstructing_topk(monkeypatch):
    from torchturbo.models.modules.moe.utils import RoutingResult

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    ids = torch.tensor([[3, 1]])
    probs = torch.tensor([[0.0, 0.0, 0.0, 1.0]], requires_grad=True)
    mask = torch.tensor([[False, True, False, True]])
    captured = {}

    class Backend(integration.RoutedExpertsBackend):
        def execute(self, experts, x, indices, weights):
            captured.update(indices=indices, weights=weights)
            return x * weights.sum()

    def unexpected_topk(*args, **kwargs):
        pytest.fail("Router IDs must not be reconstructed")

    monkeypatch.setattr(torch.Tensor, "topk", unexpected_topk)
    layer = make_model()[0]
    output = Backend(object(), 128).route(
        layer, torch.ones(1, 4), RoutingResult(probs, mask, ids)
    )
    assert captured["indices"] is ids
    torch.testing.assert_close(captured["weights"], torch.tensor([[1.0, 0.0]]))
    output.sum().backward()
    torch.testing.assert_close(probs.grad, torch.tensor([[0.0, 4.0, 0.0, 4.0]]))


@pytest.mark.parametrize("enable_moon", [False, True])
def test_upstream_executor_uses_current_parameter_views(monkeypatch, enable_moon):
    from torchturbo.optimizations import _expert_cleanup

    if enable_moon:
        monkeypatch.setitem(
            sys.modules, "npu_patch", SimpleNamespace(EXTERNAL_SHMEM_RUNTIME=True)
        )
    calls = []
    provider = SimpleNamespace(
        build_official_placement=object(), OfficialTransferBackend=object()
    )
    monkeypatch.setitem(
        sys.modules, "torchturbo.optimizations.moonep.megamoe_provider", provider
    )

    class Executor:
        training = True

        def train(self, mode):
            self.training = mode
            return self

        def __init__(self, **options):
            assert options["create_parameters"] is False
            assert options["expert_weight_layout"] == "EIO"
            assert options["apply_probs_before_fc2"] is True
            assert options["enable_hot_replica"] is enable_moon
            calls.append(options)

        def __call__(
            self,
            hidden,
            indices,
            probabilities,
            *,
            expert_weights,
            tokens_per_expert=None,
        ):
            first, second = expert_weights
            assert first.shape == (2, 4, 6)
            assert second.shape == (2, 3, 4)
            return (
                hidden
                * (first.mean() + second.mean())
                * probabilities.sum(-1, keepdim=True)
            )

        def close(self):
            calls.append("closed")

    monkeypatch.setitem(
        sys.modules, "hyper_multicore", SimpleNamespace(MegaMoeExperts=Executor)
    )
    monkeypatch.setattr(_expert_cleanup, "register", lambda *args: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    config = SimpleNamespace(
        hidden_size=4,
        moe_ffn_hidden_size=3,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_swiglu_clamp_gate_after_activation=False,
    )
    experts = SimpleNamespace(
        config=config,
        swiglu_limit=0.0,
        training=True,
        weight1=torch.nn.Parameter(torch.ones(8, 6, dtype=torch.bfloat16)),
        weight2=torch.nn.Parameter(torch.ones(6, 4, dtype=torch.bfloat16)),
    )
    identities = (id(experts.weight1), id(experts.weight2))
    backend = MegaMoeBackend(object(), 128, (("none", 0.0),), enable_moonep=enable_moon)
    inputs = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.full((2, 2), 0.5, requires_grad=True)
    ids = torch.tensor([[0, 1], [2, 3]])
    first = backend.execute(experts, inputs, ids, probs)
    first.sum().backward()
    assert inputs.grad is not None and probs.grad is not None
    assert experts.weight1.grad is not None and experts.weight2.grad is not None
    with torch.no_grad():
        experts.weight1.add_(1.0)
    second = backend.execute(experts, inputs, ids, probs)
    torch.testing.assert_close(second, first * 1.5)
    assert identities == (id(experts.weight1), id(experts.weight2))
    assert len(calls) == 1
    backend.close()
    assert calls[-1] == "closed"


def test_incompatible_shmem_images_are_rejected_before_native_loading(monkeypatch):
    monkeypatch.setitem(sys.modules, "npu_patch", SimpleNamespace())
    backend = MegaMoeBackend(object(), 128, (("none", 0.0),))
    with pytest.raises(RuntimeError, match="not symbol-isolated"):
        backend._executor(SimpleNamespace())


def test_same_limit_can_use_distinct_clipping_orders(monkeypatch):
    from torchturbo.optimizations import _expert_cleanup

    monkeypatch.setitem(
        sys.modules, "hyper_multicore", SimpleNamespace(MegaMoeExperts=SimpleNamespace)
    )
    monkeypatch.setattr(_expert_cleanup, "register", lambda *args: None)
    config = SimpleNamespace(
        hidden_size=4,
        moe_ffn_hidden_size=3,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_swiglu_clamp_gate_after_activation=False,
    )
    experts = SimpleNamespace(config=config, swiglu_limit=0.25)
    backend = MegaMoeBackend(
        object(), 128, (("pre_activation", 0.25), ("post_activation", 0.25))
    )
    pre = backend._executor(experts)
    config.moe_swiglu_clamp_gate_after_activation = True
    post = backend._executor(experts)
    assert pre is not post
    assert pre.swiglu_limit == post.swiglu_limit == 0.25
    assert (
        not getattr(pre, "clamp_after_activation", False)
        and post.clamp_after_activation
    )


def test_native_dispatch_reuses_the_current_global_count_agreement(monkeypatch):
    from torchturbo.models.modules.moe import token_dispatcher as module

    dispatcher = module.MoEAlltoAllTokenDispatcher.__new__(
        module.MoEAlltoAllTokenDispatcher
    )
    dispatcher.config = SimpleNamespace(moe_router_topk=2, moe_permute_fusion=True)
    dispatcher._ep_group = None
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    dispatcher.ep_size, dispatcher.tp_size = 2, 1
    dispatcher.num_experts, dispatcher.num_local_experts = 4, 2
    dispatcher.local_expert_indices = [2, 3]
    dispatcher.dtoh_point = dispatcher.sync_point = "before_ep_alltoall"
    dispatcher.sync_point_priority = {"before_ep_alltoall": 0}
    mask = torch.tensor([[False, False, True, True], [False, False, True, True]])
    counts = torch.tensor([[1, 1, 1, 1], [0, 0, 2, 2]])

    def unexpected_collective(*args, **kwargs):
        pytest.fail("Current source counts must not be gathered twice")

    monkeypatch.setattr(
        module, "gather_from_sequence_parallel_region", unexpected_collective
    )
    actual = dispatcher.preprocess(mask, counts)
    torch.testing.assert_close(actual, torch.tensor([3, 3]))
    torch.testing.assert_close(dispatcher.input_splits, torch.tensor([0, 4]))
    torch.testing.assert_close(dispatcher.output_splits, torch.tensor([2, 4]))


def test_routed_forward_preserves_selected_zero_weights_and_gradients(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    experts = GroupedMLP(2, False, MoEConfig(hidden_size=4, moe_ffn_hidden_size=2))
    ids = torch.tensor([[0, 1], [0, 1]])
    probs = torch.tensor([[0.0, 0.5001], [0.25, 0.75]], requires_grad=True)
    routing = RoutingResult(probs, torch.ones_like(probs, dtype=torch.bool), ids)
    hidden = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)

    class Backend(RoutedExpertsBackend):
        def execute(self, module, tokens, indices, selected_probs):
            assert module is experts
            assert indices is ids
            assert selected_probs.dtype == torch.float32
            torch.testing.assert_close(selected_probs, probs)
            return tokens.float() * selected_probs.sum(-1, keepdim=True)

    backend = Backend(object(), 128)
    layer = SimpleNamespace(experts=experts, config=experts.config)
    result = backend.route(layer, hidden, routing)
    result.sum().backward()
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 4.0))
    assert hidden.grad is not None


def test_compute_experts_uses_native_stages_only_when_backend_declines():
    hidden = torch.ones(2, 4)
    routing = object()
    calls = []

    class Backend:
        def route(self, layer, tokens, selected):
            calls.append("route")
            return None

    layer = SimpleNamespace(_routed_experts_backend=Backend())
    layer.compute_native_experts = lambda x, r: calls.append("native") or x * 2
    result = MoELayer._compute_experts(layer, hidden, routing)
    assert calls == ["route", "native"]
    torch.testing.assert_close(result, hidden * 2)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_gradient_return_dtype_roundtrips(dtype):
    config = EngineConfig(
        moe=EngineMoEConfig(
            enable_mega_moe=True, enable_moonep=True, moonep_grad_reduce_dtype=dtype
        )
    )
    restored = EngineConfig.from_dict(config.to_dict())
    assert restored.moe.moonep_grad_reduce_dtype == dtype
    assert restored.moe.enable_mega_moe and restored.moe.enable_moonep


def test_gradient_return_dtype_default_and_validation():
    assert EngineConfig().moe.moonep_grad_reduce_dtype == "fp32"
    with pytest.raises(ValueError, match="moonep_grad_reduce_dtype"):
        EngineMoEConfig(moonep_grad_reduce_dtype="fp16")


@pytest.mark.parametrize(
    "length,batch,sp,expected",
    [
        (131072, 1, 8, 16384),
        (4097, 2, 8, 1152),
        (4096, 2, 1, 8192),
    ],
)
def test_automatic_local_token_capacity(tmp_path, length, batch, sp, expected):
    path = tmp_path / "engine.yaml"
    path.write_text(
        f"engine_config:\n  moe:\n    enable_mega_moe: true\n  parallel_plan:\n    sp_size: {sp}\n"
    )
    config = build_from_args(TorchTurboArgs(config_path=str(path)))
    args = SimpleNamespace(
        packing=True,
        packing_length=length,
        max_length=length,
        per_device_train_batch_size=batch,
    )
    integration.configure_token_bound(config, args)
    assert config.moe.local_token_capacity == expected


def test_expert_binding_follows_replacement_before_sharding(monkeypatch):
    from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend

    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    model = nn.Linear(4, 4)
    calls = []
    ep_group = object()
    monkeypatch.setattr(backend, "_init_mesh", lambda plan: calls.append("mesh"))

    def replace_modules(value, plan):
        assert value is model
        calls.append("replace")
        return value

    def bind_expert_backends(value, config, group):
        assert value is model and config is backend.config and group is ep_group
        calls.append("bind")
        raise RuntimeError("stop before checkpointing and sharding")

    monkeypatch.setattr(backend, "_replace_modules", replace_modules)
    monkeypatch.setattr(backend, "get_ep_info", lambda: {"ep_group": ep_group})
    monkeypatch.setattr(integration, "bind_expert_backends", bind_expert_backends)
    with pytest.raises(RuntimeError, match="stop before checkpointing"):
        backend.prepare_model(model)
    assert calls == ["mesh", "replace", "bind"]


def test_missing_ep_group_never_uses_world(monkeypatch):
    def unexpected_world(group):
        pytest.fail("Missing EP group must never query WORLD")

    monkeypatch.setattr(integration.dist, "get_world_size", unexpected_world)
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.moe.local_token_capacity = 128
    with pytest.raises(ValueError, match="explicit EP group"):
        integration.bind_expert_backends(make_model(), config, None)
    with pytest.raises(ValueError, match="explicit EP group"):
        MegaMoeBackend(None, 128, (("none", 0.0),))


def test_local_expert_count_must_match_ep_group(monkeypatch):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 4)
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.moe.local_token_capacity = 128
    with pytest.raises(ValueError, match="Expected 1 local experts"):
        integration.bind_expert_backends(make_model(), config, object())


def test_clamp_manifest_must_match_across_ranks(monkeypatch):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(integration.dist, "is_initialized", lambda: True)

    def gather(manifests, local, group):
        manifests[:] = [local, (("post_activation", 4.0),)]

    monkeypatch.setattr(integration.dist, "all_gather_object", gather)
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.moe.local_token_capacity = 128
    with pytest.raises(ValueError, match="mode/limit manifests differ"):
        integration.bind_expert_backends(make_model(), config, object())


def test_capacity_overflow_fails_before_collectives(monkeypatch):
    monkeypatch.setattr(integration.dist, "is_initialized", lambda: True)
    backend = RoutedExpertsBackend(object(), 128)
    # Neither expert hooks nor geometry collectives may run on this input.
    with pytest.raises(ValueError, match="actual_tokens=129 exceeds token_bound=128"):
        backend.route(None, torch.ones(129, 4), None)


def test_native_grouped_experts_require_token_counts():
    experts = make_model()[0].experts
    with pytest.raises(ValueError, match="tokens_per_expert"):
        experts(torch.ones(2, 4), None, torch.ones(2))


def test_cleanup_preserves_multiple_owners_and_failed_retry():
    from torchturbo.optimizations._expert_cleanup import _Cleanup

    cleanup = _Cleanup()
    root = object()
    unrelated = object()
    calls = []
    attempts = 0

    def first():
        calls.append("first")

    def second():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("release failed")
        calls.append("second")

    cleanup.callbacks = {first: root, second: root}
    cleanup.close(unrelated)
    assert calls == []
    with pytest.raises(RuntimeError, match="release failed"):
        cleanup.close(root)
    assert first not in cleanup.callbacks and second in cleanup.callbacks
    cleanup.close(root)
    assert calls == ["first", "second"]
    assert not cleanup.callbacks


def test_shutdown_hook_runs_before_original_and_ignores_other_groups(monkeypatch):
    from torchturbo.optimizations import _expert_cleanup as module

    calls = []
    root = object()
    unrelated = object()

    class ProcessGroup:
        def shutdown(group):
            calls.append(("shutdown", group))

    monkeypatch.setattr(module.c10d, "ProcessGroup", ProcessGroup)
    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    cleanup = module._Cleanup()
    monkeypatch.setattr(module, "_CLEANUP", cleanup)
    module.register(lambda: calls.append(("first", root)), root)
    module.register(lambda: calls.append(("second", root)), root)
    ProcessGroup.shutdown(unrelated)
    assert calls == [("shutdown", unrelated)]
    calls.clear()
    ProcessGroup.shutdown(root)
    assert calls == [("first", root), ("second", root), ("shutdown", root)]
    assert not cleanup.callbacks


def test_native_moonep_dispatcher_width_is_bounded(monkeypatch):
    pytest.importorskip("hyper_multicore.moonep.placement")
    from torchturbo.optimizations.moonep import native_backend as module

    backend = module.NativeMoonEPBackend(object(), 128)
    experts = make_model()[0].experts
    original = SimpleNamespace(num_local_experts=2, dtoh_stream=object())
    backend.register_dispatcher(experts, original)
    creations = []

    def new_dispatcher(width, indices, config, group, *, dtoh_stream):
        assert dtoh_stream is original.dtoh_stream
        creations.append(width)
        return object()

    monkeypatch.setattr(module, "MoEAlltoAllTokenDispatcher", new_dispatcher)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    for width in (2, 3, 4, 3, 2, 4):
        backend._dispatcher(experts, experts.config, width)
    assert creations == [3, 4]
    with pytest.raises(ValueError, match="virtual width"):
        backend._dispatcher(experts, experts.config, 5)


def test_native_moonep_rejects_incompatible_runtime(monkeypatch):
    pytest.importorskip("hyper_multicore.moonep.placement")
    from torchturbo.optimizations.moonep.native_backend import NativeMoonEPBackend

    monkeypatch.setitem(
        sys.modules, "npu_patch", SimpleNamespace(EXTERNAL_SHMEM_RUNTIME=False)
    )
    backend = NativeMoonEPBackend(object(), 128)
    with pytest.raises(RuntimeError, match="not symbol-isolated"):
        backend.route(None, torch.ones(1, 4), None)


def test_real_process_group_remains_usable_during_expert_cleanup(tmp_path, monkeypatch):
    from datetime import timedelta
    from torchturbo.optimizations import _expert_cleanup as module

    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the real ProcessGroup lifecycle check")
    assert not torch.distributed.is_initialized()
    monkeypatch.setattr(module, "_CLEANUP", module._Cleanup())
    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    # Restore the Torch method after this isolated lifecycle test.
    monkeypatch.setattr(
        module.c10d.ProcessGroup, "shutdown", module.c10d.ProcessGroup.shutdown
    )
    torch.distributed.init_process_group(
        "gloo",
        init_method=(tmp_path / "process_group").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=15),
    )
    calls = []

    def release():
        value = torch.tensor([1.0])
        torch.distributed.all_reduce(value)
        assert value.item() == 1.0
        calls.append("released before communicator shutdown")

    module.register(release, torch.distributed.group.WORLD)
    try:
        torch.distributed.destroy_process_group()
        assert calls == ["released before communicator shutdown"]
        assert not module._CLEANUP.callbacks
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_clamp_manifest_preserves_layer_order_across_ranks(monkeypatch):
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(integration.dist, "is_initialized", lambda: True)

    def gather(manifests, local, group):
        manifests[:] = [local, tuple(reversed(local))]

    monkeypatch.setattr(integration.dist, "all_gather_object", gather)
    model = make_model()
    for layer, after in zip(model, (False, True)):
        layer.config = replace(
            layer.config, moe_swiglu_clamp_gate_after_activation=after
        )
        layer.experts.config = layer.config
        layer.experts.swiglu_limit = 4.0
    config = EngineConfig(moe=EngineMoEConfig(enable_mega_moe=True))
    config.moe.local_token_capacity = 128
    with pytest.raises(ValueError, match="mode/limit manifests differ"):
        integration.bind_expert_backends(model, config, object())


def test_virtual_dispatchers_share_stream_but_not_layout_state(monkeypatch):
    from torchturbo.models.modules.moe import token_dispatcher as module

    def unexpected_new_stream():
        pytest.fail(
            "A virtual layout must reuse its layer's existing count-copy stream"
        )

    runtime = SimpleNamespace(
        device=torch.device("cpu"), new_stream=unexpected_new_stream
    )
    monkeypatch.setattr(module, "get_device_runtime", lambda: runtime)
    stream = object()
    config = MoEConfig(num_moe_experts=4, expert_model_parallel_size=2)
    first = module.MoEAlltoAllTokenDispatcher(
        2, [0, 1], config, object(), dtoh_stream=stream
    )
    second = module.MoEAlltoAllTokenDispatcher(
        3, [0, 1, 2], replace(config, num_moe_experts=6), object(), dtoh_stream=stream
    )
    assert first.dtoh_stream is stream and second.dtoh_stream is stream
    first.hidden_shape = torch.Size([5, 4])
    assert second.hidden_shape is None
    assert first.num_experts == 4 and second.num_experts == 6


def test_native_transfers_use_injected_official_bank_api(monkeypatch):
    """Publish every projection and preserve FP32 partials until owner reduction."""
    pytest.importorskip("hyper_multicore")
    from types import ModuleType
    from unittest.mock import Mock
    from torchturbo.optimizations.moonep import official_transfer

    module = ModuleType("npu_patch")
    launchers = ModuleType("npu_patch.launchers")
    launchers._env_use_udma = lambda _kind: False
    native = SimpleNamespace(prefetch_banks=Mock(), reduce_banks=Mock())
    module._C = native
    npu = SimpleNamespace(
        get_device_properties=lambda _device: SimpleNamespace(cube_core_num=24),
        Event=Mock(),
    )
    allocator = SimpleNamespace(
        empty=lambda shape, dtype: torch.empty(shape, dtype=dtype), free=Mock()
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(official_transfer.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(official_transfer.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setitem(sys.modules, "npu_patch", module)
    monkeypatch.setitem(sys.modules, "npu_patch.launchers", launchers)
    resource = official_transfer.ExpertTransfer(
        None, 4, ((3, 4), (2, 3)), torch.device("cpu"), allocator=allocator
    )
    resource.bind()
    resource.in_use = True
    resource.vector_cores = 48
    weights = (torch.randn(6, 4).bfloat16(), torch.randn(4, 3).bfloat16())
    copies = torch.tensor([[2], [-1]], dtype=torch.int32)
    native.prefetch_banks.return_value = tuple(p.bank[:3] for p in resource.projections)
    resource.prefetch(weights, copies)
    args = native.prefetch_banks.call_args.args
    assert args[-2:] == (48, False)
    assert [tuple(w.shape) for w in args[0]] == [(2, 3, 4), (2, 2, 3)]
    for actual, original in zip(args[0], weights):
        assert actual.data_ptr() == original.data_ptr()
    gradients = (torch.randn(3, 3, 4), torch.randn(3, 2, 3))
    native.reduce_banks.return_value = tuple(g[:2].bfloat16() for g in gradients)
    output_dtypes = (torch.bfloat16, torch.bfloat16)
    resource.return_generated_gradients(gradients, copies, output_dtypes=output_dtypes)
    call = native.reduce_banks.call_args
    assert call.args[-2:] == (48, False)
    assert call.kwargs == {"output_dtypes": output_dtypes}
    for actual, original in zip(call.args[0], gradients):
        assert actual is original
        assert actual.dtype == torch.float32
    launchers._env_use_udma = lambda _kind: True
    resource.prefetch(weights, copies)
    assert native.prefetch_banks.call_args.args[-2:] == (24, True)
