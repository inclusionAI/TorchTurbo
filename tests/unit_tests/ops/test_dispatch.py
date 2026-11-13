from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import torchturbo.runtime.runtime as runtime_impl
from torchturbo.ops import (
    OpDispatcher,
    OpImpl,
    OpImplKind,
    OpPolicy,
    OpRegistry,
    OpSpec,
    get_ops,
    get_or_create_ops,
)
from torchturbo.ops.providers import iter_provider_modules
from torchturbo.ops.registry import registry as ops_registry
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime
from torchturbo.runtime.cuda import CudaRuntime
from torchturbo.runtime.npu import NpuRuntime


def _reference_rms_norm(hidden_states, weight, eps):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


class FakeGroupedMatmulImpl(OpImpl):
    def __call__(self, x, weight, group_list, *, trans_b=False):
        assert group_list.tolist() == [1]
        assert trans_b is True
        return x + weight


class CachedAddImpl(OpImpl):
    available_calls = 0

    def available(self) -> bool:
        type(self).available_calls += 1
        return True

    def __call__(self, x):
        return x + 1


class UnavailableAddImpl(OpImpl):
    def available(self) -> bool:
        return False

    def __call__(self, x):
        return x


class FallbackAddImpl(OpImpl):
    def __call__(self, x):
        return x + 1


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(logging.getLogger("torchturbo"), "propagate", True)
    monkeypatch.setattr(runtime_impl, "_SELECTED_RUNTIME", None)
    monkeypatch.setattr(
        importlib.import_module("torchturbo.ops.ops"), "_OPS", None
    )
    runtime_impl.is_device_available.cache_clear()
    get_or_create_ops(
        device_runtime=CpuRuntime(),
    )
    yield
    runtime_impl.is_device_available.cache_clear()


def test_rms_norm_uses_reference_fallback_on_cpu():
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)

    actual = get_ops().rms_norm(hidden_states, weight, 1e-6)
    expected = _reference_rms_norm(hidden_states, weight, 1e-6)

    assert torch.allclose(actual, expected)


def test_provider_modules_are_discovered_from_provider_packages():
    cpu_modules = iter_provider_modules(TurboDevice.CPU)
    cuda_modules = iter_provider_modules(TurboDevice.CUDA)
    npu_modules = iter_provider_modules(TurboDevice.NPU)

    assert "torchturbo.ops.providers.native" not in cpu_modules
    assert "torchturbo.ops.providers.native.rms_norm" in cpu_modules
    assert "torchturbo.ops.providers.native.gated_activation" in cpu_modules
    assert "torchturbo.ops.providers.native.swiglu" in cpu_modules
    assert "torchturbo.ops.providers.native.attention" in cpu_modules
    assert "torchturbo.ops.providers.cuda.attention" in cuda_modules
    assert "torchturbo.ops.providers.npu.attention" in npu_modules
    assert "torchturbo.ops.providers.cuda.gated_activation" in cuda_modules
    assert "torchturbo.ops.providers.cuda.swiglu" in cuda_modules
    assert "torchturbo.ops.providers.cuda.cross_entropy_loss" in cuda_modules
    assert "torchturbo.ops.providers.npu.swiglu" in npu_modules
    assert (
        "torchturbo.ops.providers.npu.matmul_add"
        in npu_modules
    )
    assert "torchturbo.ops.providers.npu.cross_entropy_loss" in npu_modules
    assert "torchturbo.ops.providers.npu.moe_permutation" in npu_modules
    assert "torchturbo.ops.providers.npu.rotary_position_embedding" in npu_modules
    assert "torchturbo.ops.providers.npu.lightning_indexer" in npu_modules
    assert "torchturbo.ops.providers.npu.sparse_flash_mla" in npu_modules
    assert "torchturbo.ops.providers.native.cross_entropy_loss" in cpu_modules
    assert (
        "torchturbo.ops.providers.native.matmul_add"
        in cpu_modules
    )


def test_custom_policy_errors_when_no_cpu_custom_impl_exists():
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.ones(4, dtype=torch.float32)

    custom_ops = OpDispatcher(
        device_runtime=CpuRuntime(),
        registry=ops_registry,
        default_policy=OpPolicy.CUSTOM,
    )

    with pytest.raises(RuntimeError, match="No implementation"):
        custom_ops.rms_norm(hidden_states, weight, 1e-6)


def test_invalid_policy_errors_when_initializing_dispatcher():
    with pytest.raises(TypeError, match="policy must be an OpPolicy"):
        OpDispatcher(
            device_runtime=CpuRuntime(),
            registry=ops_registry,
            default_policy="unknown",
        )


def test_swiglu_uses_reference_fallback_on_cpu():
    x = torch.randn(2, 8, dtype=torch.float32)
    actual = get_ops().swiglu(x)

    gate, up = torch.chunk(x, 2, dim=-1)
    expected = F.silu(gate) * up

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("activation_func", [F.silu, F.gelu])
def test_gated_activation_reference_supports_limit(activation_func):
    x = torch.randn(2, 8, dtype=torch.float32)
    actual = get_ops().gated_activation(
        x,
        activation_func=activation_func,
        limit=0.5,
    )

    gate, up = torch.chunk(x, 2, dim=-1)
    expected = activation_func(torch.clamp(gate.float(), max=0.5)) * torch.clamp(
        up.float(), min=-0.5, max=0.5
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("activation_func", [F.silu, F.gelu])
def test_gated_activation_reference_can_clamp_gate_after_activation(activation_func):
    x = torch.randn(2, 8, dtype=torch.float32) * 4
    actual = get_ops().gated_activation(
        x,
        activation_func=activation_func,
        limit=0.5,
        clamp_gate_after_activation=True,
    )

    gate, up = torch.chunk(x, 2, dim=-1)
    expected = torch.clamp(activation_func(gate), max=0.5) * torch.clamp(
        up, min=-0.5, max=0.5
    )

    assert torch.allclose(actual, expected)


def test_swiglu_custom_policy_errors_when_no_cpu_custom_impl_exists():
    x = torch.randn(2, 8, dtype=torch.float32)

    custom_ops = OpDispatcher(
        device_runtime=CpuRuntime(),
        registry=ops_registry,
        default_policy=OpPolicy.CUSTOM,
    )

    with pytest.raises(RuntimeError, match="No implementation"):
        custom_ops.swiglu(x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cross_entropy_loss_uses_reference_fallback_on_cpu(dtype):
    logits = torch.randn(4, 6, dtype=dtype)
    labels = torch.tensor([1, 4, 0, 3], dtype=torch.long)

    actual = get_ops().cross_entropy_loss(logits, labels, reduction="sum")
    expected = F.cross_entropy(logits.float(), labels, reduction="sum")

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected)


def test_matmul_add_uses_reference_fallback_on_cpu():
    lhs = torch.randn(3, 5, dtype=torch.bfloat16)
    rhs = torch.randn(3, 4, dtype=torch.bfloat16)
    accumulator = torch.zeros(5, 4, dtype=torch.float32)

    get_ops().matmul_add(
        accumulator,
        lhs,
        rhs,
        transpose_lhs=True,
    )
    expected = lhs.float().t().matmul(rhs.float())

    assert torch.allclose(accumulator, expected)


@pytest.fixture
def recurrent_providers(monkeypatch):
    from torchturbo.ops.providers.native.causal_conv1d import FlaCausalConv1d
    from torchturbo.ops.providers.native.gated_delta_rule import FlaGatedDeltaRule

    registry = OpRegistry()
    calls = {}
    for provider in (FlaCausalConv1d, FlaGatedDeltaRule):
        spec = provider.op_spec()
        registry.register(ops_registry.get_by_impl_name(spec.impl_name))
        monkeypatch.setattr(provider, "available", lambda self: True)
        calls[spec.op_name] = Mock(return_value=(None, None))
        monkeypatch.setattr(provider, "__call__", calls[spec.op_name])
    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())
    return dispatcher, calls


def _call_recurrent_op(dispatcher, op_name, **kwargs):
    if op_name == "causal_conv1d":
        return dispatcher.causal_conv1d(
            torch.zeros(1, 6, 2), torch.ones(2, 3), **kwargs
        )
    qkv = torch.zeros(1, 6, 1, 2)
    gates = torch.zeros(1, 6, 1)
    return dispatcher.gated_delta_rule(qkv, qkv, qkv, gates, gates, **kwargs)


def test_recurrent_dispatch_shares_padding_metadata_and_refreshes_mutated_cu(
    recurrent_providers,
):
    dispatcher, calls = recurrent_providers
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    _call_recurrent_op(dispatcher, "causal_conv1d", cu_seqlens=cu)
    before = calls["causal_conv1d"].call_args.kwargs["packed_params"]
    _call_recurrent_op(dispatcher, "gated_delta_rule", cu_seqlens=cu)

    assert calls["gated_delta_rule"].call_args.kwargs["packed_params"] is before
    assert before.cu_seqlens_list == [0, 2, 5, 6]
    assert before.cu_seqlens.tolist() == [0, 2, 5, 6]
    assert before.chunk_indices_list["64"] == [0, 0, 1, 0, 2, 0]
    assert cu.tolist() == [0, 2, 5]
    assert cu.dtype == torch.int32
    for call in calls.values():
        assert call.call_args.kwargs.get("cu_seqlens") is None

    cu[1] = 3
    _call_recurrent_op(dispatcher, "gated_delta_rule", cu_seqlens=cu)
    after = calls["gated_delta_rule"].call_args.kwargs["packed_params"]
    _call_recurrent_op(dispatcher, "causal_conv1d", cu_seqlens=cu)

    assert after is not before
    assert calls["causal_conv1d"].call_args.kwargs["packed_params"] is after
    assert before.cu_seqlens_list == [0, 2, 5, 6]
    assert after.cu_seqlens_list == [0, 3, 5, 6]
    assert cu.tolist() == [0, 3, 5]


@pytest.mark.parametrize("op_name", ["causal_conv1d", "gated_delta_rule"])
def test_recurrent_dispatch_without_cu_passes_no_metadata(recurrent_providers, op_name):
    dispatcher, calls = recurrent_providers

    _call_recurrent_op(dispatcher, op_name)

    kwargs = calls[op_name].call_args.kwargs
    assert kwargs["packed_params"] is None
    assert kwargs.get("cu_seqlens") is None


@pytest.mark.parametrize("op_name", ["causal_conv1d", "gated_delta_rule"])
def test_recurrent_dispatch_preserves_explicit_metadata(
    recurrent_providers, monkeypatch, op_name,
):
    from torchturbo.ops import packed_params as packed_module

    dispatcher, calls = recurrent_providers
    packed = packed_module.get_packed_params(torch.tensor([0, 2, 5]), 6)
    build = Mock(side_effect=AssertionError("Explicit metadata must not be rebuilt."))
    monkeypatch.setattr(packed_module, "build_packed_params", build)

    _call_recurrent_op(dispatcher, op_name, packed_params=packed)

    assert calls[op_name].call_args.kwargs["packed_params"] is packed
    assert calls[op_name].call_args.kwargs.get("cu_seqlens") is None
    build.assert_not_called()


@pytest.mark.parametrize("op_name", ["causal_conv1d", "gated_delta_rule"])
def test_recurrent_dispatch_rejects_raw_and_prepared_metadata(
    recurrent_providers, op_name,
):
    from torchturbo.ops.packed_params import get_packed_params

    dispatcher, calls = recurrent_providers
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    packed = get_packed_params(cu, 6)

    with pytest.raises(ValueError, match="cu_seqlens.*packed_params"):
        _call_recurrent_op(dispatcher, op_name, cu_seqlens=cu, packed_params=packed)

    calls[op_name].assert_not_called()


def test_registry_register_is_idempotent_by_impl_name():
    registry = OpRegistry()

    first = OpSpec(
        op_name="add",
        impl_name="same_name",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    replacement = OpSpec(
        op_name="add",
        impl_name="same_name",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )

    registry.register(first)
    registry.register(replacement)

    assert isinstance(registry.list(), list)
    assert isinstance(
        registry.list(spec_filter=lambda spec: spec.op_name == "add"),
        list,
    )
    assert registry.list(spec_filter=lambda spec: spec.op_name == "add")[0] is replacement
    assert registry.get_by_impl_name("same_name") is replacement


def test_registry_list_returns_highest_priority_filtered_spec():
    registry = OpRegistry()

    low = OpSpec(
        op_name="add",
        impl_name="low_priority",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    high = OpSpec(
        op_name="add",
        impl_name="high_priority",
        kind=OpImplKind.NATIVE,
        priority=10,
        devices=(),
    )

    registry.register(low)
    registry.register(high)

    assert registry.list(spec_filter=lambda spec: spec.op_name == "add") == [
        high,
        low,
    ]


def test_registry_list_prefers_device_specific_and_custom_specs_at_equal_priority():
    registry = OpRegistry()

    fallback_native = OpSpec(
        op_name="add",
        impl_name="fallback_native",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    fallback_custom = OpSpec(
        op_name="add",
        impl_name="fallback_custom",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(),
    )
    device_native = OpSpec(
        op_name="add",
        impl_name="device_native",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    device_custom = OpSpec(
        op_name="add",
        impl_name="device_custom",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(TurboDevice.CPU,),
    )

    registry.register(fallback_native)
    registry.register(device_native)
    registry.register(fallback_custom)
    registry.register(device_custom)

    assert registry.list(spec_filter=lambda spec: spec.op_name == "add") == [
        device_custom,
        device_native,
        fallback_custom,
        fallback_native,
    ]


def test_registry_rejects_duplicate_impl_name_across_ops():
    registry = OpRegistry()

    registry.register(
        OpSpec(
            op_name="add",
            impl_name="same_name",
            kind=OpImplKind.NATIVE,
            priority=0,
            devices=(),
        )
    )

    with pytest.raises(ValueError, match="already registered for op"):
        registry.register(
            OpSpec(
                op_name="mul",
                impl_name="same_name",
                kind=OpImplKind.NATIVE,
                priority=0,
                devices=(),
            )
        )


def test_grouped_matmul_facade_dispatches_registered_impl(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="grouped_matmul",
        impl_name="fake_grouped_matmul",
        kind=OpImplKind.NATIVE,
        priority=100,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: FakeGroupedMatmulImpl},
    )
    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())

    x = torch.ones(1)
    weight = torch.full((1,), 2.0)
    group_list = torch.tensor([1])

    actual = dispatcher.grouped_matmul(x, weight, group_list, trans_b=True)

    assert torch.equal(actual, torch.tensor([3.0]))


def test_dispatcher_caches_selected_impl_after_first_resolution(monkeypatch):
    CachedAddImpl.available_calls = 0
    registry = OpRegistry()
    spec = OpSpec(
        op_name="cached_add",
        impl_name="cached_add_impl",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: CachedAddImpl},
    )
    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())

    x = torch.tensor([1.0])

    assert torch.equal(dispatcher.cached_add(x), torch.tensor([2.0]))
    assert torch.equal(dispatcher.cached_add(x), torch.tensor([2.0]))
    assert CachedAddImpl.available_calls == 1


def test_dispatcher_uses_config_override_and_call_override(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    registry = OpRegistry()
    overridden = OpSpec(
        op_name="add",
        impl_name="overridden_add",
        kind=OpImplKind.CUSTOM,
        priority=100,
        devices=(TurboDevice.CPU,),
    )
    call_override = OpSpec(
        op_name="add",
        impl_name="call_override_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(overridden)
    registry.register(call_override)

    class OverriddenAddImpl(OpImpl):
        def __call__(self, x):
            return x + 10

    class CallOverrideAddImpl(OpImpl):
        def __call__(self, x):
            return x + 20

    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {
            overridden.impl_name: OverriddenAddImpl,
            call_override.impl_name: CallOverrideAddImpl,
        },
    )
    dispatcher = OpDispatcher(
        registry=registry,
        device_runtime=CpuRuntime(),
        default_policy=OpPolicy.NATIVE,
        overrides={"add": overridden.impl_name},
    )

    x = torch.tensor([1.0])
    assert torch.equal(dispatcher.add(x), torch.tensor([11.0]))
    assert torch.equal(
        dispatcher.add(x, impl=call_override.impl_name), torch.tensor([21.0])
    )
    assert dispatcher.overrides == {"add": "overridden_add"}
    dispatcher.log_ops_selection()
    assert "overridden" in caplog.text


def test_dispatcher_rejects_invalid_override(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="native_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)

    class NativeAddImpl(OpImpl):
        def __call__(self, x):
            return x

    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: NativeAddImpl},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*No OpSpec registered",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": "missing_add"},
        )


def test_dispatcher_rejects_impl_overridden_for_different_op(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="multiply",
        impl_name="native_multiply",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*registered for op 'multiply'",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": spec.impl_name},
        )


def test_dispatcher_rejects_impl_unsupported_on_current_device(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="npu_add",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(TurboDevice.NPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*does not support device=cpu",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": spec.impl_name},
        )


def test_dispatcher_rejects_unavailable_override(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="unavailable_add",
        kind=OpImplKind.CUSTOM,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {spec.impl_name: UnavailableAddImpl},
    )

    with pytest.raises(
        RuntimeError,
        match=r"Invalid override.*unavailable=\['unavailable_add'\]",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"add": spec.impl_name},
        )


def test_dispatcher_rejects_override_for_unknown_op(monkeypatch):
    registry = OpRegistry()
    spec = OpSpec(
        op_name="add",
        impl_name="native_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(TurboDevice.CPU,),
    )
    registry.register(spec)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {},
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid override.*not 'aardvark'",
    ):
        OpDispatcher(
            registry=registry,
            device_runtime=CpuRuntime(),
            overrides={"aardvark": spec.impl_name},
        )


def test_process_dispatcher_rejects_different_overrides():
    with pytest.raises(
        RuntimeError,
        match="incompatible configuration.*overrides",
    ):
        get_or_create_ops(
            device_runtime=CpuRuntime(),
            overrides={"rms_norm": "torch_rms_norm"},
        )


def test_dispatcher_logs_selected_ops(caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")

    dispatcher = OpDispatcher(
        registry=ops_registry,
        device_runtime=CpuRuntime(),
    )

    selected = dispatcher._candidate_impl_cache[("rms_norm", "cpu", None)]
    assert selected.spec.impl_name == "torch_rms_norm"
    assert {
        key[0] for key in dispatcher._candidate_impl_cache
    } >= {
        "attention_with_max_logits", "cross_entropy_loss",
        "grouped_matmul", "rms_norm", "swiglu",
    }
    dispatcher.log_ops_selection()

    assert caplog.text.count("TorchTurbo ops selection (device=cpu, policy=auto):") == 1
    message = caplog.records[-1].message
    lines = message.split("\n\n", maxsplit=1)[0].splitlines()
    assert lines[2].startswith("Op")
    assert {"Implementation", "Kind", "Priority", "Status"} <= set(lines[2].split())
    assert lines[1] == lines[3] == lines[-1]
    assert "rms_norm" in message
    assert "torch_rms_norm" in message
    rows = [line.split() for line in lines]
    assert any(
        row[:2] == ["attention_with_max_logits", "torch_attention_with_max_logits"]
        for row in rows
    )
    assert "selected" in message


def test_dispatcher_logs_unresolved_ops_without_failing_configuration(caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    registry = OpRegistry()
    registry.register(
        OpSpec(
            op_name="missing_op",
            impl_name="missing_impl",
            kind=OpImplKind.CUSTOM,
            priority=100,
            devices=(TurboDevice.CPU,),
        )
    )

    dispatcher = OpDispatcher(
        registry=registry,
        device_runtime=CpuRuntime(),
    )
    dispatcher.log_ops_selection()

    assert "missing_op" in caplog.text
    assert "unavailable" in caplog.text
    assert "Unavailable details:" in caplog.text
    with pytest.raises(RuntimeError, match="No OpImpl class registered"):
        dispatcher.missing_op()


def test_dispatcher_marks_fallback_selection(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    registry = OpRegistry()
    custom = OpSpec(
        op_name="add",
        impl_name="unavailable_add",
        kind=OpImplKind.CUSTOM,
        priority=100,
        devices=(TurboDevice.CPU,),
    )
    native = OpSpec(
        op_name="add",
        impl_name="fallback_add",
        kind=OpImplKind.NATIVE,
        priority=0,
        devices=(),
    )
    registry.register(custom)
    registry.register(native)
    monkeypatch.setattr(
        OpDispatcher,
        "_load_provider_impl_classes",
        lambda self, device: {
            custom.impl_name: UnavailableAddImpl,
            native.impl_name: FallbackAddImpl,
        },
    )

    dispatcher = OpDispatcher(registry=registry, device_runtime=CpuRuntime())
    dispatcher.log_ops_selection()

    assert "add" in caplog.text
    assert "fallback_add" in caplog.text
    assert "fallback" in caplog.text


def test_dispatcher_can_use_injected_device_runtime(monkeypatch):
    def _boom():
        raise AssertionError("global runtime probe should not be used")

    monkeypatch.setattr(runtime_impl, "get_device_runtime", _boom)

    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)

    actual = get_ops().rms_norm(hidden_states, weight, 1e-6)
    expected = _reference_rms_norm(hidden_states, weight, 1e-6)

    assert torch.allclose(actual, expected)


@pytest.fixture
def attention_providers(monkeypatch):
    from torchturbo.ops.providers.cuda.attention import (
        FlexAttentionWithMaxLogits,
        TransformerEngineAttentionWithMaxLogits,
    )
    from torchturbo.ops.providers.native.attention import TorchAttentionWithMaxLogits
    from torchturbo.ops.providers.npu.attention import NpuAttentionWithMaxLogits

    provider_classes = (
        TorchAttentionWithMaxLogits,
        TransformerEngineAttentionWithMaxLogits,
        FlexAttentionWithMaxLogits,
        NpuAttentionWithMaxLogits,
    )
    registry = OpRegistry()
    availability = {}
    compute = {}
    query = torch.ones(1, 3, 2, 4)
    maxima = torch.ones(2)
    for provider in provider_classes:
        spec = provider.op_spec()
        registry.register(ops_registry.get_by_impl_name(spec.impl_name))
        availability[spec.impl_name] = Mock(return_value=True)
        monkeypatch.setattr(provider, "available", availability[spec.impl_name])
        compute[spec.impl_name] = Mock(return_value=(query, maxima))
        monkeypatch.setattr(provider, "__call__", compute[spec.impl_name])

    return SimpleNamespace(
        registry=registry,
        availability=availability,
        compute=compute,
        query=query,
        maxima=maxima,
    )


def test_attention_provider_registration_keeps_capabilities_and_devices_separate(
    attention_providers,
):
    expected = {
        "torch_attention_with_max_logits": (
            "attention_with_max_logits", OpImplKind.NATIVE, 0, (TurboDevice.CPU,),
        ),
        "te_attention_with_max_logits": (
            "attention_with_max_logits", OpImplKind.CUSTOM, 100, (TurboDevice.CUDA,),
        ),
        "flex_attention_with_max_logits": (
            "attention_with_max_logits", OpImplKind.NATIVE, 0, (TurboDevice.CUDA,),
        ),
        "npu_attention_with_max_logits": (
            "attention_with_max_logits", OpImplKind.CUSTOM, 100, (TurboDevice.NPU,),
        ),
    }

    for name, metadata in expected.items():
        spec = attention_providers.registry.get_by_impl_name(name)
        assert (spec.op_name, spec.kind, spec.priority, spec.devices) == metadata


@pytest.mark.parametrize(
    ("runtime_type", "maximum_impl"),
    [
        (CpuRuntime, "torch_attention_with_max_logits"),
        (CudaRuntime, "te_attention_with_max_logits"),
        (NpuRuntime, "npu_attention_with_max_logits"),
    ],
)
def test_maximum_attention_op_selects_and_caches(
    attention_providers, runtime_type, maximum_impl,
):
    runtime = runtime_type()
    dispatcher = OpDispatcher(
        registry=attention_providers.registry, device_runtime=runtime,
    )
    query = attention_providers.query
    device = runtime.turbo_device.value
    maximum = dispatcher._candidate_impl_cache[("attention_with_max_logits", device, None)]

    for _ in range(2):
        output, maxima = dispatcher.attention_with_max_logits(
            query, query, query, None, 0.5, 0.0, True,
        )
        torch.testing.assert_close(output, query)
        assert maxima is attention_providers.maxima

    assert maximum.spec.impl_name == maximum_impl
    assert dispatcher._candidate_impl_cache[("attention_with_max_logits", device, None)] is maximum
    attention_providers.availability[maximum_impl].assert_called_once_with()
    assert attention_providers.compute[maximum_impl].call_count == 2


def test_maximum_attention_override_precedence(attention_providers):
    dispatcher = OpDispatcher(
        registry=attention_providers.registry,
        device_runtime=CudaRuntime(),
        overrides={"attention_with_max_logits": "flex_attention_with_max_logits"},
    )
    query = attention_providers.query

    dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)
    dispatcher.attention_with_max_logits(
        query, query, query, None, 0.5, 0.0, True, impl="te_attention_with_max_logits",
    )
    dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)

    assert attention_providers.compute["flex_attention_with_max_logits"].call_count == 2
    attention_providers.compute["te_attention_with_max_logits"].assert_called_once()
    assert set(dispatcher._candidate_impl_cache) == {
        ("attention_with_max_logits", "cuda", "flex_attention_with_max_logits"),
        ("attention_with_max_logits", "cuda", "te_attention_with_max_logits"),
    }


def test_native_attention_policy_selects_flex_instead_of_available_te(attention_providers):
    dispatcher = OpDispatcher(
        registry=attention_providers.registry,
        device_runtime=CudaRuntime(),
        default_policy=OpPolicy.NATIVE,
    )
    query = attention_providers.query

    dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)

    attention_providers.compute["flex_attention_with_max_logits"].assert_called_once()
    attention_providers.availability["te_attention_with_max_logits"].assert_not_called()
    attention_providers.compute["te_attention_with_max_logits"].assert_not_called()


@pytest.mark.parametrize(
    ("runtime_type", "maximum_impl"),
    [
        (CudaRuntime, "te_attention_with_max_logits"),
        (NpuRuntime, "npu_attention_with_max_logits"),
    ],
)
def test_custom_attention_policy_selects_custom_maximum_impl(
    attention_providers, runtime_type, maximum_impl,
):
    dispatcher = OpDispatcher(
        registry=attention_providers.registry,
        device_runtime=runtime_type(),
        default_policy=OpPolicy.CUSTOM,
    )
    query = attention_providers.query

    dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)
    attention_providers.compute[maximum_impl].assert_called_once()
    attention_providers.compute["flex_attention_with_max_logits"].assert_not_called()


def test_auto_attention_falls_back_only_when_te_is_unavailable(attention_providers, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    attention_providers.availability["te_attention_with_max_logits"].return_value = False
    dispatcher = OpDispatcher(
        registry=attention_providers.registry, device_runtime=CudaRuntime(),
    )
    query = attention_providers.query

    dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)
    attention_providers.compute["flex_attention_with_max_logits"].assert_called_once()
    attention_providers.compute["te_attention_with_max_logits"].assert_not_called()
    dispatcher.log_ops_selection()
    assert any(
        "attention_with_max_logits" in line and "fallback" in line
        for line in caplog.records[-1].message.splitlines()
    )

    custom = OpDispatcher(
        registry=attention_providers.registry,
        device_runtime=CudaRuntime(),
        default_policy=OpPolicy.CUSTOM,
    )
    with pytest.raises(RuntimeError, match="No implementation.*policy=custom"):
        custom.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)
    with pytest.raises(RuntimeError, match="No implementation.*impl=te_attention_with_max_logits"):
        dispatcher.attention_with_max_logits(
            query, query, query, None, 0.5, 0.0, True, impl="te_attention_with_max_logits",
        )
    assert attention_providers.compute["flex_attention_with_max_logits"].call_count == 1


def test_attention_does_not_fall_back_after_te_execution_fails(attention_providers):
    failure = RuntimeError("TE kernel execution failed")
    attention_providers.compute["te_attention_with_max_logits"].side_effect = failure
    dispatcher = OpDispatcher(
        registry=attention_providers.registry, device_runtime=CudaRuntime(),
    )
    query = attention_providers.query

    with pytest.raises(RuntimeError) as raised:
        dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)

    assert raised.value is failure
    attention_providers.compute["flex_attention_with_max_logits"].assert_not_called()
    attention_providers.compute["torch_attention_with_max_logits"].assert_not_called()


def test_unavailable_cuda_attention_cannot_fall_back_to_cpu(attention_providers):
    for name in ("te_attention_with_max_logits", "flex_attention_with_max_logits"):
        attention_providers.availability[name].return_value = False
    dispatcher = OpDispatcher(
        registry=attention_providers.registry, device_runtime=CudaRuntime(),
    )
    query = attention_providers.query

    with pytest.raises(RuntimeError, match="No implementation.*device=cuda"):
        dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)
    with pytest.raises(RuntimeError, match="does not support device=cuda"):
        dispatcher.attention_with_max_logits(
            query, query, query, None, 0.5, 0.0, True, impl="torch_attention_with_max_logits",
        )
    attention_providers.availability["torch_attention_with_max_logits"].assert_not_called()
    attention_providers.compute["torch_attention_with_max_logits"].assert_not_called()


def test_cpu_attention_never_selects_cuda_providers(attention_providers):
    dispatcher = OpDispatcher(
        registry=attention_providers.registry, device_runtime=CpuRuntime(),
    )
    query = attention_providers.query

    dispatcher.attention_with_max_logits(query, query, query, None, 0.5, 0.0, True)

    attention_providers.compute["torch_attention_with_max_logits"].assert_called_once()
    for name in ("te_attention_with_max_logits", "flex_attention_with_max_logits"):
        attention_providers.availability[name].assert_not_called()
        attention_providers.compute[name].assert_not_called()
        with pytest.raises(RuntimeError, match="does not support device=cpu"):
            dispatcher.attention_with_max_logits(
                query, query, query, None, 0.5, 0.0, True, impl=name,
            )
