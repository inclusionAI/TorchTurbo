from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType

import pytest
import torch

from torchturbo.ops import (
    OpBinding,
    OpDispatcher,
    OpPlugin,
    OpPolicy,
    get_or_create_ops,
)
from torchturbo.ops.registry import registry
from torchturbo.runtime import TurboDevice
from torchturbo.runtime.cpu import CpuRuntime


@pytest.fixture
def plugin_module(tmp_path, monkeypatch):
    monkeypatch.setattr(logging.getLogger("torchturbo"), "propagate", True)
    module = ModuleType("external_tt_plugin")
    module.square = lambda x: x.square()
    module.detached = lambda x: x.detach().square()
    module.unavailable = lambda: False
    module.invalid_availability = lambda: "yes"

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return 2 * x * grad_output

    module.setup_context = setup_context
    module.backward = backward
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def binding(**kwargs):
    return OpBinding(
        **{
            "op_name": "plugin_square",
            "impl_name": "external::square",
            "target": "external_tt_plugin:square",
            "devices": (TurboDevice.CPU,),
            **kwargs,
        }
    )


def dispatcher(module, *bindings, **kwargs):
    module.plugin = OpPlugin(name="external", bindings=bindings or (binding(),))
    return OpDispatcher(
        device_runtime=CpuRuntime(),
        registry=registry,
        plugins=(f"{module.__name__}:plugin",),
        **kwargs,
    )


def test_plugin_preserves_autograd_and_isolates_registry(plugin_module):
    OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    before = registry.list()
    ops = dispatcher(plugin_module)
    x = torch.randn(5, dtype=torch.double, requires_grad=True)
    torch.testing.assert_close(ops.plugin_square(x), x.square())
    assert torch.autograd.gradcheck(ops.plugin_square, (x,))
    assert registry.list() == before
    other = OpDispatcher(device_runtime=CpuRuntime(), registry=registry)
    with pytest.raises(AttributeError):
        _ = other.plugin_square


def test_plugin_loads_real_python_modules(tmp_path, monkeypatch):
    (tmp_path / "plain_kernels.py").write_text(
        "def add(x, y, *, alpha=1):\n    return x + alpha * y\n"
    )
    (tmp_path / "project_plugin.py").write_text(
        "from torchturbo.ops import OpBinding, OpPlugin\n"
        "from torchturbo.runtime import TurboDevice\n"
        "plugin = OpPlugin(name='plain', bindings=(OpBinding(\n"
        "    op_name='scaled_add', impl_name='plain::add',\n"
        "    target='plain_kernels:add', devices=(TurboDevice.CPU,)),))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        ops = OpDispatcher(
            device_runtime=CpuRuntime(), registry=registry,
            plugins=("project_plugin:plugin",),
        )
        x = torch.ones(3, requires_grad=True)
        result = ops.scaled_add(x, x, alpha=2)
        torch.testing.assert_close(result, 3 * x)
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 3))
    finally:
        sys.modules.pop("plain_kernels", None)
        sys.modules.pop("project_plugin", None)


def test_explicit_backward_wraps_black_box(plugin_module):
    ops = dispatcher(
        plugin_module,
        binding(target="external_tt_plugin:detached",
                backward="external_tt_plugin:backward",
                setup_context="external_tt_plugin:setup_context"),
    )
    x = torch.randn(4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(ops.plugin_square, (x,))


def test_explicit_backward_takes_precedence(plugin_module):
    plugin_module.custom_backward = lambda ctx, grad: grad * 7
    ops = dispatcher(
        plugin_module, binding(backward="external_tt_plugin:custom_backward")
    )
    x = torch.ones(3, requires_grad=True)
    ops.plugin_square(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 7))


def test_detached_output_errors_with_source(plugin_module):
    ops = dispatcher(plugin_module, binding(target="external_tt_plugin:detached"))
    with pytest.raises(RuntimeError, match="external::square.*autograd"):
        ops.plugin_square(torch.ones(2, requires_grad=True))


def test_policy_override_and_callsite_selection(plugin_module):
    custom = binding(op_name="swiglu", target="external_tt_plugin:square", priority=500)
    ops = dispatcher(plugin_module, custom)
    x = torch.randn(2, 4)
    torch.testing.assert_close(ops.swiglu(x), x.square())
    assert ops.swiglu(x, impl="torch_swiglu").shape == (2, 2)
    native = dispatcher(plugin_module, custom, default_policy=OpPolicy.NATIVE)
    assert native.swiglu(x).shape == (2, 2)
    override = dispatcher(plugin_module, custom, overrides={"swiglu": "torch_swiglu"})
    assert override.swiglu(x).shape == (2, 2)
    torch.testing.assert_close(override.swiglu(x, impl="external::square"), x.square())


def test_missing_target_falls_back_but_explicit_does_not(plugin_module):
    ops = dispatcher(plugin_module, binding(
        op_name="swiglu", target="missing_tt_plugin_dependency:square", priority=500
    ))
    x = torch.randn(2, 4)
    assert ops.swiglu(x).shape == (2, 2)
    with pytest.raises(RuntimeError, match="missing_tt_plugin_dependency") as error:
        ops.swiglu(x, impl="external::square")
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    with pytest.raises(RuntimeError, match="Invalid override"):
        dispatcher(plugin_module, plugin_module.plugin.bindings[0],
                   overrides={"swiglu": "external::square"})


def test_device_filtered_target_not_imported(plugin_module):
    ops = dispatcher(plugin_module, binding(
        target="missing_tt_plugin_dependency:square", devices=(TurboDevice.NPU,)
    ))
    with pytest.raises(RuntimeError, match="does not support"):
        ops.plugin_square(torch.ones(2), impl="external::square")


@pytest.mark.parametrize("target", ["external_tt_plugin:absent", "external_tt_plugin:not_callable"])
def test_invalid_target_is_not_unavailability(plugin_module, target):
    plugin_module.not_callable = 3
    with pytest.raises((ValueError, TypeError), match="external::square"):
        dispatcher(plugin_module, binding(target=target))


def test_availability_failure_and_invalid_return(plugin_module):
    ops = dispatcher(plugin_module, binding(availability="external_tt_plugin:unavailable"))
    with pytest.raises(RuntimeError, match="availability"):
        ops.plugin_square(torch.ones(2))
    with pytest.raises(TypeError, match="bool"):
        dispatcher(plugin_module, binding(availability="external_tt_plugin:invalid_availability"))


def test_execution_error_never_falls_back(plugin_module):
    def fail(x):
        raise ValueError("kernel failed")

    plugin_module.fail = fail
    ops = dispatcher(plugin_module, binding(
        op_name="swiglu", target="external_tt_plugin:fail", priority=500
    ))
    with pytest.raises(ValueError, match="kernel failed") as error:
        ops.swiglu(torch.ones(2, 4))
    assert any("external::square" in note for note in error.value.__notes__)


def test_conflicting_bindings_rejected(plugin_module):
    with pytest.raises(ValueError, match="external::square"):
        dispatcher(plugin_module, binding(), binding())


@pytest.mark.parametrize("op_name", ["registry", "log_ops_selection", "_private", "not-valid"])
def test_reserved_op_names_rejected(plugin_module, op_name):
    with pytest.raises(ValueError, match="op_name"):
        dispatcher(plugin_module, binding(op_name=op_name))


def test_invalid_plugin_version_and_reference(plugin_module):
    plugin_module.plugin = OpPlugin(name="external", bindings=(binding(),), api_version=2)
    with pytest.raises(ValueError, match="api_version"):
        OpDispatcher(device_runtime=CpuRuntime(), registry=registry,
                     plugins=("external_tt_plugin:plugin",))
    with pytest.raises(ValueError, match="module:attribute"):
        OpDispatcher(device_runtime=CpuRuntime(), registry=registry, plugins=("external_tt_plugin",))


def test_setup_context_requires_backward():
    with pytest.raises(ValueError, match="backward"):
        binding(setup_context="external_tt_plugin:setup_context")


def test_singleton_compares_plugins_and_preserves_failed_initialization(plugin_module, monkeypatch):
    singleton = importlib.import_module("torchturbo.ops.ops")
    monkeypatch.setattr(singleton, "_OPS", None)
    plugin_module.plugin = OpPlugin(name="external", bindings=(binding(),))
    ops = get_or_create_ops(device_runtime=CpuRuntime(), plugins=("external_tt_plugin:plugin",))
    assert get_or_create_ops(device_runtime=CpuRuntime(), plugins=("external_tt_plugin:plugin",)) is ops
    with pytest.raises(RuntimeError, match="plugins"):
        get_or_create_ops(device_runtime=CpuRuntime())
    monkeypatch.setattr(singleton, "_OPS", None)
    with pytest.raises((ValueError, ModuleNotFoundError)):
        get_or_create_ops(device_runtime=CpuRuntime(), plugins=("absent_plugin:plugin",))
    assert singleton._OPS is None


def test_selection_log_includes_plugin_source(plugin_module, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo")
    ops = dispatcher(plugin_module)
    ops.log_ops_selection()
    assert "external::square" in caplog.text
    assert "external_tt_plugin:square" in caplog.text
    assert "builtin" in caplog.text


def test_fallback_log_keeps_missing_dependency_reason(plugin_module, caplog):
    caplog.set_level(logging.INFO, logger="torchturbo.ops.dispatcher")
    ops = dispatcher(plugin_module, binding(
        op_name="swiglu", target="missing_tt_plugin_dependency:square", priority=500
    ))
    ops.log_ops_selection()
    assert "fallback" in caplog.text
    assert "missing dependency 'missing_tt_plugin_dependency'" in caplog.text


def test_missing_dependency_chain_survives_later_unavailable_candidate(plugin_module):
    ops = dispatcher(
        plugin_module,
        binding(target="missing_tt_plugin_dependency:square", priority=100),
        binding(impl_name="external::unavailable", availability="external_tt_plugin:unavailable"),
    )
    with pytest.raises(RuntimeError, match="missing_tt_plugin_dependency") as error:
        ops.plugin_square(torch.ones(2))
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_same_priority_uses_impl_name_not_registration_order(plugin_module):
    plugin_module.cube = lambda x: x.pow(3)
    first = binding(impl_name="external::z", target="external_tt_plugin:cube")
    second = binding(impl_name="external::a")
    x = torch.tensor([2.0])
    torch.testing.assert_close(dispatcher(plugin_module, first, second).plugin_square(x), x.square())
    torch.testing.assert_close(dispatcher(plugin_module, second, first).plugin_square(x), x.square())


@pytest.mark.parametrize("mode", ["device", "policy", "priority"])
def test_filtered_targets_are_never_imported(plugin_module, tmp_path, monkeypatch, mode):
    (tmp_path / "unselected_kernel.py").write_text("raise RuntimeError('must not import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    skipped = binding(target="unselected_kernel:kernel", priority=-100)
    options = {}
    bindings = (skipped,)
    if mode == "device":
        bindings = (binding(target=skipped.target, devices=(TurboDevice.NPU,)),)
    elif mode == "policy":
        options["default_policy"] = OpPolicy.NATIVE
    else:
        bindings = (skipped, binding(impl_name="external::selected"))
    dispatcher(plugin_module, *bindings, **options)
    assert "unselected_kernel" not in sys.modules


def test_strict_custom_does_not_fall_back_to_native(plugin_module):
    ops = dispatcher(plugin_module, binding(
        op_name="swiglu", availability="external_tt_plugin:unavailable"
    ), default_policy=OpPolicy.CUSTOM)
    with pytest.raises(RuntimeError, match="policy=custom"):
        ops.swiglu(torch.ones(2, 4))


def test_existing_autograd_function_is_preserved(plugin_module):
    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.square()

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * 2 * x

    plugin_module.square = Square.apply
    ops = dispatcher(plugin_module)
    x = torch.randn(3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(ops.plugin_square, (x,))


def test_explicit_backward_handles_multiple_inputs_outputs_and_context(plugin_module):
    plugin_module.pair = lambda x, y, scale: ((x * y * scale).detach(), (x + y).detach())

    def setup(ctx, inputs, output):
        x, y, ctx.scale = inputs
        ctx.save_for_backward(x, y)

    def backward(ctx, grad_product, grad_sum):
        x, y = ctx.saved_tensors
        return grad_product * y * ctx.scale + grad_sum, grad_product * x * ctx.scale + grad_sum, None

    plugin_module.setup_pair = setup
    plugin_module.backward_pair = backward
    ops = dispatcher(plugin_module, binding(
        target="external_tt_plugin:pair", backward="external_tt_plugin:backward_pair",
        setup_context="external_tt_plugin:setup_pair",
    ))
    x = torch.randn(3, dtype=torch.double, requires_grad=True)
    y = torch.randn(3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, y: ops.plugin_square(x, y, 2), (x, y))
    assert torch.autograd.gradcheck(lambda x, y: ops.plugin_square(x, y, scale=2), (x, y))


def test_explicit_backward_accepts_existing_facade_keywords(plugin_module):
    def forward(x, weight, group_list, *, trans_b=False):
        matrix = weight.t() if trans_b else weight
        return (x @ matrix).detach()

    def setup(ctx, inputs, output):
        x, weight, _group_list, ctx.trans_b = inputs
        ctx.save_for_backward(x, weight)

    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        if ctx.trans_b:
            return grad @ weight, grad.t() @ x, None, None
        return grad @ weight.t(), x.t() @ grad, None, None

    plugin_module.matmul = forward
    plugin_module.matmul_setup = setup
    plugin_module.matmul_backward = backward
    ops = dispatcher(plugin_module, binding(
        op_name="grouped_matmul", target="external_tt_plugin:matmul",
        setup_context="external_tt_plugin:matmul_setup",
        backward="external_tt_plugin:matmul_backward", priority=500,
    ))
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, weight: ops.grouped_matmul(x, weight, None, trans_b=True), (x, weight)
    )


def test_explicit_backward_rejects_nested_tensor_inputs(plugin_module):
    plugin_module.nested = lambda values: values[0].detach().square()
    ops = dispatcher(plugin_module, binding(
        target="external_tt_plugin:nested", backward="external_tt_plugin:backward",
    ))
    with pytest.raises(TypeError, match="nested tensor"):
        ops.plugin_square([torch.ones(2, requires_grad=True)])


def test_explicit_backward_rejects_nested_outputs(plugin_module):
    plugin_module.nested = lambda x: {"value": x.detach().square()}
    ops = dispatcher(plugin_module, binding(
        target="external_tt_plugin:nested", backward="external_tt_plugin:backward",
    ))
    with pytest.raises(TypeError, match="flat tuple"):
        ops.plugin_square(torch.ones(2, requires_grad=True))


def test_duplicate_plugin_ids_and_references(plugin_module):
    plugin_module.plugin = OpPlugin(name="external", bindings=(binding(),))
    plugin_module.alias = plugin_module.plugin
    with pytest.raises(ValueError, match="Duplicate plugin ID"):
        OpDispatcher(device_runtime=CpuRuntime(), registry=registry,
                     plugins=("external_tt_plugin:plugin", "external_tt_plugin:alias"))
    with pytest.raises(ValueError, match="Duplicate plugin references"):
        OpDispatcher(device_runtime=CpuRuntime(), registry=registry,
                     plugins=("external_tt_plugin:plugin",) * 2)


def test_plugin_cannot_override_builtin_impl(plugin_module):
    from torchturbo.ops import OpRegistry, OpSpec

    local = OpRegistry()
    local.register(OpSpec(op_name="plugin_square", impl_name="external::square"))
    plugin_module.plugin = OpPlugin(name="external", bindings=(binding(),))
    with pytest.raises(ValueError, match="Duplicate op implementation"):
        OpDispatcher(device_runtime=CpuRuntime(), registry=local,
                     plugins=("external_tt_plugin:plugin",))


def test_dependency_error_keeps_original_cause(plugin_module, tmp_path, monkeypatch):
    (tmp_path / "broken_kernel.py").write_text("raise ImportError('ABI failure')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="ABI failure") as error:
        dispatcher(plugin_module, binding(target="broken_kernel:square"))
    assert isinstance(error.value.__cause__, ImportError)


def test_availability_check_is_cached(plugin_module):
    calls = []

    def available():
        calls.append(True)
        return True

    plugin_module.available = available
    ops = dispatcher(plugin_module, binding(availability="external_tt_plugin:available"))
    for _ in range(3):
        torch.testing.assert_close(ops.plugin_square(torch.ones(2)), torch.ones(2))
    assert len(calls) == 1


def test_inline_plugin_singleton_compares_binding_content(monkeypatch):
    singleton = importlib.import_module("torchturbo.ops.ops")
    monkeypatch.setattr(singleton, "_OPS", None)
    config = {"name": "external", "bindings": [{
        "op_name": "square", "impl_name": "external::square",
        "target": "torch:square", "devices": ["cpu"],
    }]}
    ops = get_or_create_ops(device_runtime=CpuRuntime(), plugins=[config])
    assert get_or_create_ops(device_runtime=CpuRuntime(), plugins=[config]) is ops
    config["bindings"][0]["target"] = "torch:neg"
    with pytest.raises(RuntimeError, match="plugins"):
        get_or_create_ops(device_runtime=CpuRuntime(), plugins=[config])
    torch.testing.assert_close(ops.square(torch.tensor([2.])), torch.tensor([4.]))


@pytest.mark.parametrize("field,value", [("devices", "cpu"), ("devices", ["invalid"]),
                                       ("kind", "invalid"), ("unknown", True)])
def test_inline_binding_rejects_invalid_configuration(field, value):
    config = {"name": "external", "bindings": [{
        "op_name": "square", "impl_name": "external::square",
        "target": "torch:square", "devices": ["cpu"], field: value,
    }]}
    with pytest.raises((TypeError, ValueError)):
        OpDispatcher(device_runtime=CpuRuntime(), registry=registry, plugins=[config])
