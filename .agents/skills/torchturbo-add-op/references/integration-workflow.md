# TorchTurbo operator integration workflow

## 1. Discover the existing pattern

Before creating files, search for the op name and for an operator with similar tensor shapes, autograd behavior, or kernel constraints:

```bash
rg -n 'op_name="<op_name>"|def <op_name>|ops\.<op_name>' src tests internal
```

Reuse an existing operator when the requested computation already has stable semantics. Do not add a second logical op under a slightly different name for the same calculation.

Read the nearest existing provider pair as a template:

| Need | Reference files |
| --- | --- |
| Simple elementwise or norm op, native plus NPU fused kernel | `providers/native/rms_norm.py`, `providers/npu/rms_norm.py` |
| Op with several device implementations at different priorities | `providers/npu/matmul_add.py` (200 and 100), `providers/native/matmul_add.py` |
| `torch.compile` based device implementation | `providers/cuda/swiglu.py`, `providers/cuda/gated_activation.py` |
| Op whose native baseline runs only on CPU | `providers/native/attention.py` (`devices=(TurboDevice.CPU,)`) |
| Op needing argument normalization in the facade | `dispatcher.py` `gated_delta_rule` / `causal_conv1d` (`cu_seqlens` -> `packed_params`) |
| Device implementation backed by local kernel code | `providers/npu/gated_delta_rule.py` with `providers/npu/_fla_npu_gdn/` |

## 2. Define the semantic contract first

Write down, and ideally encode in tests, the operator contract before choosing implementations:

- Stable `op_name` and the exact positional, keyword-only, and default arguments.
- Supported input shapes, dtypes, devices, and layouts.
- Output shapes, dtypes, devices, and whether results are returned, mutated in place, or both.
- Empty, singleton-dimension, and non-contiguous input behavior where relevant.
- Autograd behavior and gradient semantics.
- Error cases and exception types.
- Any state, packed metadata, or engine lifecycle requirement.

All native and custom implementations must satisfy this same contract. `impl_name` must be unique across the whole registry and cannot be `auto`, `native`, or `custom`; `OpSpec.__post_init__` rejects the reserved names and `OpRegistry.register` rejects an `impl_name` already bound to a different op.

## 3. Add the native provider

Create an ordinary module such as `src/torchturbo/ops/providers/native/<op_name>.py` and define each `OpImpl` subclass directly in it. Discovery via `pkgutil.walk_packages` imports the module automatically; there is no provider list to maintain.

A runtime-independent PyTorch baseline uses `devices=()` and `priority=0`:

```python
from __future__ import annotations

import torch

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="scaled_add",
    impl_name="torch_scaled_add",
    kind=OpImplKind.NATIVE,
    priority=0,
    devices=(),
)
class TorchScaledAdd(OpImpl):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        return torch.add(x, y, alpha=alpha)
```

Narrow `devices` on a native provider only when the baseline genuinely cannot run on every runtime. A native implementation should stay expressed in PyTorch operations and act as the numerical baseline for custom implementations.

## 4. Add hardware custom providers

Create `src/torchturbo/ops/providers/cuda/<op_name>.py` or `src/torchturbo/ops/providers/npu/<op_name>.py` only when a device implementation is part of the task.

```python
from __future__ import annotations

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="scaled_add",
    impl_name="npu_scaled_add",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuScaledAdd(OpImpl):
    def available(self) -> bool:
        try:
            from example_npu_kernels import scaled_add  # noqa: F401
        except ImportError:
            return False
        return True

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        from example_npu_kernels import scaled_add

        return scaled_add(x, y, alpha=alpha)
```

Rules:

- Match the native call signature and semantics exactly.
- Declare `devices=(TurboDevice.CUDA,)` or `devices=(TurboDevice.NPU,)` to match the provider directory.
- Name the implementation after the backing library or kernel, following existing names such as `npu_rms_norm`, `mindspeed_grouped_matmul`, `ascendc_causal_conv1d`, `torch_compile_swiglu`, `grouped_gemm`.
- Never import `torch_npu`, MindSpeed, Transformer Engine, `grouped_gemm`, or other optional packages at module scope. Import them inside `available()` and `__call__()`.
- `available()` decides only whether this implementation can safely run. It must not initialize or mutate the global runtime.
- Put no device detection or fallback logic inside a provider; that is the dispatcher's job.

Priorities in use today: `0` for native baselines, `50` for `torch.compile` variants, `100` for the common fused device kernel, `200` when a device has two kernels and one should win (`mindspeed_matmul_add` over `torch_npu_matmul_add`). A device implementation at the same priority as a device-agnostic one still wins, because `OpRegistry.list` ranks non-empty `devices` first, then `custom` over `native`. Prefer a distinct priority over relying on the tie-break, and cover the intended winner in a test when several candidates exist.

Provider filenames and every path component must not start with `_`; `_iter_package_modules` skips those, which is how `npu/_fla_npu_gdn/` stays a private helper package. Re-exporting an `OpImpl` defined in another module does not register it either, because `_index_provider_module` only indexes classes whose `__module__` is the provider module.

## 5. Decide whether to add an explicit facade

`OpDispatcher.__getattr__` exposes every registered op dynamically, so model code can call a new op without editing `dispatcher.py`:

```python
result = ops.scaled_add(x, y, alpha=0.5)
```

Add an explicit method to `OpDispatcher` when the op is a stable public interface, needs a typed signature and IDE support, or needs operator-specific argument normalization:

```python
def scaled_add(
    self,
    x,
    y,
    *,
    alpha: float = 1.0,
    impl: str | None = None,
):
    return self._dispatch(
        "scaled_add",
        x,
        y,
        alpha=alpha,
        impl=impl,
    )
```

Keep `impl` keyword-only and pass it through to `_dispatch`. A facade may normalize arguments the way `gated_delta_rule` converts `cu_seqlens` into `packed_params`, but it must not select devices, query hardware availability, or implement fallback.

## 6. Integrate calls into model or engine code

After Engine initialization, model and shared components use the process proxy:

```python
from torchturbo.ops import ops


output = ops.scaled_add(x, y, alpha=0.5)
```

An Engine instance also exposes `engine.ops.<op_name>(...)`. A standalone script must initialize the process dispatcher first:

```python
from torchturbo.ops import OpPolicy, get_or_create_ops
from torchturbo.runtime import get_device_runtime


runtime = get_device_runtime()
dispatcher = get_or_create_ops(
    device_runtime=runtime,
    default_policy=OpPolicy.AUTO,
)
```

Never call `ops.<op_name>()` while a provider or model module is being imported. Never write a device branch on `is_npu_available()` or `is_cuda_available()`: those report machine capability, not the process-selected runtime, so they pick wrong when CUDA and NPU coexist. When behavior genuinely depends on the current device, compare `get_device_runtime().turbo_device` against a `TurboDevice` value.

Configuration changes are not needed for every operator. When a caller needs deterministic selection, set `OpsConfig.overrides` by op name and `impl_name` in `src/torchturbo/engine/config.py`, or pass `impl=` per call, which takes precedence over the config override. Both are explicit selections: an unavailable or device-incompatible choice must fail rather than fall back, and an invalid override fails during `OpDispatcher` construction. The default policy also comes from `--tt_op_policy` or `TORCH_TURBO_OP_POLICY`, which map to the `ops.policy` config path; a top-level `op_policy` config key is rejected.

## 7. Add focused tests

Cover the native contract and dispatcher integration at minimum. For a runtime-independent op, prefer a CPU test in `tests/ops/test_dispatch.py`, where the autouse fixture already resets the process dispatcher to `CpuRuntime`:

```python
def test_scaled_add_uses_native_dispatch():
    dispatcher = get_ops()
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    actual = dispatcher.scaled_add(x, y, alpha=0.5)
    expected = torch.add(x, y, alpha=0.5)

    torch.testing.assert_close(actual, expected)
```

Hardware tests belong under `tests/ops/npu/` or `tests/ops/providers/npu/`, matching the existing split. Use `pytest.importorskip()` or import optional packages inside the test body so CPU collection does not fail when `torch_npu` or MindSpeed is absent.

Depending on the change, also cover:

- Numerical alignment with the native baseline for the relevant dtypes and tolerances.
- Backward and gradient alignment when autograd is part of the contract.
- Shape, dtype, device, mutation, reduction, and error semantics.
- `OpPolicy.AUTO` fallback from an unavailable custom candidate to the native one. Build an isolated `OpRegistry` with fake `OpImpl` classes when hardware is unavailable, as `tests/ops/test_dispatch.py` does.
- `OpPolicy.NATIVE` and `OpPolicy.CUSTOM` raising when the strict policy has no eligible implementation.
- `OpsConfig.overrides` and call-site `impl=` selection, including the error path for a mismatched op or device.

Keep shared semantic tests on CPU and limit hardware tests to device-specific alignment. Do not duplicate a full suite per provider.

## 8. Validate the change

Run the smallest meaningful set first:

```bash
PYTHONPATH=src pytest -q tests/ops/test_dispatch.py
git diff --check
```

If provider discovery or runtime matching changed, also run:

```bash
PYTHONPATH=src pytest -q tests/runtime/test_runtime.py
```

Run hardware tests only on a machine with the required device and packages. Finish by reviewing `git diff` for accidental policy changes, device branching, import-time side effects, or unrelated formatting churn.

## Common failure patterns

- `No implementation for op=...`: check the provider directory matches the runtime, the filename has no leading `_`, the class is defined directly in the provider module, `op_name` and `impl_name` are correct, `devices` includes the current `TurboDevice`, the active policy does not filter out the `kind`, and `available()` is not returning `False` from a missing optional dependency.
- `Ops dispatcher is not configured`: the call ran before Engine or `get_or_create_ops()` initialization. Fix the lifecycle instead of adding fallback behavior.
- `policy=native` failing on NPU: expected when the op has no native implementation for that runtime. Use `auto` for fallback or `custom` to require the fused implementation.
- Selection log showing `fallback`: a higher-ranked candidate returned `available() == False`. Read the `Unavailable details` section of the Engine startup log rather than bypassing the dispatcher.
- `Op implementation ... is defined by both ...`: two provider modules define a class with the same `impl_name`; make the name unique.
