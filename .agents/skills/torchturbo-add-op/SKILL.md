---
name: torchturbo-add-op
description: Add or update TorchTurbo operator providers and dispatcher integration when introducing a new op or a device-specific implementation.
metadata:
  short-description: Integrate a TorchTurbo operator
---

# TorchTurbo operator integration

Use this skill for work under the TorchTurbo ops architecture: defining a new logical operator, adding a native or hardware provider, exposing a dispatcher facade, or wiring an operator into model and engine code. Do not use it to add a new device runtime (`TurboDevice` value, `DeviceRuntime` subclass, provider directory mapping).

Start from the repository root and read `AGENTS.md`. Inspect the current implementation before editing:

- `src/torchturbo/ops/base.py` for `OpImpl`, `OpImplKind`, `OpSpec`, `OpPolicy`
- `src/torchturbo/ops/registry.py` for `register_op`, global `impl_name` uniqueness, candidate ranking
- `src/torchturbo/ops/dispatcher.py` for provider discovery, policy filtering, caching, explicit facades
- `src/torchturbo/ops/providers/__init__.py` for automatic provider discovery
- the closest existing native provider and its CUDA or NPU counterpart
- the closest existing tests under `tests/ops/`

For the detailed implementation workflow and acceptance checks, read [references/integration-workflow.md](references/integration-workflow.md).

Preserve these invariants:

- Define the semantic contract before writing a provider. Every implementation of an op must share the same call signature and behavior; the dispatcher selects an implementation, it does not reconcile differences.
- Provide a native PyTorch implementation as the correctness baseline and `auto` fallback whenever the op can be expressed in PyTorch.
- Keep optional hardware dependencies out of module scope; import them lazily inside `available()` and `__call__()`.
- Route calls through `ops.<op_name>(...)`, `engine.ops.<op_name>(...)`, or an injected `OpDispatcher`. Do not branch on `is_npu_available()` or `is_cuda_available()`: device availability does not mean the device is the process-selected runtime.
- Preserve strict policy semantics. `native` and `custom` do not cross-fall back, and neither explicit `impl=` nor `OpsConfig.overrides` falls back.
- Treat `impl` as a dispatcher-reserved keyword; never use it as an operator argument name.
- Keep provider filenames and every path component free of a leading `_`, and define each `OpImpl` subclass directly in its provider module, or discovery will skip it.
