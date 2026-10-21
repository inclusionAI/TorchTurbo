# Standalone MegaMoE backend

Install the bundled component once into the training Python environment:

```bash
python -m pip install --no-deps src/torchturbo/optimizations/mega_moe/packages/tt_megamoe-0.1.1+ascend910.cann9.20261021-cp311-cp311-linux_aarch64.whl
```

The wheel requires Linux aarch64, Python 3.11, PyTorch 2.10, torch_npu 2.10.0.post2
and CANN 9.0 on Ascend 910_93. It retains the complete component, including MHC,
its native operators, scheduling/runtime code, licenses and source provenance.
It does not import or require the HyperParallel Python package. The native ABI
vendor names are retained. No expanded component source tree is needed in this
repository or at runtime.

Enable through the Swift configuration:

```yaml
engine_config:
  mega_moe: true
```

The default is false. The Swift launcher locates the installed package and
prepares its required CANN OPP/library search paths before importing Torch, restarting each worker
once. These standard loader paths are internal; no MegaMoE-specific environment
switches are required or read. It neither installs packages automatically nor
creates extra ranks.
Direct Engine API callers must activate the installed package's `set_env.bash`
before Python starts and configure `extra.mega_moe_local_tokens` explicitly.
Swift derives this token bound from packing/max length, batch size and SP.

The main router API remains unchanged. A fixed-shape mask adapter selects expert
IDs and gathers the corresponding final probabilities, including selected zero
weights. Shared experts and auxiliary losses keep their existing implementation.
The backend runs inside `experts(...)`, preserving FSDP hooks, parameter identity,
state_dict names and optimizer ownership. Short inputs use zero-weight padding;
inputs above the fixed aligned token capacity fail explicitly.

Supported experts use BF16, SwiGLU and output routing weights. No clipping,
pre-activation clipping and post-activation clipping remain separate semantics
selected by the model's existing configuration.

## Validation

In the configured Ascend training environment, run from the repository root:

```bash
python -u tests/npu/run_mega_moe_tests.py >test.log 2>&1 &
tail -f test.log
```

The entry point installs the bundled wheel into a temporary isolated directory,
runs the CPU tests, then runs EP2/EP8 precision tests. It checks output, input,
routing-weight and expert-weight gradients with hot routes and checkpointing.
All stdout/stderr, rank metrics, case PASS/FAIL records and the final
`TEST_SUMMARY` are in `test.log`. It creates no per-case result directories.
The exit code is nonzero on failure. At least eight available NPUs are needed.
MegaMoE performance acceptance belongs to the standalone component and is not
part of TorchTurbo's tests. Precision thresholds are unchanged.

Run the independent configuration/hook/semantic checks without multi-rank jobs:

```bash
python -u tests/npu/run_mega_moe_tests.py --unit-only >unit_test.log 2>&1 &
```
