# Ahead-of-time gated activation

`ascendc_gated_activation` implements the existing `gated_activation` contract
on Ascend A2/A3. The source is adapted from Huawei ops-transformer's
`mc2/mega_moe/op_kernel/arch22/utils/gated_activation.hpp`, revision
`a739e89cc1ba834107d072b8c20ccc8358887c19`. The upstream source uses the CANN
Open Software License 2.0. The original repository is not modified.

The standalone kernel removes the CATLASS epilogue dependency and adds
forward/backward launches for FP32, FP16 and BF16. Input is split into
contiguous gate/up halves along the last dimension. SiLU and NPU's tanh GELU
are supported. A positive limit clamps up symmetrically and upper-bounds
gate before or after activation. Gradients pass at clamp endpoints.
Without a limit, activation and activation gradients retain the intermediate
dtype rounding of the Torch golden. With a limit, intermediates are FP32.
The operation supports first-order autograd; double backward is not supported.

Empty tensors, odd widths and other dtypes use the existing Torch expression
inside the provider. Other noncontiguous inputs are made contiguous. Token
counts and widths are runtime arguments and never trigger compilation.

The kernel packs aligned rows into 4096-element vector tiles using strided
global-memory copies. Unaligned widths use padded single-row tiles; copies
still transfer only valid elements. Forward skips derivative computation.
Launch tiling matches the device kernel and uses the vector core count
reported by CANN (`aclrtGetDeviceInfo`). FP32 scratch and explicit dtype
rounding preserve the existing numerical contract.

Independent input/output queues synchronize only the required MTE2, vector,
and MTE3 dependencies. They replace full-pipeline barriers and allow outgoing
copies to overlap subsequent computation. The activation and SiLU derivative
share one exponential denominator. Result storage reuses activation scratch
after its last read, leaving room for separate IO buffers in unified memory.

## Build before testing or training

Use the same Python, torch_npu and CANN environment as training. From the
TorchTurbo repository root, after sourcing CANN's `set_env.sh`:

```bash
export PYTHON=/path/to/project/.venv/bin/python
bash src/torchturbo/ops/providers/npu/ascendc/build.sh \
    /path/to/workspace/ascendc/lib Ascend910_9382
export TORCHTURBO_ASCENDC_LIB_DIR=/path/to/workspace/ascendc/lib
PYTHONPATH=src "$PYTHON" -m pytest -q tests/ops/npu/test_ascendc_gated_activation.py
```

Pass the target SoC version as the second argument, or omit it to detect NPU 0.
The output directory contains the build tree, `libtorchturbo_gated_activation.so`
and `manifest.json` with source/library hashes and dependency versions.
Rebuild after changing source, SoC, CANN, PyTorch or torch_npu. Each process
loads one library version; restart workers after changing the library.

At runtime, `torch.ops.load_library` loads the prebuilt shared object once.
There is no JIT compilation or compiler subprocess. Without the environment
variable, the loader looks in `/opt/torchturbo/ascendc/lib`, populated during
the NPU Docker image build. Development builds in other locations must set
`TORCHTURBO_ASCENDC_LIB_DIR` explicitly.
A missing library makes the provider unavailable; an explicit selection
fails. A present but incompatible library raises its loading error.

Select the implementation with the normal engine configuration:

```yaml
engine_config:
  ops:
    overrides:
      gated_activation: ascendc_gated_activation
```

The provider priority is 90, below Triton's 100, so installing the library
does not replace Triton in an otherwise unchanged AUTO configuration.
Host launches go through torch_npu's task queue on the current NPU stream.
No device-wide synchronization is added to forward/backward.
