# Ahead-of-time AscendC operators

AscendC operators are compiled during NPU Docker image construction and loaded
from prebuilt shared libraries at runtime. TorchTurbo never compiles them during
training or on the first operator call.

## Installation directory

The default installation contains:

```text
/opt/torchturbo/ascendc/lib/
  libtorchturbo_gated_activation.so
  manifest.json
```

`_config.py` is the single source of truth for the default directory. It uses
only the Python standard library. Both `install.sh` and `_library.py` use its
`library_directory()` function:

| Setting | Installation and runtime lookup |
|---|---|
| `TORCHTURBO_ASCENDC_LIB_DIR` unset or empty | `/opt/torchturbo/ascendc/lib` |
| `TORCHTURBO_ASCENDC_LIB_DIR` set | The specified directory |

Docker does not hard-code the directory or set this environment variable.
Use the override only for custom deployments or development libraries, and
keep it consistent between installation and runtime. The loader does not
search additional directories if the selected library is missing.

From the repository root, inspect the effective directory without importing
PyTorch or initializing an NPU:

```bash
python3 src/torchturbo/ops/providers/npu/ascendc/_config.py
```

## Source layout

Shared infrastructure lives in this directory. Each operator owns a subdirectory:

```text
ascendc/
  CMakeLists.txt          # Toolchain, shared linking rules, operator subdirectories
  build.sh               # Offline build and source/library manifest
  install.sh             # Explicit-SoC build and runtime-only installation
  _config.py             # Shared default directory and environment override
  _library.py            # Library location and lazy loading, keyed by operator name
  __init__.py
  gated_activation/
    __init__.py          # Package export: FusedGatedActivation
    CMakeLists.txt        # Operator build targets
    gated_activation.py  # Fused activation autograd interface
    binding.cpp          # Torch registration and NPU queue launch
    gated_activation.cpp # Kernel entry points
    gated_activation.hpp # Device computation
    README.md
```

Provider registration lives in the corresponding `npu/<operator>.py` module;
gated activation's Triton and AscendC providers share `npu/gated_activation.py`.
The provider imports the autograd interface through the package:

```python
from .ascendc.gated_activation import FusedGatedActivation
```

The package exports the class implemented in `gated_activation.py`. Importing
it does not load the shared library; the first forward call loads it through
`_library.py`. This class is a `torch.autograd.Function`, not an `OpImpl`, so
provider discovery does not register it as another implementation.

## Docker image build

The NPU Dockerfile builds these libraries after the final PyTorch installation
and installs them into `/opt/torchturbo/ascendc/lib`. Build from the repository
root, setting `--build-arg TORCHTURBO_ASCENDC_SOC=Ascend910_9382` (or the deployment
SoC). Image builds pass an explicit target and do not query an NPU.

Alternatively, starting from the repository root:

```bash
cd docker/npu/

docker build --platform linux/arm64 \
    -f Dockerfile_aarch64 \
    --build-arg TORCHTURBO_ASCENDC_SOC=Ascend910_9382 \
    -t torchturbo:aarch64 ../..
```

The build context must be the repository root so Docker can copy the operator
sources. Change the SoC argument to match the deployment device. Rebuild when
the kernel sources, target SoC, CANN, PyTorch or torch_npu versions change.

## Manual installation

To perform the same runtime-only installation outside Docker:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHON=python3 bash src/torchturbo/ops/providers/npu/ascendc/install.sh \
    Ascend910_9382
```

Run these commands from the repository root with write access to the selected
installation directory. `install.sh` requires the target SoC and preserves only shared libraries and
the manifest in the destination. It removes its temporary build output on exit.
The destination comes from `_config.py`, which uses only the Python standard
library and is also used by the runtime loader. Override
`TORCHTURBO_ASCENDC_LIB_DIR` for a custom installation or development build;
use the same override at runtime. For example:

```bash
export TORCHTURBO_ASCENDC_LIB_DIR=/path/to/workspace/ascendc/lib
PYTHON=python3 bash src/torchturbo/ops/providers/npu/ascendc/install.sh Ascend910_9382
```

## Development builds

To retain intermediate build output for debugging, use `build.sh` with an
explicit output directory instead of `install.sh`:

```bash
export PYTHON=/path/to/project/.venv/bin/python
bash src/torchturbo/ops/providers/npu/ascendc/build.sh \
    /path/to/workspace/ascendc/lib Ascend910_9382
export TORCHTURBO_ASCENDC_LIB_DIR=/path/to/workspace/ascendc/lib
```

Source CANN's `set_env.sh` first. An optional second argument selects the SoC;
otherwise the build detects NPU 0. Use the same environment as training.
The manifest records the target SoC, torch/torch_npu versions, source hashes
and each `libtorchturbo_<operator>.so` hash. `build.sh` retains its build tree,
generated headers and static libraries; `install.sh` installs only runtime artifacts.
Runtime only loads existing libraries; it never compiles.

`_library.py` resolves libraries from `TORCHTURBO_ASCENDC_LIB_DIR`, or from
`/opt/torchturbo/ascendc/lib` when unset. Each process loads an operator's library once.
Existing development builds in `ascendc/lib/` require an explicit environment
variable override; the loader does not search multiple directories.
Restart workers after changing the library or its directory.

The installed library does not change provider priority. Triton remains the
higher-priority AUTO implementation; select AscendC explicitly when needed:

```yaml
engine_config:
  ops:
    overrides:
      gated_activation: ascendc_gated_activation
```

## Adding an operator

For a new operator, add its directory with a CMake target and call
`torchturbo_ascendc_library(target, kernel_target)`, then add its subdirectory
to the root CMake file. Keep provider registration in `npu/<operator>.py`;
private Python helpers start with `_` to avoid provider discovery.
Libraries sharing the `torchturbo_ascendc` Torch namespace use
`TORCH_LIBRARY_FRAGMENT` for their schemas.

See [gated activation](gated_activation/README.md) for its contract and tests.
