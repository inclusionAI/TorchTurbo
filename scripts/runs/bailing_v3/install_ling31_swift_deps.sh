#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRITON_ASCEND_WHEEL=${TRITON_ASCEND_WHEEL:-https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl}

source "${REPO_ROOT}/scripts/runs/deps/swift_deps.sh"
python3 -m pip install --no-deps triton==3.5.0
python3 -m pip install --force-reinstall --no-deps "${TRITON_ASCEND_WHEEL}"
python3 -m pip install --no-deps -e "${REPO_ROOT}"
python3 -c "import torch, torch_npu, transformers, swift, torchturbo; from fla.modules.fused_norm_gate import rms_norm_gated; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('transformers:', transformers.__version__); print('ms-swift:', swift.__version__); print('torchturbo:', torchturbo.__file__); print('FLA NPU import: OK')"
