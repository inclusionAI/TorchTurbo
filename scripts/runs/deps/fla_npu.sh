#!/bin/bash
set -e

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${deps_dir}/../../.." && pwd)"
wheel_file="docker/npu/pkg/flash_linear_attention_npu_a3-26.9.1+dev36e67d2-py3-none-manylinux_2_34_aarch64.whl"
wheel_path="${repo_root}/${wheel_file}"

if head -n 1 "${wheel_path}" | grep -Fxq 'version https://git-lfs.github.com/spec/v1'; then
    git -C "${repo_root}" lfs pull --include="${wheel_file}" --exclude=""
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install --no-cache-dir --no-deps "${wheel_path}"

python - <<'PY'
from importlib.metadata import version

import torch_npu

from fla_npu.ops.ascendc import chunk_kda_bwd, chunk_kda_fwd
from fla_npu.ops.triton import l2norm

print(f"flash-linear-attention-npu-a3 {version('flash-linear-attention-npu-a3')}")
print("KDA forward, backward and L2Norm imports succeeded.")
PY
