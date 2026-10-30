#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
python -m pip install --no-deps --index-url https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/ 'cann-shmem==1.6.0'
python "$repo_root/src/torchturbo/optimizations/mega_moe/build.py" "$@"
