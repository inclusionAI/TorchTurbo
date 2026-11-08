#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reuse the installed HMC runtime; never build or initialize a second SHMEM runtime.
set -euo pipefail
if (( $# > 1 )); then
    echo "Usage: MOONEP_SOURCE=/path/to/MoonEP bash moonep.sh [source-directory]" >&2
    exit 2
fi
source_dir=${1:-${MOONEP_SOURCE:?Set MOONEP_SOURCE to the matching official Ascend/MoonEP source checkout}}
source_dir=$(cd "$source_dir" && pwd)
if [[ ! -f "$source_dir/npu_patch/scripts/build_base.sh" ]]; then
    echo "No npu_patch/scripts/build_base.sh found in MOONEP_SOURCE=$source_dir" >&2
    exit 1
fi
if [[ -z "${SHMEM_HOME_PATH:-}" ]]; then
    : "${HYPER_MULTICORE_SOURCE:?Set SHMEM_HOME_PATH or the HYPER_MULTICORE_SOURCE used for the installed wheel}"
    SHMEM_HOME_PATH=$(python - "$HYPER_MULTICORE_SOURCE" <<'PY'
import json,sys
from pathlib import Path
paths=list((Path(sys.argv[1])/'build/native/common').glob('*/sdk.json'))
if len(paths)!=1:
    raise SystemExit('Multiple or missing HMC SDK handoffs; set SHMEM_HOME_PATH to the matching SDK explicitly')
print(json.loads(paths[0].read_text())['install_root'])
PY
)
fi
PRIVATE_SHMEM_LIBRARY=$(python - <<'PY'
import importlib.util
from pathlib import Path
spec=importlib.util.find_spec('hyper_multicore')
if spec is None or spec.origin is None:
    raise SystemExit('Install hyper-multicore first with mega_moe.sh')
path=Path(spec.origin).parent/'shmem/lib/shmem/libhyper_multicore_shmem.so'
if not path.is_file():
    raise SystemExit(f'Missing installed HMC runtime: {path}')
print(path)
PY
)
export SHMEM_HOME_PATH PRIVATE_SHMEM_LIBRARY
export MOONEP_BUILD_ONLY=1
BUILD_DIR=${BUILD_DIR:-"$source_dir/build/torchturbo-moon"}
export BUILD_DIR
marker=$(mktemp)
trap 'rm -f "$marker"' EXIT
bash "$source_dir/npu_patch/scripts/build_base.sh" -soc_type "${MOONEP_SOC_TYPE:-Ascend910_93}"
shopt -s nullglob
wheels=()
for wheel in "$BUILD_DIR"/wheels/moonep-*.whl; do
    if [[ "$wheel" -nt "$marker" ]]; then
        wheels+=("$wheel")
    fi
done
if (( ${#wheels[@]} != 1 )); then
    echo "Expected exactly one newly built MoonEP wheel in $BUILD_DIR/wheels" >&2
    exit 1
fi
python -m pip install --no-deps --force-reinstall "${wheels[0]}"
