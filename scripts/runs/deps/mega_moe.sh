#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build and install once in the intended Torch/PTA environment.
set -euo pipefail
source_dir=${HYPER_MULTICORE_SOURCE:?Set HYPER_MULTICORE_SOURCE to the matching hyper-multicore source checkout}
source_dir=$(cd "$source_dir" && pwd)
if [[ ! -f "$source_dir/build.sh" ]]; then
    echo "No build.sh found in HYPER_MULTICORE_SOURCE=$source_dir" >&2
    exit 1
fi
python -m pip install -r "$source_dir/requirements.txt" setuptools wheel 'packaging>=22' decorator attrs psutil
cd "$source_dir"
marker=$(mktemp)
trap 'rm -f "$marker"' EXIT
bash build.sh --soc-list "${SOC_LIST:-ascend910_93}" --jobs "${MAX_JOBS:-8}" "$@"
shopt -s nullglob
wheels=()
for wheel in dist/hyper_multicore-*.whl; do
    if [[ "$wheel" -nt "$marker" ]]; then
        wheels+=("$wheel")
    fi
done
if (( ${#wheels[@]} != 1 )); then
    echo "Expected exactly one newly built hyper-multicore wheel in $source_dir/dist" >&2
    exit 1
fi
python -m pip install --no-deps --force-reinstall "${wheels[0]}"
