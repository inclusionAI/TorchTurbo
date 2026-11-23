#!/usr/bin/env bash
# Build offline and install only runtime artifacts, without querying an NPU.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
soc_version=${1:?Usage: bash install.sh SOC_VERSION (for example Ascend910_9382)}
python_bin=${PYTHON:-python3}
install_dir=$("$python_bin" "$source_dir/_config.py")
build_output=$(mktemp -d)
trap 'rm -rf -- "$build_output"' EXIT
bash "$source_dir/build.sh" "$build_output" "$soc_version"
install -d "$install_dir"
install -m 0644 "$build_output"/libtorchturbo_*.so "$build_output/manifest.json" "$install_dir/"
