#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:?Usage: bash build.sh OUTPUT_DIR [SOC_VERSION]}
python_bin=${PYTHON:-python3}
soc_version=${2:-$("$python_bin" -c 'import torch, torch_npu; print(torch.npu.get_device_name(0))')}
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
# CANN's template preprocessor rewrites object files in place. A fresh tree
# avoids reusing those transformed objects on subsequent offline builds.
build_dir=$(mktemp -d "$output_dir/build.XXXXXX")
torch_prefix=$("$python_bin" -c 'import torch; print(torch.utils.cmake_prefix_path)')
npu_path=$("$python_bin" -c 'import pathlib,torch_npu; print(pathlib.Path(torch_npu.__file__).parent)')
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$torch_prefix" \
    -DTORCH_NPU_PATH="$npu_path" -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    -DSOC_VERSION="$soc_version" -DASCEND_PYTHON_EXECUTABLE="$python_bin" \
    -DCMAKE_INSTALL_PREFIX="$output_dir"
cmake --build "$build_dir" -j "${BUILD_JOBS:-4}"
cmake --install "$build_dir"
"$python_bin" - "$source_dir" "$output_dir" "$soc_version" <<'PY'
import hashlib, json, pathlib, sys
import torch, torch_npu
src, dst, soc = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
files = sorted(p for p in src.rglob('*') if p.is_file() and
               p.suffix in {'.cpp', '.hpp', '.py', '.sh', '.txt'} and
               not {'lib', '__pycache__'}.intersection(p.relative_to(src).parts))
manifest = dict(soc=soc, torch=torch.__version__, torch_npu=torch_npu.__version__,
                sources={str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                libraries={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(dst.glob('libtorchturbo_*.so'))})
(dst/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
