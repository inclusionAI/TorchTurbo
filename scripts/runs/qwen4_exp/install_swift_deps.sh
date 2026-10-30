#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../deps/swift_deps.sh"

pip install qwen-vl-utils==0.0.14
