#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../deps/swift_deps.sh"

rm -rf ~/.cache/huggingface/modules/transformers_modules/
