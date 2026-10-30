#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../deps/lf_deps.sh"

rm -rf ~/.cache/ ~/.triton/
