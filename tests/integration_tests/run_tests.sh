#!/usr/bin/env bash
# Run the cases of a suite: prepare assets, train, then compare against golden.
#bash /Users/yxjin/Code/torchturbo_func_test/tests/integration_tests/run_tests.sh --suite base_suite --platform npu --ngpu 2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/scripts/run_tests.py" "$@"
