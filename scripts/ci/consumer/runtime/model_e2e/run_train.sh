#!/usr/bin/env bash
# NPU E2E job:
#   environment check      - executor: runtime/env.py probe
#   training orchestration - this script: install -> prepare -> train -> report
#   environment cleanup    - executor: runtime/env.py cleanup
#
# Training stages:
#   1. install    - install torchturbo and model dependencies
#   2. prepare    - rewrite the engine config, then obtain the weights
#   3. train      - launch training under a watchdog
#   4. report     - render facts, job summary, and annotation

set -uo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CI_ROOT="$(cd "$RUNTIME_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$CI_ROOT/../.." && pwd)"
MODEL_HELPER="$RUNTIME_DIR/model_e2e/prepare_model.py"
RESULT_JSON="${RESULT_JSON:-/tmp/ci-result.json}"
GITHUB_OUTPUT="${GITHUB_OUTPUT:-/dev/null}"

: "${MODEL_TYPE:=}"; : "${CASE_ID:=}"; : "${FRAMEWORK:=lf}"; : "${ENV_NAME:=}"; : "${RUNNER_LABEL:=}"
: "${IMAGE:=}"; : "${LAYERS:=0}"; : "${MODEL_HINT:=}"; : "${REPO_IN:=}"
: "${SEQ_LEN:=1024}"; : "${FINETUNING_TYPE:=full}"; : "${DATASET_IN:=gsm8k}"; : "${DATASET:=${DATASET_IN}}"
: "${FETCH_TIMEOUT:=1800}"; : "${CASE_TIMEOUT:=7200}"
: "${INSTALL_SCRIPT:=}"; : "${TRAIN_SCRIPT:=}"; : "${HF_REPO:=}"
: "${INSTALL_ARGS:=}"; : "${TRAIN_ARGS:=}"
: "${TORCH_BEFORE:=/tmp/torch-before.json}"

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf-modules}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

result() { python3 "$CI_ROOT/tools/report.py" "$@"; }

parse_kv() {  # parse_kv <KEY=value lines>
  local key value
  while IFS='=' read -r key value; do
    [ -n "$key" ] && printf -v "$key" '%s' "$value"
  done <<< "$1"
}

# ---------------------------------------------------------------- Stage 1: install
# Install torchturbo and model dependencies, then verify torch was not replaced.
pip_step() {  # pip_step <name> [install script] [install args]
  local name="$1" script="${2:-}" extra="${3:-}" log start rc
  log="/tmp/$(echo "$name" | tr -cd '[:alnum:]' | head -c 20).log"
  start=$(date +%s)
  local -a extra_args=()
  [ -n "$extra" ] && read -r -a extra_args <<< "$extra"
  python -m pip config list 2>/dev/null || true
  env | grep -iE '^(PIP_|.*_PROXY=)' || true
  if [ -n "$script" ]; then
    test -f "$script" || { echo "::error::缺少 $script"; return 1; }
    bash "$script" "${extra_args[@]}" 2>&1 | tee "$log"
  else
    python -m pip install --no-cache-dir --no-build-isolation -e ".[dev]" 2>&1 | tee "$log"
  fi
  rc=${PIPESTATUS[0]}
  grep -E '^(Collecting|Downloading|Using cached|Installing collected|Requirement already satisfied)' \
    "$log" | sed 's/ (from .*//' || true
  python - >/tmp/after.txt 2>&1 <<'PROBE' || true
import importlib
for name in "torch torch_npu numpy triton fla fla_npu transformers llamafactory deepspeed omegaconf pytest".split():
    try:
        print(f"{name}={getattr(importlib.import_module(name), '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=MISSING({type(exc).__name__})")
PROBE
  cat /tmp/after.txt
  if [ "$rc" -eq 0 ]; then
    python3 "$CI_ROOT/consumer/runtime/env.py" check --file "$TORCH_BEFORE" || rc=1
  fi
  result phase --file "$RESULT_JSON" --name "$name" \
    --seconds "$(( $(date +%s) - start ))" --rc "$rc"
  echo "::notice::${name} 耗时 $(( $(date +%s) - start ))s（rc=${rc}）"
  return "$rc"
}

cmd_install() {
  # shellcheck disable=SC2016  # backticks are literal text for the log line
  pip_step 'torchturbo `-e .[dev]`' || return 1
  # Model install scripts must not mutate shared caches. Strip destructive
  # commands here, in place -- the checkout is disposable, and the script's
  # own relative paths stay valid.
  python3 "$RUNTIME_DIR/model_e2e/patch_run_scripts.py" install --src "$INSTALL_SCRIPT" \
    || return 1
  pip_step "模型依赖 \`${INSTALL_SCRIPT}\`" "$INSTALL_SCRIPT" "$INSTALL_ARGS"
}

# ---------------------------------------------------------------- Stage 2: prepare
# Rewrite the declared plan into the engine config, then obtain the weights.
PREP_NPUS_PER_NODE=""
PREP_MODEL=""
PREP_BYTES=0
PREP_SHARDS=0
PREP_HIT=no
PREP_SP=""
PREP_EP=""
PREP_TT_CONFIG=""
PREP_SFS_UUID="?"

cmd_prepare() {
  [ -f "$TRAIN_SCRIPT" ] || { echo "::error::缺少 $TRAIN_SCRIPT"; return 1; }
  local uuid hf_repo="${REPO_IN:-$HF_REPO}"
  uuid=$(findmnt -nT /root/.cache -o SOURCE 2>/dev/null | sed -n '1p' | cut -d. -f1)
  echo "sfs_uuid=${uuid:-?}"
  if [ -z "$MODEL_HINT" ] && { [ -z "$hf_repo" ] || [ "$hf_repo" = "-" ]; }; then
    echo "::error::$MODEL_TYPE 没登记 hf_repo，必须显式传 train_model_path"
    return 1
  fi

  fail_fetch() {  # fail_fetch <reason> <seconds>
    echo "::error::$1"
    echo "skip_reason=$1" >> "$GITHUB_OUTPUT"
    result set --file "$RESULT_JSON" skip_reason="$1"
    result phase --file "$RESULT_JSON" --name '模型获取' --seconds "$2" --rc 1
    return 1
  }

  local MODEL="" BYTES=0 SHARDS=0 HIT=no CARDS="" SP="" EP="" TT_CONFIG=""
  local configured prepared start trc
  # The Plan supplies cards/sp/ep. Real runner labels carry a location suffix
  # whose last segment parses as a rank count, so inferring from labels is wrong.
  case "$PLAN_CARDS" in
    ''|-|*[!0-9]*)
      echo "::error::Plan 里 ${CASE_ID} 没有 cards/sp/ep，无法确定卡数与并行策略"
      return 1 ;;
  esac
  if ! configured=$(python3 "$RUNTIME_DIR/model_e2e/patch_run_scripts.py" config \
      --src "$TRAIN_SCRIPT" --cards "$PLAN_CARDS" --layers "${LAYERS:-0}" \
      --sp "$PLAN_SP" --ep "$PLAN_EP"); then
    echo "::error::引擎配置改写失败（cards=$PLAN_CARDS sp=$PLAN_SP ep=$PLAN_EP）"
    return 1
  fi
  parse_kv "$configured"
  NPUS_PER_NODE=$CARDS
  echo "cards=${NPUS_PER_NODE} (Plan)"
  local -a args=(fetch --repo "$hf_repo" --layers "${LAYERS:-0}")
  [ -n "$MODEL_HINT" ] && args+=(--hint "$MODEL_HINT")
  start=$(date +%s)
  prepared=$(timeout "$FETCH_TIMEOUT" python3 "$MODEL_HELPER" "${args[@]}")
  trc=$?
  if [ "$trc" -eq 124 ]; then
    fail_fetch "model-fetch-timeout(${FETCH_TIMEOUT}s)" "$(( $(date +%s) - start ))"
    return $?
  fi
  if [ "$trc" -ne 0 ]; then
    echo "::error::模型或配置预处理失败（rc=${trc}）"
    return 1
  fi
  parse_kv "$prepared"
  if [ -z "$MODEL" ] || [ -z "$SP" ] || [ -z "$EP" ] || [ -z "$TT_CONFIG" ]; then
    echo "::error::模型预处理输出不完整"
    return 1
  fi
  echo "model: $MODEL bytes=$BYTES shards=$SHARDS hit=$HIT"
  PREP_NPUS_PER_NODE=$NPUS_PER_NODE
  PREP_MODEL=$MODEL
  PREP_BYTES=$BYTES
  PREP_SHARDS=$SHARDS
  PREP_HIT=$HIT
  PREP_SP=$SP
  PREP_EP=$EP
  PREP_TT_CONFIG=$TT_CONFIG
  PREP_SFS_UUID=${uuid:-?}
}

# ---------------------------------------------------------------- Stage 3: train
# Launch the training script and supervise its process group and output log.
SUP_TRAIN_PID=""
SUP_TAIL_PID=""

supervisor_stop() {
  if [ -n "$SUP_TAIL_PID" ]; then
    kill "$SUP_TAIL_PID" 2>/dev/null || true
    SUP_TAIL_PID=""
  fi
  if [ -n "$SUP_TRAIN_PID" ] && kill -0 "$SUP_TRAIN_PID" 2>/dev/null; then
    echo "[supervisor] tearing down process group $SUP_TRAIN_PID" >&2
    kill -TERM -"$SUP_TRAIN_PID" 2>/dev/null || kill -TERM "$SUP_TRAIN_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$SUP_TRAIN_PID" 2>/dev/null && [ "$waited" -lt 10 ]; do
      sleep 1
      waited=$((waited + 1))
    done
    kill -KILL -"$SUP_TRAIN_PID" 2>/dev/null || kill -KILL "$SUP_TRAIN_PID" 2>/dev/null || true
  fi
  SUP_TRAIN_PID=""
}

supervisor_cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  supervisor_stop
  return "$rc"
}

run_supervised() {  # run_supervised <launch script> <log> [launch args]
  local script="$1" log="$2" extra="${3:-}" start stuck=0 busy=0 child rc elapsed last_loss stalled=no watchdog=no
  local -a extra_args=()
  [ -n "$extra" ] && read -r -a extra_args <<< "$extra"
  : > "$log"
  SUP_TRAIN_PID=""
  SUP_TAIL_PID=""
  trap supervisor_cleanup EXIT
  trap 'supervisor_cleanup; exit 130' INT
  trap 'supervisor_cleanup; exit 143' TERM
  if command -v setsid >/dev/null 2>&1 && command -v pgrep >/dev/null 2>&1; then
    setsid bash "$script" "${extra_args[@]}" >"$log" 2>&1 &
    watchdog=yes
  else
    bash "$script" "${extra_args[@]}" >"$log" 2>&1 &
  fi
  SUP_TRAIN_PID=$!
  tail -f "$log" &
  SUP_TAIL_PID=$!
  start=$(date +%s)
  while kill -0 "$SUP_TRAIN_PID" 2>/dev/null; do
    if [ "$watchdog" = yes ]; then
      busy=0
      for child in $(pgrep -P "$SUP_TRAIN_PID" 2>/dev/null); do
        [ "$(cat /proc/"$child"/comm 2>/dev/null)" = tee ] || busy=$((busy + 1))
      done
      if [ "$busy" -eq 0 ]; then stuck=$((stuck + 1)); else stuck=0; fi
      [ "$stuck" -ge 3 ] && break
    fi
    sleep 5
  done
  if [ "$watchdog" = yes ] && kill -0 "$SUP_TRAIN_PID" 2>/dev/null; then
    stalled=yes
    rc=1
  else
    wait "$SUP_TRAIN_PID"
    rc=$?
  fi
  elapsed=$(( $(date +%s) - start ))
  last_loss=$(grep -oE "'loss': '?[0-9.]+" "$log" 2>/dev/null | tail -1 | tr -d "'" || true)
  supervisor_stop
  trap - EXIT INT TERM
  TRAIN_ELAPSED=$elapsed TRAIN_WATCHDOG=$watchdog TRAIN_STALLED=$stalled TRAIN_LAST_LOSS=$last_loss
  echo "[supervisor] rc=${rc} elapsed=${elapsed}s watchdog=${watchdog} stalled=${stalled}" >&2
  return "$rc"
}

prefer_cached_dataset() {
  local root cache hub key hit
  root="$REPO_ROOT"
  cache="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"
  hub=$(python3 - "$root/dataset/dataset_info.json" "$DATASET" <<'PY'
import json
import sys

try:
    entry = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], {})
except (OSError, ValueError):
    entry = {}
print(entry.get("hf_hub_url", ""))
PY
)
  [ -n "$hub" ] || return 0
  key=${hub//\//___}
  hit=$(find "$cache/$key" -mindepth 3 -maxdepth 3 -type d -print -quit 2>/dev/null)
  [ -n "$hit" ] || return 0
  export HF_DATASETS_OFFLINE=1
  echo "dataset cache: $hit (offline)"
}

cmd_train() {
  [ -n "$PREP_MODEL" ] || { echo "::error::缺少预处理结果，无法训练"; return 1; }
  prefer_cached_dataset
  export MODEL="$PREP_MODEL" NPUS_PER_NODE="$PREP_NPUS_PER_NODE" DATASET="$DATASET_IN" \
    SEQ_LEN FINETUNING_TYPE TT_CONFIG="$PREP_TT_CONFIG"
  # A smoke run must not fill the pod's overlay with checkpoints or tb events.
  # Runs after preprocess, which reads the unpatched launcher for TT_CONFIG.
  python3 "$RUNTIME_DIR/model_e2e/patch_run_scripts.py" train --src "$TRAIN_SCRIPT" \
    || return 1
  run_supervised "$TRAIN_SCRIPT" /tmp/train.log "$TRAIN_ARGS"
  local rc=$? elapsed="$TRAIN_ELAPSED" last_loss="$TRAIN_LAST_LOSS"
  echo "::notice::训练 rc=${rc} 用时 ${elapsed}s；hit=${PREP_HIT}；${last_loss:-无 loss 记录}"
  local output_dir="${OUTPUT_DIR:-/tmp/tblf-sft/$(basename "$PREP_MODEL")}"
  df -h "$output_dir" 2>/dev/null | sed 's/^/  df: /'
  du -sh "$output_dir" 2>/dev/null | sed 's/^/  used: /'
  du -sh "$output_dir"/* 2>/dev/null | sed 's/^/    /'
  result set --file "$RESULT_JSON" \
    cache_hit="$PREP_HIT" sfs_uuid="$PREP_SFS_UUID" shards="$PREP_SHARDS" \
    model_size="$(( PREP_BYTES / 1048576 )) MiB" sp="$PREP_SP" ep="$PREP_EP" \
    output_dir="$output_dir" model_path="$PREP_MODEL" last_loss="$last_loss" \
    watchdog="$TRAIN_WATCHDOG" stalled="$TRAIN_STALLED" skip_reason=""
  result phase --file "$RESULT_JSON" --name '训练' --seconds "$elapsed" --rc "$rc"
  return "$rc"
}

# ---------------------------------------------------------------- Stage 4: report
# Render the Job Summary and emit the final annotation from recorded facts.
cmd_report() {
  result report --file "$RESULT_JSON" --train-log /tmp/train.log \
    --versions /tmp/after.txt >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
  result notice --file "$RESULT_JSON" --model-type "${MODEL_TYPE:-}" || true
}

# Run stages 3-4 for one E2E case; orchestration lives outside the stage bodies.
cmd_run() {
  cmd_prepare || return $?
  cmd_train
}

# Run one fully resolved ModelE2ESpec supplied by executor.py.
cmd_case() {
  local rc=0
  : "${MODEL_TYPE:?MODEL_TYPE is required}"
  : "${CASE_ID:?CASE_ID is required}"
  : "${FRAMEWORK:?FRAMEWORK is required}"
  : "${ENV_NAME:?ENV_NAME is required}"
  : "${INSTALL_SCRIPT:?INSTALL_SCRIPT is required}"
  : "${TRAIN_SCRIPT:?TRAIN_SCRIPT is required}"
  : "${RUN_DIR:?RUN_DIR is required}"
  : "${PLAN_CARDS:?PLAN_CARDS is required}"
  result init --file "$RESULT_JSON" \
    kind=train model_type="$MODEL_TYPE" case_id="$CASE_ID" framework="$FRAMEWORK" env="$ENV_NAME" \
    runner_label="$RUNNER_LABEL" image="$IMAGE" layers="$LAYERS" dataset="$DATASET_IN"
  if ! cmd_install; then
    rc=1
  else
    ( GITHUB_OUTPUT=/dev/null timeout --foreground --kill-after=30s \
        "$CASE_TIMEOUT" bash "$0" run )
    rc=$?
  fi
  cmd_report || true
  return "$rc"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    run) cmd_run ;;
    case) cmd_case ;;
    *) echo "用法：$0 {run|case}" >&2; exit 2 ;;
  esac
fi
