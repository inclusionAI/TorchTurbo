#!/bin/bash
# Qwen3.8-Flash-Next text SFT with TorchTurbo via ms-swift.
# The checkpoint uses the qwen4_exp architecture and the qwen3_5 template.
#
# Usage:
#   bash scripts/runs/qwen4_exp/swift_text_sft.sh
#   MAX_STEPS=5 bash scripts/runs/qwen4_exp/swift_text_sft.sh
#   bash scripts/runs/qwen4_exp/swift_text_sft.sh --tt_sp_size 16

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"

: "${MODEL:?Set MODEL to your Qwen3.8-Flash-Next checkpoint}"
: "${DATASET:?Set DATASET to your Swift dataset or data file}"
TT_CONFIG=${TT_CONFIG:-"${SCRIPT_DIR}/swift_text_sft.yaml"}
PYTHON=${PYTHON:-python3}
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
SEQ_LEN=${SEQ_LEN:-4096}
MAX_STEPS=${MAX_STEPS:-100}
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23576}

OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/qwen38-flash-next/swift_text_sft_node${NNODES}_seq${SEQ_LEN}"}
LOG_FILE="${OUTPUT_DIR}/log-${NODE_RANK}.txt"

source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE=AI_CPU
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-29972}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export MULTI_STREAM_MEMORY_REUSE=1
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
export ASCEND_PROCESS_LOG_PATH="${OUTPUT_DIR}/ascend_log"
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"
if [ "${NODE_RANK}" -eq 0 ]; then
    cp -f "$0" "${OUTPUT_DIR}/"
    cp -f "${TT_CONFIG}" "${OUTPUT_DIR}/"
fi
{ date "+%Y%m%d-%H%M%S"; echo "${OUTPUT_DIR}"; } >> "${LOG_FILE}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

SWIFT_ARGS=(
    sft
    --model_type turbo_qwen4_exp
    --model "${MODEL}"
    --template qwen3_5
    --dataset "${DATASET}"
    --load_from_cache_file true
    --max_length "${SEQ_LEN}"
    --packing true
    --packing_length "${SEQ_LEN}"
    --attn_impl flash_attn
    --dataset_num_proc "${DATASET_NUM_PROC:-16}"
    --tuner_type "${FINETUNING_TYPE}"
    --num_train_epochs 1
    --max_steps "${MAX_STEPS}"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
    --eval_strategy no
    --weight_decay 0.1
    --adam_beta2 0.95
    --learning_rate "${LEARNING_RATE:-1e-5}"
    --lr_scheduler_type cosine
    --warmup_ratio 0.05
    --logging_steps 1
    --save_strategy steps
    --save_steps "${SAVE_STEPS:-100}"
    --save_total_limit 2
    --bf16 true
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
    --report_to tensorboard
    --output_dir "${OUTPUT_DIR}"
    --ddp_timeout 1800000000
    --tt_config "${TT_CONFIG}"
    --tt_freeze_vision_tower true
    --tt_freeze_multi_modal_projector true
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    SWIFT_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${PYTHON}" -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" \
    "${RUN_SWIFT}" "${SWIFT_ARGS[@]}" "$@" \
    2>&1 | tee "${LOG_FILE}"
