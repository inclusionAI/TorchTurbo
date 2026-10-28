#!/bin/bash
# Ling-3.1-flash SFT with TorchTurbo via ms-swift.
#
# Tiny smoke run:
#   MAX_STEPS=1 PACKING=false bash scripts/runs/bailing_v3/ling31_swift_sft.sh
# Full run:
#   DATASET=/storage/lf02381138/sampled_sft_data_count_151372_0630.jsonl \
#     bash scripts/runs/bailing_v3/ling31_swift_sft.sh

set -o pipefail
set -eux

ulimit -l unlimited

MODEL=${MODEL:-/storage/lf02381138/ling31/Ling-3.1-flash-bf16}
DATASET=${DATASET:-/storage/lf02381138/sampled_sft_data_count_151372_0630.tiny.jsonl}
DATASET_NAME=$(basename "${DATASET%.jsonl}")
OUTPUT_DIR=${OUTPUT_DIR:-/storage/lf02381138/ling31_work/outputs/ling31_sft/${DATASET_NAME}}
SEQ_LEN=${SEQ_LEN:-32768}
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-2000}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-96}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/ling31_sft_bf16.yaml"}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=1

export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
export NNODES=${NNODES:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/ascend_log

CUR_TIME=$(date "+%Y%m%d-%H%M%S")
LOG_FILE=${LOG_FILE_PREFIX:-$OUTPUT_DIR/rank$NODE_RANK}-$CUR_TIME.log
PYTHON=${PYTHON:-python3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
export PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH

PROFILE_ARGS=()
if [[ "${PROFILING:-false}" == "true" ]]; then
    PROFILE_ARGS+=(--tt_profiling)
fi

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}
"

mkdir -p "$OUTPUT_DIR"
echo "$CUR_TIME" >> "$LOG_FILE"
echo "$OUTPUT_DIR" >> "$LOG_FILE"

"${PYTHON}" -m torch.distributed.run $DISTRIBUTED_ARGS "${RUN_SWIFT}" \
    --tt_config "${TT_CONFIG}" \
    "${PROFILE_ARGS[@]}" \
    --model_type turbo_bailing_hybrid \
    --model "$MODEL" \
    --check_model false \
    --template ling_v3 \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --max_length "${SEQ_LEN}" \
    --packing "${PACKING:-true}" \
    --packing_length "${SEQ_LEN}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --tuner_type "${FINETUNING_TYPE}" \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-7}' \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --bf16 true \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --train_dataloader_shuffle "${SHUFFLE:-true}" \
    --report_to tensorboard \
    --output_dir "$OUTPUT_DIR" \
    --ddp_timeout 1800000000 \
    2>&1 | tee -a "$LOG_FILE"
