#!/bin/bash
# Qwen3.5 dense text SFT training with torchturbo
#
# Usage:
#   bash scripts/runs/qwen3_5_dense/install_lf_deps.sh
#   bash scripts/runs/qwen3_5_dense/lf_text_sft.sh

set -o pipefail
set -eux

# model name or path
MODEL=${MODEL:-/models/Qwen3.5-4B}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/tblf-sft/$(basename "$MODEL")}
# sequence length
SEQ_LEN=${SEQ_LEN:-4096}
# finetuning type
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (must be registered in ${DATASET_DIR}/dataset_info.json)
DATASET=${DATASET:-gsm8k}
# turbo engine flags (sp/ep, freeze, cpu offload) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/lf_text_sft.yaml"}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
# AI_CPU is default for A3; AIV does not support multi-comm-domain parallelism
export HCCL_OP_EXPANSION_MODE="AI_CPU"
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2

export VIDEO_PLACEHOLDER="<qwen3_5_video_placeholder>"
export AUDIO_PLACEHOLDER="<qwen3_5_audio_placeholder>"
export IMAGE_PLACEHOLDER="<qwen3_5_image_placeholder>"

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error
export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log


CUR_TIME=$(date "+%Y%m%d-%H%M%S")
LOG_FILE=${LOG_FILE_PREFIX:-$OUTPUT_DIR/rank$NODE_RANK}-$CUR_TIME.log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_LF="${REPO_ROOT}/scripts/runs/run_lf.py"
DATASET_DIR="${REPO_ROOT}/dataset"
export PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank $NODE_RANK
"

mkdir -p $OUTPUT_DIR
echo $CUR_TIME >> $LOG_FILE
echo $OUTPUT_DIR >> $LOG_FILE

# Optional engine instrumentation (runtime-only flags, not settable via yaml).
#   TT_PROFILING=1            -> NPU trace for one step after a 10-step warmup
#   MEM_SNAPSHOT_STEPS=<n>    -> torch.npu memory snapshot at step n
TT_EXTRA_ARGS=""
if [ "${TT_PROFILING:-0}" = "1" ]; then
    TT_EXTRA_ARGS="$TT_EXTRA_ARGS --tt_profiling"
fi
if [ -n "${MEM_SNAPSHOT_STEPS:-}" ]; then
    TT_EXTRA_ARGS="$TT_EXTRA_ARGS --tt_memory_snapshot_steps ${MEM_SNAPSHOT_STEPS}"
fi

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    $TT_EXTRA_ARGS \
    --tt_config ${TT_CONFIG} \
    --report_to tensorboard \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $MODEL \
    --do_train \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --dataset $DATASET \
    --preprocessing_num_workers 16 \
    --cutoff_len ${SEQ_LEN} \
    --template qwen3_5_nothink \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1.25e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --ddp_timeout 1800000000 \
    --bf16 \
    --neat_packing \
    --trust_remote_code \
    2>&1 | tee $LOG_FILE
