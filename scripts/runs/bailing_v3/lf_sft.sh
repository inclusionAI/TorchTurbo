#!/bin/bash
# Bailing-MoE-V3 SFT training with torchturbo
#
# Usage:
#   bash scripts/runs/bailing_v3/install_lf_deps.sh
#   bash scripts/runs/bailing_v3/lf_sft.sh

set -o pipefail
set -eux

# model name or path
MODEL=${MODEL:-/models/Ling-3.0-flash}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/tblf-sft/$(basename "$MODEL")}
# sequence length
SEQ_LEN=${SEQ_LEN:-4096}
# finetuning type
FINETUNING_TYPE=${FINETUNING_TYPE:-full}
# training dataset (must be registered in ${DATASET_DIR}/dataset_info.json)
DATASET=${DATASET:-science}
# turbo engine flags (sp/ep, freeze, cpu offload) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/sft.yaml"}
# optimizer shared by Muon and its fallback AdamW groups; also 2D-matrix LR
# adjustment matters, so keep LR small for full fine-tuning.
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1.25e-5}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}
# checkpointing cadence; 200000000 disables intermediate saves.
SAVE_STEPS=${SAVE_STEPS:-200000000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
SEED=${SEED:-42}
# neat-packing flag; set to empty string to disable.
NEAT_PACKING=${NEAT_PACKING:-"--neat_packing"}
# tensorboard event directory (shared across nodes).
LOGGING_DIR=${LOGGING_DIR:-}

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=14500
export ACL_DEVICE_SYNC_TIMEOUT=14400
# AI_CPU is used with multi-comm-domain parallelism.
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export MULTI_STREAM_MEMORY_REUSE=2

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
DATASET_DIR=${DATASET_DIR:-"${REPO_ROOT}/dataset"}
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

TT_LOGGING_ARGS=""
if [ -n "$LOGGING_DIR" ]; then
    TT_LOGGING_ARGS="--logging_dir ${LOGGING_DIR}"
fi

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_config ${TT_CONFIG} \
    --report_to tensorboard \
    ${TT_LOGGING_ARGS} \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $MODEL \
    --do_train \
    --flash_attn fa2 \
    --dataset_dir ${DATASET_DIR} \
    --dataset $DATASET \
    --preprocessing_num_workers 16 \
    --cutoff_len ${SEQ_LEN} \
    --template bailing_v3 \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --learning_rate ${LEARNING_RATE} \
    --min_lr_ratio ${MIN_LR_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --plot_loss \
    --seed ${SEED} \
    --ddp_timeout 1800000000 \
    --bf16 \
    ${NEAT_PACKING} \
    --trust_remote_code \
    2>&1 | tee $LOG_FILE
