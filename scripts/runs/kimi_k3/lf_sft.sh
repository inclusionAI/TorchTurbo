#!/usr/bin/env bash
# Kimi-K3 SFT with torchturbo via LlamaFactory (single-node smoke test).
#
# Usage:
#   KIMI_K3_MODEL_PATH=/path/to/weights bash scripts/runs/kimi_k3/lf_sft.sh
#
# Notes:
# - K3 has no M-RoPE, so packing_mode is packing_sep (neat_packing collator
#   is not adapted for it).
# - --trust_remote_code loads the K3 modeling shipped with the weights repo.

set -o pipefail
set -ex

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
export MULTI_STREAM_MEMORY_REUSE=2
export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export TRANSFORMERS_VERBOSITY=error

# ==============================================================================
# Configuration (override via env)
# ==============================================================================
model_name_or_path=${KIMI_K3_MODEL_PATH:?"set KIMI_K3_MODEL_PATH to the K3 weights dir"}
seq_length=${SEQ_LENGTH:-$((1024 * 4))}
sp_size=${SP_SIZE:-2}
ep_size=${EP_SIZE:-8}          # e.g. 896 experts / EP8 = 112 experts per rank
dataset=${DATASET:-llava_1k_zh}
packing_mode=${PACKING_MODE:-packing_sep}
finetuning_type=${FINETUNING_TYPE:-full}
cpu_offload=${CPU_OFFLOAD:-false}
max_steps=${MAX_STEPS:-10}

NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23556}

output_dir=${OUTPUT_DIR:-./output/kimi_k3/lf_sft}
mkdir -p "$output_dir"
export ASCEND_PROCESS_LOG_PATH=$output_dir/log
log_file=$output_dir/log-$NODE_RANK.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_LF="${SCRIPT_DIR}/../run_lf.py"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset/npu}"
TORCHTURBO_SRC_DIR="${TORCHTURBO_SRC_DIR:-${REPO_ROOT}/src}"

if [ ! -d "${TORCHTURBO_SRC_DIR}/torchturbo" ]; then
    echo "Error: TORCHTURBO_SRC_DIR does not point to torchturbo src: ${TORCHTURBO_SRC_DIR}" >&2
    exit 1
fi

export PYTHONPATH="${TORCHTURBO_SRC_DIR}:$PYTHONPATH"
if [ -n "${LLAMA_FACTORY_SRC:-}" ]; then
    export PYTHONPATH="${LLAMA_FACTORY_SRC}:$PYTHONPATH"
fi

OFFLOAD_ARG=""
if [ "$cpu_offload" = "true" ]; then
    OFFLOAD_ARG="--tt_cpu_offload"
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank $NODE_RANK
"

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size $sp_size \
    --tt_ep_size $ep_size \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_steps $max_steps \
    --trust_remote_code \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 16 \
    --cutoff_len ${seq_length} \
    --packing_mode ${packing_mode} \
    --template kimi_k3 \
    --finetuning_type ${finetuning_type} \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1.25e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --bf16 \
    --tt_freeze_vision_tower \
    --ddp_timeout 1800000000 \
    $OFFLOAD_ARG 2>&1 | tee $log_file