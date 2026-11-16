#!/bin/bash
# GLM-5.3 MoE-DSA LoRA SFT with torchturbo via ms-swift
#
# Usage:
#   bash scripts/runs/glm5_3/install_swift_deps.sh
#   bash scripts/runs/glm5_3/swift_sft_lora.sh

set -o pipefail
set -eux

# model name or path
MODEL=${MODEL:-/models/GLM-5.3-BF16}
# sequence length
SEQ_LEN=${SEQ_LEN:-65536}
# output directory
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/tbsf-sft-lora/$(basename "${MODEL}")}
# tuner type: 'full' | 'lora'
FINETUNING_TYPE=${FINETUNING_TYPE:-lora}
# training dataset (ms-swift dataset id "repo:subset", or path to a local jsonl/parquet)
DATASET=${DATASET:-/datasets/AM-Thinking-v1-Distilled/science.jsonl}
# turbo engine flags (sp/ep, cpu offload, chunk loss, ...) live in the sibling yaml.
TT_CONFIG=${TT_CONFIG:-"$(dirname "$0")/swift_sft_lora.yaml"}

dataset_num_proc=96
dataloader_num_workers=8       # swift args: --dataloader_num_workers
model_type=turbo_glm_moe_dsa   # turbo-registered model_type (loader = TurboModelLoader)
template=${SWIFT_TEMPLATE:-glm5_1}
lora_rank=8
lora_alpha=16
lora_dropout=0.0
target_modules=all

# python interpreter (defaults to python3)
PYTHON=${PYTHON:-python3}

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
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=1
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23668}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NPUS_PER_NODE}}
export WORLD_SIZE=$((${NPUS_PER_NODE} * ${NNODES}))
export WANDB_DISABLED=True
export TRANSFORMERS_VERBOSITY=error
export ASCEND_PROCESS_LOG_PATH=$OUTPUT_DIR/log
export HF_DATASETS_CACHE=$OUTPUT_DIR/datasets_cache

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SWIFT="${REPO_ROOT}/scripts/runs/run_swift.py"
# ms-swift repo root (the dir containing the `swift` package); empty if pip-installed.
SWIFT_SRC_DIR=${SWIFT_SRC_DIR:-}
export PYTHONPATH="${REPO_ROOT}/src${SWIFT_SRC_DIR:+:${SWIFT_SRC_DIR}}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/log-$NODE_RANK.txt"
if [ "${NODE_RANK:-0}" -eq 0 ]; then
    cp -f "$0" "$OUTPUT_DIR/"
fi
{ date "+%Y%m%d-%H%M%S"; echo "$OUTPUT_DIR"; } > "$LOG_FILE"

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${RUN_SWIFT}" \
    --model_type "${model_type}" \
    --model "${MODEL}" \
    --check_model false \
    --template "${template}" \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --max_length "${SEQ_LEN}" \
    --packing true \
    --packing_length "${SEQ_LEN}" \
    --attn_impl flash_attn \
    --dataset_num_proc "${dataset_num_proc}" \
    --tuner_type "${FINETUNING_TYPE}" \
    --lora_rank "${lora_rank}" \
    --lora_alpha "${lora_alpha}" \
    --lora_dropout "${lora_dropout}" \
    --target_modules "${target_modules}" \
    --use_rslora true \
    --tt_config "${TT_CONFIG}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr":1e-7}' \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 200000000 \
    --save_total_limit 1 \
    --bf16 true \
    --dataloader_num_workers "${dataloader_num_workers}" \
    --report_to tensorboard \
    --output_dir "${OUTPUT_DIR}" \
    --ddp_timeout 1800000000 \
    2>&1 | tee -a "${LOG_FILE}"
