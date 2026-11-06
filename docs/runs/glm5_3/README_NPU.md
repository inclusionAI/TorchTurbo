# GLM-5.3 NPU

## Preparing the Model

Download the GLM-5.3 BF16 model from Hugging Face to your local machine:

```bash
pip install -U huggingface_hub
hf download zai-org/GLM-5.3-BF16 --local-dir /path/to/models/GLM-5.3-BF16
```

If network access is restricted, use an HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`.

The downloaded model directory will be mounted into the container in the "Starting the Container" section, with the in-container path `/models/<model_name>`.

## Preparing the Dataset

### SFT dataset

The SFT examples use the pure-text `science` dataset. The Hugging Face repository `a-m-team/AM-Thinking-v1-Distilled` contains multiple JSONL files, so download only `science.jsonl`:

```bash
hf download a-m-team/AM-Thinking-v1-Distilled \
    --repo-type dataset \
    --include "science.jsonl" \
    --local-dir /path/to/datasets/AM-Thinking-v1-Distilled
```

After downloading, edit the `science` entry in `dataset/dataset_info.json`. Replace `hf_hub_url` and `data_files` with the in-container `file_name`, and keep the ShareGPT column and tag mappings:

```json
"science": {
  "file_name": "/datasets/AM-Thinking-v1-Distilled/science.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "assistant"
  }
}
```

`file_name` is the in-container path of `science.jsonl`. LLaMA-Factory resolves the dataset through this entry; ms-swift does not use `dataset_info.json` and accepts the JSONL path directly.

## Starting the Container

Mount the models, datasets, and this repository into a Docker container with NPU/GPU devices attached (the container name `env-torchturbo` is just an example):

```bash
docker run -it \
    --name env-torchturbo \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci8 --device /dev/davinci9 \
    --device /dev/davinci10 --device /dev/davinci11 \
    --device /dev/davinci12 --device /dev/davinci13 \
    --device /dev/davinci14 --device /dev/davinci15 \
    --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /path/to/models:/models \
    -v /path/to/torchturbo:/torchturbo \
    -v /path/to/datasets:/datasets \
    <image> bash
```

The example above attaches 16 NPUs (`/dev/davinci0`-`/dev/davinci15`), matching the default `NPUS_PER_NODE=16` of the training scripts; adjust the devices according to the actual number of cards on your machine. The in-container model path is `/models/<model_name>` (e.g. `/models/GLM-5.3-BF16`).

## Running Training

### SFT with LLaMA-Factory

```bash
# Install dependencies
bash scripts/runs/glm5_3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/glm5_3/lf_sft.sh
```

### LoRA SFT with LLaMA-Factory

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/glm5_3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/glm5_3/lf_sft_lora.sh
```

### LoRA SFT with ms-swift

```bash
# Install dependencies
bash scripts/runs/glm5_3/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/glm5_3/swift_sft_lora.sh
```

### Basic Variables

All training scripts share the same set of basic variables, although their default values differ:

| Variable | Example | Description |
|---|---|---|
| `MODEL` | `/models/GLM-5.3-BF16` | Model path, corresponding to the container mount path in "Preparing the Model" |
| `OUTPUT_DIR` | `/tmp/tblf-sft/...`, `/tmp/tblf-sft-lora/...`, or `/tmp/tbsf-sft-lora/...` | Output directory where checkpoints and logs are stored |
| `SEQ_LEN` | `4096` or `65536` | Training sequence length, adjusted according to memory and data length |
| `FINETUNING_TYPE` | `full` or `lora` | Finetuning type; only supports `full` and `lora` |
| `DATASET` | `science` or a local path | LLaMA-Factory dataset ID registered in `dataset_info.json`, or an ms-swift dataset/local path |
| `TT_CONFIG` | `${SCRIPT_DIR}/lf_sft.yaml`, `${SCRIPT_DIR}/lf_sft_lora.yaml`, or `${SCRIPT_DIR}/swift_sft_lora.yaml` | torchturbo engine config file, containing `engine_config.parallel_plan` (sp_size/ep_size), etc. |

Override the variables as needed when launching:

```bash
# LLaMA-Factory
DATASET=science MODEL=/models/GLM-5.3-BF16 OUTPUT_DIR=/torchturbo/output/glm53-lf-sft \
    bash scripts/runs/glm5_3/lf_sft.sh

# ms-swift
DATASET=/datasets/AM-Thinking-v1-Distilled/science.jsonl \
    MODEL=/models/GLM-5.3-BF16 OUTPUT_DIR=/torchturbo/output/glm53-swift-sft-lora \
    bash scripts/runs/glm5_3/swift_sft_lora.sh
```
