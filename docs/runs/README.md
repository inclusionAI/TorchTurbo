# NPU Training Guide

## Preparing the Model

Install the Hugging Face CLI on your local machine:

```bash
pip install -U huggingface_hub
```

If network access is restricted, use an HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`.

Select the appropriate model guide below for download instructions and training commands:

| Model guide | LLaMA-Factory examples | ms-swift examples |
|---|---|---|
| [Bailing-MoE-V3](bailing_v3/README_NPU.md) | Text SFT (full), Text DPO, PT | Text SFT (full), Text DPO, PT |
| [Qwen3.5-MoE](qwen3_5_moe/README_NPU.md) | Text SFT (full) | Multimodal SFT (full), Text DPO |
| [GLM-5.3](glm5_3/README_NPU.md) | Text SFT (full, LoRA) | Text SFT (LoRA) |

The downloaded model directory will be mounted into the container in [Starting the Container](#starting-the-container), with the in-container path `/models/<model_name>`.

## Preparing the Dataset

Download the dataset used by the selected training example. The examples below use `/path/to/datasets` on the host, mounted as `/datasets` inside the container.

| Training task | Example datasets |
|---|---|
| [Text SFT](#text-sft) | Science, GSM8K |
| [Multimodal SFT](#multimodal-sft) | LLaVA Chinese |
| [DPO](#dpo) | Orca DPO Pairs |
| [PT](#pt) | Wikipedia |

### Text SFT

#### Science

The Hugging Face repository `a-m-team/AM-Thinking-v1-Distilled` contains multiple JSONL files. Download `science.jsonl` for text SFT:

```bash
hf download a-m-team/AM-Thinking-v1-Distilled \
    --repo-type dataset \
    --include "science.jsonl" \
    --local-dir /path/to/datasets/AM-Thinking-v1-Distilled
```

##### LLaMA-Factory

After downloading, edit the `science` entry in the dataset configuration file, [dataset/dataset_info.json](../../dataset/dataset_info.json). Replace `hf_hub_url` and `data_files` with the in-container `file_name`, and keep the ShareGPT column and tag mappings:

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

`file_name` is the in-container path of `science.jsonl`. LLaMA-Factory resolves the dataset through this entry.

##### ms-swift

ms-swift does not use `dataset_info.json` and accepts the JSONL path directly. The scripts use `/datasets/AM-Thinking-v1-Distilled/science.jsonl` by default. Set `DATASET` to a different path if needed.

#### GSM8K

Download the `main` subset of `openai/gsm8k`:

```bash
hf download openai/gsm8k --repo-type dataset --include "main/*" --local-dir /path/to/datasets/gsm8k
```

##### LLaMA-Factory

After downloading, edit the `gsm8k` entry in the dataset configuration file, [dataset/dataset_info.json](../../dataset/dataset_info.json). Replace `hf_hub_url` with the in-container `file_name`, and keep the column mappings:

```json
"gsm8k": {
  "file_name": "/datasets/gsm8k/main/train-00000-of-00001.parquet",
  "columns": { "prompt": "question", "response": "answer" }
}
```

### Multimodal SFT

#### LLaVA Chinese

Download `BUAADreamer/llava-en-zh-2k` for multimodal SFT:

```bash
hf download BUAADreamer/llava-en-zh-2k --repo-type dataset --local-dir /path/to/datasets/llava-en-zh-2k
```

##### LLaMA-Factory

Edit `dataset/dataset_info.json` and point the `llava_1k_zh` entry to the local file, keeping the ShareGPT column and tag mappings:

```json
"llava_1k_zh": {
  "file_name": "/datasets/llava-en-zh-2k/zh/train-00000-of-00001.parquet",
  "formatting": "sharegpt",
  "columns": { "messages": "messages", "images": "images" },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

##### ms-swift

The script uses `hf::BUAADreamer/llava-en-zh-2k:zh` by default. To use the downloaded data, set `DATASET=/datasets/llava-en-zh-2k/zh/train-00000-of-00001.parquet` when launching training.

### DPO

#### Orca DPO Pairs

Download the JSONL from `Intel/orca_dpo_pairs`:

```bash
hf download Intel/orca_dpo_pairs \
    --repo-type dataset \
    --include "orca_rlhf.jsonl" \
    --local-dir /path/to/datasets/orca_dpo_pairs
```

##### LLaMA-Factory

The scripts use the `orca_rlhf` entry in `dataset/dataset_info.json`. For an offline run, replace `hf_hub_url` with the in-container `file_name` and keep the ranking flag and column mappings:

```json
"orca_rlhf": {
  "file_name": "/datasets/orca_dpo_pairs/orca_rlhf.jsonl",
  "ranking": true,
  "columns": {
    "prompt": "question",
    "system": "system",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

##### ms-swift

Convert the downloaded JSONL to the format used by ms-swift. After starting the container, run the converter from the repository root:

```bash
python scripts/dataset/convert_text_dpo_to_swift.py \
    /datasets/orca_dpo_pairs/orca_rlhf.jsonl
```

The converter writes `swift_orca_rlhf.jsonl` next to the input file, with `messages` for the preferred response and `rejected_messages` for the rejected response. This file can be selected through `DATASET` when launching training.

### PT

#### Wikipedia

Download `olm/olm-wikipedia-20221220` to a local directory:

```bash
hf download olm/olm-wikipedia-20221220 \
    --repo-type dataset \
    --local-dir /path/to/datasets/olm-wikipedia-20221220
```

##### LLaMA-Factory

After downloading, edit the `wikipedia_en` entry in the dataset configuration file, [dataset/dataset_info.json](../../dataset/dataset_info.json). Replace `hf_hub_url` and `ms_hub_url` with the in-container `file_name`, and keep the `text` to `prompt` mapping:

```json
"wikipedia_en": {
  "file_name": "/datasets/olm-wikipedia-20221220/data",
  "columns": { "prompt": "text" }
}
```

This configuration loads all Parquet files in the downloaded `data/` directory.

##### ms-swift

The script uses `/datasets/olm-wikipedia-20221220/data` by default. Set `DATASET` to a different path if needed.

## Starting the Container

Mount the models, datasets, and this repository into a Docker container with NPU devices attached (the container name `env-torchturbo` is just an example):

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

The example above attaches 16 NPUs (`/dev/davinci0`-`/dev/davinci15`), matching the default `NPUS_PER_NODE=16` of the training scripts; adjust the devices according to the actual number of cards on your machine. The in-container model path is `/models/<model_name>`.

Run dependency installation, data conversion, and training commands inside the container from the repository root:

```bash
cd /torchturbo
```

## Running Training

Install the dependencies and start training using the commands in the model guide.

### Basic Variables

The training scripts share the following basic variables. Refer to the selected script for its defaults.

| Variable | Example | Description |
|---|---|---|
| `MODEL` | `/models/<model_name>` | Model path inside the container |
| `OUTPUT_DIR` | `/torchturbo/output/my-run` | Output directory where checkpoints and logs are stored |
| `SEQ_LEN` | `4096` | Training sequence length, adjusted according to memory and data length |
| `FINETUNING_TYPE` | `full` or `lora` | Full-parameter or LoRA fine-tuning |
| `DATASET` | `science` or a local path | LLaMA-Factory dataset entry name, or an ms-swift dataset ID/local path |
| `TT_CONFIG` | `scripts/runs/<model>/<config>.yaml` | TorchTurbo engine config file for parallelism, activation checkpointing, CPU offload, and other engine settings |

For configuration precedence and engine options, see [Configuration](../../README.md#configuration). An example YAML is provided in [config_example.yaml](../../scripts/runs/config_example.yaml).

Override the variables as needed when launching (using Bailing SFT as an example):

```bash
# LLaMA-Factory
DATASET=science MODEL=/models/Ling-3.0-flash OUTPUT_DIR=/torchturbo/output \
    bash scripts/runs/bailing_v3/lf_sft.sh

# ms-swift
DATASET=/datasets/AM-Thinking-v1-Distilled/science.jsonl \
    MODEL=/models/Ling-3.0-flash OUTPUT_DIR=/torchturbo/output \
    bash scripts/runs/bailing_v3/swift_sft.sh
```
