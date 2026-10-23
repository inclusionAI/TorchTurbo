# Bailing-MoE-V3 NPU

## Preparing the Model

Download the Bailing V3 (Ling-3.0-flash) model from Hugging Face to your local machine:

```bash
pip install -U huggingface_hub
hf download inclusionAI/Ling-3.0-flash --local-dir /path/to/models/Ling-3.0-flash
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

### DPO dataset

The DPO examples use `Intel/orca_dpo_pairs`. Download its JSONL:

```bash
hf download Intel/orca_dpo_pairs \
    --repo-type dataset \
    --include "orca_rlhf.jsonl" \
    --local-dir /path/to/datasets/orca_dpo_pairs
```

LLaMA-Factory uses the existing `orca_rlhf` entry in `dataset/dataset_info.json`. For an offline run, replace `hf_hub_url` with the in-container `file_name` and keep the ranking flag and column mappings:

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

ms-swift uses a different preference schema: `messages` contains the system, user, and preferred assistant messages, while `rejected_messages` repeats the context with the rejected assistant message. Convert the downloaded JSONL with the repository's converter:

```bash
python scripts/dataset/convert_text_dpo_to_swift.py \
    /datasets/orca_dpo_pairs/orca_rlhf.jsonl
```

The converter writes `swift_orca_rlhf.jsonl` next to the input file, which is the ms-swift script's default dataset path. It reads `question` (or `query`), `system`, `chosen`, and `rejected`, and reports the number of complete preference rows converted and incomplete rows dropped.

### PT dataset

The PT examples use `olm/olm-wikipedia-20221220`. Download it to a local directory:

```bash
hf download olm/olm-wikipedia-20221220 \
    --repo-type dataset \
    --local-dir /path/to/datasets/olm-wikipedia-20221220
```

LLaMA-Factory uses the existing `wikipedia_en` entry in `dataset/dataset_info.json`. For an offline run, point its `file_name` to one or more downloaded Parquet shards and keep the `text` to `prompt` mapping. The ms-swift script accepts the dataset's `data/` directory by default; `DATASET` can also point to another local JSONL, Parquet, text file, or directory.

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

The example above attaches 16 NPUs (`/dev/davinci0`-`/dev/davinci15`), matching the default `NPUS_PER_NODE=16` of the training scripts; adjust the devices according to the actual number of cards on your machine. The in-container model path is `/models/<model_name>` (e.g. `/models/Ling-3.0-flash`).

## Running Training

### SFT with LLaMA-Factory

```bash
# Install dependencies
bash scripts/runs/bailing_v3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/bailing_v3/lf_sft.sh
```

### SFT with ms-swift

```bash
# Install dependencies
bash scripts/runs/bailing_v3/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/bailing_v3/swift_sft.sh
```

### DPO with LLaMA-Factory

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/bailing_v3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/bailing_v3/lf_dpo.sh
```

### DPO with ms-swift

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/bailing_v3/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/bailing_v3/swift_dpo.sh
```

### PT with LLaMA-Factory

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/bailing_v3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/bailing_v3/lf_pt.sh
```

### PT with ms-swift

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/bailing_v3/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/bailing_v3/swift_pt.sh
```

### Basic Variables

All training scripts share the same set of basic variables, although their default values differ:

| Variable | Example | Description |
|---|---|---|
| `MODEL` | `/models/Ling-3.0-flash` | Model path, corresponding to the container mount path in "Preparing the Model" |
| `OUTPUT_DIR` | `/tmp/tblf-sft/...` or `/tmp/tbsf-dpo/...` | Output directory where checkpoints and logs are stored |
| `SEQ_LEN` | `4096`, `65536`, or `8192` | Training sequence length, adjusted according to memory and data length |
| `FINETUNING_TYPE` | `full` | Finetuning type; only supports `full` and `lora` |
| `DATASET` | `science`, `orca_rlhf`, or a local path | LLaMA-Factory dataset ID registered in `dataset_info.json`, or an ms-swift dataset/local path |
| `TT_CONFIG` | `${SCRIPT_DIR}/sft.yaml`, `${SCRIPT_DIR}/dpo.yaml`, or `${SCRIPT_DIR}/pt.yaml` | Sibling torchturbo engine config file shared by both frameworks for that stage |

Stage-specific defaults are:

| Stage | Shared engine config | LLaMA-Factory defaults | ms-swift defaults |
|---|---|---|---|
| SFT | `sft.yaml` | `SEQ_LEN=4096`, `DATASET=science` | `SEQ_LEN=32768`, `DATASET=/datasets/AM-Thinking-v1-Distilled/science.jsonl` |
| DPO | `dpo.yaml` | `SEQ_LEN=65536`, `DATASET=orca_rlhf` | `SEQ_LEN=65536`, `DATASET=/datasets/orca_dpo_pairs/swift_orca_rlhf.jsonl` |
| PT | `pt.yaml` | `SEQ_LEN=8192`, `DATASET=wikipedia_en` | `SEQ_LEN=8192`, `DATASET=/datasets/olm-wikipedia-20221220/data` |

Override the variables as needed when launching:

```bash
# LLaMA-Factory
DATASET=science MODEL=/models/Ling-3.0-flash OUTPUT_DIR=/torchturbo/output \
    bash scripts/runs/bailing_v3/lf_sft.sh

# ms-swift
DATASET=/datasets/AM-Thinking-v1-Distilled/science.jsonl \
    MODEL=/models/Ling-3.0-flash OUTPUT_DIR=/torchturbo/output \
    bash scripts/runs/bailing_v3/swift_sft.sh
```
