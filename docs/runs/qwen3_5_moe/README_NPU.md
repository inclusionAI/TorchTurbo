# Qwen3.5-MoE NPU

## Preparing the Model

Use the [Hugging Face CLI](../README.md#preparing-the-model) to download a Qwen3.5 model to your local machine (taking 35B-A3B as an example; 122B-A10B and 397B-A17B are the same):

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/models/Qwen3.5-35B-A3B
# hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/models/Qwen3.5-122B-A10B
# hf download Qwen/Qwen3.5-397B-A17B --local-dir /path/to/models/Qwen3.5-397B-A17B
```

The LLaMA-Factory text SFT example defaults to `/models/Qwen3.5-35B-A3B`; the ms-swift examples default to `/models/Qwen3.5-397B-A17B`.

## Preparing the Dataset

- Text SFT with LLaMA-Factory: [GSM8K](../README.md#gsm8k).
- Multimodal SFT with ms-swift: [LLaVA Chinese](../README.md#llava-chinese).
- Text DPO with ms-swift: [Orca DPO Pairs](../README.md#orca-dpo-pairs).

## Starting the Container

Follow [Starting the Container](../README.md#starting-the-container) to mount the model, dataset, and repository. Run the commands below from `/torchturbo` inside the container.

## Running Training

### Text SFT with LLaMA-Factory

The engine config is [lf_text_sft.yaml](../../../scripts/runs/qwen3_5_moe/lf_text_sft.yaml).

```bash
# Install dependencies
bash scripts/runs/qwen3_5_moe/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/lf_text_sft.sh
```

### Multimodal SFT with ms-swift

The engine config is [swift_mllm_sft.yaml](../../../scripts/runs/qwen3_5_moe/swift_mllm_sft.yaml).

```bash
# Install dependencies
bash scripts/runs/qwen3_5_moe/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/swift_mllm_sft.sh
```

### Text DPO with ms-swift

The engine config is [swift_text_dpo.yaml](../../../scripts/runs/qwen3_5_moe/swift_text_dpo.yaml).

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/qwen3_5_moe/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/swift_text_dpo.sh
```

To change the model path, dataset, output directory, or sequence length, see [Basic Variables](../README.md#basic-variables).
