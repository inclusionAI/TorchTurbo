# Qwen3.5-MoE NPU

See the [NPU Training Guide](../README_NPU.md) for dataset preparation, container setup, and common training options.

## Model Preparation

Use the [Hugging Face CLI](../README_NPU.md#preparing-the-model) to download a Qwen3.5 model to your local machine (taking 35B-A3B as an example; 122B-A10B and 397B-A17B are the same):

```bash
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/models/Qwen3.5-35B-A3B
# hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/models/Qwen3.5-122B-A10B
# hf download Qwen/Qwen3.5-397B-A17B --local-dir /path/to/models/Qwen3.5-397B-A17B
```

The LLaMA-Factory text SFT example defaults to `/models/Qwen3.5-35B-A3B`; the ms-swift examples default to `/models/Qwen3.5-397B-A17B`.

## Training Examples

### Text SFT with LLaMA-Factory

This example uses [GSM8K](../README_NPU.md#gsm8k). The engine config is [lf_text_sft.yaml](../../../scripts/runs/qwen3_5_moe/lf_text_sft.yaml).

```bash
# Install dependencies
bash scripts/runs/qwen3_5_moe/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/lf_text_sft.sh
```

### Multimodal SFT with ms-swift

This example uses [LLaVA Chinese](../README_NPU.md#llava-chinese). The engine config is [swift_mllm_sft.yaml](../../../scripts/runs/qwen3_5_moe/swift_mllm_sft.yaml).

```bash
# Install dependencies
bash scripts/runs/qwen3_5_moe/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/swift_mllm_sft.sh
```

### Text DPO with ms-swift

This example uses [Orca DPO Pairs](../README_NPU.md#orca-dpo-pairs). The engine config is [swift_text_dpo.yaml](../../../scripts/runs/qwen3_5_moe/swift_text_dpo.yaml).

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/qwen3_5_moe/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/swift_text_dpo.sh
```
