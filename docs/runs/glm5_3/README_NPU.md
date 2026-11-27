# GLM-5.3 NPU

See the [NPU Training Guide](../README_NPU.md) for dataset preparation, container setup, and common training options.

## Model Preparation

Use the [Hugging Face CLI](../README_NPU.md#preparing-the-model) to download the GLM-5.3 BF16 model to your local machine:

```bash
hf download zai-org/GLM-5.3-BF16 --local-dir /path/to/models/GLM-5.3-BF16
```

## Training Examples

The examples use the [Science](../README_NPU.md#science) dataset.

### SFT with LLaMA-Factory

The engine config is [lf_sft.yaml](../../../scripts/runs/glm5_3/lf_sft.yaml).

```bash
# Install dependencies
bash scripts/runs/glm5_3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/glm5_3/lf_sft.sh
```

### LoRA SFT with LLaMA-Factory

The engine config is [lf_sft_lora.yaml](../../../scripts/runs/glm5_3/lf_sft_lora.yaml).

```bash
# Install dependencies if they have not already been installed
bash scripts/runs/glm5_3/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/glm5_3/lf_sft_lora.sh
```

### LoRA SFT with ms-swift

The engine config is [swift_sft_lora.yaml](../../../scripts/runs/glm5_3/swift_sft_lora.yaml).

```bash
# Install dependencies
bash scripts/runs/glm5_3/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/glm5_3/swift_sft_lora.sh
```
