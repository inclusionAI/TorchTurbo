# GLM-5.3 NPU

## Preparing the Model

Use the [Hugging Face CLI](../README.md#preparing-the-model) to download the GLM-5.3 BF16 model to your local machine:

```bash
hf download zai-org/GLM-5.3-BF16 --local-dir /path/to/models/GLM-5.3-BF16
```

## Preparing the Dataset

The full SFT and both LoRA SFT examples use the [Science](../README.md#science) dataset.

## Starting the Container

Follow [Starting the Container](../README.md#starting-the-container) to mount the model, dataset, and repository. Run the commands below from `/torchturbo` inside the container.

## Running Training

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

To change the model path, dataset, output directory, or sequence length, see [Basic Variables](../README.md#basic-variables).
