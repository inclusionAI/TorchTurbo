# Bailing-MoE-V3 NPU

## Preparing the Model

Use the [Hugging Face CLI](../README.md#preparing-the-model) to download the Bailing V3 (Ling-3.0-flash) model to your local machine:

```bash
hf download inclusionAI/Ling-3.0-flash --local-dir /path/to/models/Ling-3.0-flash
```

## Preparing the Dataset

- SFT: [Science](../README.md#science).
- DPO: [Orca DPO Pairs](../README.md#orca-dpo-pairs).
- PT: [Wikipedia](../README.md#wikipedia).

## Starting the Container

Follow [Starting the Container](../README.md#starting-the-container) to mount the model, dataset, and repository. Run the commands below from `/torchturbo` inside the container.

## Running Training

The LLaMA-Factory and ms-swift scripts share an engine config for each stage:

| Stage | Engine config |
|---|---|
| SFT | [sft.yaml](../../../scripts/runs/bailing_v3/sft.yaml) |
| DPO | [dpo.yaml](../../../scripts/runs/bailing_v3/dpo.yaml) |
| PT | [pt.yaml](../../../scripts/runs/bailing_v3/pt.yaml) |

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

To change the model path, dataset, output directory, or sequence length, see [Basic Variables](../README.md#basic-variables).
