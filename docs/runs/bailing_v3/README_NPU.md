# Bailing-MoE-V3 NPU

See the [NPU Training Guide](../README_NPU.md) for dataset preparation, container setup, and common training options.

## Model Preparation

Use the [Hugging Face CLI](../README_NPU.md#preparing-the-model) to download the Bailing V3 (Ling-3.0-flash) model to your local machine:

```bash
hf download inclusionAI/Ling-3.0-flash --local-dir /path/to/models/Ling-3.0-flash
```

## Training Examples

The LLaMA-Factory and ms-swift scripts share an engine config for each stage:

| Stage | Dataset | Engine config |
|---|---|---|
| SFT | [Science](../README_NPU.md#science) | [sft.yaml](../../../scripts/runs/bailing_v3/sft.yaml) |
| DPO | [Orca DPO Pairs](../README_NPU.md#orca-dpo-pairs) | [dpo.yaml](../../../scripts/runs/bailing_v3/dpo.yaml) |
| PT | [Wikipedia](../README_NPU.md#wikipedia) | [pt.yaml](../../../scripts/runs/bailing_v3/pt.yaml) |

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
