# TorchTurbo

A supernode-native post-training engine for rapidly adapting and training large language and vision models.

## Overview

Post-training large models presents several challenges. Model architectures continue to evolve and introduce innovations such as mixture-of-experts (MoE), multimodality, and sparse attention. Post-training workflows, including reinforcement learning (RL), supervised fine-tuning (SFT), continual pre-training (CPT), and direct preference optimization (DPO), must flexibly adapt to these changing architectures while combining N-dimensional parallelism for high-performance training. Rapid model iteration therefore calls for a simple, general-purpose, high-performance distributed training engine that keeps the focus on model and training logic.

At the same time, AI infrastructure is entering the SuperNode era. The powerful communication capabilities of SuperNodes provide better support for training ultra-large MoE models. Against this backdrop, we introduce TorchTurbo, a distributed training engine centered on model and training logic. TorchTurbo improves model adaptation and training efficiency through deeply optimized fused operators and SuperNode affinity, taking full advantage of low inter-node communication overhead within a SuperNode.

Built on PyTorch FSDP2 and Hugging Face Transformers, TorchTurbo combines **fully sharded data parallelism (FSDP), expert parallelism (EP), and Ulysses sequence parallelism (SP)** to provide flexible and efficient parallel training strategies for multimodal models. Its architecture is shown below:

![TorchTurbo training architecture](docs/assets/architecture-en.svg)

Core principles:

1. **Model-faithful:** Use clean upstream Hugging Face model implementations, replacing modules only where necessary for high performance, such as MoE, attention, and fused operations. This approach simplifies maintenance and extension.
2. **Simple parallel-training transformations:** Build straightforward training-parallel strategies by adding EP and SP on top of FSDP2 to achieve high training throughput.
3. **Pluggable into mainstream training frameworks:** Keep the training engine focused on model and training logic.
4. **SuperNode-native:** Optimize the training engine for SuperNode architectures.

### Model

Each model directory is responsible for three tasks:

- Registering the model type
- Implementing the `TurboModel` or `TurboVLMModel` contract and model-specific hooks
- Providing the model's default strategy through `get_default_parallel_plan()`

Models declare their sequence parallel input contract through
`get_sequence_parallel_input_slicing_spec()`. Before the model call, the spec's
padded fields in the global batch are right-padded to a multiple of the SP size;
packed sequence metadata keeps describing the valid prefix, so attention, MoE
statistics, and losses ignore the tail. The backend installs slicing at the spec's
boundary before activation checkpointing and sharding. The default spec slices
token IDs and positions at the model entry; VLMs slice at their media-fusion
boundary. Loss targets use the same shards, while media tensors and packed
sequence metadata stay global.

Typical structure:

```text
src/torchturbo/models/<model_name>/
├── __init__.py
├── model.py
├── parallel_plan.py
├── configuration_*.py       # Optional local Transformers configuration
├── modeling_*.py            # Optional local Transformers implementation
├── lf_template.py            # Optional LlamaFactory integration
├── swift_template.py         # Optional ms-swift integration
└── npu.py / gpu.py           # Optional device-specific patches
```

When a model needs local copies of upstream Transformers code, its directory maintains the corresponding `configuration_*.py` and `modeling_*.py` files.

### ParallelPlan

A model declares its distributed execution strategy through `FSDPParallelPlan`:

```python
from torch.distributed._tensor import Shard
from torchturbo.parallel import FSDPParallelPlan, FSDPPlacementSpec, FSDPPlanSpec


def get_parallel_plan() -> FSDPParallelPlan:
    return FSDPParallelPlan(
        plan={
            "model.layers.*.mlp.experts": FSDPPlanSpec(
                placement=FSDPPlacementSpec(Shard(0), "edp"),
            ),
        },
        sp_size=1,
        ep_size=1,
    )
```

The wrapper class returns this plan from `get_default_parallel_plan()`. The engine then merges `EngineConfig.parallel_plan` into it, including runtime `sp_size` and `ep_size` values supplied through environment variables, a configuration file, or CLI arguments.

Vision balancing is configured on a compatible module's plan entry. For Bailing
V3 VL, enable it on `model.visual` in the file passed to `--tt_config`:

```yaml
engine_config:
  parallel_plan:
    plan:
      model.visual:
        vision_balance:
          mode: patch_sq
```

`patch` weights whole media by patch count; `patch_sq` uses squared patch count.
The model's Python plan supplies `VisionBalanceSpec(io=BailingVisionBalanceIO())`;
its default `mode=None` leaves balancing disabled. A mode-only file override
preserves that I/O instance. Python callers use `VisionBalanceMode.PATCH` or
`VisionBalanceMode.PATCH_SQ`; file values are converted to the enum when parsed.

Models implement `BaseVisionBalanceIO` to read and replace call inputs, extract
and restore all output streams, and build valid dummy media. `VisionBatch`
declares packed patch rows, media grids, and per-medium lengths for each output
stream. `register_vision_balance` uses the instance to preserve the model's
argument and return formats while exchanging media and restoring their order.
The boundary must be called in the same order on every WORLD rank. Bailing V3 VL
provides this implementation; other towers require their own I/O contract.

### Backend

Currently available backend:

- `fsdp2`: Provides weight loading, sharding, saving, and optimizer preparation based on PyTorch FSDP2 and DTensor

Future backends should be registered through `src/torchturbo/engine/backend_registry.py`.

### Adapter

Currently registered adapters:

- `llamafactory_v0`: Hooks into LlamaFactory's model loading, training, and saving workflows
- `msswift`: Integrates with ms-swift as a plugin

TorchTurbo currently supports training only. Evaluation, prediction, and generation
are not supported, with or without sequence parallelism. The LlamaFactory and
ms-swift adapters reject configurations unless `eval_strategy="no"`,
`do_eval=False`, `do_predict=False`, `predict_with_generate=False`, and
`eval_on_start=False`.

For ms-swift DPO, `--router_aux_loss_coef` sets the default for TorchTurbo's
`moe.aux_loss_coeff`; `--tt_config` and explicit `--tt_moe_aux_loss_coeff` take
precedence. The engine's routers apply the auxiliary gradients and report
`aux_loss` / `aux_loss_scaled` separately from the DPO loss.

SFT samples are always isolated, including with LlamaFactory's
`packing=True, neat_packing=False`. `train_on_prompt=True` enables prompt
supervision within each sample; predictions never cross sample boundaries.

The `src/torchturbo/adapters/rl/` package provides reusable packed-sequence and SP helpers for reinforcement learning frameworks; it is a utilities package rather than a registered adapter.

RL callers can suspend the prepared FSDP2 optimizer between training windows.

The `cpu_offload` training configuration keeps optimizer steps and checkpoint
save/load available. Calling `engine.offload_optimizer()` temporarily suspends
the optimizer and rejects those operations until `engine.restore_optimizer()`.
Restoring resumes the configured training policy, including CPU offload.

```python
optimizer.zero_grad(set_to_none=True)
engine.offload_optimizer()
# Run inference, then release its allocations before restoring training state.
engine.restore_optimizer()
```

Call these synchronous, idempotent APIs on every training worker at an idle
training boundary. They release AdamW, AdamWSwap, Muon, and child optimizer
device state and pause QK clipping statistics, while preserving parameter and
scheduler bindings. Model weights and inference resources remain
caller-managed. With Muon, keep FSDP parameters in place: moving them with
`module.to()` replaces their storage and makes the next step fail, so use
`full` CPU offload when weights must leave the device.

Muon supports `full` and `optimizer` CPU offload, and
`optimizer_memory_fraction=0` selects CPU Newton-Schulz and CPU exchange. These
paths are experimental until they are fully validated.

To register a LlamaFactory template for a model, add an `lf_template.py` file to the model directory. When an adapter is activated, it scans registered models and automatically calls the corresponding `register()` function.

## Quick Start

### Build Docker

```bash
cd docker/npu/

docker build \
--platform linux/arm64 \
-f Dockerfile_aarch64 \
-t torchturbo:aarch64 \
```

Platform dependency scripts are available in `scripts/runs/deps/`, while model-specific installers are located under `scripts/runs/<model>/`.

### Start Training

When training is launched through an entry point provided by TorchTurbo, the training framework is invoked as a dependency.

The LlamaFactory entry point, `scripts/runs/run_lf.py`, explicitly activates `llamafactory_v0` and then calls LlamaFactory's `run_exp()`.    
The ms-swift entry point, `scripts/runs/run_swift.py`, parses TorchTurbo arguments and activates the `msswift` plugin before dispatching the selected stage.

See the [`Qwen3.5 MoE training guide`](docs/runs/qwen3_5_moe/README_NPU.md).

### Configuration

TorchTurbo-specific configuration can be provided in three ways. The precedence order, from lowest to highest, is:

```text
Adapter defaults < TORCH_TURBO_* environment variables < --tt_config YAML file < explicit --tt_* CLI arguments
```

The `--tt_*` arguments are defined in [`src/torchturbo/engine/args.py`](src/torchturbo/engine/args.py).

See [`scripts/runs/config_example.yaml`](scripts/runs/config_example.yaml) for an example YAML configuration:

```bash
torchrun --nproc_per_node=8 scripts/runs/run_lf.py \
  --tt_config scripts/runs/config_example.yaml \
  <LLaMA-Factory arguments>
```

The `--tt_*` arguments are removed before the remaining arguments are forwarded to the training framework, so they do not interfere with the original framework's argument parser.

### Run Scripts

Common training scripts are organized by model under `scripts/runs/`. The `run_lf.py` and `run_swift.py` files are shared entry points:

```text
scripts/runs/
├── run_lf.py                 # LlamaFactory entry point
├── run_swift.py              # ms-swift entry point
├── config_example.yaml       # Example --tt_config file
├── deps/                     # Base dependencies
│   ├── gpu.sh                # GPU dependencies
│   ├── npu.sh                # NPU dependencies
│   ├── lf_deps.sh            # LlamaFactory dependencies
│   └── swift_deps.sh         # ms-swift dependencies
└── qwen3_5_moe/
    ├── install_lf_deps.sh    # Model runtime dependency installer
    ├── install_swift_deps.sh # ms-swift dependency installer
    ├── lf_text_sft.sh        # LlamaFactory SFT training script
    └── swift_mllm_sft.sh     # ms-swift multimodal SFT script
```

Example:

```bash
bash scripts/runs/qwen3_5_moe/lf_text_sft.sh
```

Before running a script, you will typically need to check:

- `model_name_or_path`
- `output_dir`
- TorchTurbo's `PYTHONPATH`
- `NNODES` / `NPUS_PER_NODE` / `MASTER_ADDR` / `MASTER_PORT`
- `TORCH_TURBO_SP_SIZE` / `TORCH_TURBO_EP_SIZE` (or `--tt_sp_size` / `--tt_ep_size`)
- `TORCH_TURBO_GRADIENT_CHECKPOINTING` / `--tt_gradient_checkpointing`
- `--tt_cpu_offload` or `backend_config.cpu_offload` in the `--tt_config` file

### Checkpoints

Training checkpoints let you resume an interrupted run. Model weights are saved
in DCP format. For inference or sharing, export the model to HF format; LoRA/DoRA
weights are merged into the model during export.

```text
checkpoint-<step>/
├── model_distcp/       # Model state
├── optimizer_distcp/   # Optimizer state
├── scheduler.pt        # LR scheduler state
├── torchturbo/         # Model and PEFT configurations
└── huggingface/        # Exported HF model
```

`--tt_hf_export` controls when to export HF models. The corresponding config field
is `checkpoint.hf_export`:

- `every` (default): export an HF model with every checkpoint.
- `final`: export an HF model only at the end of training. Intermediate checkpoints still save DCP.

LlamaFactory saves the final HF model to `<output_dir>/huggingface/`.
ms-swift saves it inside the last `checkpoint-*/huggingface/`.

To resume training, keep the entire checkpoint directory, including model,
optimizer, scheduler, and training framework state. Checkpoints saved with
`save_only_model=true` contain only the HF model and cannot be used with
`resume_from_checkpoint`. To continue training from their weights, start a new
run in a new `output_dir` with `model_name_or_path` set to
`<checkpoint>/huggingface`. DCP loading
requires all state needed by the current model and ignores extra entries.
DCP also saves frozen parameters, so LoRA checkpoints need storage for the full
base model.

`<checkpoint>/torchturbo/checkpoint.json` records the source model path, the
training model config, structural engine settings (backend, mixed precision,
EP size, MTP, and freeze policy), and the PEFT adapter configs. When resuming
from DCP state, the current PEFT config must match the saved one, so adapter
structure and LoRA scaling (`lora_alpha`, `use_rslora`, `alpha_pattern`) stay
the same. Only dropout, `bias`, initialization settings, and bookkeeping fields
such as `peft_version` and `base_model_name_or_path` may differ; the freeze
policy may change as well. `scripts/checkpoint/dcp_to_hf.py` uses the file to
rebuild the model. Checkpoints without this file skip the resume check and need
explicit export arguments: `--model_name_or_path`, the
`--tt_*` settings used for training, and for LoRA, `--adapter_config` or all of
the script's LoRA options.

Use this script to convert a DCP checkpoint to an HF model:

```bash
torchrun --nproc_per_node=8 scripts/checkpoint/dcp_to_hf.py \
  --checkpoint_dir output/my_run/checkpoint-1000 \
  --output_dir exports/my_run/step-1000
```

The exported model is saved in `exports/my_run/step-1000/huggingface/`. The source
model must remain accessible; if it has moved, pass its new path with
`--model_name_or_path`. Export uses the saved expert parallelism (EP) size and MTP
settings. The total number of processes must be a multiple of the EP size.

Use a separate export directory to avoid affecting automatic cleanup of training
checkpoints. The script rejects identical input and output directories, paths
where one directory contains the other, output directories that hold training
checkpoint state, and output directories inside the source model directory.

### Supported Models

Built-in models are discovered from `src/torchturbo/models/`. Public environment setup and run instructions are currently available under `docs/runs/`:

```text
docs/runs/
└── qwen3_5_moe/
└── bailing_v3/
...
```

## Environment Variables

| Environment Variable | Description |
| --- | --- |
| `TORCH_TURBO_DISABLE=1` | Disable detection-based activation through `activate()` without an explicit adapter name |
| `TORCH_TURBO_BACKEND=fsdp2` | Select the distributed backend |
| `TORCH_TURBO_MIXED_PRECISION=bf16` | Set the mixed-precision policy |
| `TORCH_TURBO_OP_POLICY=auto` | Set the operator dispatch policy: `auto`, `native`, or `custom` |
| `TORCH_TURBO_GRADIENT_CHECKPOINTING=true` | Enable activation checkpointing |
| `TORCH_TURBO_AC_USE_REENTRANT=true` | Use reentrant activation checkpointing to reduce cross-step memory peaks for MoE and linear attention |
| `TORCH_TURBO_USE_DETERMINISTIC_ALGORITHMS=true` | Require the selected device runtime to use deterministic algorithms |
| `TORCH_TURBO_SP_SIZE=2` | Set the sequence-parallel size |
| `TORCH_TURBO_EP_SIZE=8` | Set the expert-parallel size |
| `TORCH_TURBO_GBS=64` | Set the global batch size in micro-batch units; the engine derives gradient accumulation from GBS, SP size, and world size |
| `TORCH_TURBO_RESHARD_AFTER_FWD=0` | Control whether parameters are resharded after the forward pass |
| `TORCH_TURBO_PREFETCH_STRATEGY=auto` | Set the FSDP prefetch topology: `native`, `auto` (default), `expert_lookahead`, or `expert_with_layer` |
| `TORCH_TURBO_PREFETCH_DIRECTION=forward_backward` | Set the passes in which prefetching is applied: `forward` or `forward_backward` (default) |
| `TORCH_TURBO_FORWARD_PREFETCH_DEPTH=1` | Set the number of layers prefetched for each source during the forward pass (default: 1) |
| `TORCH_TURBO_BACKWARD_PREFETCH_DEPTH=1` | Set the number of layers prefetched for each source during the backward pass (default: 1; applies only to `forward_backward`) |
| `TORCH_TURBO_DEBUG=1` | Enable additional rank-0 debug output in the MoE layer |

## Command-Line Arguments

Most environment variables have corresponding higher-priority `--tt_*` CLI arguments. All arguments are defined in [`src/torchturbo/engine/args.py`](src/torchturbo/engine/args.py) and grouped below by category:

| Category | Arguments |
| --- | --- |
| Basic configuration | `--tt_config` / `--tt_backend` / `--tt_mixed_precision` / `--tt_op_policy` / `--tt_use_deterministic_algorithms` |
| Parallelism | `--tt_sp_size` / `--tt_ep_size` |
| Batch size | `--tt_gbs` |
| Memory | `--tt_gradient_checkpointing` / `--tt_ac_use_reentrant` / `--tt_cpu_offload` |
| Prefetching | `--tt_prefetch_strategy` / `--tt_prefetch_direction` / `--tt_forward_prefetch_depth` / `--tt_backward_prefetch_depth` |
| MTP / MoE loss | `--tt_enable_mtp` / `--tt_mtp_loss_scaling_factor` / `--tt_mtp_log_loss` / `--tt_moe_aux_loss_coeff` / `--tt_moe_z_loss_coeff` / `--tt_enable_chunk_loss` / `--tt_loss_chunk_size` |
| DeepSeek-V4 DSpark (needs MTP and a DSpark checkpoint) | `--tt_enable_dspark` / `--tt_dspark_loss_scaling_factor` / `--tt_dspark_num_anchors` / `--tt_dspark_loss_decay_gamma` / `--tt_dspark_ce_loss_alpha` / `--tt_dspark_l1_loss_alpha` / `--tt_dspark_confidence_loss_alpha` |
| VLM freezing | `--tt_freeze_vision_tower` / `--tt_freeze_multi_modal_projector` / `--tt_freeze_language_model` |
| Diffusion models (LLaDA2) | `--tt_cap_lambda` / `--tt_bd_mask_low` / `--tt_bd_mask_high` / `--tt_bd_block_size` |
| Checkpointing | `--tt_hf_export` |
| Profiling, metrics, and logging | `--tt_profiling` / `--tt_profiling_config` / `--tt_memory_snapshot_steps` / `--tt_log_mfu` / `--tt_card_name` / `--tt_log_level` |

`--tt_profiling` profiles one step into `{output_dir}/profiling/`. Add
`--tt_profiling_config` to collect extra details (a JSON dict; quote it in the
shell so it survives as one token): `{"with_shape": true}` records tensor
shapes (operator- and kernel-level; raises the NPU profiler level to Level1) and
`{"with_stack": true}` records the Python call stack. Both default to false and
only take effect together with `--tt_profiling`.

DeepSeek-V4 MTP and DSpark jointly train the backbone and draft modules, including
the shared embedding and output head. DSpark stops gradients through its reference
distributions while allowing gradients through the backbone context features.

## Adding a New Model

1. Create a model directory under `src/torchturbo/models/`.
2. In `model.py`, inherit from `TurboModel` for language models or `TurboVLMModel` for multimodal models, and register the class with `@register_model()`.
3. Define the default parallel strategy in `parallel_plan.py` and return it from the class method `get_default_parallel_plan()`.
4. Add `configuration_*.py` and `modeling_*.py` if needed.
5. Add `lf_template.py` or `swift_template.py` when the corresponding framework requires model-specific registration.
6. Override hooks such as `on_parallelized()` or `on_materialized()` only when the model needs setup at those lifecycle points.
7. Add launch scripts under `scripts/runs/<model>/` and a run guide under `docs/runs/<model>/`.

Minimal wrapper class:

```python
from transformers import AutoModelForCausalLM

from torchturbo.parallel import FSDPParallelPlan

from .configuration_my_model import MyConfig
from .modeling_my_model import MyModelForCausalLM
from ..base import TurboModel
from ..registry import register_model


@register_model("my_model", config_cls=MyConfig, auto_class=AutoModelForCausalLM)
class TurboMyModelForCausalLM(TurboModel, MyModelForCausalLM):
    config_class = MyConfig

    @classmethod
    def get_default_parallel_plan(cls) -> FSDPParallelPlan:
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()

    def on_materialized(self) -> None:
        """Initialize runtime buffers after parameters are materialized."""
```

`on_materialized()` is optional; omit it when the model has no runtime buffers to rebuild. It runs after checkpoint loading or device materialization. Use `on_parallelized()` instead for setup that must happen immediately after module replacement and sharding.

## Citation

If you find this project useful, please consider citing it as follows:

```bibtex
@misc{torchturbo2026,
  title        = {TorchTurbo: A supernode-native post-training engine for rapidly adapting and training large language and vision models.},
  author       = {Tai An and Haishan Feng and Xuejin He and Chen Long and Yicheng Shan and Wei Tang and Chao Wang and Xiong Xu and Yifei Yang and Liangji Zhuang},
  year         = {2026},
  url          = {},
  license      = {Apache-2.0}
}
```

This project benefits from Megatron-LM. We thank its authors for their contributions.

## License

Apache-2.0
