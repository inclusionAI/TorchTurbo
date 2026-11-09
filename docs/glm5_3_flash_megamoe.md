# GLM-5.3-Flash MegaMoE / MoonEP

The optional BF16 NPU backend retains `Glm5NextGroupedRouter`, shared experts,
HF checkpoint conversion, and existing FSDP/EP parameter ownership. Only routed
expert dispatch/compute/combine is replaced. The default is disabled.

The backend is adapted from TorchTurbo commit
`24943c3937cd1600eab32525f029dd69c3c9c1eb` and its integration precursor
`9bc327961961df9eea842870e5d92b068af3fe93`. This branch retains the existing
three-result router protocol and CLI configuration system.

## Runtime

Provision outside the repository before training:

- Linux aarch64, Python 3.11, Torch 2.10.0, torch_npu 2.10.0.post4, CXX11 ABI=1.
- CANN 9.1 development toolkit and the matching device runtime.
- `tt-megamoe==0.1.3+author.2094e90`, `cann-shmem==1.6.0`.
- The allocator built with the selected toolkit:

```bash
source /path/to/cann-9.1/set_env.sh
python -m tt_megamoe.build_allocator
```

Keep the model's KDA dependencies from `docs/glm5_3_flash_npu_kda.md` available.
The validated environment uses Triton-Ascend 3.2.2 and the fixed-workspace FLA
`0.6.0+main.954438d1` wheel. A mixed stock-Triton/Ascend installation can report
the distribution as installed while `triton._C.libtriton.ascend` cannot import;
provision a clean package directory in that case.

The source-bearing component wheel does not contain the compiled allocator.
Neither dependencies nor allocator compilation occur during training.
The LF entry point prepares the component's native library paths before importing
Torch/CANN. Direct engine callers must prepare those paths before starting Python.
Missing allocator/native libraries fail during installation before loading model
weights. EP=1 uses the ordinary local NPU grouped-MoE implementation and does not
validate the distributed fused kernel or MoonEP transfers.

## Launch full-parameter SFT

```bash
SP_SIZE=2 EP_SIZE=8 SEQ_LENGTH=2048 MAX_STEPS=2 \
MODEL_NAME_OR_PATH=/mnt/sfs_turbo/pretrained_models/GLM-5.3-Flash-8 \
bash scripts/runs/lf_glm5_3_flash_text_sft_megamoe.sh
```

The launcher defaults to one 16-NPU worker, the 8-layer checkpoint, full-parameter
training, BF16 and non-reentrant checkpointing. It accepts ordinary LF arguments
at the end. Set `LLAMAFACTORY_SRC_DIR`, `DATASET`, `OUTPUT_DIR`, `NNODES`,
`DATASET_DIR`, `NODE_RANK`, `MASTER_ADDR`, `NPUS_PER_NODE`, `SP_SIZE`, `EP_SIZE`,
`MICRO_BATCH_SIZE`, `GRAD_ACCUM_STEPS` as needed. `CANN_ENV_SH` optionally selects
the toolkit. `MEGAMOE_RUNTIME` optionally points to an isolated installation whose
packages are in its `python/` subdirectory.

For an existing full-parameter invocation through `scripts/runs/run_lf.py`, add
`--tt_mega_moe true`. `TORCH_TURBO_MEGA_MOE` overrides the CLI flag, consistently
in early bootstrap and EngineConfig. False/absent leaves the ordinary MoE path
and requires no MegaMoE component. Expert LoRA wrappers and TE are unsupported.

## Model and lifetime contracts

- GLM uses 288 routed experts, top-8, FP32 sigmoid routing, selection-only
  correction bias, normalized uncorrected probabilities and scaling 2.5.
- `hidden_size=4096`, expert intermediate size 2048, pre-activation SwiGLU
  limit 10, and one shared expert retain their original meaning.
- EP must divide both the world size and 288. EP8/16/32 are valid expert counts;
  EP64 is invalid for this model. Mesh and transport support must also be checked.
- Installation occurs after module replacement/PEFT and before checkpoint/FSDP.
  Calls still enter `experts(...)`, preserving expert FSDP hooks.
- Shared workspaces require matching routed-expert geometry across layers.
- Non-reentrant/no-checkpoint backward refetches borrowed weights from the saved
  placement after FSDP restores local parameters; gradients return to owners.
  Reentrant checkpointing uses the reference shared-weight lifetime path.
- LF capacity includes neat-packing's extra token, padding to 8, SP padding,
  batch size, evaluation/prediction batch size, and final 128-token alignment.
  Example: cutoff=2048, neat packing, SP2, batch1 gives capacity 1152.
- Direct engine callers set `config.extra['mega_moe_local_tokens']` to a positive
  128-aligned bound before preparing the model. Larger actual inputs fail.
- Approximate cross-layer shared weight-pool budgets at the real GLM dimensions
  are 3.375/1.6875/0.84375 GiB per rank for EP8/16/32, excluding execution
  workspaces, model/optimizer state and FSDP communication buffers.

## Verification

Configuration/bootstrap/routing contracts:

```bash
PYTHONPATH=src TORCH_TURBO_DISABLE=1 python -m pytest -q \
  tests/models/glm5_3_flash/test_megamoe_contract.py
```

With the matching component/runtime, verify the original GLM router, shared and
routed expert output/input/router/expert gradients using 288 experts:

```bash
PYTHONPATH=src TORCH_TURBO_DISABLE=1 OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tests/npu/glm5_megamoe_precision.py
```

The reference holds all experts and uses reduced H/I=256 by default. `--hidden`
and `--intermediate` select larger dimensions; actual GLM dimensions substantially
increase CPU/NPU memory requirements. A one-process invocation only verifies
local routing. Success prints `GLM_MEGAMOE_RESULT` with `status=PASS` per rank.
The relative L2 threshold is 3% for output and all checked gradients.

For borrowed expert gradients, two-layer shared-buffer lifetime, DP2×EP8 FSDP2
resharding and all three checkpoint modes:

```bash
for mode in none nonreentrant reentrant; do
  PYTHONPATH=src TORCH_TURBO_DISABLE=1 OMP_NUM_THREADS=1 \
  python -m torch.distributed.run --standalone --nproc_per_node=16 \
    tests/npu/mega_moe_precision.py --experts 288 --tokens 256 --top-k 8 \
    --planner-ub-bytes 16384 --mode pre_activation --limit 10 \
    --routing hot --identity --fsdp --checkpoint-mode "$mode"
done
```

The retained author runner `tests/npu/run_mega_moe_16card.sh` separately uses
384 experts and logical EP64 planner matrices. It does not establish GLM
full-model accuracy, actual-size compute, EP32 cross-node transport or speedup.
Record correctness before comparing warmed-up training time and peak HBM.

The component precision reference accumulates weighted outputs in FP32 and
rounds once, matching the native unpermute operator. Repeated BF16 `index_add`
rounds once per expert; with two layers and limit=10 this can change downstream
clipping masks. The 2% component gradient threshold is unchanged. GLM's separate
precision test retains the original model's forward as its reference.

## Recorded validation: 2026-11-03

Multi-NPU checks ran in the vllm container on `33.215.119.173`, with CANN 9.1,
Torch 2.10.0 and torch_npu 2.10.0.post4. The optional packages were installed in
`work/megamoe_20261103/runtime/python`; the allocator build source was verified
as `2094e90519b7efe77825306abf34d17a0b8de31b`.

| Check | Result |
| --- | --- |
| CLI/config/LF capacity/bootstrap/FSDP ordering/weight conversion | 59 passed; 4 existing CPU grouped-MoE cases deselected |
| Logical EP64/384 planner, allocation tables and virtual routing | 1 passed |
| GLM router/shared/routed experts, E=288, H/I=256, EP8 | 8/8 ranks passed; maximum relative L2 0.550% |
| GLM E=288, H=4096, I=2048, EP8 | 8/8 ranks passed; maximum relative L2 0.551% |
| Two layers, DP2×EP8 FSDP2, hot routing, limit=10, planner budget 16 KiB | 16/16 ranks passed in each checkpoint mode: none/non-reentrant/reentrant; maximum relative L2 0.582% |
| Borrowed expert backward | At least 2 gradient-return operations per rank in each checkpoint mode |

The older FSDP test file has 12 failures reproduced against the unmodified HEAD;
they concern mock-model `get_norm()` and logger expectations, not this backend.
The new installation-order test and the updated checkpoint-order fixture pass.

An 8-layer checkpoint completed two full-parameter SFT steps in both modes,
using 16 NPUs, SP2/EP8, sequence length 128, BF16 mixed precision,
non-reentrant checkpointing, synthetic ShareGPT data, seed 42 and learning rate
1e-5. Both runs exited zero and saved HF checkpoints.

| Step | MegaMoE loss | Ordinary MoE loss | MegaMoE grad norm | Ordinary grad norm |
| --- | --- | --- | --- | --- |
| 1 | 11.1875 | 11.1875 | 34.210300 | 34.208157 |
| 2 | 9.5625 | 9.5625 | 37.836132 | 37.848614 |

The maximum grad-norm relative difference is 0.033%. The enabled run logged the
actual `MegaMoE + MoonEP initialized` backend on EP8 with capacity 128 and
pre-activation limit 10. Logs, JSON summaries, the smoke-run script and derived
checkpoints are under `work/megamoe_20261103/cann91/`.

These short runs include different compilation/warmup costs and are correctness
checks. Steady-state speedup, the full 45-layer model, long sequences and EP32
cross-node transport have not been measured.
