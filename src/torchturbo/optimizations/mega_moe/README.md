# MegaMoE with MoonEP

Enable the BF16 NPU routed-expert backend with `--tt_mega_moe true` in
`scripts/runs/run_lf.py`, or `TORCH_TURBO_MEGA_MOE=1` (environment overrides CLI).
The switch defaults to false; disabled runs do not require native components.
See `docs/glm5_3_flash_megamoe.md` for the GLM full-parameter launcher.

The backend retains the model router, shared experts, expert parameter ownership,
and pre-activation SwiGLU clipping. Expert LoRA wrappers and TE are unsupported.
It installs after module replacement/PEFT and before checkpoint/FSDP wrapping.

Reentrant checkpointing retains the shared-weight fast path. With checkpointing
disabled or non-reentrant, backward refetches borrowed experts using the saved
placement and FSDP-restored parameters. This adds communication without retaining
a borrowed-weight copy for every layer. Use the same checkpoint mode throughout
configuration and invocation; GLM's launcher retains non-reentrant checkpointing.

## Runtime dependency

Provision `tt-megamoe==0.1.3+author.2094e90` and `cann-shmem==1.6.0` in the
container image or its startup script before launching training or verification.
TorchTurbo does not bundle wheels or install dependencies at runtime.

The matching environment is Linux aarch64, Python 3.11, Torch 2.10.0,
torch_npu 2.10.0.post4, CANN 9.1 and CXX11 ABI=1. The author 2094e905 routing
library must be built and installed with the runtime; the older allocator ABI
is incompatible. The source-bearing component wheel requires
`python -m tt_megamoe.build_allocator` with `bisheng` after installation.

## Representative verification

On one allocated 16-NPU worker with the training dependencies and pytest:

```bash
bash tests/npu/run_mega_moe_16card.sh
```

This runs two checks:

1. `test_mega_moe_planner.py`: EP64/384-expert summary, planner tables and virtual
   routing on one NPU, compared with an independent integer reference.
2. `mega_moe_precision.py`: two layers sharing buffers, DP2 x EP8 FSDP2 with
   resharding, and 384-expert hot routing. Compare output and input/router/expert
   gradients against ordinary operations with checkpointing disabled, non-reentrant
   and reentrant. A 16 KiB planner budget forces paging. The test requires actual
   expert gradient returns, so bypassing borrowed experts cannot produce a pass.

`OUTPUT_DIR` selects the log directory. Success ends with `VERIFY_PASS` and
16 passing `PRECISION_RESULT` entries for each checkpoint mode. Failures stop later stages. This validates
representative MoE behavior; full SFT, EP64 cross-node transport and performance
still require the full-model experiment.
