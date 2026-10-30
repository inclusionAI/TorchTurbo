# Kimi K3 (Kimi Linear) support for TorchTurbo

Multi-modal MoE model: hybrid KDA / Gated MLA attention, AttnRes,
Stable LatentMoE (896 experts), MoonViT-V2 vision tower.

## Install

The modeling code hard-depends on `flash-linear-attention` (fla); the import
in `modeling_kimi_linear.py` asserts it at load time and points here on
failure:

```bash
pip install "flash-linear-attention>=0.6.0"
```

## Verified environment matrix (Ascend NPU)

| Component | Verified version | Notes |
|---|---|---|
| flash-linear-attention (fla) | 0.6.0 | `kda_implementation=fused` (Triton) verified: 10-step smoke, loss matches naive |
| triton-ascend | 3.2.2 | required for the fused path; **3.2.1 + fla 0.5.2 deadlocks (aicore timeout)** |
| transformers | >= 4.56.0 | asserted at import |
| NPU stack | any image with the two versions above | e.g. recent Ascend training images |

Old NPU stacks without a working triton-ascend: set
`kda_implementation: naive` in the model config to use the pure-PyTorch
fallback (`torchturbo/ops/kda/chunk_kda_naive.py`).

## Smoke baseline

2-layer / 896-expert cut model, 16 NPU, SP=1/EP=16, 10 steps:
loss 13.76 -> 8.67, grad_norm 15-25 — reproduced across SP=1/SP=2,
EP=8/EP=16, Triton/naive KDA, CPU offload on/off, AC on/off.

## Local patches for known issues (not upstreamed)

The patches below are intentionally NOT part of this PR: they touch
shared engine code and currently lack a verified reproducer on the smoke
matrix. Apply them locally if you hit the symptom; each will be
upstreamed separately once it has a reproducer.

### 1. Full-checkpoint placement on NPU breaks low-memory loading

Symptom: with `transfer_to_npu` active (the NPU runtime default), HF
`from_pretrained` materializes the whole checkpoint on-device before
FSDP sharding — OOM for large models (e.g. full 69-layer K3 on 64GB
cards) even though the engine's distributed loader
(`DistParamLoader`) would stream shards instead.

Patch `src/torchturbo/runtime/npu.py`, in `NpuRuntime.initialize`:

```python
if os.environ.get("TORCHTURBO_DISABLE_TRANSFER_TO_NPU", "") != "1":
    from torch_npu.contrib import transfer_to_npu as _  # noqa: F401
```

Then launch with `TORCHTURBO_DISABLE_TRANSFER_TO_NPU=1`.

### 2. `nested_gather` missing in newer transformers

Symptom: `ImportError: cannot import name 'nested_gather' from
'transformers.trainer'` raised from `_report_engine_metrics`.

Patch `src/torchturbo/adapters/llamafactory_v0/trainer.py`:

```python
try:
    from transformers.trainer import nested_gather

    def gather_fn(tensors):
        return nested_gather(tensors, self.args.parallel_mode)
except ImportError:
    def gather_fn(tensors):  # report local metrics only
        return tensors
```

## Known limitations

- **Multimodal token accounting is a placeholder**: the LlamaFactory
  mm_plugin currently reports 1 token per image
  (`_get_image_token_count` returns 1). Text-only SFT is unaffected.
  For image-containing data, packing/truncation budgets will be
  inaccurate until this lands in a follow-up — the K3 vision processor
  already exposes the real count via
  `get_resize_config(media)["num_tokens"]`, so the fix is a small
  plugin change.
