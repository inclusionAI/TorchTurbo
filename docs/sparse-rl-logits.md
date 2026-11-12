# Native sparse logits for masked RL losses

## Scope and ownership

This feature lives in TorchTurbo, not in a training-launcher patch. The engine
selects output-head rows, computes sampled-token log-probabilities and entropy,
and restores the packed layout with its existing sequence-parallel autograd
collectives. A calling framework supplies the configuration, packed masks, old
log-probabilities, and an explicit masked-loss contract. TorchTurbo has no dependency on a training framework.

The initial model implementation is
`TurboQwen3_5ForConditionalGeneration` on **text-only** inputs. Other models must
implement `TurboModel.forward_sparse_logits`; the base implementation fails
before projecting dense logits. This does not enable Qwen3.5 MoE or arbitrary
models by model-name matching.

The supported layouts are SP1 and contiguous Ulysses SP8, with EP1. The feature
is opt-in and applies only to `forward_backward_batch` training. Ordinary model
forward, forward-only evaluation, `compute_logprobs`, and SFT paths retain their
existing behavior. No runtime method replacement, AST rewriting, import hook,
source-hash allowlist, or context-global training switch is installed.

## Why selecting rows is equivalent

A masked objective reads only the predictions at active mask positions:

```text
full input context -> transformer -> hidden states [T, H]
                                  -> select active predictor rows [R, H]
                                  -> lm_head -> logits [R, V]
```

Every selected row still uses the **entire vocabulary**. Prompt and tool-result
tokens remain transformer inputs; their representations can influence selected
predictions and receive gradients through attention. This is not context
truncation, vocabulary pruning, or sampling fewer loss tokens. Equality holds
only when the caller's entire objective depends on the selected positions.

For packed sequences of lengths `L1, L2, ...`, `ppo_loss_mask` has length
`sum(Li - 1)`. It describes next-token targets. The selected hidden state is the
**preceding predictor**, not the target token itself. No predictor crosses a
packed-sequence boundary. The engine checks the rolled labels on SP shards.

The vocabulary-sized output allocation becomes proportional to `R * V`
instead of `T_local * V`. Existing chunked/checkpointed logprob and entropy
computation bounds additional FP32 activations. Transformer activations, model
weights, optimizer state, and inference KV caches are not reduced by this
operation. Dense masks may give little benefit. No production memory reduction
is claimed from the CPU correctness tests.

## Configuration and caller contract

```python
from torchturbo.engine import EngineConfig, SparseLogitsConfig
from torchturbo.engine.config import LogprobComputationConfig

config = EngineConfig(
    sparse_logits=SparseLogitsConfig(enabled=True),
    logprob_computation=LogprobComputationConfig(
        chunk_size=32, checkpoint_chunks=True,
    ),
)
config.mtp.enabled = False
# Use config when constructing and preparing the ordinary TorchTurboEngine.

engine.forward_backward_batch(
    micro_batches,
    process_fn=None,
    loss_fn=masked_loss,
    loss_weight_fn=masked_loss_weight,
    loss_uses_ppo_mask=True,
)
```

The default is disabled. Logprob computation defaults to `chunk_size=1024`
and `checkpoint_chunks=False`; the example opts into smaller checkpointed chunks.
These settings are available through Python dataclasses and YAML only, with no
additional CLI flags, environment variables, or `TorchTurboArgs` fields.

These fields also accept the normal config-file form:

```yaml
engine_config:
  sparse_logits:
    enabled: true
  mtp:
    enabled: false
  logprob_computation:
    chunk_size: 32
    checkpoint_chunks: true
```

- The original micro-batch contains full packed `input_ids`, `cu_seqlens`,
  `ppo_loss_mask`, and finite `old_logp`. Both mask and old logprobs have shifted
  packed shape `[sum(Li - 1)]`. Existing packed/SP preparation remains in use.
- `loss_uses_ppo_mask=True` explicitly asserts that policy **and entropy** loss
  read only nonzero mask positions. Raw-logit callbacks and auxiliary losses
  are rejected. It is not inferred by inspecting a Python closure.
- At inactive positions, returned logprobs equal detached `old_logp`, making
  unused PPO ratios finite and equal to one. Entropy is zero. These values are
  **not current-policy evaluations** and must not feed unmasked losses or
  unmasked metrics. Use dense evaluation when those values are needed.
- Every loss-bearing row in each SP shard is projected, including counts
  above 4096. There is no fixed row cap, silent dense fallback, or token
  truncation. `SparseLogitsConfig` only accepts `enabled`; remove the retired
  `max_rows` key from existing configurations and update framework bridges
  together with TorchTurbo.
- The output head still materializes all selected logits at once. Removing
  the row cap does not implement head chunking or guarantee memory safety for
  large vocabularies and long trajectories.
- A shard with no active rows projects one dummy row. Its zero-valued dependency
  keeps the backbone/head backward and SP collective participation intact.
  Entirely masked batches are supported. All-singleton packed batches are not.
- Nonempty image/video inputs, incompatible packed shapes, enabled MTP losses,
  unsupported SP/EP layouts, and training with disabled gradients fail clearly.

### MTP behavior

The Qwen outer forward gains `compute_mtp: bool = True`. The ordinary path keeps
its previous default. The native sparse model API calls the normal module
`__call__` with `compute_mtp=False` and `logits_to_keep=<selected rows>`, preserving
FSDP/module hooks. MTP modules, parameters, state-dict keys, and training mode
are retained. Only the unused auxiliary computation is skipped; MTP is not
deleted or globally disabled by changing model mode.

## Tests and limitations

```bash
TORCH_TURBO_DISABLE=1 PYTHONPATH=src python -m pytest -q \
  tests/engine/test_sparse_logits.py tests/engine/test_logprob_computation.py \
  tests/engine/test_engine_config_loading.py

TORCH_TURBO_DISABLE=1 PYTHONPATH=src python \
  tests/engine/fixtures/sparse_distributed.py \
  --world 8 --backend gloo --mtp --out /tmp/native-sparse-sp8
```

The unit suite checks dense/sparse active logprobs, entropy, masked objectives
and all parameter gradients, including 4097, 5117 and 8192 active rows. It also
covers packed boundaries, padding, singleton mixtures, empty/all-active masks,
configuration errors, and the caller contract. The real Qwen wrapper/outer
forward is tested with a small causal backbone, including MTP state preservation,
module hooks, tuple outputs, two optimizer steps and unchanged dense evaluation.

The distributed fixture uses engine/backend packed-input preparation and the
model-declared SP slicing boundary. It checks uneven and empty shards on eight
CPU/Gloo workers. These are small-model correctness tests, not a full Qwen
backbone or production memory benchmark.

Full-size policy training, accelerator memory, BF16/HCCL, FSDP2 and cross-node
acceptance require separate validation. The fixture's accelerator options do not
imply those modes have been validated. Removing the row cap does not guarantee
that larger micro-batches or more active rows will fit in device memory.
