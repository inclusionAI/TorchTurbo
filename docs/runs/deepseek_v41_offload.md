# DeepSeek V4.1 offload

This follows internal `dsv41_sft_rl@e49ff83a`. BF16 CPU tables are partitioned
by EP row owner, with physical rows aligned to 2048. Requested unique rows
are routed to owners and returned through all-to-all. The whole Engram module
retains its own checkpoint/no-shard boundary and original gate precision.
There is no node-shared full-table or WORLD gradient replication backend.

```yaml
engine_config:
  activation_checkpointing:
    enabled: true
    use_reentrant: false
    offload_input_to_cpu: true
    offload_large_inputs_to_cpu: false
  backend_config:
    cpu_offload:
      enabled: true
      mode: optimizer
```

First-input mode: `--tt_ac_offload_input_to_cpu`. The original large-input
mode, `--tt_ac_offload_large_inputs_to_cpu`, requires reentrant checkpointing
and uses size-gated async pinned copies and events. These modes are not
interchangeable. V4.1's cross-layer graph requires non-reentrant checkpointing;
use first-input mode with that configuration. The original large-input
contract is retained without a replacement algorithm.

Enable table training with `--tt_train_engram true`. Actual table
`requires_grad` controls autograd. Gradients return to EP owners and merge
across EDP replicas with original averaging. ActiveRowAdam retains visited-row
FP32 moments, follows dense AdamW LR/betas/eps and shares global L2 clipping.
Tables are excluded from dense optimizer groups. Original BF16/FP32 and
no-full-CPU-offload training constraints remain.

Use engine model/optimizer/scheduler checkpoint entry points. Trainable table
weights use EP-stamped DCP keys and `model_distcp/engram_layout.json`. Sparse
state uses checkpoint-root `engram_optimizer_ep<N>.pt`, including owner
geometry and hash fingerprint. HF export streams disjoint EP ranges into
BF16 tables with logical checkpoint shapes.

The NPU memory limit uses `--tt_npu_memory_fraction` or the original legacy
environment variable. MegaMoE uses YAML `engine_config.mega_moe`; adapters
derive `config.extra['mega_moe_local_tokens']` from input bounds.

See [alignment details](../dsv41-internal-alignment.md) for stage boundaries,
public infrastructure adapters, source checksums and pending NPU checks.
