GLM-5.3-Flash NPU k-pool indexing uses precompressed SP communication by default.

Each complete pool belongs to the SP rank containing its last token. Pool
grouping starts at each packed segment's first token, rather than at a rank
boundary. The owner collects any preceding raw K/Gate tokens that cross rank
boundaries, applies the original FP32 softmax and dtype rounding, and sends
only the pooled key in the main all-gather. Short segments do not participate
in indexer projection/communication when the whole batch can skip scoring.

Collective branches and padded sizes come from the same global segment layout
on every rank. Ranks with no local pool still participate, and a pool can span
more than two ranks. Cached metadata contains only layout-dependent indices;
projected tensors and scoring results are never cached across forwards.

For aligned long sequences with pool size p, the main collective transfers
1/(2*p) as many elements as gathering token K and Gate, with the same dtype.
Boundary exchanges and padding for uneven pool ownership add overhead.

Set `TORCHTURBO_GLM5_KPOOL_EXCHANGE=tokens` on **all SP ranks** to use the
previous token K/Gate exchange for A/B comparisons. The default is `pooled`.
This option changes only communication placement, not the selected set.

`kpool_indices` returns valid token IDs in ascending order followed by -1
padding. It sorts K/p selected pool IDs once for the whole query slice, then
expands them into contiguous token IDs and appends the incomplete causal tail.
Full DSA layers pass this local sorted result directly to sparse attention.
Shared layers retain token sorting because inputs may come from a reference
fallback. TopK score computation, causal tile boundaries, and candidate layout
are unchanged, including tie behavior before the final index ordering.

CPU validation, without Transformers or Ascend imports:

```bash
python -m pytest tests/standalone/test_glm5_kpool_optimization.py \
  tests/standalone/test_glm5_kpool_sp_optimization.py \
  --noconftest -p no:cacheprovider -q
```

Distributed NPU validation against a saved original `npu.py`:

```bash
torchrun --nproc_per_node=4 --master_port=29543 \
  tests/standalone/glm5_kpool_sp_worker.py --device npu \
  --source src/torchturbo/models/glm5_3_flash/npu.py \
  --baseline /absolute/path/to/original/npu.py
```

The same worker supports `--bench --lengths 65536 131072` to measure pooled
communication/preparation and index sorting separately, using real indexer
dimensions on 16 ranks. It reports the median of five synchronized runs and
the slowest rank in each run; these are microbenchmarks, not full-step timing.

For LoRA sparse-attention output, gradient and checkpoint comparisons:

```bash
PYTHONPATH=src:/opt/MindSpeed torchrun --nproc_per_node=2 --master_port=29545 \
  tests/standalone/glm5_kpool_attention_worker.py \
  --source src/torchturbo/models/glm5_3_flash/npu.py \
  --baseline /absolute/path/to/original/npu.py \
  --output /absolute/path/to/attention_results.json
```

Full training uses `scripts/runs/lf_glm5_3_flash_text_sft_lora_8.sh` with
`SEQ_LENGTH=65536` or `131072`. A batch packed from only short segments does
not exercise long-sequence indexer scoring; use a real long segment when
validating this path. Keep weights, inputs, SP topology and random seed fixed
for before/after comparisons.
