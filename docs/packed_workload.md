# Automatic packed-workload balancing

Cached packing datasets are balanced automatically during dataloader creation
when an explicit global batch size identifies optimizer-step boundaries and
there is more than one DP rank. No feature toggle is required.

Ordinary datasets, mixed packed/unpacked dataset wrappers, single-DP execution,
and legacy layouts without an explicit global batch size keep the original
sampler. This prevents interpreting an unknown block as an optimizer step.

Metadata recognition uses the cached packing contract (`dataset`, `packed_idx`,
`packing_length`) and supports `ConcatDataset` and `Subset`. It does not load
sample tensors or repeat tokenization. Scheduling preserves optimizer-batch
membership and each existing microbatch's sample grouping.

This is CPU-side, backend-independent scheduling. The cost estimate is a proxy;
actual throughput gains depend on the model and length distribution. Floating
point reduction order and random-number assignment can differ after reassignment,
so bitwise full-training equivalence is not promised.

```bash
python -u tests/run_packed_workload_tests.py >test.log 2>&1 &
```
