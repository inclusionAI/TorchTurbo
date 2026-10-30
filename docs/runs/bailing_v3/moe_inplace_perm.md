# MoE In-Place Chunk Permutation (`moe.inplace_perm`)

## What it does

In the MoE all-to-all token dispatcher, received tokens are reordered twice
per layer: once to group chunks by local expert after the dispatch all-to-all,
and once to restore the original order before the combine all-to-all. The
stock path performs this reorder with `split + cat`, which allocates a full
sorted copy of the token buffer while the unsorted buffer is still alive.

`moe.inplace_perm` replaces the copy with an in-place cycle-rotation reorder
that uses a single spare slot sized at the largest chunk. Peak temporary
memory at the reorder step drops from one full token buffer to a few MiB,
while the reorder result stays bit-exact identical to `split + cat`.

## When to enable it

Enable it when training runs out of memory at the MoE dispatch/combine
reorder. This happens with dropless routing at large EP sizes, where skewed
expert loads inflate the per-rank token buffer (observed at EP=64, where the
coexisting sorted copy exceeded free HBM by ~2 GiB).

At small EP sizes the spike is modest and the option is memory-neutral to
mildly beneficial; it is off by default and can be left off when memory is
not a concern.

## How to enable

Add to the engine config YAML:

```yaml
engine_config:
  moe:
    inplace_perm: true
```

The option applies to the `alltoall` MoE token dispatcher used by the
bailing_hybrid models and is honored in both the deferred-probs (NPU) and
the non-deferred dispatch paths.

## Behavior and constraints

- The reorder output is bit-exact identical to the stock `split + cat` path;
  no numerics change.
- The in-place reorder is autograd-safe: backward walks the same permutation
  in reverse through a custom `torch.autograd.Function`.
- Rows are moved in place, so callers must not hold views of the pre-reorder
  buffer across the call (the dispatcher owns both buffers at these sites).
- The option is independent of `moe.permute_fusion` and can be combined with
  it.

## Tests

`tests/models/test_inplace_perm.py` checks the reorder against
`index_select` for forward and backward, and verifies that backward restores
the original gradient order.
