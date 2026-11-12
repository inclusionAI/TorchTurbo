# Optional training garbage collection freeze

Large training processes retain many Python objects after model and optimizer
initialization. Repeated cyclic GC scans of those objects can pause a rank and
make its peers wait at distributed operations. The optional `freeze` mode
excludes initialization survivors from later scans during a trainer invocation.
It does not change model arithmetic or introduce per-step collection.

`auto` is the default and preserves Python's existing GC behavior for all models.
Enable `freeze` explicitly with either CLI or YAML:

```bash
--tt_gc_mode freeze
```

```yaml
engine_config:
  garbage_collection:
    mode: freeze
```

The ms-swift and LlamaFactory adapters share one implementation. A temporary
trainer callback runs `gc.collect()` and `gc.freeze()` at `on_train_begin`, after
model and optimizer setup. An enclosing context removes the callback and
unfreezes an owned frozen heap on normal completion or exception. A caller's
existing frozen heap is left unchanged. Automatic collection settings and
thresholds stay unchanged, and objects created after freezing remain subject to
normal GC. Custom training loops can use the same runtime context directly.

Freezing affects the whole Python process, including its other models and
threads. Cycles involving frozen objects may remain alive until unfreezing, so
measure long-run host and device memory before adopting it for another workload.
Overlapping training scopes must be nested; concurrent trainers in one process
are not independently isolated by this option.

See [Python's GC documentation](https://docs.python.org/3/library/gc.html#gc.freeze)
for the underlying mechanism. This opt-in feature does not establish a speedup
or convergence result for every model.
