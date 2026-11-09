# GLM-5.3-Flash NPU KDA

After `on_parallelized`, `glm5_3_flash` selects the NPU attention forward for each
model instance. `text_config.kda_backend="auto"` (the default) and `"fla"` use
FLA `chunk_kda` on NPU; `"torch"` explicitly selects the head-parallel PyTorch
reference core. CPU behavior is unchanged.

Use the same fixed-workspace runtime as `origin/bailing_v3_vl_sft`
(`2d186ac199e3ed7e1e4e859d668b96a8ccebea33`):

- `triton_ascend-3.2.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`
- `flash_linear_attention-0.6.0+main.954438d1-py3-none-any.whl`

Both wheels are in `/mnt/sfs_turbo/torchturbo/stable/ling_v3_vl_deps/` and the
existing `scripts/runs/npu_deps/glm5_3_flash_deps.sh` installs them. FLA 0.5.2
previously failed during KDA backward compilation on this platform. Missing FLA
raises an error; the fused path does not silently fall back to the slow core.

## Implementation

Each rank processes a disjoint head range over the full sequence and slices the
convolution weights, `A_log` and `dt_bias` to that range. Q/K/V, raw forget gate
and beta share one input SP all-to-all when their dtypes agree, followed by one
inverse output exchange: two forward collectives instead of six. Mixed-dtype
inputs retain separate exchanges without rounding beta or gate values. Entirely
empty batches also retain the separate path and its unused-gradient behavior.

All valid segments across batch rows are packed into one tensor, and each of
Q/K/V uses a single FLA Triton `causal_conv1d` call with INT32 `cu_seqlens`.
There is no Python convolution loop over segments in the default `auto`/`fla`
path. The same boundaries then go to `chunk_kda`, resetting convolution and KDA
state at every packed sample/position reset and excluding padding. Outputs are
scattered back before inverse SP exchange and the existing output projection.

The FP32 checkpoint weights remain FP32 parameters. The convolution operands
are explicitly cast as PyTorch conv1d would cast them under the active device
autocast policy. FLA convolution runs with `activation=None`, followed by SiLU:
this preserves GLM's convolution-output rounding boundary before activation.
Fusing the activation would change BF16 rounding. `cu_seqlens_cpu` is supplied to
avoid rediscovering segment lengths on the host. Position indices and cumulative
lengths are shared across layers/recomputation in an eight-entry cache keyed by
all segment boundaries, sequence length and concrete device. It stores no
activations and does not reuse metadata after segment-list mutations.

As in bailing, Q/K normalization runs outside FLA because the Triton norm
backward faults on this NPU. GLM's exact formula
`x / sqrt(sum(x*x) + 1e-6)` is evaluated in FP32 and cast back to the input dtype;
using `F.normalize` would change GLM's epsilon semantics. KDA fuses the forget
gate with the original lower-bound sigmoid or unbounded softplus formula.
The gate cumsum autotuner gets BS=64 and BS=128 configurations, following bailing,
to avoid the stock BS=32 grid limit for long sequences.

## Validation

The standalone CPU contracts require only PyTorch and pytest, avoiding training
engine/Transformers imports. They execute selected, unmodified production
definitions with CPU oracles for FLA kernels and cover packing, forward/input
and parameter gradients, mixed-dtype exchanges, FP32 convolution weights,
BF16 autocast and non-reentrant checkpointing:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/standalone/test_glm5_kda_optimization.py -q --noconftest
```

These tests do not compile or execute Triton-Ascend kernels. No NPU environment
was available when the optimization was implemented; device correctness and
performance still require the following tests in the training container.

Inside an Ascend container, source the same CANN and custom operator environment
as `scripts/runs/lf_glm5_3_flash_text_sft_lora.sh`, set `PYTHONPATH` to the repository's
`src`, and run:

```bash
GLM5_TEST_NPU=1 python -m pytest tests/models/glm5_3_flash/test_npu.py tests/models/glm5_3_flash/test_npu_kda.py -q
GLM5_SP_ONLY=kda torchrun --nproc_per_node=2 tests/models/glm5_3_flash/npu_sp_worker.py
GLM5_SP_ONLY=kda GLM5_SP_EMPTY=1 torchrun --nproc_per_node=2 tests/models/glm5_3_flash/npu_sp_worker.py
```

The SP worker defaults to devices 2 and 3; set `GLM5_SP_DEVICE_OFFSET=0` when
running with only two visible devices. Tests cover bounded/unbounded gates,
explicit `auto`/`fla`/`torch` dispatch, packed samples crossing the 64-token chunk
boundary, mixed FP32/BF16 checkpoint parameters, all parameter/input gradients,
and LoRA with 64 heads × 128 dimensions at 1024 tokens under non-reentrant
checkpoint recomputation. CPU contract tests run without importing FLA.

The NPU KDA tests also directly compare varlen convolution outputs, input
gradients and weight gradients against per-segment conv1d at lengths 1/3/65,
with FP32/BF16 weights and autocast enabled/disabled.

For A/B diagnosis, the default is varlen convolution plus merged exchanges.
Set either option independently or both to restore the corresponding reference
algorithm without changing the checkpoint or the KDA core:

```bash
export TORCHTURBO_GLM5_KDA_CONV=reference       # default: varlen
export TORCHTURBO_GLM5_KDA_EXCHANGE=separate   # default: merged
```

`kda_backend="torch"` keeps per-segment PyTorch convolution and does not require
FLA convolution. Missing FLA or device kernel errors in the default fused path
propagate rather than silently switching implementations. Keep the existing
FP32 GLM normalization and gated norm; the bailing gated norm is not substituted.
For throughput measurements, disable profiling after warmup, hold effective
batch size and valid token count constant, and compare multiple steps. Expected
structural changes are three Q/K/V convolution calls regardless of segment
count and two KDA SP forward collectives; these are not measured speedup claims.

## Sparse attention backward

The NoPE DSA path keeps its fused forward and all k-pool/tail selections. With
`index_topk=2048`, `index_kpool=4` and always-selected tails, a row has up to
2051 valid indices; the forward accepts `-1` padding to 2112 columns. The deployed
CANN sparse backward instead consumes `min(K, causal_length)` entries without
stopping at `-1`. A 3116-token exception dump showed its first invalid consumed
entry at query row 2051 (zero-based), followed by an MTE DDR-address fault.

`_NpuNopeSparseAttention` sends only the first `index_topk` columns to fused
backward. Every row in this prefix has the expected number of valid causal
indices. The original forward's output, softmax maximum and softmax sum are
passed unchanged: this preserves the gradient interaction with the remaining
selected tokens. Their Q and shared-K/V gradients are added in FP32 query chunks,
with `dS_i = P_i * (dO @ V_i - dO @ O)` using the full forward's normalization.
This avoids a full reference backward, dropping tails, or changing attention
weights by duplicating indices. No CANN installation or launch-script changes
are required.

The fused path requires an `index_topk` from 1024 through 8192 in multiples of
1024, divisible by the pool size, and always-selected tails. Other index
configurations retain the reference forward. Existing dtype/shape guards and
full-sequence SP gathering remain in effect.

CPU regression tests check the consumed-prefix rule, dense-reference outputs
and gradients at 2051/2052/3116 tokens, BF16/autocast, and LoRA under non-reentrant
checkpointing. Run them with:

```bash
PYTHONPATH=src python -m pytest tests/models/glm5_3_flash/test_npu_sparse_backward.py -q
```

The same module includes opt-in real-NPU backward comparisons at 2051, 2052 and
3116 tokens with 64 heads and a 512-dimensional shared K/V. These require the
training container's CANN/custom-operator environment and `GLM5_TEST_NPU=1`;
passing CPU tests does not substitute for this device validation.
