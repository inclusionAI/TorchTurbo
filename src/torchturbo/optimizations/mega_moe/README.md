# MegaMoE + MoonEP

`engine_config.mega_moe: true` enables the fused MoonEP + MegaMoE backend.
MoonEP balances each microbatch from its current routing counts; MegaMoE overlaps
token transfers with grouped GEMMs, SwiGLU and output combination. The PR contains
these operators, their scheduling code, TorchTurbo integration and relevant tests.

## Build and install

Requirements: Linux aarch64, Python 3.11, PyTorch 2.10 with matching torch_npu,
CANN 9.0.0 and Ascend 910_93. Use an environment with TorchTurbo and its normal
NPU dependencies installed. Routing metadata and selective gradient initialization
are compiled as AscendC kernels; reductions and token indexing use existing NPU operators.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/runs/deps/mega_moe.sh --jobs 16
```

The dependency script installs official `cann-shmem==1.6.0` and compiles the
MegaMoE and MoonEP operators into this feature's `runtime` directory. Run it
when preparing the training environment. Training startup never invokes pip,
downloads or compilation; it validates dependencies and activates loader paths.
No separate MegaMoE wheel is required.

The Ascend image builds and installs the operators after its final Torch build:

```bash
docker build -f docker/npu/Dockerfile_aarch64 -t torchturbo:npu .
```

The official SHMEM wheel omits internal headers needed to compile device kernels.
The build obtains headers from the exact official commit recorded by that wheel:
`aee374a396b389626f68fab74ec8c555e18978a7` (v1.6.0). They stay in the ignored build
directory. SHMEM itself is neither rebuilt nor patched nor bundled in this PR.

For an offline build, install the unmodified official wheel first, then provide
an official SHMEM Git checkout containing that commit:

```bash
python -m pip install --no-deps /path/to/cann_shmem-1.6.0-cp311-cp311-linux_aarch64.whl
python src/torchturbo/optimizations/mega_moe/build.py \
  --shmem-source /path/to/cann-shmem-checkout --jobs 16
```

Build with the same Python environment used for training. Generated libraries
link to that environment's official SHMEM runtime. Source provenance and the
validated official wheel hash are in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Enable

```yaml
engine_config:
  mega_moe: true
```

Swift activates the native OPP and library paths before importing Torch. Direct
API callers can call `activate()` from `runtime/bootstrap.py` via `runpy.run_path`
before importing Torch; it reexecutes the process once with the required paths.
Set `extra.mega_moe_local_tokens` to a positive 128-aligned bound. Swift derives
it from packing/max length, batch size and SP size.

The backend executes inside existing expert FSDP hooks. Parameters, checkpoint
names, optimizer ownership and selected experts remain unchanged. Router weights
are cast at the same boundary as the original dispatcher. Short inputs receive
zero-weight padding; inputs exceeding the declared bound raise an error.
EP1 uses local routing and grouped GEMMs because no remote expert transfer exists.

Experts must use BF16 SwiGLU with probabilities applied to outputs. Layers sharing
scratch must have matching geometry and clipping convention. Unclipped, pre-SiLU
clipping and post-SiLU clipping preserve distinct forward/backward semantics.
Shared scratch supports both reentrant and non-reentrant activation checkpointing,
as well as training without checkpointing. Autograd saves home parameters and route
placement rather than mutable borrowed-weight views. Backward reuses matching
prefetched weights or restores them when another layer has overwritten the slots.
The usual reentrant forward/recompute/backward path needs no extra transfers.


LoRA is not supported by MoonEP + MegaMoE. When a PEFT configuration is
provided together with `mega_moe: true`, model preparation emits a warning
and disables MegaMoE before installing its backend. Training uses the native
MoE modules with LoRA enabled, regardless of adapter dropout or target modules.

CPU parameter and optimizer offload use the existing FSDP and optimizer policies.
The communication workspace and SHMEM buffers remain on the NPU.

### Operator dispatch and configuration

Expert execution calls `ops.mega_moe(...)` inside the original `experts(...)`
module call. The existing dispatcher selects `moonep_mega_moe` on NPU with
`ops.policy: auto` or `custom`; `native` uses the PyTorch token-all-to-all and
SwiGLU baseline. Explicit `impl=` and `ops.overrides.mega_moe` retain the normal
strict dispatcher behavior. Provider discovery does not initialize SHMEM or
allocate device buffers. The NPU provider checks official SHMEM and native
libraries before selection; the enabled training bootstrap reports missing
components with the dependency setup command.

Swift derives `extra.mega_moe_local_tokens` from sequence length, batch size and
SP. An explicit value is preserved when it is 128-aligned and at least the
derived bound; an invalid or undersized value fails during adapter setup. Direct
engine callers supply this same existing config entry before model preparation.

## Memory and routing

Placement uses live routing counts, independent of dataset identity or fixed
expert order. Large inputs can borrow expert weights when destination load
exceeds receive capacity. Small copy workloads reserve a topology-derived
lossless bound, subject to a 512 MiB payload limit, to avoid expensive whole-expert
transfers. Plan widths depend on topology. Borrowed weight slots become gradient
return scratch after backward finishes reading the weights.

One fixed SHMEM heap is reserved from token bounds, EP topology, expert dimensions
and activation modes. It is additional device memory outside PyTorch allocator
statistics. No generic SHMEM heap-rebuild framework or standalone MegaMoE module
is included. Autograd leases and collective teardown protect live buffers.

## Validate

```bash
python -u tests/npu/run_mega_moe_tests.py
# Configuration and semantic checks only:
python -u tests/npu/run_mega_moe_tests.py --unit-only
```

The suite covers EP2/EP8, uniform/hot routing, clipping and checkpointing against
an independent scalar reference (2% BF16 relative L2). The module benchmark uses
stricter comparisons against the original dispatcher: output/input/probability
gradients <=0.1%, weight gradients <=0.5%, and all results finite.

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tests/npu/benchmark_mega_moe_moonep.py \
  --tokens 4096 --hidden 2048 --intermediate 1024 --experts 128 --topk 4 \
  --routing hot --warmup 30 --repeat 60 --mutation-check
```

Benchmark with idle devices. Optional profiles collect with `analyse_flag=False`;
parse them offline after training. Performance and end-to-end accuracy must be
measured for the actual training configuration.

The MoonEP planner queries the device UB capacity through CANN. Small plans stay in UB; larger plans keep tables in device memory and use bounded UB pages. Allocation tables that do not fit use the existing NPU cumulative-sum operator. This selection depends only on tensor shapes and hardware capacity, not on a dataset.
