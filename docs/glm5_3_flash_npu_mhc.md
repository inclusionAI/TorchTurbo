# GLM5.3 Flash NPU mHC pre

`TORCHTURBO_ENABLE_NPU_MHC=1` opts into the non-cross-layer
`mhc_pre_sinkhorn` training provider. The default is `0`.

## Model behavior

`on_parallelized()` installs an instance-scoped `_pre` on each text decoder
layer. Attention and FFN use their own current `hc_*_fn/scale/base` parameters.
Repeated patching preserves the original `_pre` and does not wrap it again.
The decoder's existing forward, activation checkpoint wrappers, residual
transpose, HC head, state dict keys and no-shard gradient synchronization are
unchanged.

The provider imports the wheel's three-output Python autograd wrapper from
`cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn.mhc_pre_sinkhorn`, rather than
the top-level eight-output `torch.ops` binding. It passes both epsilon values
explicitly (`hc_eps=1e-6`, `norm_eps=1e-5` for the supplied checkpoints),
converts current parameters to contiguous FP32 without detaching, and makes
the BSND input contiguous. Post and residual coefficients are cast back to
the input dtype; the residual coefficients are reshaped, without transposing.
Frozen mHC parameters still allow the hidden-input gradient to reach LoRA.

The conservative training contract is nonempty BSND BF16/FP16 on NPU, N=4,
D divisible by 128, 20 Sinkhorn iterations and matching parameter shapes.
CPU, unsupported inputs, no-grad, cache-free forward and missing dependencies
fall back to the saved reference implementation. Fallback reasons are logged
once on rank zero. Disabling the flag avoids the model's runtime probe.
An error after entering the fused kernel is propagated, never silently retried
with reference operators.

## Runtime requirements

Install the supplied wheel in the target container:

```bash
python -m pip install --no-deps /mnt/sfs_turbo/qcx02367502/DeepSeekV41_SFT/cann-ops-transformer-pkg/ops-transformer-master-31bfd6d1_20261021_cann900/cann_ops_transformer-1.0.0-py3-none-any.whl
```

This wheel supplies Python/C++ wrappers. The selected CANN/runtime or a custom
operator installation must additionally supply these ACLNN entry points and
their kernels:

```text
aclnnMhcPreSinkhorn
aclnnMhcPreSinkhornGetWorkspaceSize
aclnnMhcPreSinkhornBackward
aclnnMhcPreSinkhornBackwardGetWorkspaceSize
```

The provider checks op API symbols and loads both C++ bindings before declaring
availability. This does not replace a successful forward/backward kernel test.
Unavailable results are cached per process; restart training after installing
or changing the runtime. Both `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`
must be configured before the process starts.

The 8-layer launcher accepts optional `CANN_ENV_SH` and `MHC_ENV_SH` overrides.
Its default CANN 8.5 source and the user's existing training settings are kept.
With a verified runtime, launch with:

```bash
CANN_ENV_SH=/path/to/compatible/cann/set_env.sh \
MHC_ENV_SH=/path/to/non-premix/vendor/bin/set_env.bash \
TORCHTURBO_ENABLE_NPU_MHC=1 \
bash scripts/runs/lf_glm5_3_flash_text_sft_lora_8.sh --max_steps 2
```

The paths above are placeholders; the provided premix vendor package is not a
replacement for the non-premix runtime.

## Validation

Run inside an NPU container, from the repository root:

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" GLM5_TEST_NPU=1 \
python -m pytest tests/models/glm5_3_flash/test_npu_mhc.py -q
```

For real fused kernel validation, additionally set `GLM5_TEST_NPU_MHC=1`.
These tests assert availability and count actual fused entry calls, so a
reference fallback cannot pass as a fused result. The cases compare all three
outputs and hidden/fn/scale/base gradients, both attention and FFN, frozen
mHC parameters, BF16/FP16, D=128/4096, noncontiguous inputs and non-reentrant
checkpoint recomputation. They use output tolerances `rtol=0.01, atol=0.003`
and a gradient relative L2 bound of 3%; these acceptance limits still need
confirmation against the actual target kernel.

On 2026-10-23 the supplied wheel was installed in the current container on
33.215.116.134. Native CANN 9.0.0-beta.1 and the supplied dsv41/dsv41exp vendor
packages did not expose the four non-premix symbols above. The dsv41exp package
exposes `aclnnMhcPreSinkhornPremix` and its backward instead. Initial wheel
compilation also reported `ACL_HIFLOAT4` absent from the container's CANN
headers. No CANN headers, wheel sources or shared toolchains were modified.

Real fused numerical alignment, model/parallel/checkpoint alignment and
performance are pending a compatible non-premix runtime. Logs and the
short-training runner are under `work/mhc_integration_20261023/`; reference
fallback results must not be interpreted as fusion accuracy or speedup.

Validation recorded on the current container:

| Check | Result |
| --- | --- |
| New adapter/provider tests, including NPU fallback | 7 passed, 8 real-kernel cases skipped |
| Real-kernel tests explicitly enabled, fail-fast | Failed availability assertion: four non-premix ACLNN symbols missing |
| Existing regression subset | 53 passed, 20 skipped, 7 failed; all 7 failures reproduced with the pre-change source |
| 8-layer launcher, default CANN 8.5 | Failed before completing step 1, AICore 507015; dump identifies `CrossEntropyLoss_bf16_1000_high_performance` |
| Same launcher with CANN 9 override | Completed 2 optimizer steps, exit 0; mHC reference fallback |

The successful short run used 16 NPUs, SP=2, EP=8, LoRA, BF16, sequence
length 128, FSDP2 and non-reentrant activation checkpointing. Step losses were
10.0498 and 10.0488; gradient norms were 1.4096688 and 1.4109547. First-step
timing includes compilation and is not a fusion performance measurement.
The CANN 9 run used `CANN_ENV_SH=/usr/local/Ascend/cann/set_env.sh`; the
launcher's default runtime was not changed.

The baseline regression failures are four CPU-FP32 MoE tests selecting an
NPU-BF16-only permutation provider, and three logger/caplog assertions. They
were reproduced using an isolated copy of the source with the original
`npu.py` and without the new mHC provider.
