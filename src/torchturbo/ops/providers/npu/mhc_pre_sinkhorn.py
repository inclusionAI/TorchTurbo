"""Training-only BSND mHC pre + Sinkhorn, using the wheel's autograd API.

The Python wheel and the ACLNN forward/backward libraries are separate
dependencies. Probe both before selecting this optional provider. Kernel
execution errors are deliberately allowed to propagate.
"""

from __future__ import annotations

import ctypes
import importlib
import os
from functools import lru_cache
from pathlib import Path

import torch

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


def _check_aclnn_libraries():
    """Check the same custom/default op API libraries used by the wheel."""
    names = ("libcust_opapi.so", "libopapi_transformer.so", "libopapi.so")
    libraries = list(names)
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if directory:
            libraries.extend(str(Path(directory) / name) for name in names)
    vendors = [
        Path(path)
        for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if path
    ]
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        libraries.extend(str(Path(home) / "lib64" / name) for name in names[1:])
    libraries.extend(str(path / "op_api/lib/libcust_opapi.so") for path in vendors)
    required = {
        "aclnnMhcPreSinkhorn",
        "aclnnMhcPreSinkhornGetWorkspaceSize",
        "aclnnMhcPreSinkhornBackward",
        "aclnnMhcPreSinkhornBackwardGetWorkspaceSize",
    }
    for library in dict.fromkeys(libraries):
        try:
            handle = ctypes.CDLL(library, mode=os.RTLD_LAZY | os.RTLD_LOCAL)
        except OSError:
            continue
        required.difference_update(name for name in tuple(required) if hasattr(handle, name))
        if not required:
            return
    raise OSError("missing ACLNN mHC symbols: " + ", ".join(sorted(required)))


@lru_cache(maxsize=1)
def _mhc_runtime():
    try:
        _check_aclnn_libraries()
        # Do not use the package's top-level torch.ops handle (eight outputs,
        # no autograd). This submodule exports the three-output Python wrapper.
        forward = importlib.import_module(
            "cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn.mhc_pre_sinkhorn"
        )
        backward = importlib.import_module(
            "cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_backward.mhc_pre_sinkhorn_backward"
        )
        forward.mhc_pre_sinkhorn_op_builder.load(verbose=False)
        backward.mhc_pre_sinkhorn_backward_op_builder.load(verbose=False)
        return forward.mhc_pre_sinkhorn, None
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        return None, f"{type(error).__name__}: {error}"


def mhc_pre_sinkhorn_unavailable_reason():
    return _mhc_runtime()[1]


def mhc_pre_sinkhorn_unsupported_reason(x, phi, alpha, bias, hc_mult, num_iters):
    """Conservative arch22 training contract; contiguous copies are supported."""
    if x.device.type != "npu":
        return "input is not on NPU"
    if x.dtype not in (torch.bfloat16, torch.float16):
        return "input dtype must be BF16 or FP16"
    if x.ndim != 4 or any(size == 0 for size in x.shape):
        return "input must be nonempty BSND"
    if hc_mult != 4 or x.shape[2] != 4:
        return "training requires hc_mult=4 and N=4"
    if x.shape[3] % 128:
        return "training requires D divisible by 128"
    if num_iters != 20:
        return "training requires 20 Sinkhorn iterations"
    expected = ((24, 4 * x.shape[3]), (3,), (24,))
    for name, parameter, shape in zip(("phi", "alpha", "bias"), (phi, alpha, bias), expected):
        if tuple(parameter.shape) != shape or parameter.device != x.device:
            return f"{name} must have shape {shape} on the input device"
        if not parameter.is_floating_point():
            return f"{name} must be floating point"
    if not torch.is_grad_enabled():
        return "no_grad uses the reference pre"
    if not any(t.requires_grad for t in (x, phi, alpha, bias)):
        return "cache-free forward is not enabled"
    return None


@register_op(
    op_name="mhc_pre_sinkhorn",
    impl_name="npu_mhc_pre_sinkhorn",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class NpuMhcPreSinkhorn(OpImpl):
    def available(self) -> bool:
        return mhc_pre_sinkhorn_unavailable_reason() is None

    def __call__(self, x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
        reason = mhc_pre_sinkhorn_unsupported_reason(x, phi, alpha, bias, hc_mult, num_iters)
        if reason is not None:
            raise ValueError(reason)
        function, reason = _mhc_runtime()
        if function is None:
            raise RuntimeError(f"NPU mhc_pre_sinkhorn unavailable: {reason}")
        # Keep casts differentiable, including frozen mHC parameters with a
        # trainable upstream LoRA input. Read current parameters at every call.
        return function(
            x.contiguous(),
            phi.float().contiguous(),
            alpha.float().contiguous(),
            bias.float().contiguous(),
            hc_mult,
            num_iters,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
        )
