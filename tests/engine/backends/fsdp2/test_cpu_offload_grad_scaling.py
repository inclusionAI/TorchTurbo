"""Accelerator scaling must update CPU gradient storage before returning."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.runtime import DeviceRuntime, TurboDevice, get_device_runtime


@pytest.fixture
def accelerator() -> DeviceRuntime:
    runtime = get_device_runtime()
    if runtime.turbo_device not in (TurboDevice.CUDA, TurboDevice.NPU):
        pytest.skip("requires CUDA or NPU for CPU-offload gradient transfers")
    runtime.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return runtime


def _check_scaling(
    runtime: DeviceRuntime,
    dtype: torch.dtype,
    pinned: bool,
    shapes: tuple[tuple[int, ...], ...],
    repeats: int,
) -> None:
    backend = object.__new__(FSDP2Backend)
    backend._device_runtime = runtime
    gradients = []
    expected = []
    for index, shape in enumerate(shapes):
        generator = torch.Generator(device="cpu").manual_seed(2701 + index)
        source = torch.randn(shape, dtype=dtype, device="cpu", generator=generator)
        gradient = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pinned)
        gradient.copy_(source)
        assert gradient.is_pinned() == pinned
        gradients.append(gradient)
        expected.append(source)

    aliases = [gradient.view_as(gradient) for gradient in gradients]
    pointers = [gradient.data_ptr() for gradient in gradients]
    for repeat in range(repeats):
        # Power-of-two factors avoid device-dependent scalar rounding in BF16.
        # Repeated reductions stay in the normal range for both tested dtypes.
        for factor in (1 / 16, 1 / 2):
            backend._scale_offloaded_chunk_on_device(gradients, runtime.device, factor)
            expected = [value * factor for value in expected]
            for index, (gradient, alias, reference, pointer) in enumerate(
                zip(gradients, aliases, expected, pointers, strict=True)
            ):
                context = (
                    f"repeat={repeat}, factor={factor}, tensor={index}, "
                    f"dtype={dtype}, pinned={pinned}"
                )
                assert gradient.device.type == "cpu", context
                assert gradient.data_ptr() == pointer, context
                # Read CPU storage immediately, without adding a device sync.
                # The helper must complete the transfer before it returns.
                torch.testing.assert_close(
                    gradient, reference, rtol=0, atol=0, msg=context
                )
                torch.testing.assert_close(
                    alias, reference, rtol=0, atol=0, msg=context
                )


def test_cpu_offload_scaling_from_fresh_process(accelerator):
    # A separate interpreter keeps earlier accelerator tests from warming the
    # first multiply/copy execution that exposed missing CPU gradient updates.
    script = """
import sys
import torch
from torchturbo.runtime import get_device_runtime
from tests.engine.backends.fsdp2.test_cpu_offload_grad_scaling import _check_scaling

torch.set_num_threads(1)
runtime = get_device_runtime()
runtime.set_device(int(sys.argv[1]))
runtime.use_deterministic_algorithms()
shapes = tuple((size,) for size in (1, 3, 127, 1024, 4097, 65536, 131072, 262144))
_check_scaling(runtime, torch.float32, False, shapes, repeats=10)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(accelerator.current_device())],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pinned", [False, True], ids=["unpinned", "pinned"])
def test_cpu_offload_scaling_preserves_gradient_storage(accelerator, dtype, pinned):
    shapes = (
        (1,),
        (3,),
        (127,),
        (32, 32),
        (17, 241),
        (64, 1024),
        (2, 256, 256),
        (512, 512),
    )
    _check_scaling(accelerator, dtype, pinned, shapes, repeats=10)
