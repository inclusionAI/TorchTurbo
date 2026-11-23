from collections.abc import Callable
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_with_cpu_input(function: Callable, *args: Any, **kwargs: Any) -> Any:
    # Offload only the first positional tensor saved by non-reentrant checkpoint.

    # Other inputs keep their existing storage and dtype. Transfers are synchronous.
    # The pack closure stores an identity, not a reference to the device input.
    if not args or not isinstance(args[0], torch.Tensor):
        raise ValueError("CPU checkpoint input offload requires a positional tensor input.")
    if kwargs.get("use_reentrant") is not False:
        raise ValueError("CPU checkpoint input offload requires use_reentrant=False.")
    input_id = id(args[0])

    def pack(tensor):
        if id(tensor) == input_id and tensor.device.type != "cpu":
            return tensor.device, tensor.detach().to("cpu")
        return tensor.detach()

    def unpack(saved):
        if isinstance(saved, tuple):
            device, tensor = saved
            return tensor.to(device)
        return saved

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return checkpoint(function, *args, **kwargs)


# Only the packed residual streams ([B, S, N, D] activations checkpointed
# between layers) are worth a host round trip; smaller checkpoint inputs
# stay on device.
LARGE_INPUT_MIN_BYTES = 64 * 2**20


def checkpoint_with_large_input_offload(
    function: Callable,
    *args: Any,
    min_bytes: int = LARGE_INPUT_MIN_BYTES,
    **kwargs: Any,
) -> Any:
    """Reentrant checkpoint that parks large tensor inputs in pinned host memory.

    The packed HC residual streams dominate the checkpointed inputs at long
    sequence lengths (39 live layers x 0.31 GiB at the 128k sp16 shape and
    they are the top device-memory holder in backward). Size-gated async
    copies move each stream to the host between forward and backward, and the
    unpack streams it back before recompute. Copies are bitwise, so outputs
    and gradients match the unoffloaded path exactly. ``min_bytes`` is the
    size gate (exposed so tests can trigger it with small tensors).
    """
    if kwargs.get("use_reentrant") is not True:
        raise ValueError("Large-input offload requires use_reentrant=True.")

    def pack(tensor):
        if (
            tensor.device.type == "cpu"
            or not tensor.is_contiguous()
            or tensor.numel() * tensor.element_size() < min_bytes
        ):
            return tensor.detach()
        host = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu"
        )
        host.copy_(tensor, non_blocking=True)
        if tensor.device.type == "npu":
            copied = torch.npu.Event()
        else:
            copied = torch.cuda.Event()
        copied.record()
        return tensor.device, host, copied

    def unpack(saved):
        if not isinstance(saved, tuple):
            return saved
        device, host, copied = saved
        # Backward runs long after the forward-side copy was enqueued, so
        # this only orders the host read against an already-finished copy.
        copied.synchronize()
        return host.to(device, non_blocking=True)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return checkpoint(function, *args, **kwargs)
