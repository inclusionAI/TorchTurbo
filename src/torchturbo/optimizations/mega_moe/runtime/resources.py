"""Shape-bound scratch owned exclusively by the fused MoonEP backend."""

from typing import Any

import torch

from .activation import resolve_mode
from .modules.mega_moe.spec import bind_mega_moe_spec
from .modules.mega_moe.workspace import MegaMoeWorkspace, _spec_workspace_bytes


class ExecutionResources:
    """Reserve one workspace per activation mode without creating expert weights."""

    def __init__(self, tensor: torch.Tensor, **specification: Any) -> None:
        limit = specification["swiglu_limit"]
        if limit is not None:
            if not 0 < limit < float("inf"):
                raise ValueError("SwiGLU clamp must be positive and finite")
            limit = float(torch.tensor(limit, dtype=torch.bfloat16))
        specification["swiglu_limit"] = limit
        specification["swiglu_mode"] = resolve_mode(specification["swiglu_mode"], limit)
        self.spec = bind_mega_moe_spec(specification, tensor)
        self.symmetric_bytes = _spec_workspace_bytes(specification, tensor.element_size())
        self.workspace = MegaMoeWorkspace(shared=True)
        self.workspace.capacity_floor = self.spec.receive_capacity

    def close(self) -> None:
        """Release buffers only after their final backward has completed."""
        self.workspace.close()
