"""A local distributed job for tests that materialize node-shared tables.

Model construction is local; only materialization needs a process group.
"""

import pytest
import torch.distributed as dist

from torchturbo.models.modules.cpu_resident import NodeSharedCPUResidentParameter
from torchturbo.ops import get_or_create_ops
from torchturbo.runtime.cpu import CpuRuntime
from tests.support.distributed import destroy_process_group


@pytest.fixture(autouse=True)
def configure_cpu_ops(request):
    if request.node.get_closest_marker("cpu") is not None:
        get_or_create_ops(device_runtime=CpuRuntime())


@pytest.fixture
def node_group(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    init_file = tmp_path_factory.mktemp("qwen-node-group") / "init"
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        destroy_process_group()
        NodeSharedCPUResidentParameter._node_group = None
