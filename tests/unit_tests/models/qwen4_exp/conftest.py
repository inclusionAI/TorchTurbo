"""A local distributed job for tests that materialize node-shared tables.

Model construction is local; only materialization needs a process group.
"""

import pytest
import torch.distributed as dist

from torchturbo.models.modules.cpu_resident import NodeSharedCPUResidentParameter


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
        dist.destroy_process_group()
        NodeSharedCPUResidentParameter._node_group = None
