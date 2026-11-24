"""Shared loading of prebuilt AscendC operator libraries."""
from functools import lru_cache
from pathlib import Path

import torch

from ._config import library_directory


def library_path(name: str) -> Path:
    return library_directory() / f"libtorchturbo_{name}.so"


@lru_cache(maxsize=None)
def load_library(name: str) -> None:
    path = library_path(name)
    if not path.is_file():
        raise RuntimeError(
            f"AscendC library missing: {path}. Use the TorchTurbo NPU image or build the library "
            "with ascendc/build.sh and set TORCHTURBO_ASCENDC_LIB_DIR before training."
        )
    torch.ops.load_library(str(path.resolve()))
