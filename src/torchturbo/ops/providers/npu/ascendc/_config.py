"""AscendC installation settings shared by build scripts and runtime loading."""
import os
from pathlib import Path


DEFAULT_LIBRARY_DIR = Path("/opt/torchturbo/ascendc/lib")


def library_directory() -> Path:
    directory = os.environ.get("TORCHTURBO_ASCENDC_LIB_DIR")
    return Path(directory) if directory else DEFAULT_LIBRARY_DIR


if __name__ == "__main__":
    print(library_directory())
