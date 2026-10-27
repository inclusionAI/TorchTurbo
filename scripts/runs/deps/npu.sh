#!/bin/bash

# The Ascend base image provides torch_npu and CANN. TorchTurbo has no
# MindSpeed runtime dependency.
set -e
python3 -c "import torch_npu"
