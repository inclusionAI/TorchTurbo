#!/bin/bash

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${deps_dir}/platform.sh"

# dependency of swift

pip install deepspeed==0.16.0
pip install tensorboard
pip install psutil==7.2.2
pip install omegaconf
pip install pyparsing
pip install cycler
pip install kiwisolver
pip install torchvision==0.25.0
pip install torchaudio==2.10.0
pip install accelerate==1.11.0
pip install trl==0.29.0
pip install matplotlib

pip install ms_swift==4.4.2
pip install transformers==5.9.0
pip install peft==0.18.0
pip install datasets==5.0.1
pip install binpacking==1.5.2
pip install safetensors==0.7.0

pip install pyarrow==21.0.0
