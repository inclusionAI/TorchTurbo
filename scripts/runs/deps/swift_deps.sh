#!/bin/bash

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${deps_dir}/platform.sh"

# dependency of swift

pip install --retries 12 --timeout 60 deepspeed==0.16.0
pip install --retries 12 --timeout 60 tensorboard
pip install --retries 12 --timeout 60 psutil==7.2.2
pip install --retries 12 --timeout 60 omegaconf
pip install --retries 12 --timeout 60 pyparsing
pip install --retries 12 --timeout 60 cycler
pip install --retries 12 --timeout 60 kiwisolver
pip install --retries 12 --timeout 60 torchvision==0.25.0
pip install --retries 12 --timeout 60 torchaudio==2.10.0
pip install --retries 12 --timeout 60 accelerate==1.11.0
pip install --retries 12 --timeout 60 trl==0.29.0
pip install --retries 12 --timeout 60 matplotlib

pip install --retries 12 --timeout 60 ms_swift==4.4.2
pip install --retries 12 --timeout 60 transformers==5.9.0
pip install --retries 12 --timeout 60 peft==0.18.0
pip install --retries 12 --timeout 60 datasets==5.0.1
pip install --retries 12 --timeout 60 binpacking==1.5.2
pip install --retries 12 --timeout 60 safetensors==0.7.0

pip install --retries 12 --timeout 60 numpy==2.0
