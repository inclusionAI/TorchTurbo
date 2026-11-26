#!/bin/bash

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${deps_dir}/platform.sh"

# Use Transformers 5.5.4 while 5.6.0 mishandles Qwen3.5 FA2 with s_aux=None.
pip install --retries 12 --timeout 60 transformers==5.5.4

# dependency of llama_factory
pip install --retries 12 --timeout 60 safetensors==0.7.0
pip install --retries 12 --timeout 60 deepspeed==0.16.0
pip install --retries 12 --timeout 60 tensorboard
pip install --retries 12 --timeout 60 psutil==7.2.2
pip install --retries 12 --timeout 60 omegaconf
pip install --retries 12 --timeout 60 pyparsing
pip install --retries 12 --timeout 60 cycler
pip install --retries 12 --timeout 60 kiwisolver
pip install --retries 12 --timeout 60 peft==0.18.1
pip install --retries 12 --timeout 60 torchaudio==2.10.0
pip install --retries 12 --timeout 60 datasets==4.0.0
pip install --retries 12 --timeout 60 accelerate==1.11.0
pip install --retries 12 --timeout 60 trl==0.24.0
pip install --retries 12 --timeout 60 matplotlib

pip install --retries 12 --timeout 60 llamafactory==0.9.5

pip install --retries 12 --timeout 60 numpy==2.0
