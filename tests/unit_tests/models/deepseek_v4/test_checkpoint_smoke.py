"""Optional DeepSeekV4 real-checkpoint metadata smoke test.

This test is intentionally opt-in because the small_dsv4 checkpoint is still
large. Enable it with:

    TT_DSV4_SMOKE_MODEL=/path/to/small_dsv4 pytest tests/models/deepseek_v4/test_checkpoint_smoke.py -q
"""

import os
import sys
import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeekV4 checkpoint smoke tests", allow_module_level=True)
    raise

import pytest

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_small_dsv4_checkpoint_metadata_smoke():
    model_path = os.environ.get("TT_DSV4_SMOKE_MODEL")
    if not model_path:
        pytest.skip("set TT_DSV4_SMOKE_MODEL to run the real-checkpoint smoke test")

    model_dir = Path(model_path)
    if not (model_dir / "config.json").is_file():
        pytest.skip(f"DeepSeekV4 smoke model config not found: {model_dir / 'config.json'}")

    import torchturbo  # noqa: F401
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    assert config.model_type == "deepseek_v4"
    assert config.num_hidden_layers > 0
    assert config.num_nextn_predict_layers >= 0
    assert AutoModelForCausalLM._model_mapping[type(config)].__name__ == "TurboDeepseekV4ForCausalLM"

    index_path = model_dir / "model.safetensors.index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    required = {
        "embed.weight": (config.vocab_size, config.hidden_size),
        "layers.0.attn.attn_sink": (config.num_attention_heads,),
        "layers.0.attn.wq_a.weight": (config.q_lora_rank, config.hidden_size),
        "layers.0.attn.wkv.weight": (config.head_dim, config.hidden_size),
        "head.weight": (config.vocab_size, config.hidden_size),
    }
    for name, expected_shape in required.items():
        assert name in weight_map
        shard_path = model_dir / weight_map[name]
        assert shard_path.is_file()
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(name)
        assert tuple(tensor.shape) == expected_shape
        assert tensor.dtype in (torch.bfloat16, torch.float32, torch.int64, torch.int32)
