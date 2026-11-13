"""Compare the DeepSeekV4 Turbo wrapper with the base modeling class.

This is the DeepSeekV4-specific counterpart of the old ad-hoc
run_turbo_only_vs_hf_precision.py idea: instantiate both implementations
directly, without FSDP2, copy identical random weights, and compare logits.
"""

import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("torch is required for DeepSeekV4 Turbo-vs-base tests", allow_module_level=True)
    raise

os.environ.setdefault("TORCH_TURBO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from .test_alignment import (  # noqa: E402
        TEST_CONFIG_MODELING,
        align_weights,
        create_test_input,
        get_modeling_model,
        run_forward_modeling,
        seed_all,
    )
except ImportError:  # pragma: no cover - direct script execution
    from test_alignment import (  # noqa: E402
        TEST_CONFIG_MODELING,
        align_weights,
        create_test_input,
        get_modeling_model,
        run_forward_modeling,
        seed_all,
    )


def get_turbo_model(config_dict, device="cpu"):
    from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from torchturbo.models.deepseek_v4 import model as turbo_model_module
    from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as modeling_module
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime.cpu import CpuRuntime
    try:
        from .test_alignment import hadamard_transform_impl
    except ImportError:  # pragma: no cover - direct script execution
        from test_alignment import hadamard_transform_impl

    # The Turbo wrapper's RMSNorm goes through the ops dispatcher, so a CPU
    # dispatcher must exist before the model is constructed.
    get_or_create_ops(device_runtime=CpuRuntime())

    # This is a CPU precision test. In NPU containers torch_npu can be importable
    # even when the test tensors live on CPU, so avoid globally patching RMSNorm
    # to npu_rms_norm here.
    turbo_model_module.is_npu_available = False

    modeling_module.hadamard_transform_ref = lambda x, hadamard_mat, scale=1.0: hadamard_transform_impl(
        x, scale
    )
    modeling_module.rotate_activation = lambda x, hadamard_mat: hadamard_transform_impl(
        x, scale=x.size(-1) ** -0.5
    )

    config = DeepseekV4Config(**config_dict)
    model = turbo_model_module.TurboDeepseekV4ForCausalLM(config).float()
    model = model.to(device=device)
    model.eval()
    return model


def test_turbo_wrapper_matches_base_modeling_cpu():
    seed = 43
    device = "cpu"
    batch_size = 1
    seq_len = 128

    seed_all(seed)
    base_model, _ = get_modeling_model(TEST_CONFIG_MODELING, device)
    turbo_model = get_turbo_model(TEST_CONFIG_MODELING, device)
    align_weights(base_model, turbo_model)

    input_ids = create_test_input(batch_size, seq_len, TEST_CONFIG_MODELING["vocab_size"], device, seed)
    base_logits = run_forward_modeling(base_model, input_ids)
    turbo_logits = run_forward_modeling(turbo_model, input_ids)

    torch.testing.assert_close(turbo_logits, base_logits, atol=1e-5, rtol=1e-5)
