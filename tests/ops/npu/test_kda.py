"""Real-NPU regression for provider-side KDA packed-sequence batching."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu.ops.ascendc")

from torchturbo.ops import OpDispatcher, OpPolicy
from torchturbo.ops.providers.npu import kda
from torchturbo.ops.registry import registry
from torchturbo.runtime.npu import NpuRuntime


@pytest.mark.parametrize("num_sequences", [1025, 2049])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_large_varlen_kda_matches_smaller_groups(monkeypatch, num_sequences, output_final_state):
    if not torch.npu.is_available():
        pytest.skip("NPU device not available")
    torch.manual_seed(42)
    lengths = ([1, 63, 64, 65, 0] * ((num_sequences + 4) // 5))[:num_sequences]
    num_tokens = sum(lengths)
    boundaries = torch.tensor([0, *lengths], dtype=torch.int32, device="npu").cumsum(0)
    original = [
        torch.randn(1, num_tokens, 2, 128, device="npu", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    original.extend([
        torch.rand(1, num_tokens, 2, device="npu", dtype=torch.float32),
        torch.randn(2, device="npu", dtype=torch.float32),
        torch.randn(256, device="npu", dtype=torch.float32),
    ])
    ops = OpDispatcher(registry=registry, device_runtime=NpuRuntime(), default_policy=OpPolicy.AUTO)

    def run(max_sequences):
        monkeypatch.setattr(kda, "_MAX_KDA_SEQUENCES", max_sequences)
        inputs = [tensor.detach().clone().requires_grad_() for tensor in original]
        output, final_state = ops.kda(
            q=inputs[0], k=inputs[1], v=inputs[2], g=inputs[3], beta=inputs[4],
            A_log=inputs[5], dt_bias=inputs[6], cu_seqlens=boundaries,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, impl="ascendc_kda",
        )
        output.float().square().sum().backward()
        result = [output.detach(), *(tensor.grad.detach() for tensor in inputs)]
        if output_final_state:
            assert final_state.shape == (num_sequences, 2, 128, 128)
            result.append(final_state.detach())
        return result

    baseline = run(127)
    actual = run(1024)
    torch.npu.synchronize()
    names = ("output", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "dt_bias", "final_state")
    for name, expected, result in zip(names, baseline, actual):
        torch.testing.assert_close(
            result.float(), expected.float(), rtol=2e-2, atol=2e-2,
            msg=lambda message, name=name: f"{name}: {message}",
        )
