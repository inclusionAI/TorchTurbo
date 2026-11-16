"""The base test suite: cases that exercise core training features end to end."""

from run_tests import TestCaseSpec

TEST_CASES: list[TestCaseSpec] = [
    TestCaseSpec(
        test_name="bailing_v3/sft_16npu",
        test_descr="Bailing V3 tiny pretrained SFT on 16 NPUs",
        framework="swift",
        mode="train",
        test_type="frozen-start",
        compare=True,
        metrics={
            "loss": {"atol": 1.0e-5, "rtol": 1.0e-3},
            "grad_norm": {"atol": 1.0e-5, "rtol": 0.1},
        },
        ngpu=16,
    ),
    TestCaseSpec(
        test_name="qwen3_5_moe/sft_2npu",
        test_descr="Qwen3.5-MoE VL-template SFT on 2 NPUs",
        framework="llamafactory",
        mode="train",
        test_type="regular",
        compare=True,
        metrics={
            "loss": {"atol": 1.0e-5, "rtol": 1.0e-3},
            "grad_norm": {"atol": 1.0e-5, "rtol": 1.0e-3},
        },
    ),
    TestCaseSpec(
        test_name="qwen3_5_moe/sft_2cpu",
        test_descr="Qwen3.5-MoE VL-template SFT on 2 CPU ranks",
        framework="swift",
        mode="train",
        test_type="regular",
        compare=True,
        metrics={
            "loss": {"atol": 1.0e-6, "rtol": 1.0e-6},
            "grad_norm": {"atol": 2.0e-4, "rtol": 1.0e-4},
        },
        platform="cpu",
    ),
]
