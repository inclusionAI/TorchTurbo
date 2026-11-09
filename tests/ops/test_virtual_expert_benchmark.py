"""Check benchmark resource setup before initializing an NPU device."""

import os
import sys

import pytest


@pytest.mark.parametrize("ep_size,suite,modes,shape,configured,expected", [
    (2, "all", ["alltoall", "ve_shmem"], [], None, "3G"),
    (8, "moe", ["ve_shmem"], [], None, "1G"),
    (2, "moe", ["ve_native"], ["--hidden", "1024", "--ffn", "4096"], None, "10G"),
    (8, "all", ["ve_shmem"], ["--hidden", "304", "--ffn", "24528"], None, "5G"),
    (8, "all", ["ve_shmem"], ["--hidden", "261", "--ffn", "28569"], None, "5G"),
    (8, "moe", ["ve_shmem"], ["--hidden", "261", "--ffn", "28569"], None, "4G"),
    (2, "moe", ["ve_shmem"], [], "4G", "4G"),
    (2, "planner", ["ve_shmem"], [], None, None),
    (2, "moe", ["alltoall"], [], None, None),
])
def test_shmem_heap_is_sized_before_npu_initialization(
    monkeypatch, tmp_path, ep_size, suite, modes, shape, configured, expected,
):
    from tests.ops.npu import benchmark_virtual_experts as benchmark

    key = "NPU_SHMEM_SYMMETRIC_SIZE"
    monkeypatch.delenv(key, raising=False)
    if configured is not None:
        monkeypatch.setenv(key, configured)
    monkeypatch.setenv("WORLD_SIZE", str(ep_size))
    monkeypatch.setattr(sys, "argv", [
        "benchmark_virtual_experts.py", "--output", str(tmp_path / "results.json"),
        "--suite", suite, "--moe-modes", *modes, *shape,
    ])

    class DeviceInitializationReached(Exception):
        pass

    def stop_before_device_initialization(runtime):
        raise DeviceInitializationReached(os.environ.get(key))

    monkeypatch.setattr(benchmark.NpuRuntime, "initialize", stop_before_device_initialization)
    with pytest.raises(DeviceInitializationReached) as reached:
        benchmark.main()
    assert reached.value.args == (expected,)


@pytest.mark.parametrize("suite", ["planner", "moe", "all"])
@pytest.mark.parametrize("hot_options", [
    ["--hot-fractions", "0.5"],
    ["--hot-multipliers", "64"],
    ["--hot-experts", "256"],
    ["--hot-fractions", "0.00390625"],
])
def test_empty_routing_matrix_is_rejected_before_npu_initialization(
    monkeypatch, tmp_path, capsys, suite, hot_options,
):
    from tests.ops.npu import benchmark_virtual_experts as benchmark

    output = tmp_path / "results.json"
    monkeypatch.delenv("NPU_SHMEM_SYMMETRIC_SIZE", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(sys, "argv", [
        "benchmark_virtual_experts.py", "--output", str(output),
        "--suite", suite, "--routing", "hotspot", *hot_options,
    ])

    def reject_device_initialization(runtime):
        pytest.fail("Empty routing matrix reached NPU initialization.")

    monkeypatch.setattr(benchmark.NpuRuntime, "initialize", reject_device_initialization)
    with pytest.raises(SystemExit) as rejected:
        benchmark.main()
    assert rejected.value.code == 2
    captured = capsys.readouterr()
    assert "No valid routing cases" in captured.err
    assert "BENCHMARK_COMPLETE" not in captured.out
    assert not output.exists()


@pytest.mark.parametrize("routing_options", [
    ["--routing", "hotspot", "--hot-fractions", "0.5", "0.125"],
    ["--routing", "balanced", "hotspot", "--hot-fractions", "0.5"],
    ["--routing", "random", "hotspot", "--hot-experts", "256"],
])
def test_partially_filtered_routing_matrix_reaches_npu_initialization(
    monkeypatch, tmp_path, routing_options,
):
    from tests.ops.npu import benchmark_virtual_experts as benchmark

    monkeypatch.delenv("NPU_SHMEM_SYMMETRIC_SIZE", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(sys, "argv", [
        "benchmark_virtual_experts.py", "--output", str(tmp_path / "results.json"),
        *routing_options,
    ])

    class DeviceInitializationReached(Exception):
        pass

    def stop_before_device_initialization(runtime):
        raise DeviceInitializationReached

    monkeypatch.setattr(benchmark.NpuRuntime, "initialize", stop_before_device_initialization)
    with pytest.raises(DeviceInitializationReached):
        benchmark.main()
