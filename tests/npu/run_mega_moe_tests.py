"""Source-build MegaMoE + MoonEP precision tests; stdout is the report."""
import argparse
import importlib.util
import runpy
import struct
from types import SimpleNamespace
import os
from pathlib import Path
import shlex
import subprocess
import sys


def run_phase(name: str, command: list[str], env: dict[str,str]) -> bool:
    print("\nPHASE_START "+name,flush=True)
    print("COMMAND "+shlex.join(command),flush=True)
    result=subprocess.run(command,env=env,stderr=subprocess.STDOUT)
    print(f"PHASE_RESULT {'PASS' if result.returncode==0 else 'FAIL'} {name} exit_code={result.returncode}",flush=True)
    return result.returncode==0

def run_npu_precision(root: Path, env: dict[str,str], checkpoint_only: bool = False, compatibility_only: bool = False) -> bool:
    cases=[]
    for mode,limit in [("none","0"),("pre_activation",".25"),("post_activation",".25")]:
        for routing in ["uniform","hot"]:
            for checkpoint in [False,True]:
                name=f"{mode}_{routing}_checkpoint{int(checkpoint)}"
                command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",str(root/"tests/npu/mega_moe_precision.py"),"--mode",mode,"--limit",limit,"--routing",routing]
                if checkpoint:
                    command.extend(["--identity","--checkpoint"])
                cases.append((name,command))
        cases.append((mode+"_ep8_checkpoint",[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=8",str(root/"tests/npu/mega_moe_precision.py"),"--mode",mode,"--limit",limit,"--routing","hot","--identity","--checkpoint"]))
    # Larger hot inputs trigger real expert copies instead of the small-load path.
    for checkpoint_mode in ["none", "reentrant", "nonreentrant"]:
        for layers in [1, 2]:
            name=f"hot_copies_{checkpoint_mode}_layers{layers}"
            command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",str(root/"tests/npu/mega_moe_precision.py"),"--routing","hot","--tokens","2048","--layers",str(layers),"--checkpoint-mode",checkpoint_mode,"--require-copies"]
            cases.append((name,command))
    for early_stop in [True, False]:
        name=f"hot_copies_nonreentrant_stack_early_stop{int(early_stop)}"
        command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",str(root/"tests/npu/mega_moe_precision.py"),"--routing","hot","--tokens","2048","--layers","2","--checkpoint-mode","nonreentrant","--checkpoint-scope","stack","--require-copies"]
        if not early_stop:
            command.append("--disable-early-stop")
        cases.append((name,command))
    if checkpoint_only:
        cases = [case for case in cases if case[0].startswith("hot_copies_")]
    if compatibility_only:
        entry = str(root / "tests/npu/mega_moe_compatibility.py")
        prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", entry]
        cases = [
            ("compatibility_full_adamw", prefix + ["--offload", "full"]),
            ("compatibility_optimizer_adamw", prefix + ["--offload", "optimizer", "--checkpoint", "reentrant"]),
            ("compatibility_full_muon", prefix + ["--offload", "full", "--checkpoint", "reentrant", "--optimizer", "muon"]),
        ]
    failures=[]
    for index,(name,command) in enumerate(cases,1):
        print(f"\nCASE_START {index}/{len(cases)} {name}",flush=True)
        result=subprocess.run(command,env=env,stderr=subprocess.STDOUT,timeout=300)
        status="PASS" if result.returncode==0 else "FAIL"
        print(f"CASE_RESULT {status} {name} exit_code={result.returncode}",flush=True)
        if result.returncode:
            failures.append(name)
    print(f"\nCASE_SUMMARY total={len(cases)} passed={len(cases)-len(failures)} failed={len(failures)}",flush=True)
    for name in failures:
        print("FAILED_CASE "+name,flush=True)
    return not failures

def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser=argparse.ArgumentParser()
    parser.add_argument("--unit-only",action="store_true",help="Run only the configuration, hook and semantic checks; no multi-rank jobs")
    parser.add_argument("--checkpoint-only", action="store_true", help="Run only the eight borrowed-expert checkpoint NPU regressions after CPU checks")
    parser.add_argument("--compatibility-only", action="store_true", help="Run FSDP CPU-offload and optimizer combinations after CPU checks")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    runtime=root/"src/torchturbo/optimizations/mega_moe/runtime"
    if not args.unit_only:
        runpy.run_path(str(runtime/"bootstrap.py"))["activate"]()
    env={**os.environ,"PYTHONPATH":str(root/"src")+os.pathsep+os.environ.get("PYTHONPATH",""),"PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"1"}
    results=[run_phase("unit_checks",[sys.executable,"-m","pytest","-vv","--color=no","-p","no:cacheprovider",str(Path(__file__).resolve()),str(root/"tests/npu/test_mega_moe_moonep.py"),str(root/"tests/ops/test_mega_moe.py")],env)]
    if not args.unit_only and all(results):
        results.append(run_npu_precision(root,env,args.checkpoint_only,args.compatibility_only))
    passed=all(results) and len(results)==(1 if args.unit_only else 2)
    print(f"TEST_SUMMARY {'PASS' if passed else 'FAIL'} phases={len(results)}",flush=True)
    return int(not passed)

if __name__=="__main__":
    raise SystemExit(main())

# Pytest imports this module to collect the independent checks below.
# Direct invocation exits above after running them in an isolated child process.
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torchturbo.engine.args import TorchTurboArgs
from torchturbo.engine.config import EngineConfig, build_from_args
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.optimizations.mega_moe import integration
from torchturbo.optimizations import mega_moe

@pytest.mark.parametrize("enabled", [None, False, True])
def test_yaml_opt_in(tmp_path, enabled):
    path = tmp_path / "engine.yaml"
    body = "engine_config: {}\n" if enabled is None else (
        f"engine_config:\n  mega_moe: {str(enabled).lower()}\n"
    )
    path.write_text(body)
    config = build_from_args(TorchTurboArgs(config_path=str(path)))
    assert config.mega_moe is (enabled is True)
    assert EngineConfig.from_dict({}).mega_moe is False
    assert EngineConfig.from_dict({"mega_moe": True}).mega_moe is True


@pytest.mark.parametrize("length,batch,sp,expected", [
    (131072, 1, 8, 16384), (4097, 2, 8, 1152), (4096, 2, 1, 8192),
])
def test_automatic_local_token_capacity(tmp_path, length, batch, sp, expected):
    path = tmp_path / "engine.yaml"
    path.write_text(f"engine_config:\n  mega_moe: true\n  parallel_plan:\n    sp_size: {sp}\n")
    config = build_from_args(TorchTurboArgs(config_path=str(path)))
    args = SimpleNamespace(packing=True, packing_length=length,
                           max_length=length, per_device_train_batch_size=batch)
    integration.configure_token_bound(config, args)
    assert config.extra["mega_moe_local_tokens"] == expected


def test_expert_hooks_parameter_identity_and_route_order(monkeypatch):
    from torchturbo.models.modules.moe import experts as expert_module
    from torchturbo.ops.providers.npu.mega_moe import MoonEPMegaMoe

    monkeypatch.setattr(expert_module, "ops", SimpleNamespace(mega_moe=MoonEPMegaMoe()))
    config = MoEConfig(hidden_size=4, moe_ffn_hidden_size=8,
                       num_moe_experts=4, moe_router_topk=2,
                       gated_linear_unit=True, activation_func=F.silu)
    layer = MoELayer.__new__(MoELayer)
    nn.Module.__init__(layer)
    layer.config = config
    layer.experts = GroupedMLP(4, False, config)
    model = nn.Sequential(layer)
    identities = {name: id(value) for name, value in model.named_parameters()}
    keys = set(model.state_dict())
    engine_config = EngineConfig(mega_moe=True)
    engine_config.activation_checkpointing.enabled = True
    engine_config.activation_checkpointing.use_reentrant = True
    monkeypatch.setattr(integration.dist, "get_world_size", lambda group: 1)
    engine_config.extra["mega_moe_local_tokens"] = 128
    integration.install(model, engine_config, ep_group=None)
    assert identities == {name: id(value) for name, value in model.named_parameters()}
    assert keys == set(model.state_dict())
    observed = []
    layer.experts.register_forward_pre_hook(lambda *_: observed.append("unshard"))
    layer.experts.register_forward_hook(lambda *_: observed.append("reshard"))

    from torchturbo.optimizations.mega_moe.moonep.backend import MoonEPMegaMoeBackend

    def execute(self, experts, hidden, indices, probs):
        observed.append("execute")
        assert indices.tolist() == [[1, 3], [0, 2]]
        torch.testing.assert_close(probs, torch.tensor([[0.2, 0.8], [0.7, 0.3]]))
        assert experts.weight1 is layer.experts.weight1
        assert experts.weight2 is layer.experts.weight2
        assert self.token_bound == 128
        return hidden * experts.weight1.mean()

    monkeypatch.setattr(MoonEPMegaMoeBackend, "execute", execute)
    hidden = torch.ones(2, 4, requires_grad=True)
    probs = torch.tensor([[0., 0.2, 0., 0.8], [0.7, 0., 0.3, 0.]])
    routing_map = probs.ne(0)

    def reject_nonzero(*args, **kwargs):
        raise AssertionError("MegaMoE routing conversion must have a fixed output shape.")

    monkeypatch.setattr(torch, "nonzero", reject_nonzero)
    output, bias = layer._routed_experts_backend.route(
        layer.experts, hidden, probs, routing_map
    )
    assert observed == ["unshard", "execute", "reshard"]
    assert bias is None
    output.sum().backward()
    assert layer.experts.weight1.grad is not None


def test_missing_capacity_rejected():
    with pytest.raises(ValueError, match="token capacity"):
        integration.install(nn.Sequential(), EngineConfig(mega_moe=True), None)


def test_disabled_bootstrap_ignores_legacy_environment(tmp_path, monkeypatch):
    import os
    import runpy
    from pathlib import Path

    monkeypatch.setenv("TORCHTURBO_MOE_BACKEND", "mega_moe")
    bootstrap = Path(integration.__file__).with_name("bootstrap.py")
    activate = runpy.run_path(str(bootstrap))["activate"]
    path = tmp_path / "disabled.yaml"
    path.write_text("engine_config: {}\n")
    before = dict(os.environ)
    assert activate(["--tt_config", str(path)]) is False
    assert dict(os.environ) == before
    assert EngineConfig().mega_moe is False


def test_selected_zero_weight_preserves_expert_identity():
    captured={}
    class Experts:
        config=SimpleNamespace(moe_router_topk=2)
        def __call__(self,hidden,indices,weights):
            captured.update(indices=indices,weights=weights)
            return hidden,None
    backend=integration.RoutedExpertsBackend(None,128)
    mask=torch.tensor([[False,True,False,True]])
    probs=torch.tensor([[0.,0.,0.,1.]])
    backend.route(Experts(),torch.ones(1,4),probs,mask)
    assert captured["indices"].tolist()==[[1,3]]
    assert captured["weights"].tolist()==[[0.,1.]]


def test_bootstrap_resolves_source_build(tmp_path,monkeypatch):
    import importlib.util
    import os
    from pathlib import Path
    import runpy
    package=Path(importlib.util.find_spec("torchturbo.optimizations.mega_moe.runtime").origin).parent.resolve()
    original_find_spec = importlib.util.find_spec
    shmem_lib = tmp_path / "shmem/backends/910"
    shmem_lib.mkdir(parents=True)
    for name in ("libshmem.so", "libshmem_utils.so"):
        (shmem_lib / name).touch()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs:
                        SimpleNamespace(origin=str(tmp_path / "shmem/__init__.py"))
                        if name == "shmem" else original_find_spec(name, *args, **kwargs))
    config=tmp_path/"enabled.yaml"
    config.write_text("engine_config:\n  mega_moe: true\n")
    initial_keys=set(os.environ)
    monkeypatch.setenv("LD_LIBRARY_PATH","")
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH","")
    calls=[]
    monkeypatch.setattr(os,"execv",lambda executable,arguments:calls.append((executable,arguments)))
    activate=runpy.run_path(str(Path(integration.__file__).with_name("bootstrap.py")))["activate"]
    assert activate(["--tt_config",str(config)]) is True
    assert not any(key.startswith(("TT_", "_TT_")) for key in set(os.environ)-initial_keys)
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0]==str(package/"lib/vendors/mega_moe/op_api/lib")
    assert len(calls)==1
    assert activate(["--tt_config",str(config)]) is True
    assert len(calls)==1


@pytest.mark.parametrize("mode,limit,expected", [
    (None, None, "none"), (None, 4., "post_activation"),
    ("none", None, "none"), ("pre_activation", 4., "pre_activation"),
    ("post_activation", 4., "post_activation"),
])
def test_mode_resolution(mode, limit, expected):
    activation = Path(importlib.util.find_spec("torchturbo.optimizations.mega_moe.runtime").origin).parent / "activation.py"
    assert runpy.run_path(str(activation))["resolve_mode"](mode, limit) == expected


@pytest.mark.parametrize("mode,limit", [
    ("other", 4.), ("none", 4.), ("pre_activation", None),
    ("post_activation", None),
])
def test_incompatible_mode_and_limit_rejected(mode, limit):
    activation = Path(importlib.util.find_spec("torchturbo.optimizations.mega_moe.runtime").origin).parent / "activation.py"
    with pytest.raises(ValueError):
        runpy.run_path(str(activation))["resolve_mode"](mode, limit)


@pytest.mark.parametrize("mode,limit,signed", [
    ("none", None, 0.), ("pre_activation", 4., -4.),
    ("post_activation", 4., 4.),
])
def test_serialized_mode_keeps_descriptor_layout(monkeypatch, mode, limit, signed):
    component = Path(importlib.util.find_spec("torchturbo.optimizations.mega_moe.runtime").origin).parent.parent
    monkeypatch.syspath_prepend(str(component))
    from torchturbo.optimizations.mega_moe.runtime.tasks.swiglu import _encode_clamp_limit

    bits = _encode_clamp_limit(limit, mode)
    assert struct.unpack("<f", struct.pack("<I", bits))[0] == signed


def test_clipped_gradient_tile_reserves_vector_api_workspace(monkeypatch):
    import ctypes
    component = Path(importlib.util.find_spec("torchturbo.optimizations.mega_moe.runtime").origin).parent.parent
    monkeypatch.syspath_prepend(str(component))
    from torchturbo.optimizations.mega_moe.runtime.scheduler.config import SwiGluTilingDataC
    from torchturbo.optimizations.mega_moe.runtime.modules.mega_moe.backward.tiling_tables import get_swiglu_grad_tiling_bytes

    original = get_swiglu_grad_tiling_bytes(128, intermediate_size=256)
    clipped = get_swiglu_grad_tiling_bytes(128, intermediate_size=256, clipped=True)
    size = ctypes.sizeof(SwiGluTilingDataC)
    old = SwiGluTilingDataC.from_buffer_copy(original[:size])
    safe = SwiGluTilingDataC.from_buffer_copy(clipped[:size])
    assert old.baseRowLen == 19
    # Queue and FP32 storage must end before CANN's scalar-Select scratch area.
    assert safe.baseRowLen * safe.baseColLen * 40 <= 184 * 1024
    assert len(original) == len(clipped)


@pytest.mark.parametrize("mode,limit", [("none", None), ("pre_activation", .25), ("post_activation", .25)])
def test_execution_owner_records_semantic_mode(monkeypatch, mode, limit):
    from torchturbo.optimizations.mega_moe.runtime import resources
    captured={}
    def bind(specification, tensor):
        captured.update(specification)
        return SimpleNamespace(receive_capacity=256)
    monkeypatch.setattr(resources,"bind_mega_moe_spec",bind)
    owner=resources.ExecutionResources(torch.empty(0,dtype=torch.bfloat16),local_num_tokens=128,
        hidden_size=256,intermediate_size=256,num_experts=8,top_k=2,ep_size=2,ep_group=None,
        initial_capacity_factor=1.3,swiglu_limit=limit,swiglu_mode=mode)
    assert captured["swiglu_mode"]==mode
    assert owner.workspace.capacity_floor==256
