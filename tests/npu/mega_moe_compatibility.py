"""Compare fused offloaded training with native resident MoE training."""
import argparse
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / 'src/torchturbo/optimizations/mega_moe/runtime/bootstrap.py'))['activate']()
import torch
import torch_npu
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.utils.checkpoint import checkpoint
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_layer import MoELayer, MoESubmodules
from torchturbo.models.modules.moe.experts import GroupedMLP
from torchturbo.models.modules.moe.module import ModuleSpec
from torchturbo.optimizations.mega_moe.integration import install
from torchturbo.engine.backends.fsdp2.backend import FSDP2Backend
from torchturbo.engine.backends.fsdp2.config import FSDPConfig, CPUOffloadConfig
from torchturbo.engine.optimizers import OptimizerClassPlan, MuonPlan, MuonRecipe
from torchturbo.engine.config import MuonOptimizerConfig
from torchturbo.runtime.npu import NpuRuntime
from torchturbo.ops import get_or_create_ops

H, I, E, K, T = 256, 256, 8, 2, 2048


class Model(nn.Module):
    def __init__(self, config, group, cp):
        super().__init__()
        self.layer = MoELayer(config, group,
            submodules=MoESubmodules(experts=ModuleSpec(module=GroupedMLP)),
            use_fused_permute_unpermute=True, layer_number=0)
        self.cp = cp

    def forward(self, inputs):
        return checkpoint(self.layer, inputs, use_reentrant=self.cp)[0]


def local(tensor):
    return (tensor.to_local() if hasattr(tensor, 'to_local') else tensor).detach().float().cpu()


def relative(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-12))


def configure(model, dense, expert, full):
    options = dict(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                   reduce_dtype=torch.float32), reshard_after_forward=True)
    if full:
        options['offload_policy'] = CPUOffloadPolicy(pin_memory=True)
    fully_shard(model.layer.experts, mesh=expert, **options)
    fully_shard(model.layer, mesh=dense, **options)
    fully_shard(model, mesh=dense, **options)


def optimizer(model, mesh, offload, kind):
    backend = FSDP2Backend.__new__(FSDP2Backend)
    backend.fsdp_config = FSDPConfig(cpu_offload=offload)
    backend._device_runtime = NpuRuntime()
    backend._mesh = {'edp': mesh}
    if kind == 'muon':
        plan = MuonPlan(model, recipe=MuonRecipe({}), config=MuonOptimizerConfig(
            lr=1e-3, weight_decay=.01, ns_steps=5, overlap_comm_compute=True,
            adamw_fused=True))
    else:
        plan = OptimizerClassPlan(model, torch.optim.AdamW,
                                 {'lr': 1e-3, 'weight_decay': .01, 'fused': True})
    return backend.prepare_optimizer(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offload', choices=('none', 'full', 'optimizer'), default='full')
    parser.add_argument('--checkpoint', choices=('reentrant', 'nonreentrant'), default='nonreentrant')
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    parser.add_argument('--strict-gradients', action='store_true',
                        help='Require elementwise rtol=1%% and atol=1e-7; report this check separately otherwise')
    parser.add_argument('--optimizer-memory-fraction', type=float, default=.001)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    device = torch.device('npu', int(os.environ['LOCAL_RANK']))
    torch.npu.set_device(device)
    get_or_create_ops(device_runtime=NpuRuntime())
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (dist.get_world_size() // 2, 2), mesh_dim_names=('edp', 'ep'))
    dense = init_device_mesh('npu', (dist.get_world_size(),), mesh_dim_names=('dp',))
    mode = args.offload
    cp = args.checkpoint == 'reentrant'
    muon = args.optimizer == 'muon'
    fraction = args.optimizer_memory_fraction
    offload = CPUOffloadConfig(enabled=mode != 'none', mode=mode if mode != 'none' else 'full',
                               optimizer_memory_fraction=fraction)
    config = MoEConfig(hidden_size=H, moe_ffn_hidden_size=I, num_moe_experts=E,
                       expert_model_parallel_size=2,
                       moe_router_topk=K, gated_linear_unit=True, activation_func=F.silu,
                       moe_token_dispatcher_type='alltoall', moe_router_dtype='fp32')
    torch.manual_seed(123)
    model = Model(config, mesh['ep'].get_group(), cp).to(device)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.normal_(0, .02)
        model.layer.router.weight[:K].add_(.08)
        model.layer.router.weight[K:].sub_(.08)
    reference = Model(config, mesh['ep'].get_group(), cp).to(device)
    reference.load_state_dict(model.state_dict())
    install(model, SimpleNamespace(extra={'mega_moe_local_tokens': T}), mesh['ep'].get_group())
    routed = [[], []]
    for index, target in enumerate((model, reference)):
        def capture(module, args, output, *, bucket=routed[index]):
            probability = output[0].probs
            if probability.requires_grad:
                probability.retain_grad()
                bucket.append(probability)
        target.layer.router.register_forward_hook(capture)
    configure(model, dense, mesh['edp'], mode == 'full')
    configure(reference, dense, mesh['edp'], muon and mode == 'full')
    opt = optimizer(model, mesh['edp'], offload, args.optimizer)
    ref_opt = optimizer(reference, mesh['edp'], offload if muon else CPUOffloadConfig(enabled=False), args.optimizer)
    residency = {name: str(parameter.device) for name, parameter in model.named_parameters()}
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    expected_device = 'cpu' if mode == 'full' else 'npu'
    assert all(parameter.device.type == expected_device for parameter in model.parameters()), residency
    results = []
    for step in range(2):
        losses, errors, weight_errors, input_errors, diagnostics, mixed_checks = [], {}, {}, [], {}, {}
        for lane in range(2):
            torch.manual_seed(456 + rank * 19 + step * 3 + lane)
            x = torch.randn(T - lane * 128, H, device=device).abs().requires_grad_()
            xr = x.detach().clone().requires_grad_()
            rng = torch.npu.get_rng_state()
            actual = model(x)
            torch.npu.set_rng_state(rng)
            expected = reference(xr)
            loss = actual.float().square().mean() / 2
            ref_loss = expected.float().square().mean() / 2
            assert abs(float(loss - ref_loss)) < max(1e-6, abs(float(ref_loss)) * .03)
            loss.backward()
            ref_loss.backward()
            input_errors.append(relative(x.grad.float().cpu(), xr.grad.float().cpu()))
            assert input_errors[-1] < .04, ('input_grad', input_errors[-1])
            losses.append([float(loss), float(ref_loss)])
        for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            if parameter.grad is None:
                assert ref_parameter.grad is None, name
                continue
            a, b = local(parameter.grad), local(ref_parameter.grad)
            errors[name] = relative(a, b)
            diagnostics[name] = dict(actual_norm=float(a.norm()), reference_norm=float(b.norm()),
                                     difference_norm=float((a-b).norm()), max_absolute_error=float((a-b).abs().max()))
            mixed_checks[name] = bool(torch.isclose(a, b, rtol=.01, atol=1e-7).all())
            if args.strict_gradients:
                torch.testing.assert_close(a, b, rtol=.01, atol=1e-7, msg=name)
            else:
                # Retain the joint test's stated BF16 relative-gradient ceiling.
                assert errors[name] < .05, (name, errors[name])
        squares = torch.tensor([
            sum(value['actual_norm'] ** 2 for value in diagnostics.values()),
            sum(value['reference_norm'] ** 2 for value in diagnostics.values()),
            sum(value['difference_norm'] ** 2 for value in diagnostics.values()),
        ], dtype=torch.float32, device=device)
        dist.all_reduce(squares)
        actual_norm, reference_norm, difference_norm = squares.sqrt().cpu().tolist()
        aggregate = dict(actual_gradnorm=actual_norm, reference_gradnorm=reference_norm,
                         relative_l2=difference_norm / max(reference_norm, 1e-12),
                         gradnorm_relative_difference=abs(actual_norm-reference_norm) / max(reference_norm, 1e-12))
        route_a = [p.grad for p in routed[0] if p.grad is not None]
        route_b = [p.grad for p in routed[1] if p.grad is not None]
        assert len(route_a) == len(route_b) == 2, (len(route_a), len(route_b))
        route_errors = [relative(a.float().cpu(), b.float().cpu()) for a, b in zip(route_a, route_b)]
        assert max(route_errors) < .04, ('route_grad', route_errors)
        routed[0].clear(); routed[1].clear()
        opt.step(); ref_opt.step()
        assert all(parameter.device.type == expected_device for parameter in model.parameters())
        for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            weight_errors[name] = relative(local(parameter), local(ref_parameter))
            assert weight_errors[name] < .02, ('updated_weight', name, weight_errors[name])
        state_devices = [value.device.type for state in opt.state.values()
                         for key, value in state.items() if isinstance(value, torch.Tensor) and key != 'step']
        if mode != 'none':
            assert state_devices and set(state_devices) == {'cpu'}, state_devices
        results.append(dict(step=step, losses=losses, input_gradient_errors=input_errors,
                            router_probability_gradient_errors=route_errors,
                            parameter_gradient_errors=errors, updated_weight_errors=weight_errors,
                            parameter_gradient_diagnostics=diagnostics,
                            aggregate_parameter_gradient=aggregate,
                            strict_relative_1pct_passed=all(error < .01 for error in errors.values()),
                            parameter_mixed_close_passed=mixed_checks,
                            optimizer_state_devices=state_devices))
        opt.zero_grad(set_to_none=True); ref_opt.zero_grad(set_to_none=True)
    backend = model.layer._routed_experts_backend
    assert not backend.resources.in_use
    stats = dict(backend.resources.statistics)
    del actual, expected, loss, ref_loss
    backend.close()
    result = dict(rank=rank, mode=mode, checkpoint_reentrant=cp,
                  gradient_accumulation=2, optimizer_memory_fraction=fraction,
                  optimizer='muon' if muon else 'adamw', reference_uses_same_offload=muon,
                  gradient_check_kind='strict_mixed' if args.strict_gradients else 'joint_relative_5pct',
                  strict_gradient_rtol=.01, strict_gradient_atol=1e-7,
                  parameter_residency_after_reshard=residency, runtime=stats, results=results,
                  cleanup='pass')
    print('OFFLOAD_COMPATIBILITY_RESULT ' + json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
