"""Record DeepSeekV4 SFA loss / gradient norms for baseline comparison.

Runs identical fwd+bwd steps on a small kernel-constrained DeepSeekV4 (the SFA
kernels fix head_dim=512, 64 q heads, sliding_window=128 and cmp topk=512 --
only the other dims are shrunk) and dumps per-step loss / gradient norms to
JSON. The v3-baseline run uses the v3 code checkout; the smla-v2 run uses the
dev-smla-v2 checkout (one experiment folder each, per the ZCode.md workflow):

    # v3 baseline dir
    python tests/models/deepseek_v4/compare_smla_loss_gn.py --tag v3 --out v3.json
    # dev-smla-v2 dir
    python tests/models/deepseek_v4/compare_smla_loss_gn.py --tag smla-v2 --out v2.json
    # anywhere
    python tests/models/deepseek_v4/compare_smla_loss_gn.py --compare v3.json v2.json

Use --train to also apply SGD updates between steps.

Note: v2 HCA layers attend over the full visible compressed prefix while v3
selected top-512 compressed blocks, so a systematic (non-noise) deviation in
loss/gn is expected to be dominated by that semantic change until parity data
is reviewed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for _path in (SRC_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SEED = 20260917

# Fixed by the SFA kernels (head_dim=512, 64 q heads, window 128, topk 512);
# everything else is shrunk for speed.
KERNEL_TEST_CONFIG = {
    "vocab_size": 1024,
    "hidden_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 448,
    "q_lora_rank": 256,
    "o_lora_rank": 256,
    "o_groups": 4,
    "intermediate_size": 1024,
    "moe_intermediate_size": 512,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 1.0,
    "scoring_func": "sqrtsoftplus",
    "route_scale": 1.0,
    "aux_loss_alpha": 0.0,
    "sliding_window": 128,
    "compress_ratios": [4, 128],
    "index_head_dim": 128,  # LightningIndexerV2 requires 128
    "index_n_heads": 8,
    "index_topk": 512,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 2048,
    "original_max_position_embeddings": 0,
    "compress_rope_theta": 40000.0,
    "rope_factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "num_hash_layers": 1,
    "num_nextn_predict_layers": 0,
    "swiglu_limit": 10.0,
    "hc_eps": 1e-6,
    "initializer_range": 0.02,
    "use_cache": False,
}

GRAD_GROUPS = ("attn.wkv", "compressor", "attn.wq", "attn_sink", "indexer")


def seed_all(seed=SEED):
    import torch

    torch.manual_seed(seed)


def _npu_ready() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, AttributeError, RuntimeError):
        return False


def _make_batches(steps, batch_size, seq_len, device):
    import torch

    batches = []
    for step in range(steps):
        g = torch.Generator().manual_seed(SEED + step)
        input_ids = torch.randint(
            0, KERNEL_TEST_CONFIG["vocab_size"], (batch_size, seq_len), generator=g
        )
        batches.append((input_ids.to(device), input_ids.to(device)))
    return batches


def _build_model(device, dtype):
    from torchturbo.models.deepseek_v4 import modeling_deepseek_v4 as dsv4
    from torchturbo.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from torchturbo.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
    from torchturbo.ops import get_or_create_ops
    from torchturbo.runtime import get_device_runtime

    # Same hadamard substitution the alignment tests use, so every checkout
    # shares one deterministic transform implementation.
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from test_alignment import hadamard_transform_impl
    except ImportError:  # pragma: no cover - pytest context
        from .test_alignment import hadamard_transform_impl

    dsv4.hadamard_transform_ref = lambda x, hadamard_mat, scale=1.0: hadamard_transform_impl(x, scale)
    dsv4.rotate_activation = lambda x, hadamard_mat: hadamard_transform_impl(x, scale=x.size(-1) ** -0.5)

    # The model patch obtains SFA/LI through the process-wide dispatcher.
    get_or_create_ops(device_runtime=get_device_runtime())

    # model.py's patch installs the packed runtime + fused NPU forwards.
    from torchturbo.models.deepseek_v4.model import TurboDeepseekV4ForCausalLM

    TurboDeepseekV4ForCausalLM.patch_deepseek_v4_attention_for_sp(
        None, 1, 0
    )

    config = DeepseekV4Config(**KERNEL_TEST_CONFIG)
    model = DeepseekV4ForCausalLM(config).to(device=device)
    # Cast parameters only: the freqs_cis buffer is complex64 and a whole-model
    # .to(dtype) silently turns it real, breaking view_as_real downstream.
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(dtype)
    model.eval()  # no dropout; gradients still flow
    return model


def _grad_norms(model):
    group_sq = {name: 0.0 for name in GRAD_GROUPS}
    total_sq = 0.0
    for param_name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = param.grad.detach().float().pow(2).sum().item()
        total_sq += sq
        for group in GRAD_GROUPS:
            if group in param_name:
                group_sq[group] += sq
    norms = {"global": total_sq**0.5}
    norms.update({f"gn:{g}": group_sq[g] ** 0.5 for g in GRAD_GROUPS})
    return norms


def run_steps(tag, steps, batch_size, seq_len, device, dtype, train, lr):
    import torch

    seed_all()
    model = _build_model(device, dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) if train else None
    batches = _make_batches(steps, batch_size, seq_len, "cpu")

    records = []
    for step, (input_ids, labels) in enumerate(batches):
        input_ids, labels = input_ids.to(device), labels.to(device)
        # all-ones mask: single unpacked sequence per batch row; the fused TND
        # path requires a real mask tensor.
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        record = {"step": step, "loss": loss.item(), **_grad_norms(model)}
        records.append(record)
        if optimizer is not None:
            optimizer.step()
        print(
            f"[{tag}] step {step:3d} loss={record['loss']:.6f} "
            f"gn={record['global']:.4e} "
            + " ".join(f"{record[f'gn:{g}']:.3e}" for g in GRAD_GROUPS),
            flush=True,
        )
    return {"tag": tag, "records": records}


def _rel(a, b):
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def compare_results(a, b, tol):
    print(f"\n=== loss / grad-norm comparison: {a['tag']} vs {b['tag']} ===")
    header = ["step", "loss_a", "loss_b", "loss_rel"] + [f"gn:{g}" for g in GRAD_GROUPS] + ["gn:global"]
    print("{:>4} {:>12} {:>12} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(*header))
    worst = {"loss": 0.0, "gn:global": 0.0}
    for ra, rb in zip(a["records"], b["records"]):
        row = [
            ra["step"], f"{ra['loss']:.6f}", f"{rb['loss']:.6f}", f"{_rel(ra['loss'], rb['loss']):.2e}",
        ] + [f"{_rel(ra[f'gn:{g}'], rb[f'gn:{g}']):.2e}" for g in GRAD_GROUPS] + [
            f"{_rel(ra['global'], rb['global']):.2e}"
        ]
        print("{:>4} {:>12} {:>12} {:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(*row))
        worst["loss"] = max(worst["loss"], _rel(ra["loss"], rb["loss"]))
        worst["gn:global"] = max(worst["gn:global"], _rel(ra["global"], rb["global"]))
    print(f"\nworst rel diff: loss={worst['loss']:.3e}  global gn={worst['gn:global']:.3e}  (tol={tol:.1e})")
    print(
        "NOTE: v2 HCA attends over the full visible compressed prefix (v3: top-512"
        " blocks); systematic deviations may reflect that contract change, not a bug."
    )
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=None, help="Label stored in the result JSON.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--train", action="store_true", help="Apply SGD updates between steps.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=2e-2, help="Advisory tolerance.")
    parser.add_argument("--out", default=None, help="Write results JSON here.")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Compare two previously saved result files and exit.")
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0]) as fa, open(args.compare[1]) as fb:
            compare_results(json.load(fa), json.load(fb), args.tol)
        return 0

    if not _npu_ready():
        print("NPU (torch_npu) required; this comparison runs on the Ascend machine.")
        return 0

    import torch

    device, dtype = "npu", torch.bfloat16
    tag = args.tag or "run"
    result = run_steps(tag, args.steps, args.batch_size, args.seq_len, device, dtype,
                       args.train, args.lr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nresults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
