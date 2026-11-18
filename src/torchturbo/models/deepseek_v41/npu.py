"""Source BSND mHC and RMSNorm patches."""
import os
from types import MethodType
import torch
from torchturbo.logging import get_logger
from torchturbo.ops import ops
from .configuration_deepseek_v41 import text_config_of
logger = get_logger(__name__)

def _npu_hc_pre(self, hidden_states, fn, scale, base, pre_mix):
    """Fused mHC for every call: training forward, checkpoint recompute,
    and no-grad forward alike.

    Routing the no-grad path (reentrant AC's initial forward, eval) through
    the fused kernel keeps forward/backward on the same numerics under
    reentrant checkpointing and matches the non-reentrant default, where
    the initial forward already runs fused.
    """
    sinkhorn = self.hc.hc_split_sinkhorn
    hin, post, residual, pre = ops.mhc_pre_sinkhorn_premix(
        hidden_states, fn, scale, base, pre_mix,
        self.hc_mult,
        sinkhorn.hc_sinkhorn_iters,
        sinkhorn.hc_eps,
        self.hc.norm_eps,
    )
    comb = residual.view(
        hidden_states.size(0), hidden_states.size(1), self.hc_mult, self.hc_mult
    )
    return hin, post, comb, pre

def patch_deepseek_v41_npu_mhc(model: torch.nn.Module) -> None:
    """Install required mHC before activation checkpointing.

    Missing kernels or unsupported configurations fail before patching.
    """
    from torchturbo.ops.providers.npu.mhc_pre_sinkhorn_premix import (
        has_mhc_pre_sinkhorn_premix,
    )

    if not has_mhc_pre_sinkhorn_premix():
        raise RuntimeError("NPU training requires mhc_pre_sinkhorn_premix and its backward operator.")
    config = text_config_of(model.config)
    # Hard constraints of the fused backward kernel.
    if config.hc_mult != 4:
        raise ValueError(f"Fused mHC requires hc_mult=4, got {config.hc_mult}.")
    if config.hidden_size % 128 != 0:
        raise ValueError(f"Fused mHC requires hidden_size divisible by 128, got {config.hidden_size}.")
    if config.hc_sinkhorn_iters != 20:
        raise ValueError(f"Fused mHC requires hc_sinkhorn_iters=20, got {config.hc_sinkhorn_iters}.")
    from . import modeling_deepseek_v41 as dsv41

    for module in model.modules():
        if isinstance(module, dsv41.DeepseekV41DecoderLayer):
            module._hc_pre = MethodType(_npu_hc_pre, module)
    logger.info("deepseek_v41: fused NPU mhc_pre_sinkhorn_premix enabled")

def _env_flag_enabled(name: str) -> bool:
    """True when an env var is set to an explicit true value."""
    return os.environ.get(name, "").lower() in ("1", "true", "yes")

def _npu_rms_norm_forward(self, hidden_states):
    """Fused npu_rms_norm replacement for DeepseekV41RMSNorm.forward."""
    return ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)

def patch_deepseek_v41_npu_rms_norm(model: torch.nn.Module) -> None:
    """Route DeepseekV41RMSNorm instances through the fused npu_rms_norm op.

    One kernel replaces the fp32 small-op chain; the op keeps gamma in fp32,
    matching the v41 math. Instance-level like the mHC patches, so modules
    constructed outside this model keep the eager chain.
    TORCH_TURBO_DISABLE_V41_FUSED_RMS_NORM restores the eager chain.
    """
    if _env_flag_enabled("TORCH_TURBO_DISABLE_V41_FUSED_RMS_NORM"):
        return
    from . import modeling_deepseek_v41 as dsv41

    patched = 0
    for module in model.modules():
        if isinstance(module, dsv41.DeepseekV41RMSNorm):
            module.forward = MethodType(_npu_rms_norm_forward, module)
            patched += 1
    logger.info("deepseek_v41: fused NPU rms_norm enabled, patched_norms=%d", patched)

def _npu_hc_post_forward(self, x, residual, post, comb):
    """Fused mhc_post replacement for HcPost.forward (bound per instance)."""
    return ops.mhc_post(residual, comb, x, post)

def patch_deepseek_v41_npu_hc_post(model: torch.nn.Module) -> None:
    """Install the fused mhc_post HcPost patch.

    Runs after patch_deepseek_v41_npu_mhc, so hc_mult=4 and hidden_size % 128
    are already validated. Missing kernels fail before patching; the bmm
    HcPost fallback stays installed when disabled via the env flag.
    """
    if _env_flag_enabled("TORCH_TURBO_DISABLE_V41_FUSED_HC_POST"):
        return
    from torchturbo.ops.providers.npu.mhc_post import has_mhc_post

    if not has_mhc_post():
        raise RuntimeError("NPU training requires mhc_post and its backward operator.")
    from . import modeling_deepseek_v41 as dsv41

    for module in model.modules():
        if isinstance(module, dsv41.HcPost):
            module.forward = MethodType(_npu_hc_post_forward, module)
    logger.info("deepseek_v41: fused NPU mhc_post enabled")
