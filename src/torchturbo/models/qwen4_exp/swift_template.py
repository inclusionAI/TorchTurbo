"""Register Qwen4-Exp with the compatible Qwen3.5 Swift template."""

from __future__ import annotations

from ...logging import get_logger
from ..qwen3_5.swift_template import register as register_qwen3_5_template

logger = get_logger(__name__)

NATIVE_MODEL_TYPE = "qwen3_5_moe"
TEMPLATE_TYPE = "qwen3_5"


def register() -> None:
    """Point ``turbo_qwen4_exp`` at the Qwen3.5 template and loader chain."""
    register_qwen3_5_template()

    from swift.model.model_meta import MODEL_MAPPING

    donor_meta = MODEL_MAPPING.get(f"turbo_{NATIVE_MODEL_TYPE}")
    turbo_meta = MODEL_MAPPING.get("turbo_qwen4_exp")
    if donor_meta is None or turbo_meta is None:
        logger.warning_rank0(
            "qwen4_exp needs turbo_%s registered first; "
            "keeping the fallback loader/template",
            NATIVE_MODEL_TYPE,
        )
        return
    native_meta = MODEL_MAPPING[NATIVE_MODEL_TYPE]

    turbo_meta.template = TEMPLATE_TYPE
    turbo_meta.model_groups[0].template = TEMPLATE_TYPE
    # candidate_templates is computed in ModelMeta.__post_init__; patching
    # template after the fact leaves it stale, so refresh it explicitly.
    turbo_meta.candidate_templates = [TEMPLATE_TYPE]
    turbo_meta.loader = donor_meta.loader
    turbo_meta.requires = list(native_meta.requires)
    turbo_meta.tags = list(native_meta.tags)


__all__ = ["NATIVE_MODEL_TYPE", "TEMPLATE_TYPE", "register"]
