"""Build unsharded CPU checkpoints using the TorchTurbo model initialization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INTEGRATION_ROOT.parents[1]
INDEX_FILENAME = "model.safetensors.index.json"


def asset_path(value: str) -> Path:
    """Resolve local assets without replacing a named symlink with its blob."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = INTEGRATION_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path.absolute()


def save_hf_checkpoint(model: nn.Module, assets_dir: Path, output_dir: Path) -> Path:
    """Save an initialized model alongside the case's Hugging Face assets.

    Everything the training framework needs besides the weights -- tokenizer,
    chat template, and the processor configs ``AutoProcessor`` resolves -- is
    copied verbatim from ``assets_dir``. A model that needs one more artifact
    adds a file there instead of code here.

    The training framework loads weights through ``model.safetensors.index.json``.
    ``save_pretrained`` writes that index only when the model is sharded, so a
    single-file save gets its index built here from the saved weights.
    """
    from safetensors import safe_open

    output_dir.mkdir(parents=True, exist_ok=False)
    # Fresh weights use native model keys, without legacy checkpoint renaming.
    model.save_pretrained(
        output_dir, safe_serialization=True, save_original_format=False
    )
    shutil.copytree(assets_dir, output_dir, dirs_exist_ok=True)
    index_path = output_dir / INDEX_FILENAME
    if not index_path.exists():
        weight_file = output_dir / "model.safetensors"
        with safe_open(weight_file, framework="pt", device="cpu") as weights:
            keys = weights.keys()
            weight_map = {key: weight_file.name for key in keys}
            total_size = 0
            for key in keys:
                tensor = weights.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
        index_path.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                indent=2,
            )
            + "\n"
        )
    return output_dir


def turbo_model_class(model_type: str) -> type:
    """Return the TorchTurbo model class registered for ``model_type``.

    ``model_type`` is the architecture name from ``MODEL_ARGS``, which is also
    what the saved ``config.json`` carries. The training framework resolves its
    model class from that name through this same registry, so the checkpoint is
    built by exactly the class that will later load it.
    """
    from torchturbo.models.registry import get_model

    model_cls = get_model(model_type)
    if model_cls is None:
        raise NotImplementedError(
            f"No TorchTurbo model is registered for model_type {model_type!r}"
        )
    return model_cls


def generate_checkpoint(
    case: dict[str, Any], output_dir: Path, engine_config_path: Path
) -> Path:
    """Initialize the model described by ``case`` and save the checkpoint.

    ``engine_config_path`` is the engine configuration the training run itself
    receives. The engine config decides which parameters a model even has --
    MTP layers are stripped unless ``mtp.enabled`` -- so generation must read
    the same file rather than engine defaults.
    """
    from torchturbo.engine.args import TorchTurboArgs
    from torchturbo.engine.config import build_from_args

    model_cls = turbo_model_class(case["MODEL_ARGS"]["model_type"])
    config, _ = case_config_and_tokenizer(case, model_cls.config_class)
    engine_config = build_from_args(TorchTurboArgs(config_path=str(engine_config_path)))
    model = build_cpu_model(
        model_cls, config, engine_config, case["TRAIN_ARGS"]["seed"]
    )
    save_hf_checkpoint(model, asset_path(case["TRAIN_ARGS"]["tokenizer"]), output_dir)
    print(
        f"Initialized {sum(p.numel() for p in model.parameters()):,} parameters: {output_dir}"
    )
    return output_dir


def case_config_and_tokenizer(
    case: dict[str, Any], config_cls: type
) -> tuple[Any, Any]:
    """Build the model config from ``MODEL_ARGS`` and align it with the tokenizer.

    Returns the config object and the tokenizer loaded from
    ``TRAIN_ARGS.tokenizer``. The vocabulary size and the special token ids are
    derived from that tokenizer, because a freshly initialized checkpoint and
    the tokenizer saved next to it must agree on both.
    """
    from transformers import AutoTokenizer

    train_args = case["TRAIN_ARGS"]
    if train_args.get("model") or train_args.get("model_name_or_path"):
        raise ValueError(
            "Checkpoint generation requires the training model path to be unset"
        )
    if not train_args.get("tokenizer"):
        raise ValueError("TRAIN_ARGS.tokenizer is required when model is unset")
    tokenizer = AutoTokenizer.from_pretrained(
        asset_path(train_args["tokenizer"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    model_args = dict(case["MODEL_ARGS"])
    model_type = model_args.pop("model_type")
    # A vision-language config nests the language model under ``text_config``.
    # Vocabulary size and special token ids describe the language model, so they
    # belong there; setting them on the outer config would leave the text config
    # at its defaults and mismatch the tokenizer saved next to the checkpoint.
    token_args = (
        model_args["text_config"] if model_type == "qwen3_5_moe" else model_args
    )
    min_vocab_size = max(tokenizer.get_vocab().values()) + 1
    token_args.setdefault("vocab_size", min_vocab_size)
    if token_args["vocab_size"] < min_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {min_vocab_size} for this tokenizer"
        )
    for name, token_id in (
        ("pad_token_id", tokenizer.pad_token_id),
        ("eos_token_id", tokenizer.eos_token_id),
        ("bos_token_id", tokenizer.bos_token_id),
    ):
        if name in token_args and token_args[name] != token_id:
            raise ValueError(
                f"MODEL_ARGS.{name} does not match the tokenizer ({token_id})"
            )
        token_args.setdefault(name, token_id)
    return config_cls(**model_args), tokenizer


def build_cpu_model(model_cls: type, config: Any, engine_config: Any, seed: int) -> Any:
    """Initialize ``model_cls`` on CPU through the engine's own model interfaces.

    ``TurboModel.init_context`` is what the engine enters before constructing a
    model, so the checkpoint is built under the same attention implementation,
    float32 parameter dtype and MTP policy the training run uses. Three steps of
    ``TorchTurboEngine.build_model`` are deliberately left out: its meta device
    (parameters must hold real initialized values here), backend sharding (the
    checkpoint has to stay an unsharded HF-format model the weight converter can
    read) and weight loading (there is nothing to load yet).
    """
    import torch

    torch.manual_seed(seed)
    with (
        model_cls.init_context(config, engine_config=engine_config),
        torch.device("cpu"),
    ):
        model = model_cls._from_config(config)
    # The constructor's post_init already initialized the model. Do not reapply it.
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Non-finite initialized parameter: {name}")
    return model
