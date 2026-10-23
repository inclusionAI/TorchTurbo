"""TorchTurboEngine — one-stop-shop for distributed training."""

from __future__ import annotations

import dataclasses
import json
import os
import socket
from collections.abc import Callable, Iterable
from typing import Any, Iterator, Optional

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..logging import get_logger
from .backend_registry import get_available_backends, get_backend
from .backends.base import BaseBackend
from .checkpoint import CheckpointFormat
from .checkpoint.coordinator import CheckpointCoordinator
from .config import EngineConfig, FreezePolicy
from .metrics import MetricTracker
from .optimizers import MuonPlan, OptimizerClassPlan
from .optimizers.lr_scheduler import create_lr_scheduler
from .utils import has_meta_tensor
from torchturbo.adapters.rl.packed import (
    gather_logprobs_entropy,
    get_packed_mb_orig,
    prepare_packed_mb_inputs,
    shifted_packed_tensor,
)
from ..ops import get_or_create_ops
from ..runtime import BaseProfiler, get_device_runtime


logger = get_logger(__name__)


class DPBlockDistributedSampler(DistributedSampler):
    """DistributedSampler that parcels contiguous blocks over DP groups.

    Each block covers ``dp_size * rank_block`` indices: every DP group gets
    a disjoint contiguous ``rank_block`` slice, and ranks sharing a
    ``dp_rank`` (e.g. an SP group) yield the identical sequence. The
    sampler only knows the dp info and the block size; the engine picks
    ``rank_block`` as ``ga * batch_size`` with ``tt_gbs`` (whole optimizer
    steps) or ``sp_size`` (legacy full-world sweep).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        *,
        dp_size: int,
        dp_rank: int,
        rank_block: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        # Let DistributedSampler handle defaults for num_replicas / rank
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.rank_block = rank_block

        # Recompute sizing: the epoch covers whole blocks of dp_size * rank_block.
        total = len(self.dataset)  # type: ignore[arg-type]
        block = self.dp_size * self.rank_block
        if self.drop_last:
            self.num_iterations = total // block
        else:
            self.num_iterations = math.ceil(total / block)
        self.total_size = self.num_iterations * block
        # Override parent's num_samples (used by __len__ in parent)
        self.num_samples = self.num_iterations * self.rank_block

        logger.info_rank0(
            "DPBlockDistributedSampler: seed=%d, world=%d, rank=%d, "
            "dp_size=%d, dp_rank=%d, rank_block=%d, num_iterations=%d, "
            "total_samples=%d, num_samples_per_rank=%d",
            self.seed,
            self.num_replicas,
            self.rank,
            self.dp_size,
            self.dp_rank,
            self.rank_block,
            self.num_iterations,
            self.total_size,
            self.num_samples,
        )

    def __iter__(self) -> Iterator[int]:
        # Build full index list (deterministic across all ranks)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # Pad or truncate to total_size
        if len(indices) < self.total_size:
            # Cycle the full list until we reach total_size.
            indices = (indices * math.ceil(self.total_size / len(indices)))[: self.total_size]
        else:
            indices = indices[: self.total_size]

        # Each block is consecutive per-group slices; DP rank d takes
        # [d*rank_block, (d+1)*rank_block). Ranks sharing a dp_rank yield
        # the identical list.
        rank_indices = []
        for i in range(self.num_iterations):
            base = i * self.dp_size * self.rank_block + self.dp_rank * self.rank_block
            rank_indices.extend(indices[base : base + self.rank_block])

        if self.shuffle:
            g2 = torch.Generator()
            g2.manual_seed(self.seed + self.epoch + self.dp_rank)
            perm = torch.randperm(len(rank_indices), generator=g2)
            rank_indices = [rank_indices[i] for i in perm.tolist()]

        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

class EpochDrivenDataLoader(DataLoader):
    """DataLoader that forwards set_epoch to the sampler
    """
    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)


class TorchTurboEngine:
    """One-stop-shop engine for distributed training.

    Owns model preparation, distributed sharding, optimizer setup, loss
    computation, profiling, and checkpoint I/O.

    Parameters
    ----------
    config:
        An :class:`EngineConfig`, a raw ``dict``, or ``None`` (defaults).

    Example::

        engine = TorchTurboEngine({"backend": "fsdp2", "mixed_precision": "bf16"})

        # Setup
        engine.prepare_optimizer(
            optimizer_cls=torch.optim.AdamW,
            lr=1e-4,
            weight_decay=1e-2,
        )
        engine.prepare_lr_scheduler(
            scheduler_cls=CosineAnnealingLR,
            T_max=1000,
        )
    """

    def __init__(self, config: EngineConfig | dict[str, Any] | None = None) -> None:
        # Normalize to EngineConfig
        if config is None:
            self.config = EngineConfig()
        elif isinstance(config, dict):
            self.config = EngineConfig.from_dict(config)
        else:
            self.config = config

        from .plugins import PluginManager

        self.plugin_manager = PluginManager(list(self.config.plugin_paths))
        self.metric_tracker = MetricTracker()
        self._mfu_estimator: Any = None

        # Select distributed backend before configuring process-wide state.
        backend_name = self.config.backend.value
        backend_cls = get_backend(backend_name)
        if backend_cls is None:
            raise ValueError(
                f"Unknown distributed backend {backend_name!r}. "
                f"Available: {get_available_backends()}"
            )

        # Resolve the process-wide device runtime before backend construction.
        self.device_runtime = get_device_runtime()
        if self.config.use_deterministic_algorithms:
            self.device_runtime.use_deterministic_algorithms()
        self.backend: BaseBackend = backend_cls(self.config, self.device_runtime)
        self._checkpoint: CheckpointCoordinator | None = None
        self._profiler: BaseProfiler | None = None
        self.ops = get_or_create_ops(
            device_runtime=self.device_runtime,
            default_policy=self.config.ops.policy,
            overrides=self.config.ops.overrides,
        )

        # These are set by setup()
        self.model: nn.Module | None = None
        self.model_name_or_path: str | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: Any = None
        self.parallel_plan: Any = None

        logger.info_rank0(
            "TorchTurboEngine initialized: backend=%s, device=%s",
            backend_name,
            self.device_runtime.device.type,
        )

    # ------------------------------------------------------------------
    # Model preparation (meta device → shard → load weights)
    # ------------------------------------------------------------------

    def _patch_transformers(self) -> None:
        """Connect Transformers attention and packing to TorchTurbo."""
        import transformers.integrations.flash_attention as flash_integration
        import transformers.modeling_flash_attention_utils as flash_utils

        from ..models.modules.attention.interface import attention_forward
        from ..utils.flash_attn_utils import get_unpad_data

        flash_integration._flash_attention_forward = attention_forward
        # Support packed sequence IDs as well as binary padding masks.
        flash_utils._get_unpad_data = get_unpad_data

    def build_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_cls: type | None = None,
        trust_remote_code: bool = False,
        init_kwargs: dict[str, Any] | None = None,
        peft_config: Any | None = None,
        load_weights: bool = True,
    ) -> nn.Module:
        """Build a distributed model instance without registering it on the engine.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace model path (local or repo id).
        torch_dtype:
            Data type for model parameters.
        model_cls:
            Model class to use.  If ``None``, uses HuggingFace
            ``AutoModelForCausalLM`` (which will pick up registered
            torchturbo models).
        trust_remote_code:
            Passed to HuggingFace ``from_pretrained``.
        init_kwargs:
            Additional keyword arguments
        peft_config:
            Optional PEFT/LoRA configuration (``LoraConfig`` or ``dict``).
            When provided, LoRA adapters are applied **after** module
            replacement but **before** FSDP2 sharding.  Applying LoRA registers
            a custom handler for ``GroupedMLP`` (MoE experts with grouped
            matmul).
        load_weights:
            Whether to load checkpoint weights.  When ``False``, the model is
            still materialized onto real device storage but checkpoint I/O is
            skipped.

        Returns
        -------
        The sharded model instance.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        if init_kwargs is None:
            init_kwargs = {}

        # Load config
        if "config" in init_kwargs:
            config = init_kwargs["config"]
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )

        if model_cls is None:
            model_cls = AutoModelForCausalLM

        # 2. Build model on meta device (no memory)
        #
        # Try torchturbo's own model registry first (keyed by model_type).
        # This bypasses AutoModelForCausalLM.from_config entirely, avoiding
        # the type(config) mismatch that occurs when the config object was
        # created from a remote config class (e.g. via trust_remote_code).
        from transformers import PreTrainedModel

        from ..models.base import TurboModel
        from ..models.registry import get_model

        model_type = getattr(config, "model_type", None)
        turbo_cls = get_model(model_type) if model_type else None

        turbo_model_cls = turbo_cls if turbo_cls is not None else model_cls
        if not isinstance(turbo_model_cls, type) or not issubclass(
            turbo_model_cls, TurboModel
        ):
            raise TypeError(
                "TorchTurboEngine requires a TurboModel subclass. "
                f"Got {getattr(turbo_model_cls, '__name__', turbo_model_cls)!r} "
                f"for model_type={model_type!r}."
            )
        if not issubclass(turbo_model_cls, PreTrainedModel):
            raise TypeError(
                "TorchTurboModel classes must also inherit transformers.PreTrainedModel. "
                f"Got {turbo_model_cls.__name__}."
            )
        with turbo_model_cls.init_context(config, engine_config=self.config):
            logger.info_rank0("Building model on meta device: %s", model_name_or_path)
            with torch.device("meta"):
                logger.info_rank0(
                    "Using torchturbo model class: %s", turbo_model_cls.__name__
                )
                model = turbo_model_cls._from_config(config)

                # Resolve parallel plan
                parallel_plan = self._resolve_parallel_plan(model)
                self.parallel_plan = parallel_plan

                # Prepare model via backend (module replacement → LoRA → sharding).
                logger.info_rank0("Sharding model via %s backend", self.config.backend.value)
                model = self.backend.prepare_model(
                    model, parallel_plan=parallel_plan, peft_config=peft_config
                )

                self._patch_transformers()
                parallel_context = self._get_parallel_context()
                model.on_parallelized(parallel_context)

        if self._checkpoint is None:
            self._checkpoint = CheckpointCoordinator(
                self.config, self.backend.create_checkpoint_manager(),
                parallel_context=parallel_context,
            )

        logger.info_rank0(
            "%s model via %s backend",
            "Loading" if load_weights else "Materializing",
            self.config.backend.value,
        )
        if load_weights:
            self.checkpoint.load_pretrained(model, model_name_or_path)
        else:
            self.backend.materialize(model)
        self.backend.finalize_model_load(model)

        model.on_materialized()
        meta_tensor_names = has_meta_tensor(model)
        if meta_tensor_names:
            raise RuntimeError(
                "Model still has meta tensors after backend load/materialization "
                "and on_materialized(). Checkpoint loading must assign all "
                "parameters and persistent buffers; runtime buffers must be "
                "rebuilt after materialization or registered lazily with a real "
                "tensor on first use. Meta tensors: "
                f"{', '.join(meta_tensor_names)}"
            )

        logger.info_rank0("Model built: %s (%s)", type(model).__name__, torch_dtype)
        return model

    @property
    def checkpoint(self) -> CheckpointCoordinator:
        """Return checkpoint coordination after model preparation."""
        if self._checkpoint is None:
            raise RuntimeError(
                "Checkpoint coordinator is unavailable. Call prepare_model() first."
            )
        return self._checkpoint

    def _apply_freeze_policy(self, model: nn.Module, policy: FreezePolicy) -> None:
        if not policy.enabled:
            return

        get_parameter_groups = getattr(model, "get_parameter_groups", None)
        if not callable(get_parameter_groups):
            raise TypeError(
                "FreezePolicy requires models to expose get_parameter_groups(). "
                f"Got {type(model).__name__}."
            )

        parameter_groups = get_parameter_groups()
        available_groups = ", ".join(group.value for group in parameter_groups)

        for group in policy.parameter_groups:
            parameters = parameter_groups.get(group)
            if parameters is None:
                raise ValueError(
                    f"FreezePolicy requested unsupported parameter group "
                    f"{group.value!r} for {type(model).__name__}. "
                    f"Available groups: {available_groups or '<none>'}."
                )

            tensor_count = 0
            element_count = 0
            for _, param in parameters:
                param.requires_grad_(False)
                tensor_count += 1
                element_count += param.numel()

            if tensor_count == 0:
                raise ValueError(
                    f"FreezePolicy group {group.value!r} matched no parameters "
                    f"in {type(model).__name__}."
                )

            logger.info_rank0(
                "Froze %d parameter tensors (%d elements) in group %s",
                tensor_count,
                element_count,
                group.value,
            )

    def prepare_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_cls: type | None = None,
        trust_remote_code: bool = False,
        init_kwargs: dict[str, Any] | None = None,
        peft_config: Any | None = None,
    ) -> nn.Module:
        """Build and register the engine's primary training model."""
        model = self.build_model(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch_dtype,
            model_cls=model_cls,
            trust_remote_code=trust_remote_code,
            init_kwargs=init_kwargs,
            peft_config=peft_config,
            load_weights=True,
        )

        self._apply_freeze_policy(model, self.config.freeze_policy)

        self.model = model
        self.model_name_or_path = model_name_or_path

        # Synchronize printing
        if dist.is_initialized():
            dist.barrier()

        self._log_info()

        return model

    def _log_info(self) -> None:
        self.config.parallel_plan = self.parallel_plan

        logger.info_rank0("TorchTurbo engine summary:")
        logger.info_rank0(
            "EngineConfig:\n%s",
            json.dumps(
                vars(self.config),
                indent=2,
                default=lambda value: (
                    {
                        field.name: getattr(value, field.name)
                        for field in dataclasses.fields(value)
                        if field.name != "module_kwargs"
                    }
                    if dataclasses.is_dataclass(value)
                    else (
                        getattr(value, "__name__", type(value).__name__)
                        if callable(value)
                        else str(value)
                    )
                ),
            ),
        )

        self.ops.log_ops_selection()

        self._log_model_info(self.model)

    def save_model(
        self,
        model: nn.Module,
        output_dir: str,
        format: CheckpointFormat | None = None,
        *,
        final: bool = False,
        **kwargs: Any,
    ) -> None:
        """Save model weights under ``output_dir``.

        Args:
            model: Model prepared by this engine.
            output_dir: Checkpoint or export root directory.
            format: ``CheckpointFormat.HF`` writes an HF safetensors export and
                ``CheckpointFormat.NATIVE`` writes exact model state in the
                backend's native format.
                ``None`` saves the model part of a training checkpoint: the
                native state, plus an HF export when ``final`` is set or
                ``checkpoint.hf_export`` is ``"every"``.
            final: Whether this is the last checkpoint of the run. Only used
                when ``format`` is ``None``.
        """
        if self.model_name_or_path is None:
            raise RuntimeError("Cannot save before prepare_model().")
        formats = self.checkpoint.save_model(
            model,
            output_dir,
            source_path=self.model_name_or_path,
            format=format,
            final=final,
        )
        logger.info_rank0(
            "Model saved in %s (%s) via %s backend",
            output_dir,
            ", ".join(model_format.value for model_format in formats),
            self.config.backend.value,
        )

    def save_optimizer_and_scheduler(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        output_dir: str,
        **kwargs: Any,
    ) -> None:
        """Save optimizer and LR scheduler state for resume.

        ``optimizer`` is the engine-owned optimizer, before any framework
        wrapper is applied. ``model`` and ``lr_scheduler`` are the training
        loop's live objects.
        """
        self.checkpoint.save_optimizer_and_scheduler(
            model, optimizer, lr_scheduler, output_dir
        )

    def load_optimizer_and_scheduler(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        checkpoint_dir: str,
        **kwargs: Any,
    ) -> None:
        """Restore state written by :meth:`save_optimizer_and_scheduler`."""
        self.checkpoint.load_optimizer_and_scheduler(
            model, optimizer, lr_scheduler, checkpoint_dir
        )

    def load_from_checkpoint(self, model: nn.Module, checkpoint_dir: str, **kwargs: Any) -> None:
        """Restore model weights saved by :meth:`save_model` in ``checkpoint_dir``."""
        if self.model_name_or_path is None:
            raise RuntimeError("Cannot load a checkpoint before prepare_model().")
        self.checkpoint.load_model(
            model, checkpoint_dir, source_path=self.model_name_or_path
        )
        logger.info_rank0(
            "Model weights reloaded from %s via %s backend",
            checkpoint_dir,
            self.config.backend.value,
        )

    def _get_parallel_context(self) -> dict[str, Any]:
        return self.backend.get_parallel_context()

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------

    def prepare_optimizer(
        self,
        optimizer_cls: type | None = None,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """Resolve an optimizer plan and receive the backend's complete optimizer."""
        if self.model is None:
            raise RuntimeError("No model prepared. Call engine.prepare_model() first.")

        optimizer_name = self.config.optimizer.name
        if optimizer_name is not None:
            if optimizer_cls is not None or kwargs:
                raise ValueError(
                    "optimizer_cls and optimizer kwargs cannot be supplied when "
                    f"using engine-managed optimizer {optimizer_name!r}"
                )
            if optimizer_name == MuonPlan.name:
                optimizer_config = self.config.optimizer.muon
                # The training framework's own optimizer arguments never reach an
                # engine-managed optimizer, and its LR scheduler then scales from
                # the values below. Say so, or a run silently uses a different LR
                # than the one on the command line.
                logger.warning_rank0(
                    "Engine-managed optimizer %r uses lr=%s and weight_decay=%s "
                    "from engine config optimizer.muon; framework optimizer "
                    "arguments such as learning_rate and weight_decay are ignored.",
                    optimizer_name,
                    optimizer_config.lr,
                    optimizer_config.weight_decay,
                )
                recipe = self.model.get_optimizer_recipe(optimizer_name)
                plan = MuonPlan(
                    self.model,
                    recipe=recipe,
                    config=optimizer_config,
                    model_fqn_map=self.backend.get_model_fqn_map(),
                )
            else:
                raise NotImplementedError(
                    f"TorchTurboEngine cannot prepare optimizer {optimizer_name!r}"
                )
        else:
            if optimizer_cls is None:
                optimizer_cls = torch.optim.AdamW
            plan = OptimizerClassPlan(
                model=self.model,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=kwargs,
            )

        self.optimizer = self.backend.prepare_optimizer(plan)
        logger.info_rank0(
            "Created optimizer %s from %r",
            type(self.optimizer).__name__,
            plan,
        )
        return self.optimizer

    def offload_optimizer(self) -> None:
        """Temporarily suspend optimizer use and release device state for inference.

        Optimizer steps and checkpoint saves/loads are rejected until
        restore_optimizer() is called. The CPU offload training configuration
        keeps the optimizer usable and allows checkpoints.

        Call on every training worker after zero_grad(set_to_none=True), with
        no backward, optimizer step, or checkpoint in progress. Model weights
        and inference resources are managed separately by the caller. The
        engine-managed Muon optimizer requires FSDP parameters to keep their
        storage: moving the model, for example with ``module.to()``, makes its
        next step fail, so keep parameters in place or use full CPU offload.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "No optimizer created. Call engine.prepare_optimizer() first."
            )
        self.backend.offload_optimizer()

    def restore_optimizer(self) -> None:
        """End temporary suspension and restore the optimizer's training policy.

        Release inference allocations first. States configured to reside on CPU
        remain there; execution workspaces are allocated by the next step.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "No optimizer created. Call engine.prepare_optimizer() first."
            )
        self.backend.restore_optimizer()

    def prepare_lr_scheduler(
        self,
        scheduler_cls: type | str,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an LR scheduler for the optimizer.

        Must be called **after** ``prepare_optimizer()``.

        A scheduler name uses the engine's named scheduler factory. A scheduler
        class is constructed directly. The scheduler binds to ``optimizer``
        when supplied, or to the engine's optimizer otherwise. Base learning
        rates come from the bound optimizer's parameter groups.

        Example::

            optimizer = engine.prepare_optimizer(
                torch.optim.AdamW,
                lr=1e-4,
                weight_decay=1e-2,
            )
            scheduler = engine.prepare_lr_scheduler(
                CosineAnnealingLR, T_max=1000,
            )

        Returns the created scheduler (also stored as ``self.lr_scheduler``).
        """
        if self.optimizer is None:
            raise RuntimeError(
                "No optimizer created. Call engine.prepare_optimizer() first."
            )

        if optimizer is None:
            optimizer = self.optimizer
        if isinstance(scheduler_cls, str):
            self.lr_scheduler = create_lr_scheduler(scheduler_cls, optimizer, **kwargs)
        else:
            self.lr_scheduler = scheduler_cls(optimizer, **kwargs)
        logger.info_rank0("Created LR scheduler: %s", type(self.lr_scheduler).__name__)
        return self.lr_scheduler

    def _resolve_parallel_plan(self, model: nn.Module):
        """Resolve the parallel plan for a model."""
        from ..models.base import TurboModel
        from ..parallel import VisionBalanceSpec

        if not isinstance(model, TurboModel):
            raise TypeError(
                "TorchTurboEngine requires prepared models to inherit TurboModel. "
                f"Got {type(model).__name__}."
            )

        plan = type(model).get_default_parallel_plan()

        if self.config.parallel_plan is not None:
            engine_plan = self.config.parallel_plan

            for attr in ["sp_size", "ep_size"]:
                if hasattr(engine_plan, attr):
                    val = getattr(engine_plan, attr)
                    setattr(plan, attr, val)

            if hasattr(engine_plan, "plan") and engine_plan.plan:
                if not hasattr(plan, "plan") or plan.plan is None:
                    plan.plan = {}
                for fqn, engine_spec in engine_plan.plan.items():
                    base = plan.plan.get(fqn)
                    if base is not None and dataclasses.is_dataclass(base):
                        overrides = {}
                        for f in dataclasses.fields(engine_spec):
                            value = getattr(engine_spec, f.name)
                            if value is not None:
                                if (
                                    isinstance(value, VisionBalanceSpec)
                                    and base.vision_balance is not None
                                ):
                                    base_vision = base.vision_balance
                                    value = VisionBalanceSpec(
                                        mode=(
                                            value.mode
                                            if value.mode is not None
                                            else base_vision.mode
                                        ),
                                        io=(
                                            value.io
                                            if value.io is not None
                                            else base_vision.io
                                        ),
                                    )
                                overrides[f.name] = value
                        plan.plan[fqn] = dataclasses.replace(base, **overrides)
                    else:
                        plan.plan[fqn] = engine_spec

        return plan

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def start_profiling(
        self,
        output_dir: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
        profiling_config: dict[str, bool] | None = None,
    ) -> None:
        """Start TorchTurbo profiling under ``output_dir``."""
        if memory_snapshot_steps is not None and memory_snapshot_steps < 1:
            raise ValueError("memory_snapshot_steps must be greater than 0")

        memory_snapshot_enabled = memory_snapshot_steps is not None
        if not profiling_enabled and not memory_snapshot_enabled:
            return

        # profiling_config knobs are only meaningful when profiling is enabled.
        knobs = profiling_config if profiling_enabled else None
        record_shapes = bool(knobs and knobs.get("with_shape", False))
        with_stack = bool(knobs and knobs.get("with_stack", False))

        profiling_dir = os.path.join(output_dir, "profiling")
        if profiling_enabled:
            os.makedirs(profiling_dir, exist_ok=True)

        memory_snapshot_dir = os.path.join(output_dir, "memory_snapshot")
        if memory_snapshot_enabled:
            os.makedirs(memory_snapshot_dir, exist_ok=True)

        old_profiler = self._profiler
        if old_profiler is not None:
            old_profiler.stop()
            self._profiler = None

        rank = int(os.environ.get("RANK", 0))
        try:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except Exception:
            pass

        profiler = self.device_runtime.profiler(
            output_dir=profiling_dir,
            worker_name=f"rank{rank}_{socket.gethostname()}",
            profiling_enabled=profiling_enabled,
            record_shapes=record_shapes,
            with_stack=with_stack,
            memory_snapshot_steps=memory_snapshot_steps,
        )
        profiler.start()
        self._profiler = profiler

        if profiling_enabled:
            logger.info_rank0("Profiling started, output dir: %s", profiling_dir)
            if record_shapes or with_stack:
                logger.info_rank0(
                    "Profiling detail knobs: record_shapes=%s, with_stack=%s",
                    record_shapes,
                    with_stack,
                )
        if memory_snapshot_enabled:
            logger.info_rank0(
                "Memory snapshot scheduled for the first %s steps, output dir: %s",
                memory_snapshot_steps,
                memory_snapshot_dir,
            )

    def step_profiling(self) -> None:
        """Advance TorchTurbo profiling by one optimizer step."""
        if self._profiler is None:
            raise RuntimeError("No active profiler. Call engine.start_profiling() first.")
        self._profiler.step()

    def stop_profiling(self) -> None:
        """Stop TorchTurbo profiling and flush traces."""
        if self._profiler is None:
            raise RuntimeError("No active profiler. Call engine.start_profiling() first.")
        profiler = self._profiler
        profiler.stop()
        logger.info_rank0("Profiling stopped, traces flushed.")
        self._profiler = None

    # ------------------------------------------------------------------
    # Data loading (SP-aware)
    # ------------------------------------------------------------------

    def get_dp_info(self) -> dict[str, Any]:
        """Get data-parallel rank/size, accounting for sequence parallelism.

        When SP is active, ranks in the same SP group share the same data.
        The effective DP size = world_size / sp_size.

        Example::

            sp_size = 2
            info = engine.get_dp_info()
            # world_size=8 → dp_size=4, ranks [0,1] share dp_rank=0
        """
        return self.backend.get_dp_info()

    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Create a DataLoader with SP-aware distributed sampling.

        All ranks in an SP group receive the same batches. With ``tt_gbs``
        set, ``drop_last`` drops only the tail that cannot fill one more
        whole optimizer step.

        Example::

            dataloader = engine.create_dataloader(
                dataset, batch_size=4,
            )
            trainer.train()

        Parameters
        ----------
        dataset:
            A ``torch.utils.data.Dataset``.
        batch_size:
            Per-rank micro-batch size.
        """
        dp_info = self.get_dp_info()
        sp_info = self.get_sp_info()

        # Block granularity: whole optimizer steps (ga * batch_size per
        # rank) with tt_gbs, else the legacy full-world-sweep layout.
        gbs = self.config.gbs
        if gbs is not None:
            gradient_accumulation_steps = self._ga_from_gbs(
                gbs, sp_info["sp_size"], dist.get_world_size()
            )
            rank_block = gradient_accumulation_steps * batch_size
        else:
            rank_block = sp_info["sp_size"]

        sampler = DPBlockDistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            dp_size=dp_info["dp_size"],
            dp_rank=dp_info["dp_rank"],
            rank_block=rank_block,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        logger.info_rank0(
            "Created SP-aware DataLoader: dp_size=%d, dp_rank=%d, sp_size=%d",
            dp_info["dp_size"],
            dp_info["dp_rank"],
            sp_info["sp_size"],
        )

        return EpochDrivenDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    def fit(
        self,
        train_dataloader: Iterable,
        num_epochs: int = 1,
        max_steps: int | None = None,
        compute_loss_fn: Any = None,
        return_outputs: bool = False,
        hooks: list[Any] | None = None,
        max_grad_norm: float | None = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """Compatibility stub for the removed engine-owned training loop.

        TODO: Remove this compatibility surface after downstream callers have
        migrated to their own training loops.
        """
        del (
            train_dataloader,
            num_epochs,
            max_steps,
            compute_loss_fn,
            return_outputs,
            hooks,
            max_grad_norm,
            gradient_accumulation_steps,
        )
        raise NotImplementedError(
            "TorchTurboEngine.fit() is no longer supported. "
            "Drive training through the engine's per-step methods instead."
        )

    def get_sp_info(self) -> dict[str, Any]:
        """Get sequence-parallel rank, size and group."""
        return self.backend.get_sp_info()

    @staticmethod
    def _ga_from_gbs(gbs: int, sp_size: int, world_size: int) -> int:
        """ga = gbs * sp_size / world_size (= gbs / dp_size). Single source of
        truth shared by the training loop and the sampler block sizing so the
        two cannot drift apart."""
        if (gbs * sp_size) % world_size != 0:
            raise ValueError(
                f"tt_gbs({gbs}) * sp_size({sp_size}) must be divisible by "
                f"world_size({world_size}); got remainder "
                f"{(gbs * sp_size) % world_size}."
            )
        return (gbs * sp_size) // world_size

    def resolve_gradient_accumulation_steps(self, framework_ga: int) -> int:
        """Return the ``gradient_accumulation_steps`` for the training loop.

        Two modes:
        - ``tt_gbs`` set (``--tt_gbs``): derive ``ga = tt_gbs * sp_size /
          world_size``.
        - ``tt_gbs`` unset: fall back to ``framework_ga * sp_size``
        """
        sp_size = int(self.get_sp_info().get("sp_size", 1))
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        gbs = self.config.gbs
        if gbs is not None:
            if framework_ga > 1:
                raise ValueError(
                    f"Cannot set both tt_gbs ({gbs}) and "
                    f"gradient_accumulation_steps ({framework_ga}) at the same "
                    f"time; pick one — tt_gbs will derive ga from gbs*sp/world."
                )
            ga = self._ga_from_gbs(gbs, sp_size, world_size)
            logger.info_rank0(
                "gbs=%d, sp=%d: gradient_accumulation_steps = %d*%d/%d = %d "
                "(overriding framework ga=%d)",
                gbs, sp_size, gbs, sp_size, world_size, ga, framework_ga,
            )
            return ga
        logger.info_rank0(
            "sp=%d: gradient_accumulation_steps %d -> %d (step compensated)",
            sp_size, framework_ga, framework_ga * sp_size,
        )
        return framework_ga * sp_size

    def get_ep_info(self) -> dict[str, Any]:
        """Get expert-parallel rank, size and group."""
        return self.backend.get_ep_info()

    def _prepare_moe_runtime(
        self,
        model: nn.Module,
        inputs: Any,
        gradient_accumulation_steps: int,
    ) -> None:
        """Inject MoE runtime inputs and update the training aux-loss scale."""
        from ..models.modules.moe.moe_layer import BaseMoELayer

        moe_layers = [
            module for module in model.modules() if isinstance(module, BaseMoELayer)
        ]
        if not moe_layers:
            return

        from ..models.modules.moe.moe_utils import MoEAuxLossAutoScaler

        if model.training:
            aux_loss_scale = 1.0 / gradient_accumulation_steps
            MoEAuxLossAutoScaler.set_loss_scale(aux_loss_scale)

        runtime_inputs = model.get_moe_runtime_inputs(
            inputs, self._get_parallel_context()
        )
        for module in moe_layers:
            module.set_runtime_inputs(runtime_inputs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        gradient_accumulation_steps: int = 1,
        **kwargs,
    ):
        """Compute the scalar loss for a single micro-batch.

        Packed SFT batches contain one token row and validated cu/max metadata.
        Sequence boundaries reach the models only as ``cu_seq_lens_q/k``;
        batches carrying ``attention_mask`` are rejected.
        """
        inputs = model.on_prepare_inputs(inputs)
        inputs = self.backend.prepare_inputs(model, inputs)
        if model.training:
            self._record_micro(inputs)
        self._prepare_moe_runtime(
            model, inputs, gradient_accumulation_steps
        )
        result = self.backend.compute_loss(
            model,
            inputs,
            return_outputs,
            num_items_in_batch=num_items_in_batch,
            **kwargs,
        )
        if return_outputs:
            self._record_mtp_outputs(result[1], gradient_accumulation_steps)
        return result

    def collect_metrics(self, model: nn.Module) -> None:
        """Collect metrics currently pending on ``model``."""
        self.metric_tracker.collect(model)

    def report_metrics(self, logs: dict, gather: Callable) -> None:
        """Flush engine metrics into ``logs``"""
        if self.config.mfu.enabled:
            self.metric_tracker.record(self._get_mfu_estimator().report())
        self.metric_tracker.report(logs, gather)

    # ------------------------------------------------------------------
    # Online MFU / HFU estimation
    # ------------------------------------------------------------------
    def _get_mfu_estimator(self):
        """Build the MFU estimator once by extracting what it needs from the engine/model."""
        if self._mfu_estimator is not None:
            return self._mfu_estimator

        from .metrics import MFUEstimator
        self._mfu_estimator = MFUEstimator(
            flops_cls=self.model.get_flops_class(),
            engine_config=self.config,
            model_config=self.model.config,
            parallel_context=self._get_parallel_context(),
            runtime=self.device_runtime,
        )

        return self._mfu_estimator

    def _record_micro(self, inputs: Any) -> None:
        """Feed one micro-batch to the MFU estimator."""
        if self.config.mfu.enabled:
            self._get_mfu_estimator().record_micro(inputs)

    def compute_diffusion_loss(
        self,
        model: nn.Module,
        inputs: Any,
        ga: int = 1,
        return_outputs: bool = False,
        **kwargs,
    ):
        """Compute block-diffusion loss for an ``is_diffusion_model`` model.

        ``ga`` is the effective gradient-accumulation steps (already SP-adjusted
        by the dataloader); the backend divides the loss by it to match HF's
        grad-accum normalization.
        """
        return self.backend.compute_diffusion_loss(
            model, inputs, ga=ga, return_outputs=return_outputs, **kwargs
        )

    # ------------------------------------------------------------------
    # MTP loss recording (reported via the shared metric_tracker)
    # ------------------------------------------------------------------

    def _record_mtp_outputs(
        self, outputs: Any, gradient_accumulation_steps: int
    ) -> None:
        """Record MTP losses stamped on ``outputs`` into ``metric_tracker``.

        The backend sets ``outputs.mtp_loss`` (a list of per-head losses) plus
        optional ``mtp_loss_scaled`` / ``main_loss`` when MTP (multi-token
        prediction) is enabled.
        """
        mtp_loss = getattr(outputs, "mtp_loss", None)
        if not mtp_loss:
            return

        scale = float(gradient_accumulation_steps)
        metrics: dict[str, torch.Tensor] = {
            f"mtp_loss/{i}": m * scale for i, m in enumerate(mtp_loss)
        }
        # Per-micro-batch mean across heads; the tracker's reduction of this
        # equals the mean of the per-head ``mtp_loss/{i}`` series.
        metrics["mtp_loss/avg"] = (sum(mtp_loss) / len(mtp_loss)) * scale

        main_loss = getattr(outputs, "main_loss", None)
        if main_loss is not None:
            metrics["mtp_loss/main"] = main_loss * scale
        scaled_loss = getattr(outputs, "mtp_loss_scaled", None)
        if scaled_loss is not None:
            metrics["mtp_loss/scaled"] = scaled_loss * scale

        self.metric_tracker.record(metrics)

    def prepare_packed_micro_batch(
        self,
        mb: dict[str, Any],
        seqlens: list[int],
    ) -> dict[str, Any]:
        """Create padded model inputs from a packed micro-batch.

        Model inputs use the same backend padding as SFT. The micro-batch
        ``cu_seqlens`` reach the model as ``cu_seq_lens_q/k``, the same
        boundary metadata as SFT. Rolled token IDs are aligned to that layout
        and sliced for the local RL loss. Media tensors and grids stay global
        and reach the model's input hook before padding.
        """

        input_ids = mb["input_ids"]
        total_len = input_ids.shape[0]
        cu_seqlens = mb["cu_seqlens"]

        position_ids = torch.zeros_like(input_ids)
        for i, seq_len in enumerate(seqlens):
            start = int(cu_seqlens[i].item())
            position_ids[start : start + seq_len] = torch.arange(
                seq_len, device=input_ids.device
            )

        max_seqlen = max(seqlens) if seqlens else total_len
        result = {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "use_cache": False,
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seqlen,
            "max_length_k": max_seqlen,
            "_pad_length": 0,
        }
        result = self.model.on_prepare_inputs(result, source_inputs=mb)
        result = self.backend.prepare_inputs(self.model, result)
        result["_sp_pad_size"] = result["input_ids"].size(-1) - total_len

        sp_info = self.get_sp_info()
        if sp_info["sp_size"] > 1:
            from torchturbo.models.modules.attention.ulysses import slice_input_tensor

            # Roll the original sequence before adding the ignored tail. The
            # padded loss targets must remain valid vocabulary indices.
            input_ids_rolled = torch.zeros_like(result["input_ids"])
            input_ids_rolled[:, :total_len] = torch.roll(input_ids, shifts=-1, dims=0)
            result["_input_ids_rolled"] = slice_input_tensor(
                input_ids_rolled,
                dim=1,
                group=sp_info["sp_group"],
                rank=sp_info["sp_rank"],
            ).squeeze(0)
        return result

    def forward_backward_batch(
        self,
        mb_list: list[Any],
        process_fn: Callable[..., torch.Tensor],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        temperature: float = 1.0,
        forward_only: bool = False,
        device: Optional[torch.device] = None,
        auxiliary_loss_fn: Optional[
            Callable[[Any, dict[str, Any], float], torch.Tensor]
        ] = None,
    ) -> None:
        """Run packed-token forward and optional backward over micro-batches."""

        if self.model is None:
            raise RuntimeError("Call engine.prepare_model() before forward_backward_batch().")

        total_loss_weight = self._compute_total_loss_weight(mb_list, device)
        loss_multiplier = self._get_loss_multiplier()

        for mb_idx, mb_item in enumerate(mb_list):
            inputs, ctx = prepare_packed_mb_inputs(mb_item)
            outputs = self._forward_microbatch(inputs)
            logits = self._get_main_logits(outputs)

            if process_fn is None:
                logprobs, entropy = (
                    self._compute_logrobs_and_entropy(
                        logits,
                        ctx,
                        temperature=temperature,
                        grad_scaler=True,
                    )
                )
                loss = loss_fn(
                    logprobs=logprobs,
                    entropy=entropy,
                    mb_input=ctx["mb_input"],
                )
            else:
                loss = process_fn(logits, ctx, loss_fn)

            scaled_policy_loss = self._scale_loss(
                loss,
                ctx,
                total_loss_weight,
                loss_multiplier,
                loss_weight_fn,
            )
            total_loss = scaled_policy_loss

            if not forward_only and auxiliary_loss_fn is not None:
                auxiliary_loss = auxiliary_loss_fn(outputs, ctx, loss_multiplier)
                total_loss = total_loss + auxiliary_loss

            if not forward_only:
                if mb_idx == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                    logger.info_rank0(
                        "packed mb[%d] before backward, loss=%.6f",
                        mb_idx,
                        total_loss.item(),
                    )
                self._backward_loss(total_loss)

    @torch.no_grad()
    def compute_logprobs(
        self,
        mb_list: list[Any],
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute packed log-probabilities without backward."""

        if self.model is None:
            raise RuntimeError("Call engine.prepare_model() before compute_logprobs().")

        all_logprobs = []
        for mb_item in mb_list:
            inputs, ctx = prepare_packed_mb_inputs(mb_item)
            outputs = self._forward_microbatch(inputs)
            logits = self._get_main_logits(outputs)
            logprobs, _ = self._compute_logrobs_and_entropy(
                logits,
                ctx,
                temperature=temperature,
                grad_scaler=False,
            )
            all_logprobs.append(logprobs.cpu())
        return torch.cat(all_logprobs, dim=0)

    def _forward_microbatch(self, inputs: dict[str, Any]) -> Any:
        return self.model(**inputs)

    @staticmethod
    def _get_main_logits(outputs: Any) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits.squeeze(0)

    def _compute_logrobs_and_entropy(
        self,
        logits: torch.Tensor,
        ctx: Any,
        temperature: float,
        grad_scaler: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sp_info = self.get_sp_info()
        sp_size = int(sp_info.get("sp_size", 1))
        mb_input = ctx["mb_input"]
        cu_seqlens = mb_input.get("cu_seqlens")

        if sp_size > 1:
            from torchturbo.adapters.rl.sp import (
                gather_entropy_and_shift,
                gather_logprobs_and_shift,
            )

            labels = mb_input["_input_ids_rolled"]
            logprobs, entropy = gather_logprobs_entropy(logits, labels, temperature)
            logprobs = gather_logprobs_and_shift(
                logprobs,
                ctx["sp_pad_size"],
                cu_seqlens,
                sp_info,
                grad_scaler=grad_scaler,
            )
            entropy = gather_entropy_and_shift(
                entropy,
                ctx["sp_pad_size"],
                cu_seqlens,
                sp_info,
                grad_scaler=grad_scaler,
            )
            return logprobs, entropy

        input_ids = ctx["model_inputs"]["input_ids"].squeeze(0)
        if ctx["pad_length"] > 0:
            logits = logits[: -ctx["pad_length"]]
            input_ids = input_ids[: -ctx["pad_length"]]

        shifted_logits = shifted_packed_tensor(logits, cu_seqlens, 0, 1)
        shifted_labels = shifted_packed_tensor(input_ids, cu_seqlens, 1, 0)
        logprobs, entropy = gather_logprobs_entropy(
            shifted_logits,
            shifted_labels,
            temperature,
        )
        return logprobs, entropy

    def _scale_loss(
        self,
        loss: torch.Tensor,
        ctx: Any,
        total_loss_weight: torch.Tensor,
        loss_multiplier: float,
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor:
        loss_scale = loss_weight_fn(ctx["mb_input"]) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_total_loss_weight(
        self,
        mb_list: list[Any],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        sp_info = self.get_sp_info()
        sp_size = int(sp_info.get("sp_size", 1))

        total = torch.tensor(0.0, device=device)

        for mb_item in mb_list:
            mb = get_packed_mb_orig(mb_item)
            if "ppo_loss_mask" in mb:
                total += mb["ppo_loss_mask"].float().sum()
            else:
                if "input_ids" in mb:
                    local_tokens = float(mb["input_ids"].numel())
                    if sp_size > 1:
                        local_tokens *= sp_size
                    total += local_tokens

        dp_info = self.get_dp_info()
        dp_group = dp_info.get("dp_group")
        dp_size = int(dp_info.get("dp_size", 1))
        if dist.is_initialized() and dp_group is not None and dp_size > 1:
            self._sync_device()
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=dp_group)
            dist.barrier(group=dp_group)
        return total.clamp(min=1.0)

    def _get_loss_multiplier(self) -> float:
        sp_info = self.get_sp_info()
        dp_info = self.get_dp_info()
        sp_size = int(sp_info.get("sp_size", 1))
        dp_size = int(dp_info.get("dp_size", 1))
        if sp_size > 1:
            return float(dp_size) / float(sp_size)
        return float(dp_size)

    def _backward_loss(self, loss: torch.Tensor) -> None:
        try:
            from torchturbo.engine.backends.fsdp2.prefetch import (
                fsdp2_backward_prefetch_manager,
            )

            with fsdp2_backward_prefetch_manager:
                loss.backward()
        except ImportError:
            loss.backward()

    @staticmethod
    def _sync_device() -> None:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_model_info(self, model: nn.Module) -> None:
        """Log model architecture and parameter sharding map."""

        if dist.get_rank() != 0:
            return

        # --- Architecture ---
        logger.info_rank0("Model architecture:\n%s", model)

        # --- Sharding table ---
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            DTensor = None

        max_name_len = max(len(fqn) for fqn, _ in model.named_parameters())
        name_col = max(max_name_len, len("Parameter")) + 2  # +2 for padding

        header = f"{'Parameter':<{name_col}} {'Shape':<20} {'Dtype':<12} {'Device':<10} {'Mesh':<30} {'Placements'}"
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for fqn, param in model.named_parameters():
            shape = str(tuple(param.shape))
            dtype = str(param.dtype).replace("torch.", "")
            device = str(param.device)

            if DTensor is not None and isinstance(param, DTensor):
                mesh = param.device_mesh
                mesh_info = (
                    f"{mesh.mesh_dim_names}={list(mesh.mesh.shape)}"
                    if hasattr(mesh, "mesh_dim_names")
                    else str(mesh)
                )
                placements = str(param.placements)
            else:
                mesh_info = "-"
                placements = "local"

            lines.append(
                f"{fqn:<{name_col}} {shape:<20} {dtype:<12} {device:<10} {mesh_info:<30} {placements}"
            )

        lines.append(sep)
        logger.info_rank0("Parameter sharding map:\n%s", "\n".join(lines))

    def __repr__(self) -> str:
        setup_status = "ready" if self.model is not None else "not setup"
        return f"<TorchTurboEngine(backend={self.backend}, {setup_status})>"
