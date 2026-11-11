"""TorchTurboEngine — one-stop-shop for distributed training."""

from __future__ import annotations

import os
import socket
from typing import Any, Callable, Iterable, Optional

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from ..logging import get_logger
from .backend_registry import get_available_backends, get_backend
from .backends.base import BaseBackend
from .config import EngineConfig, FreezePolicy
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


class GatherAndInnerLoopSPDistributedSampler(DistributedSampler):
    """DistributedSampler subclass that gives each SP group contiguous index blocks.

    All ranks sharing the same ``dp_rank`` yield the identical sequence
    of ``sp_size`` indices per iteration block.

    Parameters
    ----------
    dataset:
        Dataset to sample from.
    sp_size:
        Sequence-parallel group size.  ``1`` = standard DistributedSampler.
    num_replicas:
        Total world size.  Defaults to ``dist.get_world_size()``.
    rank:
        Global rank.  Defaults to ``dist.get_rank()``.
    shuffle:
        Whether to shuffle indices each epoch.
    seed:
        Random seed used when *shuffle* is True.
    drop_last:
        Drop the tail to make the dataset evenly divisible.
    """

    def __init__(
        self,
        dataset: Dataset,
        sp_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
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

        if self.num_replicas % sp_size != 0:
            raise ValueError(
                f"world_size ({self.num_replicas}) must be divisible by sp_size ({sp_size})"
            )

        self.sp_size = sp_size
        self.dp_size = self.num_replicas // sp_size
        self.dp_rank = self.rank // sp_size

        # Recompute sizing: each DP group consumes sp_size samples per
        # iteration, so total iterations = dataset / (dp_size * sp_size).
        total = len(self.dataset)  # type: ignore[arg-type]
        if self.drop_last:
            self.num_iterations = total // (self.dp_size * self.sp_size)
        else:
            self.num_iterations = math.ceil(total / (self.dp_size * self.sp_size))
        self.total_size = self.num_iterations * self.dp_size * self.sp_size
        # Override parent's num_samples (used by __len__ in parent)
        self.num_samples = self.num_iterations * self.sp_size

        logger.info_rank0(
            "SPDistributedSampler: world=%d, rank=%d, sp_size=%d, "
            "dp_size=%d, dp_rank=%d, iterations=%d",
            self.num_replicas,
            self.rank,
            self.sp_size,
            self.dp_size,
            self.dp_rank,
            self.num_iterations,
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

        # Each iteration block has dp_size * sp_size indices laid out as:
        #   [dp0_s0, dp0_s1, ..., dp0_sp-1, dp1_s0, ..., dpN_sp-1]
        #
        # DP rank d gets indices at offsets [d*sp_size .. d*sp_size+sp_size-1]
        # within each block.  All ranks in the same SP group (same dp_rank)
        # yield the identical list.
        rank_indices = []
        for i in range(self.num_iterations):
            base = i * self.dp_size * self.sp_size + self.dp_rank * self.sp_size
            for s in range(self.sp_size):
                rank_indices.append(indices[base + s])

        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples


class TorchTurboEngine:
    """Facade for model preparation, parallelism, optimization, and metrics.

    Training loops are driven by the framework adapter or the caller.
    """

    def __init__(self, config: EngineConfig | dict[str, Any] | None = None) -> None:
        # Normalize to EngineConfig
        if config is None:
            self.config = EngineConfig()
        elif isinstance(config, dict):
            self.config = EngineConfig.from_dict(config)
        else:
            self.config = config

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
        self.backend: BaseBackend = backend_cls(self.config, self.device_runtime)
        self._profiler: BaseProfiler | None = None
        self._mfu_estimator = None
        self.ops = get_or_create_ops(
            device_runtime=self.device_runtime,
            default_policy=self.config.ops.policy,
        )
        self.ops.log_ops_selection()

        # These are set by setup()
        self.model: nn.Module | None = None
        self.model_name_or_path: str | None = None
        self.optimizer: Any = None
        self.lr_scheduler: Any = None

        # Hooks (can be added before or after setup)

        logger.info_rank0(
            "TorchTurboEngine initialized: backend=%s, device=%s",
            backend_name,
            self.device_runtime.device.type,
        )

    # ------------------------------------------------------------------
    # Model preparation (meta device → shard → load weights)
    # ------------------------------------------------------------------

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

        if self.config.mtp.enabled:
            n_mtp = getattr(config, "num_nextn_predict_layers", 0)
            if n_mtp > 0:
                logger.info_rank0("MTP enabled: %d MTP layer(s) will be built", n_mtp)
            else:
                logger.warning_rank0(
                    "--tt_enable_mtp passed but model config has "
                    "num_nextn_predict_layers=0 — nothing to enable."
                )
        else:
            if hasattr(config, "num_nextn_predict_layers"):
                if config.num_nextn_predict_layers > 0:
                    logger.info_rank0(
                        "MTP off (default): stripping %d MTP layer(s) — "
                        "their checkpoint weights will be skipped. "
                        "Pass --tt_enable_mtp to enable.",
                        config.num_nextn_predict_layers,
                    )
                config.num_nextn_predict_layers = 0

        if model_cls is None:
            model_cls = AutoModelForCausalLM

        # 2. Build model on meta device (no memory)
        #
        # Try torchturbo's own model registry first (keyed by model_type).
        # This bypasses AutoModelForCausalLM.from_config entirely, avoiding
        # the type(config) mismatch that occurs when the config object was
        # created from a remote config class (e.g. via trust_remote_code).
        from ..models.base import TurboModel
        from ..models.registry import get_model
        from transformers import PreTrainedModel

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

        with turbo_model_cls.init_context(config):
            logger.info_rank0("Building model on meta device: %s", model_name_or_path)
            with torch.device("meta"):
                logger.info_rank0(
                    "Using torchturbo model class: %s", turbo_model_cls.__name__
                )
                model = turbo_model_cls._from_config(config)

                # Resolve parallel plan
                parallel_plan = self._resolve_parallel_plan(model)

                # Prepare model via backend (module replacement → LoRA → sharding).
                logger.info_rank0("Sharding model via %s backend", self.config.backend.value)
                model = self.backend.prepare_model(
                    model, parallel_plan=parallel_plan, peft_config=peft_config
                )

                parallel_context = self._get_parallel_context()
                model.on_parallelized(parallel_context)

        logger.info_rank0(
            "%s model via %s backend",
            "Loading" if load_weights else "Materializing",
            self.config.backend.value,
        )
        self.backend.load_model(
            model,
            model_name_or_path=model_name_or_path,
            parallel_context=parallel_context,
            load_weights=load_weights,
        )

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

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info_rank0("Model prepared: %s (%s)", type(model).__name__, torch_dtype)
            self._log_model_info(model)

        return model

    def save_model(self, model: nn.Module, output_dir: str, **kwargs: Any) -> None:
        self.backend.save_model(
            model,
            self.model_name_or_path,
            output_dir,
            parallel_context=self._get_parallel_context(),
            **kwargs,
        )
        logger.info_rank0(
            "Model saved in %s via %s backend", output_dir, self.config.backend.value
        )

    def _get_parallel_context(self) -> dict[str, Any]:
        return {
            "dp": self.get_dp_info(),
            "sp": self.get_sp_info(),
            "ep": self.get_ep_info(),
        }

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------

    def prepare_optimizer(
        self,
        optimizer_cls: type | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create an optimizer for the prepared model.

        Must be called **after** ``prepare_model()``.

        Example::

            model = engine.prepare_model("meta-llama/Llama-2-7b-hf")
            optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=1e-4)

        Returns the created optimizer (also stored as ``self.optimizer``).
        """
        if self.model is None:
            raise RuntimeError("No model prepared. Call engine.prepare_model() first.")

        if optimizer_cls is None:
            optimizer_cls = torch.optim.AdamW

        kwargs.setdefault("lr", 1e-4)
        kwargs.setdefault("betas", (0.9, 0.999))
        kwargs.setdefault("weight_decay", 1e-2)

        self.optimizer = self.backend.prepare_optimizer(
            self.model, optimizer_cls, **kwargs
        )
        logger.info_rank0("Created optimizer: %s", type(self.optimizer).__name__)
        return self.optimizer

    def prepare_lr_scheduler(
        self,
        scheduler_cls: type,
        **kwargs: Any,
    ) -> Any:
        """Create an LR scheduler for the optimizer.

        Must be called **after** ``prepare_optimizer()``.

        Example::

            optimizer = engine.prepare_optimizer(torch.optim.AdamW, lr=1e-4)
            scheduler = engine.prepare_lr_scheduler(
                CosineAnnealingLR, T_max=1000,
            )

        Returns the created scheduler (also stored as ``self.lr_scheduler``).
        """
        if self.optimizer is None:
            raise RuntimeError(
                "No optimizer created. Call engine.prepare_optimizer() first."
            )

        self.lr_scheduler = scheduler_cls(self.optimizer, **kwargs)
        logger.info_rank0("Created LR scheduler: %s", type(self.lr_scheduler).__name__)
        return self.lr_scheduler

    def _resolve_parallel_plan(self, model: nn.Module):
        """Resolve the parallel plan for a model."""
        from ..models.base import TurboModel

        if not isinstance(model, TurboModel):
            raise TypeError(
                "TorchTurboEngine requires prepared models to inherit TurboModel. "
                f"Got {type(model).__name__}."
            )

        plan = type(model).get_default_parallel_plan()
        logger.info_rank0(
            "Auto-discovered default parallel plan from %s "
            "(plan_keys=%d, sp=%d, ep=%d)",
            type(model).__name__,
            len(getattr(plan, "plan", {})),
            getattr(plan, "sp_size", 1),
            getattr(plan, "ep_size", 1),
        )

        if self.config.parallel_plan is not None:
            logger.info_rank0("Overriding parallel plan with engine config")
            engine_plan = self.config.parallel_plan

            for attr in ["sp_size", "ep_size"]:
                if hasattr(engine_plan, attr):
                    val = getattr(engine_plan, attr)
                    setattr(plan, attr, val)

            if hasattr(engine_plan, "plan") and engine_plan.plan:
                if not hasattr(plan, "plan") or plan.plan is None:
                    plan.plan = {}
                plan.plan.update(engine_plan.plan)

        logger.info_rank0(
            "Parallel Plan for %s (sp=%d, ep=%d)",
            type(model).__name__,
            getattr(plan, "sp_size", 1),
            getattr(plan, "ep_size", 1),
        )
        return plan

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def start_profiling(
        self,
        output_dir: str,
        profiling_enabled: bool,
        memory_snapshot_steps: int | None,
    ) -> None:
        """Start TorchTurbo profiling under ``output_dir``."""
        if memory_snapshot_steps is not None and memory_snapshot_steps < 1:
            raise ValueError("memory_snapshot_steps must be greater than 0")

        memory_snapshot_enabled = memory_snapshot_steps is not None
        if not profiling_enabled and not memory_snapshot_enabled:
            return

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
            memory_snapshot_steps=memory_snapshot_steps,
        )
        profiler.start()
        self._profiler = profiler

        if profiling_enabled:
            logger.info_rank0("Profiling started, output dir: %s", profiling_dir)
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
        **kwargs: Any,
    ) -> Any:
        """Create a DataLoader with SP-aware distributed sampling.

        When ``sp_size > 1``, uses ``DistributedSampler`` with
        ``num_replicas=dp_size`` and ``rank=dp_rank``, so that all
        ranks in the same SP group receive the **same batch**.

        Example::

            dataloader = engine.create_dataloader(
                dataset, batch_size=4,
            )
            for batch in dataloader:
                loss = engine.compute_loss(model, batch)

        Parameters
        ----------
        dataset:
            A ``torch.utils.data.Dataset``.
        sp_size:
            Sequence parallel size.  ``1`` = standard DP.
        batch_size:
            Per-rank micro-batch size.
        """
        from torch.utils.data import DataLoader, DistributedSampler

        dp_info = self.get_dp_info()
        sp_info = self.get_sp_info()

        # sampler = DistributedSampler(
        #     dataset,
        #     num_replicas=dp_info["dp_size"],
        #     rank=dp_info["dp_rank"],
        #     shuffle=shuffle,
        #     drop_last=drop_last,
        # )
        # To align with version v1
        sampler = GatherAndInnerLoopSPDistributedSampler(
            dataset,
            sp_size=sp_info["sp_size"],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )

        logger.info_rank0(
            "Created SP-aware DataLoader: dp_size=%d, dp_rank=%d, sp_size=%d",
            dp_info["dp_size"],
            dp_info["dp_rank"],
            sp_info["sp_size"],
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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

    def get_ep_info(self) -> dict[str, Any]:
        """Get expert-parallel rank, size and group."""
        return self.backend.get_ep_info()


    def compute_loss(
        self, model: nn.Module, inputs: Any, return_outputs: bool = False, **kwargs
    ):
        """Compute the scalar loss for a single micro-batch."""
        if self.config.mfu.enabled and model.training:
            self._get_mfu_estimator().record_micro(inputs)
        return self.backend.compute_loss(model, inputs, return_outputs, **kwargs)

    def _get_mfu_estimator(self):
        if self._mfu_estimator is None:
            from .metrics import MFUEstimator

            self._mfu_estimator = MFUEstimator(
                recipe=self.model.get_flops_class()(self.model.config),
                config=self.config,
                sp_size=self.get_sp_info()["sp_size"],
                runtime=self.device_runtime,
            )
        return self._mfu_estimator

    def report_metrics(self, logs: dict, gather: Callable) -> None:
        """Flush pending MFU/HFU estimates into a training log dictionary."""
        if self.config.mfu.enabled and self._mfu_estimator is not None:
            self._mfu_estimator.report(logs, gather)

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
    # forward / backward API for RL
    # ------------------------------------------------------------------

    def prepare_packed_micro_batch(
        self,
        mb: dict[str, Any],
        seqlens: list[int],
    ) -> dict[str, Any]:
        """Create padded model inputs from a packed micro-batch.

        This is the packed-token counterpart of the SFT ``compute_loss`` input
        preparation.  The engine owns SP state and uses Ulysses helpers when
        sequence parallelism is active.
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

        padded_input_ids = input_ids.unsqueeze(0)
        padded_position_ids = position_ids.unsqueeze(0)
        attention_mask = mb.get("attention_mask", None)
        sp_info = self.get_sp_info()
        sp_size = int(sp_info.get("sp_size", 1))

        sp_pad_size = 0
        input_ids_rolled = None
        if sp_size > 1:
            from torchturbo.adapters.rl.sp import pad_and_slice_packed_inputs

            (
                padded_input_ids,
                padded_position_ids,
                attention_mask,
                sp_pad_size,
                input_ids_rolled,
            ) = pad_and_slice_packed_inputs(
                padded_input_ids,
                padded_position_ids,
                attention_mask,
                sp_info,
            )

        max_seqlen = max(seqlens) if seqlens else total_len
        result = {
            "input_ids": padded_input_ids,
            "position_ids": padded_position_ids,
            "use_cache": False,
            "cu_seqlens": cu_seqlens,
            "cu_seq_lens_q": cu_seqlens,
            "cu_seq_lens_k": cu_seqlens,
            "max_length_q": max_seqlen,
            "max_length_k": max_seqlen,
            "max_seqlen": max_seqlen,
            "attention_mask": attention_mask,
            "_pad_length": 0,
            "_sp_pad_size": sp_pad_size,
        }
        if input_ids_rolled is not None:
            result["_input_ids_rolled"] = input_ids_rolled.squeeze(0)
        return result

    def forward_backward_batch(
        self,
        mb_list: list[Any],
        process_fn: Callable[..., torch.Tensor],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        temperature: float = 1.0,
        forward_only: bool = False,
        device: Optional[torch.device] = None
    ) -> None:
        """Run packed-token forward and optional backward over micro-batches."""

        if self.model is None:
            raise RuntimeError("Call engine.prepare_model() before forward_backward_batch().")

        total_loss_weight = self._compute_total_loss_weight(mb_list, device)
        loss_multiplier = self._get_loss_multiplier()

        for mb_idx, mb_item in enumerate(mb_list):
            inputs, ctx = prepare_packed_mb_inputs(mb_item)
            logits = self._forward_microbatch(inputs)

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

            loss = self._scale_loss(
                loss,
                ctx,
                total_loss_weight,
                loss_multiplier,
                loss_weight_fn,
            )

            if not forward_only and loss is not None:
                if mb_idx == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                    logger.info_rank0(
                        "packed mb[%d] before backward, loss=%.6f",
                        mb_idx,
                        loss.item(),
                    )
                self._backward_loss(loss)

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
            logits = self._forward_microbatch(inputs)
            logprobs, _ = self._compute_logrobs_and_entropy(
                logits,
                ctx,
                temperature=temperature,
                grad_scaler=False,
            )
            all_logprobs.append(logprobs.cpu())
        return torch.cat(all_logprobs, dim=0)

    def _forward_microbatch(self, inputs: dict[str, Any]) -> torch.Tensor:
        outputs = self.model(**inputs)
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
        logger.debug_rank0("Model architecture:\n%s", model)

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
        logger.debug_rank0("Parameter sharding map:\n%s", "\n".join(lines))


    def __repr__(self) -> str:
        setup_status = "ready" if self.model is not None else "not setup"
        return f"<TorchTurboEngine(backend={self.backend}, {setup_status})>"
