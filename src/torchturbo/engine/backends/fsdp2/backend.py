"""FSDP2 distributed backend — ``torch.distributed.fsdp.fully_shard``."""

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, MethodType
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

try:
    from torch.nn.utils import clip_grads_with_norm_, get_total_norm
except ImportError:  # pragma: no cover - torch<2.9 compatibility
    from torch.nn.utils.clip_grad import (  # type: ignore[attr-defined]
        _clip_grads_with_norm_ as clip_grads_with_norm_,
        _get_total_norm as get_total_norm,
    )

from torchturbo.models.modules.attention.ulysses import (
    patch_forward_for_sp_input_slicing,
    slice_input_tensor,
    ulysses_pad_inputs,
)
from torchturbo.models.modules.comm import all_gather_tensor
from torchturbo.models.modules.loss import chunk_loss, logits_loss
from torchturbo.models.modules.vision import register_vision_balance
from torchturbo.parallel import (
    FSDPPlanSpec,
    FSDPParallelPlan,
)
from torchturbo.runtime import DeviceRuntime
from torchturbo.engine.model_fqn import ModelFQNMap, build_model_fqn_map
from torchturbo.engine.optimizers import (
    MuonPlan,
    OptimizerClassPlan,
    OptimizerPlan,
    OptimizersContainer,
)
from torchturbo.engine.utils import match_fqn
from torchturbo.utils.torch_utils import torch_version

from ...backend_registry import register_backend
from ...config import EngineConfig
from ...distributed import assert_finite, init_process_group_if_needed
from ....logging import get_logger
from ..base import BaseBackend
from .checkpoint import FSDP2CheckpointManager
from .config import CPUOffloadMode, FSDPConfig
from .optimizers.adamw import prepare_fsdp_adamw_optimizer
from .optimizers.memory import OptimizerMemoryController
from .prefetch import FSDPPrefetchManager

logger = get_logger(__name__)

if torch_version() >= (2, 6, 0):
    from torch.distributed.fsdp._fully_shard import (
        FSDPModule,
        register_fsdp_forward_method,
    )
else:  # pragma: no cover - torch<2.6 compatibility
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule

    try:  # pragma: no cover - torch<2.6 compatibility
        from torch.distributed._composable.fsdp.fully_shard import (
            register_fsdp_forward_method,
        )
    except ImportError:  # pragma: no cover - older torch compatibility
        register_fsdp_forward_method = None

from torch.distributed.tensor import DTensor


@dataclass
class _FSDPUnit:
    modules: list[nn.Module]
    mesh: Any
    reshard_after_forward: bool | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedFSDPPlan:
    modules: Mapping[str, FSDPPlanSpec]
    parameters: Mapping[str, FSDPPlanSpec]


def _confidence_entropy_sum(logits: torch.Tensor, labels: torch.Tensor,
                            ignore_index: int = -100):
    """CAP / DPARALLEL confidence term.

    Returns ``(entropy_sum_over_correct_tokens, num_correct_tokens)`` — both
    summable so they can be all-reduced under SP and divided globally, mirroring
    how the CE term is summed + all-reduced + /n_valid.

    Entropy is only summed over ``correct`` rows, so we compute
    ``log_softmax`` strictly on the correct-row slice of ``logits`` rather than
    on the full ``[T, V]`` tensor.
    """
    valid = labels != ignore_index
    pred = torch.argmax(logits, dim=-1)
    correct = (pred == labels) & valid
    num_correct = correct.sum()
    if num_correct.item() == 0:
        return logits.float().sum() * 0.0, num_correct
    # Slice to correct rows before log_softmax — sees only [n_correct, V].
    logits_c = logits[correct]
    log_probs = nn.functional.log_softmax(logits_c.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return entropy.sum(), num_correct


class FSDP2Backend(BaseBackend):
    """FSDP2 backend using ``torch.distributed.fsdp.fully_shard``.

    Reads FSDP-specific settings from ``config.fsdp`` and mixed precision
    from ``config.mixed_precision``.

    When ``config.parallel.sp_size > 1``, creates a sequence parallel
    process group for Ulysses SP.
    """

    name = "fsdp2"

    def __init__(self, config: EngineConfig, device_runtime: DeviceRuntime) -> None:
        super().__init__(config, device_runtime)
        self.fsdp_config: FSDPConfig = config.backend_config
        # Manages the optimizer returned by prepare_optimizer().
        self._optimizer_memory: OptimizerMemoryController | None = None

        offload = self.fsdp_config.cpu_offload
        if offload.enabled and (
            offload.mode is CPUOffloadMode.FULL
            or (
                config.optimizer.name == MuonPlan.name
                and offload.optimizer_memory_fraction == 0
            )
        ):
            init_process_group_if_needed(enable_cpu_backend=True)

        fsdp_config = self.fsdp_config
        self.mixed_precision = config.mixed_precision
        self.sharding_strategy = fsdp_config.sharding_strategy
        self.reshard_after_forward = fsdp_config.reshard_after_forward

        self.prefetch_manager = FSDPPrefetchManager(
            config=self.fsdp_config,
        )

        # Sequence parallel process group and device mesh
        self.rank = dist.get_rank()
        self._mesh: Any = None  # torch.distributed.device_mesh.DeviceMesh
        self._world_mesh: Any = None
        self._sp_size: int = 1
        self._sp_group: dist.ProcessGroup = None
        self._sp_rank: int = 0
        self._use_fla_kcp: bool = False

        # Data parallel process group.
        # exposed through :`get_dp_info`.
        self._dp_group: dist.ProcessGroup = None
        self._dp_rank: int = 0
        self._dp_size: int = 1
        self.use_transformer_engine = False
        self.gradient_scaling_factor = 1.0
        self.parallel_plan: FSDPParallelPlan = None
        self._resolved_parallel_plan: _ResolvedFSDPPlan | None = None

    def _create_sp_group(self, sp_size: int) -> dist.ProcessGroup | None:
        """Create sequence parallel process group.

        Groups consecutive ranks into SP groups of size ``sp_size``.
        E.g. with 8 GPUs and sp_size=4:
          group 0: [0, 1, 2, 3]
          group 1: [4, 5, 6, 7]
        """
        if not dist.is_initialized():
            logger.warning_rank0(
                "torch.distributed not initialized — skipping SP group creation. "
                "SP groups will be created lazily when needed."
            )
            return None

        world_size = dist.get_world_size()
        rank = self.rank

        if world_size % sp_size != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"sp_size ({sp_size}) for sequence parallelism."
            )

        # Create all SP groups (all ranks must call this)
        sp_group = None
        for start in range(0, world_size, sp_size):
            ranks = list(range(start, start + sp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                sp_group = group
        sp_rank = dist.get_rank(sp_group)

        logger.info_rank0(
            "Created SP process group: sp_size=%d, rank=%d, " "group_ranks=%s",
            sp_size,
            rank,
            list(range((rank // sp_size) * sp_size, (rank // sp_size + 1) * sp_size)),
        )

        self._sp_size = sp_size
        self._sp_group = sp_group
        self._sp_rank = sp_rank

    def _create_dp_group(self) -> dist.ProcessGroup | None:
        """Create data parallel process group for Ulysses SP.

        E.g. ``world_size=16, sp_size=2`` -> mesh ``(8, 2)`` ::
            r 0 -> (0,0)  dpg=[0,2,4,6,8,10,12,14]   spg=[0,1]
            r 1 -> (0,1)  dpg=[1,3,5,7,9,11,13,15]   spg=[0,1]
            r 2 -> (1,0)  dpg=[0,2,4,6,8,10,12,14]   spg=[2,3]
        """
        if not dist.is_initialized():
            self.logger.warning(
                "torch.distributed not initialized — skipping DP group creation."
            )
            return None

        world_size = dist.get_world_size()
        if self._sp_size <= 1:
            self._dp_group = dist.group.WORLD
            self._dp_rank = self.rank
            self._dp_size = world_size
            self.logger.debug("sp_size <= 1, _dp_group=%s", self._dp_group)
            return self._dp_group

        if world_size % self._sp_size != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"sp_size ({self._sp_size}) for data parallelism under SP."
            )

        dp_size = world_size // self._sp_size

        # DP (data) replication set for rank r = ranks sharing the same SP
        # offset: [r%sp, r%sp+sp, r%sp+2*sp, ...] (a "column" of the
        # (dp_size, sp_size) row-major layout). All ranks must call
        # dist.new_group, exactly as in :meth:`_create_sp_group`.
        dp_group = None
        for j in range(self._sp_size):
            ranks = list(range(j, world_size, self._sp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                dp_group = group
        self._dp_group = dp_group
        self._dp_rank = dist.get_rank(group=dp_group)
        self._dp_size = dp_size

        self.logger.info(
            "Created DP process group (via new_group, no extra mesh): "
            "dp_size=%d, dp_rank=%d, sp=%d",
            self._dp_size,
            self._dp_rank,
            self._sp_size,
        )
        return self._dp_group

    def _init_mesh(self, plan: Any) -> Any | None:
        """Create an N-dimensional DeviceMesh based on the parallel plan."""
        if self._mesh is not None:
            return self._mesh

        if not dist.is_initialized():
            return None

        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError:
            raise ValueError("torch.distributed.device_mesh not available.")

        world_size = dist.get_world_size()

        ep_size = getattr(plan, "ep_size", 1)
        sp_size = getattr(plan, "sp_size", 1)
        if world_size % sp_size != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"sp_size ({sp_size}) for sequence parallelism."
            )

        world_size = world_size
        device_name = self.device.type
        self._world_mesh = init_device_mesh(
            device_name,
            (world_size,),
            mesh_dim_names=("dp",),
        )
        self._mesh = init_device_mesh(
            device_name, (world_size // ep_size, ep_size), mesh_dim_names=("edp", "ep")
        )

        logger.info_rank0("Created DeviceMesh")

        self._create_sp_group(sp_size)
        self._create_dp_group()

    def _replace_modules(self, model: nn.Module, plan: FSDPParallelPlan) -> nn.Module:
        """Replace modules according to the parallel plan (e.g., MoE layers).

        This is separated from :meth:`prepare_model` so that LoRA can be
        applied *after* module replacement but *before* FSDP2 sharding.

        Parameters
        ----------
        model : nn.Module
            The model (typically on meta device).
        plan : FSDPParallelPlan
            The parallel plan containing ``replace_module`` specs.

        Returns
        -------
        nn.Module
            The model with replaced modules.
        """
        config = model.config

        if plan.plan and isinstance(plan.plan, dict):
            for fqn, submodule in list(model.named_modules()):
                if fqn == "":
                    continue

                spec = plan.match(fqn)
                if isinstance(spec, FSDPPlanSpec) and spec.replace_module:
                    kwargs = spec.module_kwargs

                    kwargs["hf_config"] = config
                    kwargs["ep_info"] = self.get_ep_info()
                    kwargs["sp_info"] = self.get_sp_info()
                    kwargs["engine_config"] = self.config
                    new_module = spec.replace_module(submodule, **kwargs)

                    if new_module is not None and new_module is not submodule:
                        parent_fqn, _, name = fqn.rpartition(".")
                        parent = model.get_submodule(parent_fqn)
                        setattr(parent, name, new_module)
                        # Per-module trace is DEBUG-only; it fires once per
                        # replaced/patched module and would otherwise inflate
                        # the log on every rank.
                        logger.debug_rank0(
                            "FSDP2Backend replaced module %s with %s",
                            fqn,
                            type(new_module).__name__,
                        )
                    else:
                        logger.debug_rank0("FSDP2Backend patched module %s in-place", fqn)

        return model

    @staticmethod
    def _compile_parallel_plan(
        plan: FSDPParallelPlan,
        model_fqn_map: ModelFQNMap,
    ) -> _ResolvedFSDPPlan:
        """Resolve model-authored patterns to exact prepared-tree FQNs."""
        for pattern, spec in plan.plan.items():
            vision_spec = spec.vision_balance
            if vision_spec is None or vision_spec.mode is None:
                continue
            matches_module = any(
                match_fqn(pattern, fqn) and plan.match(fqn) is spec
                for fqn in model_fqn_map.modules
            )
            if not matches_module:
                raise ValueError(
                    f"Vision balance pattern {pattern!r} does not resolve "
                    "to a module in the active parallel plan."
                )
            if vision_spec.io is None:
                raise ValueError(
                    f"Vision balance pattern {pattern!r} requires a model-provided "
                    "input/output adapter."
                )
        model_fqns = tuple(model_fqn_map.modules) + tuple(model_fqn_map.parameters)
        unmatched_patterns = [
            pattern
            for pattern in plan.plan
            if not any(match_fqn(pattern, fqn) for fqn in model_fqns)
        ]
        if unmatched_patterns:
            # TODO: Raise ValueError when unmatched plan patterns become errors.
            logger.warning_rank0(
                "FSDPParallelPlan patterns matched no model FQNs: %s",
                unmatched_patterns,
            )

        modules: dict[str, FSDPPlanSpec] = {}
        for model_fqn, prepared_fqn in model_fqn_map.modules.items():
            spec = plan.match(model_fqn)
            if isinstance(spec, FSDPPlanSpec):
                modules[prepared_fqn] = spec

        parameters: dict[str, FSDPPlanSpec] = {}
        for model_fqn, prepared_fqn in model_fqn_map.parameters.items():
            spec = plan.match(model_fqn)
            if isinstance(spec, FSDPPlanSpec):
                parameters[prepared_fqn] = spec
        return _ResolvedFSDPPlan(
            modules=MappingProxyType(modules),
            parameters=MappingProxyType(parameters),
        )

    def _register_no_shard_params_grad_sync(self, model: nn.Module) -> None:
        """Synchronize gradients for ``no_shard`` parameters.

        FSDP receives these parameters as ``ignored_params`` and therefore does
        not reduce their gradients. FSDP2 uses ReduceOp.AVG by default, so the
        manual synchronization uses AVG as well.
        """
        resolved_plan = self._resolved_parallel_plan
        if resolved_plan is None:
            raise RuntimeError("FSDP plan has not been compiled")
        no_shard_params, _ = self._collect_no_shard(model, resolved_plan)

        world_group = dist.group.WORLD
        for param in no_shard_params:
            if not param.requires_grad:
                continue

            def _grad_hook(p):
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=world_group)

            param.register_post_accumulate_grad_hook(_grad_hook)

    def prepare_model(
        self,
        model: nn.Module,
        parallel_plan: Any = None,
        peft_config: Any = None,
    ) -> nn.Module:
        if parallel_plan is None:
            plan = FSDPParallelPlan()
        elif not isinstance(parallel_plan, FSDPParallelPlan):
            raise ValueError(
                f"FSDP2Backend requires an FSDPParallelPlan, got {type(parallel_plan).__name__}"
            )
        else:
            plan = parallel_plan

        self.parallel_plan = plan
        self._use_fla_kcp = plan.use_fla_kcp

        # Initialize DeviceMesh and SP groups
        self._init_mesh(plan)

        # Phase 1: Module replacement (e.g., MoE layers).
        model = self._replace_modules(model, plan)

        # Phase 2: Apply LoRA (after module replacement, before sharding).
        if peft_config is not None:
            from ...peft import apply_peft

            logger.info_rank0("Applying PEFT LoRA to model")
            model = apply_peft(model, peft_config)

        # Models slice inputs at their declared boundary. Install slicing
        # first so activation checkpointing wraps it and remains effective.
        sp_info = self.get_sp_info()
        if sp_info["sp_size"] > 1:
            patch_forward_for_sp_input_slicing(
                model.get_sequence_parallel_input_slicing_spec(),
                sp_info["sp_group"],
                sp_info["sp_size"],
                sp_info["sp_rank"],
            )

        # Freeze model-authored patterns into exact targets before later
        # transformations can add wrappers or replace parameter objects.
        self._model_fqn_map = build_model_fqn_map(model)
        self._resolved_parallel_plan = self._compile_parallel_plan(
            plan, self._model_fqn_map
        )

        # Phase 3: Activation checkpointing must be applied before FSDP2 so
        # checkpoint recompute runs inside the FSDP unit, not around it. Targets
        # are explicit in the parallel plan and patch module.forward directly, so
        # FQNs stay stable for later sharding/prefetch/weight logic.
        model = self._apply_activation_checkpointing(
            model, self._resolved_parallel_plan
        )

        # Phase 4: FSDP2 sharding.
        model = self._shard_model(model, self._resolved_parallel_plan)
        self.prefetch_manager.setup(model)

        # Append after FSDP hooks so unsharding and mixed-precision casting run
        # before redistribution, and the fan-out runs after FSDP post-forward.
        for fqn, spec in self._resolved_parallel_plan.modules.items():
            vision_spec = spec.vision_balance
            if vision_spec is None or vision_spec.mode is None:
                continue
            register_vision_balance(
                model.get_submodule(fqn),
                spec=vision_spec,
                sp_size=sp_info["sp_size"],
                sp_rank=sp_info["sp_rank"],
            )
            logger.info_rank0(
                "Vision balance registered on %s (mode=%s)",
                fqn,
                vision_spec.mode.value,
            )

        ep_size = self.get_ep_info()["ep_size"]
        if ep_size > 1:
            self.gradient_scaling_factor = 1 / ep_size

        return model

    def create_checkpoint_manager(self) -> FSDP2CheckpointManager:
        if self._mesh is None or self._world_mesh is None:
            raise RuntimeError(
                "Checkpoint manager is unavailable. Call prepare_model() first."
            )
        offload = self.fsdp_config.cpu_offload
        return FSDP2CheckpointManager(
            rank=self.rank,
            mesh=self._mesh,
            world_mesh=self._world_mesh,
            parallel_context=self.get_parallel_context(),
            device_runtime=self.device_runtime,
            cpu_offload=offload.enabled and offload.mode is CPUOffloadMode.FULL,
            # The manager is created before prepare_optimizer().
            require_optimizer_not_suspended=(
                lambda: self._optimizer_memory.require_not_suspended()
            ),
        )

    def materialize(self, model: nn.Module) -> nn.Module:
        offload = self.fsdp_config.cpu_offload
        cpu_offload = offload.enabled and offload.mode is CPUOffloadMode.FULL
        device = torch.device("cpu") if cpu_offload else self.device
        model.to_empty(device=device)
        if cpu_offload:
            self._keep_non_offloaded_on_device(model)
        return model

    def _keep_non_offloaded_on_device(self, model: nn.Module) -> None:
        """Keep FSDP-unmanaged tensors on the compute device under CPU offload.

        CPUOffloadPolicy offloads only sharded DTensor parameters. Buffers and
        ignored no_shard parameters are plain tensors, so they must stay on the
        compute device for forward and backward.
        """
        target = self._device_runtime.device
        for _, submodule in model.named_modules():
            for buffer_name, buffer in list(submodule.named_buffers(recurse=False)):
                if buffer is None or buffer.device.type == target.type:
                    continue
                persistent = buffer_name not in submodule._non_persistent_buffers_set
                submodule.register_buffer(
                    buffer_name, buffer.detach().to(target), persistent=persistent
                )
            for param in submodule.parameters(recurse=False):
                if (
                    param is None
                    or isinstance(param, DTensor)
                    or param.device.type == target.type
                ):
                    continue
                param.data = param.data.to(target)

    def finalize_model_load(self, model: nn.Module) -> nn.Module:
        # Register after loading or materialization because assign=True can
        # replace parameter objects and discard hooks registered earlier.
        self._register_no_shard_params_grad_sync(model)
        return model

    def _build_fsdp_kwargs(
        self,
        ignored_params: set[nn.Parameter],
    ) -> dict[str, Any]:
        from torch.distributed.fsdp import MixedPrecisionPolicy

        fsdp_kwargs = {}

        mp_policy = None
        if self.mixed_precision.value == "bf16":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        if mp_policy:
            fsdp_kwargs["mp_policy"] = mp_policy

        if hasattr(self, "reshard_after_forward"):
            fsdp_kwargs["reshard_after_forward"] = self.reshard_after_forward

        if ignored_params:
            fsdp_kwargs["ignored_params"] = ignored_params

        offload = self.fsdp_config.cpu_offload
        if offload.enabled and offload.mode is CPUOffloadMode.FULL:
            from torch.distributed.fsdp import CPUOffloadPolicy

            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

        return fsdp_kwargs

    def _apply_fsdp(
        self,
        model: nn.Module,
        units: list[_FSDPUnit],
        fsdp_kwargs: dict[str, Any],
    ) -> None:
        from torch.distributed.fsdp import fully_shard

        unit_by_module_id: dict[int, tuple[int, _FSDPUnit]] = {}
        for idx, unit in enumerate(units):
            for module in unit.modules:
                unit_by_module_id[id(module)] = (idx, unit)

        applied: set[int] = set()

        for _, module in reversed(list(model.named_modules())):
            item = unit_by_module_id.get(id(module))
            if item is None:
                continue
            unit_idx, unit = item
            if unit_idx in applied:
                continue

            target_fsdp_kwargs = fsdp_kwargs
            if unit.reshard_after_forward is not None:
                target_fsdp_kwargs = fsdp_kwargs.copy()
                target_fsdp_kwargs["reshard_after_forward"] = (
                    unit.reshard_after_forward
                )

            fully_shard(unit.modules, mesh=unit.mesh, **target_fsdp_kwargs)
            applied.add(unit_idx)

        if len(applied) != len(units):
            raise RuntimeError(
                "Some FSDP units were not found in model.named_modules(); "
                "FSDP units must only contain registered model submodules."
            )

    def _optimize_fsdp_units(
        self,
        model: nn.Module,
        units: list[_FSDPUnit],
    ) -> list[_FSDPUnit]:
        """Return optimized FSDP units before ``fully_shard``.

        This is a deterministic unit transform following Accelerate's FSDP2
        layout: carve input embeddings, conditionally merge the final norm with
        the output embedding tail, and keep root/tail params resident because
        their backward all-gather cannot be hidden behind useful compute.
        """
        # Start from explicit/default FSDP units collected by _shard_model.
        # The optimizer only appends or merges deterministic auto units.
        optimized = list(units)

        # Direct lookup for modules that are already FSDP unit anchors. This is
        # used to merge final_norm into an explicitly configured output embed
        # unit instead of creating a second tail unit.
        unit_by_module_id = {
            id(module): unit for unit in optimized for module in unit.modules
        }

        # Full subtree coverage from existing units. Auto units must not split
        # inside an explicit unit or default layer unit.
        covered_module_ids = {
            id(child)
            for unit in optimized
            for module in unit.modules
            for child in module.modules()
        }

        # Auto units are only added for real, registered modules that are not
        # already covered by a base unit and are not marked no_shard.
        def can_add_auto_unit(module: Any) -> bool:
            return (
                isinstance(module, nn.Module)
                and id(module) not in covered_module_ids
                and not getattr(module, "_fsdp_no_shard", False)
            )

        # Accelerate-style input embedding discovery; absent/unsupported APIs
        # simply mean there is no input embed auto unit to add.
        input_embed = None
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                input_embed = get_input_embeddings()
            except (NotImplementedError, AttributeError):
                pass

        # Output embedding (``get_output_embeddings`` delegates to
        # ``get_lm_head`` by default and remains overridable per model).
        output_embed = None
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            try:
                output_embed = get_output_embeddings()
            except (NotImplementedError, AttributeError):
                pass

        # Tied embeddings share the same weight object. In that case the tail
        # resident unit should be the input embedding, not a separate output
        # embedding module.
        input_weight = getattr(input_embed, "weight", None)
        output_weight = getattr(output_embed, "weight", None)
        is_tied = input_weight is not None and input_weight is output_weight
        tail_embed = input_embed if is_tied else output_embed

        loss_chunk_enabled = self.config.loss_chunk.enabled
        lm_head = model.get_lm_head()
        if loss_chunk_enabled and is_tied:
            raise ValueError(
                "chunked LM loss does not support tied input/output "
                "embeddings. Disable it (--tt_enable_chunk_loss false) or "
                "untie the embeddings."
            )

        # MTP calls lm_head repeatedly, so only merge the tail when MTP is off.
        final_norm = self._find_final_norm(model)
        can_merge_tail = not self.config.mtp.enabled
        split_lm_head_tail = (
            loss_chunk_enabled and not is_tied and tail_embed is lm_head
        )
        merge_norm_with_tail = can_merge_tail and not split_lm_head_tail

        if not is_tied and can_add_auto_unit(input_embed):
            optimized.append(_FSDPUnit(modules=[input_embed], mesh=self._world_mesh))

        # If output/tied embedding is already an explicit unit, extend that unit
        # with final_norm. Otherwise, create one auto tail unit from the
        # available final_norm/output embedding modules.
        tail_embed_unit = (
            unit_by_module_id.get(id(tail_embed))
            if isinstance(tail_embed, nn.Module)
            else None
        )
        if tail_embed_unit is not None:
            if merge_norm_with_tail and can_add_auto_unit(final_norm):
                tail_embed_unit.modules = [final_norm] + tail_embed_unit.modules
            tail_embed_unit.reshard_after_forward = False
        else:
            tail_modules = [
                module
                for module in (
                    (final_norm if merge_norm_with_tail else None),
                    tail_embed,
                )
                if can_add_auto_unit(module)
            ]
            if tail_modules:
                optimized.append(
                    _FSDPUnit(
                        modules=tail_modules,
                        mesh=self._world_mesh,
                        reshard_after_forward=False,
                    )
                )

        if not merge_norm_with_tail and can_add_auto_unit(final_norm):
            optimized.append(
                _FSDPUnit(
                    modules=[final_norm],
                    mesh=self._world_mesh,
                    reshard_after_forward=False,
                )
            )

        if can_add_auto_unit(model):
            optimized.append(
                _FSDPUnit(
                    modules=[model],
                    mesh=self._world_mesh,
                    reshard_after_forward=False,
                )
            )

        # Root and tail units are kept resident after forward. Their backward
        # all-gathers happen at the step edges, where there is little useful
        # compute to hide them.
        resident_module_ids = {
            id(module)
            for module in (model, final_norm, tail_embed)
            if isinstance(module, nn.Module)
        }
        for unit in optimized:
            if any(id(module) in resident_module_ids for module in unit.modules):
                unit.reshard_after_forward = False

        return optimized

    def _apply_activation_checkpointing(
        self, model: nn.Module, plan: _ResolvedFSDPPlan
    ) -> nn.Module:
        from contextlib import nullcontext
        from torch.utils.checkpoint import checkpoint

        target_modules: list[tuple[str, nn.Module]] = []
        for fqn, submodule in model.named_modules():
            spec = plan.modules.get(fqn)
            if isinstance(spec, FSDPPlanSpec) and spec.activation_checkpoint:
                target_modules.append((fqn, submodule))

        if not target_modules:
            return model

        if not self.config.activation_checkpointing.enabled:
            target_fqns = [fqn or "<root>" for fqn, _ in target_modules]
            logger.warning_rank0(
                "Activation checkpoint targets are configured in the "
                "parallel plan, but activation_checkpointing.enabled is "
                "False; skipping activation checkpointing for %s.",
                target_fqns,
            )
            return model

        # TorchTurbo owns activation checkpointing here. Make sure HF's own
        # GradientCheckpointingLayer base class does not ALSO checkpoint its
        # __call__: that would double-checkpoint and, worse, re-enter the FSDP
        # forward hooks during recompute (the forward-prefetch leak).
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False
            # Match HF/LF gradient checkpointing behavior. Some parent forwards
            # read config.use_cache before calling the checkpointed layers.
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False

        def checkpoint_context_fn():
            # The recompute context is active only in backward. Expert containers
            # can be nested FSDP units under the checkpointed layer, and their
            # PRE_BACKWARD hook may not have fired yet when recompute calls
            # experts.forward(). Without this guard, expert forward-prefetch
            # edges can leak into backward.
            return (
                nullcontext(),
                self.prefetch_manager.suspend_expert_forward_prefetch(model),
            )

        def make_checkpointed_forward(module: nn.Module):
            original_forward = module.forward

            use_reentrant = bool(
                getattr(self.config.activation_checkpointing, "use_reentrant", False)
            )

            @functools.wraps(original_forward)
            def checkpointed_forward(*args, **kwargs):
                if not module.training:
                    return original_forward(*args, **kwargs)

                checkpoint_kwargs = kwargs
                for cache_key in (
                    "past_key_value",
                    "past_key_values",
                    "layer_past",
                ):
                    if checkpoint_kwargs.get(cache_key) is not None:
                        checkpoint_kwargs = dict(checkpoint_kwargs)
                        checkpoint_kwargs[cache_key] = None
                if checkpoint_kwargs.get("use_cache", False):
                    checkpoint_kwargs = dict(checkpoint_kwargs)
                    checkpoint_kwargs["use_cache"] = False

                base_fn = functools.partial(original_forward, **checkpoint_kwargs)

                if not use_reentrant:
                    return checkpoint(
                        base_fn,
                        *args,
                        use_reentrant=False,
                        context_fn=checkpoint_context_fn,
                    )

                suspend_prefetch = self.prefetch_manager.suspend_expert_forward_prefetch

                def run_fn(*a, **kw):
                    if torch.is_grad_enabled():
                        with suspend_prefetch(model):
                            return base_fn(*a, **kw)
                    return base_fn(*a, **kw)

                return checkpoint(
                    run_fn,
                    *args,
                    use_reentrant=True,
                )

            return checkpointed_forward

        for fqn, submodule in target_modules:
            if hasattr(submodule, "_tt_original_forward"):
                continue
            submodule._tt_original_forward = submodule.forward
            submodule.forward = make_checkpointed_forward(submodule)

        # When base model params are frozen (e.g. LoRA) and activation
        # checkpointing with use_reentrant=False is active, the recompute
        # inside checkpoint cannot build a gradient path if input embeddings
        # have requires_grad=False, causing "element 0 of tensors does not
        # require grad" on backward(). enable_input_require_grads() registers
        # a forward hook on the embed tokens layer that forces its output to
        # require_grad=True, so gradients always flow to LoRA adapters.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            self.logger.info(
                "Called enable_input_require_grads() after activation checkpointing"
            )

        return model

    def _collect_no_shard(
        self, model: nn.Module, plan: _ResolvedFSDPPlan
    ) -> tuple[set[nn.Parameter], list[nn.Module]]:
        """Identify ``no_shard`` params and module-level subtrees from *plan*.

        Two resolved target kinds, both driven by ``FSDPPlanSpec.no_shard``:

        - module-level: a spec matching a module FQN selects the whole subtree,
          so every parameter under it is collected (e.g. replicating a small
          vision tower).
        - param-level: a spec matching a parameter name selects only that
          parameter (e.g. deepseek_v4 ``hc_*``/``attn_sink`` scalars); the
          owning module is still sharded normally.

        Pure identification with no side effects; callers decide what to do with
        the result (mark ``_fsdp_no_shard``, feed FSDP ``ignored_params``, or
        register grad-sync hooks). ``_shard_model`` and
        ``_register_no_shard_params_grad_sync`` share this so the two never
        disagree on which params are no_shard.

        Returns ``(no_shard_params, no_shard_modules)``.
        """
        no_shard_params: set[nn.Parameter] = set()
        no_shard_modules: list[nn.Module] = []

        inactive_roots = self._get_inactive_modules_to_save(model)
        inactive_params = {
            parameter
            for root in inactive_roots
            for parameter in root.parameters()
        }

        for fqn, spec in plan.modules.items():
            if not spec.no_shard:
                continue
            submodule = model.get_submodule(fqn)
            no_shard_modules.append(submodule)
            for param in submodule.parameters():
                if param in no_shard_params or param in inactive_params:
                    continue
                no_shard_params.add(param)

        named_parameters = dict(model.named_parameters())
        for name, spec in plan.parameters.items():
            if not spec.no_shard:
                continue
            param = named_parameters[name]
            if param in no_shard_params:
                continue
            no_shard_params.add(param)

        return no_shard_params, no_shard_modules

    @staticmethod
    def _get_inactive_modules_to_save(model: nn.Module) -> tuple[nn.Module, ...]:
        """Return PEFT modules-to-save copies that do not run in forward."""
        try:
            from peft.utils.other import ModulesToSaveWrapper
        except ImportError:
            return ()

        inactive_modules: list[nn.Module] = []
        for module in model.modules():
            if not isinstance(module, ModulesToSaveWrapper):
                continue

            if not module.disable_adapters and module.active_adapters:
                active_name = module.active_adapters[0]
                active_module = module.modules_to_save[active_name]
            else:
                active_module = module.original_module

            if module.original_module is not active_module:
                inactive_modules.append(module.original_module)
            inactive_modules.extend(
                saved_module
                for saved_module in module.modules_to_save.values()
                if saved_module is not active_module
            )
        return tuple(inactive_modules)

    def _forward_for_lm_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        *,
        use_chunked_loss: bool,
    ) -> tuple[Any, list[torch.Tensor] | None]:
        """Run LM forward, optionally replacing its output head with a no-op."""
        if not use_chunked_loss:
            return model(**inputs), None

        forward_inputs = dict(inputs)
        forward_inputs["return_dict"] = True
        lm_head = model.get_lm_head()
        if getattr(lm_head, "bias", None) is not None:
            raise NotImplementedError(
                "chunked linear CE does not support lm_head bias"
            )
        module_name = next(
            name for name, module in model.named_modules() if module is lm_head
        )
        parent_name, _, name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name)

        # Shadow the attribute without changing its FSDP module registration.
        object.__setattr__(parent, name, nn.Identity())
        try:
            outputs = model(**forward_inputs)
        finally:
            object.__delattr__(parent, name)

        hidden_dtype = lm_head.weight.dtype
        hidden_states = [outputs.logits.to(hidden_dtype)]
        mtp_hidden_states = getattr(outputs, "mtp_logits", None)
        if mtp_hidden_states:
            hidden_states.extend(
                hidden_state.to(hidden_dtype)
                for hidden_state in mtp_hidden_states
            )
        outputs["logits"] = None
        if mtp_hidden_states is not None:
            outputs["mtp_logits"] = None
        return outputs, hidden_states

    def _prepare_chunk_loss_lm_head(
        self, model: nn.Module, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the full LM-head weight, FSDP input, and fallback anchor."""
        lm_head = model.get_lm_head()

        if isinstance(lm_head, FSDPModule):
            if register_fsdp_forward_method is None:
                raise RuntimeError(
                    "chunked LM loss requires a PyTorch FSDP build with "
                    "register_fsdp_forward_method"
                )

            # The model forward bypasses this unit. This synthetic forward
            # all-gathers it only after the backbone and MTP forwards finish,
            # then keeps the full weight resident until chunk loss backward.
            lm_head.set_reshard_after_forward(False, recurse=False)
            method_name = "_tt_prepare_chunk_loss_input"
            if not hasattr(lm_head, method_name):

                def prepare_chunk_loss_input(
                    module: nn.Module, reference: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    weight = getattr(module, "weight", None)
                    if not isinstance(weight, torch.Tensor):
                        raise TypeError(
                            "chunked LM loss requires the LM head to expose "
                            "a tensor weight"
                        )
                    loss_anchor = reference.new_zeros(
                        (),
                        requires_grad=(
                            weight.requires_grad and not reference.requires_grad
                        ),
                    )
                    return reference, loss_anchor

                setattr(
                    lm_head,
                    method_name,
                    MethodType(prepare_chunk_loss_input, lm_head),
                )
                register_fsdp_forward_method(lm_head, method_name)

            # FSDP's root pre-forward requires at least one positional input.
            prepared_reference, loss_anchor = getattr(lm_head, method_name)(
                reference
            )
        else:
            # Keep the caller branch-free. This constant has no autograd edge
            # and is negligible compared with the chunked projection itself.
            prepared_reference = reference
            loss_anchor = lm_head.weight.new_zeros(())

        weight = lm_head.weight
        if isinstance(weight, DTensor):
            raise RuntimeError(
                "lm_head.weight is still sharded; chunked LM loss needs "
                "the full [vocab, hidden] matrix. Give the LM head its own "
                "unit in the parallel plan."
            )
        if not isinstance(weight, torch.Tensor):
            raise TypeError("lm_head weight must be a tensor")
        return weight, prepared_reference, loss_anchor

    def _shard_model(self, model: nn.Module, plan: _ResolvedFSDPPlan) -> nn.Module:
        named_modules = list(model.named_modules())

        # no_shard identification is shared with _register_no_shard_params_grad_sync
        # via _collect_no_shard so both stay in sync. Module-level matches replicate
        # a whole subtree: mark _fsdp_no_shard so the wrap passes below skip it.
        # Param-level matches (e.g. deepseek_v4 hc_*/attn_sink) keep their owning
        # module sharded and only route the matched params to ignored_params.
        no_shard_params, no_shard_modules = self._collect_no_shard(model, plan)
        inactive_roots = self._get_inactive_modules_to_save(model)
        inactive_subtree_modules = {
            id(child)
            for root in inactive_roots
            for child in root.modules()
        }
        for submodule in no_shard_modules:
            for child in submodule.modules():
                if id(child) in inactive_subtree_modules:
                    continue
                setattr(child, "_fsdp_no_shard", True)

        fsdp_kwargs = self._build_fsdp_kwargs(no_shard_params)

        # Collect base FSDP units from explicit plan entries and the default
        # ModuleList policy. Automatic carve/merge optimizations happen later.
        units: list[_FSDPUnit] = []
        wrapped_modules: set[int] = set()
        explicit_subtree_modules: set[int] = set()

        def _add_unit(
            modules: nn.Module | list[nn.Module],
            mesh: Any,
            reshard_after_forward: bool | None = None,
        ) -> None:
            modules = modules if isinstance(modules, list) else [modules]
            unit_ids = {id(module) for module in modules}
            if unit_ids & wrapped_modules:
                return
            units.append(
                _FSDPUnit(
                    modules=modules,
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
            )
            wrapped_modules.update(unit_ids)

        # PEFT keeps frozen and inactive copies registered under each
        # ModulesToSaveWrapper. Give each copy an FSDP boundary so an ancestor
        # unit cannot own and all-gather parameters that never run in forward.
        for root in inactive_roots:
            explicit_subtree_modules.update(id(child) for child in root.modules())
            _add_unit(root, self._world_mesh)

        # Explicit sharding specs first, bottom-up.
        for fqn, submodule in reversed(named_modules):
            if fqn == "" or getattr(submodule, "_fsdp_no_shard", False):
                continue

            spec = plan.modules.get(fqn)
            if not isinstance(spec, FSDPPlanSpec):
                continue

            if spec.placement is None and not spec.standalone_world_shard:
                continue

            if spec.placement is not None:
                mesh_dim = spec.placement.mesh_dim
                if mesh_dim is None or mesh_dim == "world":
                    target_mesh = self._world_mesh
                else:
                    target_mesh = self._mesh[mesh_dim]
            else:
                target_mesh = self._world_mesh

            explicit_subtree_modules.update(
                id(child) for child in submodule.modules()
            )
            _add_unit(submodule, target_mesh)

        # Default policy: each ModuleList child is a world-mesh FSDP unit.
        # Iterate reversed: named_modules() is a pre-order DFS (ancestors before
        # descendants), so reversing makes descendants wrap first.
        for fqn, submodule in reversed(named_modules):
            if getattr(submodule, "_fsdp_no_shard", False):
                continue
            if not isinstance(submodule, nn.ModuleList):
                continue
            for idx, child in enumerate(submodule):
                if getattr(child, "_fsdp_no_shard", False):
                    continue
                if id(child) in wrapped_modules:
                    continue
                # An explicit plan entry (placement or standalone_world_shard)
                # defines an FSDP boundary. The default ModuleList policy does not
                # split ModuleList children inside that explicitly wrapped subtree.
                if id(child) in explicit_subtree_modules:
                    continue
                _add_unit(child, self._world_mesh)

        units = self._optimize_fsdp_units(model, units)
        self._apply_fsdp(model=model, units=units, fsdp_kwargs=fsdp_kwargs)

        return model

    @staticmethod
    def _find_final_norm(model: nn.Module) -> Optional[nn.Module]:
        """Find the final normalization layer, mirroring Accelerate's heuristic.

        Scans the direct children of the model's base module
        (``base_model_prefix``) and returns the last child whose class name
        contains ``"Norm"``. Returns ``None`` when the base model is absent or
        has no norm among its direct children (e.g. deeply nested VLMs), in
        which case the norm simply stays in the root unit.
        """
        declared_norm = model.get_norm()
        if isinstance(declared_norm, nn.Module):
            return declared_norm

        base_prefix = getattr(model, "base_model_prefix", "")
        base_model = getattr(model, base_prefix, None) if base_prefix else model
        if not isinstance(base_model, nn.Module):
            return None
        final_norm: Optional[nn.Module] = None
        for _, module in base_model.named_children():
            if "Norm" in type(module).__name__:
                final_norm = module
        return final_norm

    def get_dp_info(self) -> dict[str, Any]:
        """Compute data-parallel rank and size, accounting for SP.

        When SP is active, ranks in the same SP group process the same
        data. The effective DP size = world_size / sp_size.

        Returns ``{"dp_rank": ..., "dp_size": ..., "dp_group": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        return {
            "dp_rank": self._dp_rank,
            "dp_size": self._dp_size,
            "dp_group": self._dp_group,
        }

    def get_sp_info(self) -> dict[str, Any]:
        """Get sequence-parallel rank, size and group.

        Returns ``{"sp_size": ..., "sp_group": ..., "sp_rank": ...,
        "use_fla_kcp": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        if self._sp_size and self._sp_group:
            return {
                "sp_size": self._sp_size,
                "sp_group": self._sp_group,
                "sp_rank": self._sp_rank,
                "use_fla_kcp": self._use_fla_kcp,
            }
        else:
            return {
                "sp_size": 1,
                "sp_group": None,
                "sp_rank": 0,
                "use_fla_kcp": False,
            }

    def get_ep_info(self) -> dict[str, Any]:
        """Get expert-parallel rank, size and group.

        Returns ``{"ep_rank": ..., "ep_size": ..., "ep_group": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        # If DeviceMesh is initialized, extract EP dimension directly
        if self._mesh is not None and "ep" in self._mesh.mesh_dim_names:
            ep_mesh = self._mesh["ep"]
            return {
                "ep_rank": ep_mesh.get_local_rank(),
                "ep_size": ep_mesh.size(),
                "ep_group": ep_mesh.get_group(),
            }
        else:
            return {"ep_rank": 0, "ep_size": 1, "ep_group": None}

    def get_parallel_context(self) -> dict[str, Any]:
        """Assemble the parallel-context dict from this backend's own EP/SP/DP info."""
        return {
            "dp": self.get_dp_info(),
            "sp": self.get_sp_info(),
            "ep": self.get_ep_info(),
        }

    def prepare_optimizer(
        self,
        plan: OptimizerPlan,
    ) -> Optimizer:
        """Build and validate the complete optimizer, including plan-defined hooks."""
        if not isinstance(plan, OptimizerPlan):
            raise TypeError("plan must be an OptimizerPlan")

        if isinstance(plan, OptimizerClassPlan):
            groups_by_optimizer_name = plan.get_param_groups_by_optimizer_name()
            expected_names = (plan.optimizer_name,)
            if tuple(groups_by_optimizer_name) != expected_names:
                raise ValueError(
                    "optimizer parameter groups do not match OptimizerClassPlan: "
                    f"expected {expected_names!r}, "
                    f"got {tuple(groups_by_optimizer_name)!r}"
                )
            groups = groups_by_optimizer_name[plan.optimizer_name]
            offload = self.fsdp_config.cpu_offload
            if plan.optimizer_cls is torch.optim.AdamW:
                optimizer = prepare_fsdp_adamw_optimizer(
                    groups,
                    plan.optimizer_kwargs,
                    device_runtime=self.device_runtime,
                    cpu_offload=offload,
                    split_param_groups=self._split_optimizer_groups_by_topology,
                )
            else:
                if offload.enabled and offload.mode is CPUOffloadMode.OPTIMIZER:
                    raise ValueError(
                        "swap-optimizer mode currently only supports AdamW; "
                        f"got {plan.optimizer_cls.__module__}.{plan.optimizer_cls.__name__}"
                    )
                optimizer = plan.optimizer_cls(
                    self._split_optimizer_groups_by_topology(
                        tuple(group.get_optimizer_param_group() for group in groups)
                    ),
                    **plan.optimizer_kwargs,
                )
            optimizers_by_name = {plan.optimizer_name: optimizer}

        elif isinstance(plan, MuonPlan):
            from .optimizers import prepare_fsdp_muon_optimizers

            offload = self.fsdp_config.cpu_offload
            optimizers_by_name = prepare_fsdp_muon_optimizers(
                plan,
                device_runtime=self.device_runtime,
                cpu_offload=offload,
                split_param_groups=self._split_optimizer_groups_by_topology,
            )

        else:
            raise TypeError(
                f"FSDP2 does not support optimizer plan {type(plan).__name__}"
            )

        # Validate every result before returning a single optimizer or the
        # composite facade. Step hooks must bind to that final object.
        container = OptimizersContainer(plan, optimizers_by_name)
        if len(container) == 1:
            optimizer = container.optimizers[0]
        else:
            optimizer = container
        qk_clip_controller = None
        if (
            isinstance(plan, MuonPlan)
            and plan.qk_clip_plan is not None
            and plan.qk_clip_plan.layers
        ):
            from .qk_clip import install_qk_clip

            qk_clip_controller = install_qk_clip(
                plan.qk_clip_plan,
                optimizer,
                sp_size=self._sp_size,
                device=self.device,
            )
        self._optimizer_memory = OptimizerMemoryController(
            optimizer,
            self.device_runtime,
            qk_clip_controller,
        )
        return optimizer

    def offload_optimizer(self) -> None:
        """Suspend optimizer use and release device resources for inference."""
        self._optimizer_memory.offload()

    def restore_optimizer(self) -> None:
        """Resume the prepared optimizer without replacing framework bindings."""
        self._optimizer_memory.restore()

    def _split_optimizer_groups_by_topology(
        self,
        param_groups: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Split native optimizer groups by FSDP mesh topology."""
        optimizer_params = []
        edp = self._mesh["edp"]
        for param_group in param_groups:
            group_options = {
                key: value
                for key, value in param_group.items()
                if key != "params"
            }
            split = {
                "dense": [],
                "moe": [],
                "ignored": [],
            }
            for param in param_group["params"]:
                if not param.requires_grad:
                    continue
                if isinstance(param, DTensor):
                    if param.device_mesh == edp:
                        key = "moe"
                    else:
                        key = "dense"
                else:
                    key = "ignored"
                split[key].append(param)

            for params in split.values():
                if not params:
                    continue
                optimizer_params.append({"params": params, **group_options})

        return optimizer_params

    @torch.no_grad()
    def _clip_grad_norm_cpu_offload(
        self, moe_params, params, ignored_params, max_norm, norm_type
    ):
        """Grad-norm + clip for the FSDP2 CPU-offload path — fully chunked on the
        compute device (peak GPU mem = one chunk, ``CLIP_CHUNK_BYTES``).
        """
        all_params = list(moe_params) + list(params) + list(ignored_params)
        reduce_device = self.device if not isinstance(self.device, str) else torch.device(self.device)
        world_group = dist.group.WORLD

        offloaded = [p for p in all_params if hasattr(p.grad, "_local_tensor")]
        plain_grads = [p.grad for p in all_params if not hasattr(p.grad, "_local_tensor")]

        # --- Pass 1: moe gradient scaling (1/ep), chunked on the device, before
        # the norm so the norm sees the ep-corrected moe grads.
        if moe_params:
            for chunk in self._iter_offloaded_chunks(moe_params):
                self._scale_offloaded_chunk_on_device(
                    chunk, reduce_device, self.gradient_scaling_factor
                )

        # --- Pass 2: global norm, chunked on the device.
        def _accumulate(t, acc):
            if t.numel() == 0:
                return acc
            if norm_type == float("inf"):
                m = t.detach().float().abs().amax()
                return torch.maximum(m, acc)
            return acc + (t.detach().float().abs() ** norm_type).sum()

        acc = torch.zeros((), device=reduce_device, dtype=torch.float32)
        for chunk in self._iter_offloaded_chunks(offloaded):
            chunk_dev = [l.to(reduce_device, non_blocking=True) for l in chunk]
            self.device_runtime.synchronize()  # H2D complete before reading chunk_dev
            for t in chunk_dev:
                acc = _accumulate(t, acc)
        # MAX reduction can hide a rank-local NaN.
        local_norm = acc.clone()
        reduce_op = dist.ReduceOp.MAX if norm_type == float("inf") else dist.ReduceOp.SUM
        dist.all_reduce(acc, op=reduce_op, group=world_group)
        # no_shard gradients are already world-reduced and replicated.
        for grad in plain_grads:
            acc = _accumulate(grad, acc)
        total_norm = acc if norm_type == float("inf") else acc ** (1.0 / norm_type)
        assert_finite(torch.stack((local_norm, total_norm)), "gradient norm")

        # --- Pass 3: clip scaling, chunked on the device.
        clip_coef = float(max_norm) / (float(total_norm) + 1e-6)
        if clip_coef < 1.0:
            coef = min(clip_coef, 1.0)
            for chunk in self._iter_offloaded_chunks(offloaded):
                self._scale_offloaded_chunk_on_device(chunk, reduce_device, coef)
            if plain_grads:
                torch._foreach_mul_(plain_grads, coef)
        return total_norm

    def _iter_offloaded_chunks(self, params):
        """Yield budget-sized lists of CPU-local shards (``_local_tensor``) of
        offloaded (DTensor) grads from ``params``, so each chunk's GPU footprint
        stays within ``CLIP_CHUNK_BYTES``."""
        CLIP_CHUNK_BYTES = 512 * 1024 * 1024
        chunk: list = []
        chunk_bytes = 0
        for p in params:
            g = p.grad
            if g is None or not hasattr(g, "_local_tensor"):
                continue
            l = g._local_tensor
            if l.numel() == 0:
                continue
            b = l.element_size() * l.numel()
            if chunk and chunk_bytes + b > CLIP_CHUNK_BYTES:
                yield chunk
                chunk, chunk_bytes = [], 0
            chunk.append(l)
            chunk_bytes += b
        if chunk:
            yield chunk

    def _scale_offloaded_chunk_on_device(self, cpu_locals, reduce_device, factor):
        """Move one chunk of CPU-local grad shards to the compute device, scale by
        ``factor`` (one batched ``_foreach_mul_``), and copy the result back into
        the CPU local objects in place (persists to ``p.grad`` — the offloaded
        param is on CPU, so ``p.grad`` cannot be reassigned).

        non_blocking H2D enqueues the chunk's copies concurrently, but the mul is
        not guaranteed to share the H2D stream, so synchronize before reading
        ``chunk_dev``; another sync after the D2H frees the device chunk so peak
        stays at one chunk.
        """
        chunk_dev = [l.to(reduce_device, non_blocking=True) for l in cpu_locals]
        self.device_runtime.synchronize()  # H2D complete before mul reads chunk_dev
        torch._foreach_mul_(chunk_dev, factor)
        torch._foreach_copy_(cpu_locals, chunk_dev)
        self.device_runtime.synchronize()  # D2H complete; chunk_dev freed on return

    @torch.no_grad()
    def clip_grad_norm_(self, parameters, max_norm=1.0, norm_type=2.0):
        offload = self.fsdp_config.cpu_offload
        params = []
        grads = []
        moe_params = []
        moe_grads = []
        # Parameters not managed by FSDP (ignored_params) have plain tensors,
        # not DTensors; they must be handled separately to avoid mixed-type errors.
        ignored_params = []
        ignored_grads = []
        edp = self._mesh["edp"] if self._mesh is not None else None

        for param in parameters:
            # Skip parameters without gradients to ensure correctness.
            if param.grad is not None:
                if hasattr(param, "device_mesh"):
                    if edp is not None and param.device_mesh == edp:
                        moe_params.append(param)
                        moe_grads.append(param.grad)
                    else:
                        params.append(param)
                        grads.append(param.grad)
                else:
                    ignored_params.append(param)
                    ignored_grads.append(param.grad)


        if offload.enabled and offload.mode is CPUOffloadMode.FULL:
            return self._clip_grad_norm_cpu_offload(
                moe_params, params, ignored_params, max_norm, norm_type
            )

        edp_total_norm_list = []
        if moe_params:
            self._scale_moe_gradient(moe_params)
            edp_total_norm = get_total_norm(moe_grads, norm_type=norm_type)
            edp_total_norm = edp_total_norm.full_tensor().unsqueeze(0)
            edp_total_norm = all_gather_tensor(
                edp_total_norm, self.get_ep_info()["ep_group"]
            )
            edp_total_norm_list.append(edp_total_norm)

        all_norms = edp_total_norm_list
        if grads:
            dp_total_norm = get_total_norm(grads, norm_type=norm_type)
            dp_total_norm = dp_total_norm.full_tensor().unsqueeze(0)
            all_norms.append(dp_total_norm)
        # Compute norm for non-DTensor ignored grads separately.
        if ignored_grads:
            ignored_total_norm = get_total_norm(ignored_grads, norm_type=norm_type)
            all_norms.append(ignored_total_norm.unsqueeze(0))
        # Under CPU offload the sharded-grad norms come back on CPU (full_tensor /
        # all_gather run over a gloo-capable PG), while no_shard/ignored grads stay
        # on the compute device — so all_norms can span CPU and the device. Move
        # every group norm onto one device before concatenating. total_norm's own
        # device is irrelevant to clip_grads_with_norm_, which re-homes the clip
        # coefficient per grad device.
        total_norm = torch.zeros((), device=self.device)
        if all_norms:
            ref_device = all_norms[0].device
            total_norm = torch.linalg.vector_norm(
                torch.cat([norm.to(ref_device) for norm in all_norms]), norm_type
            )
        norm_values = [total_norm.to(self.device)]
        if norm_type == float("inf") and (grads or moe_grads):
            # DTensor can hide NaNs in MAX reductions inside get_total_norm.
            local_grads = [grad.to_local() for grad in grads + moe_grads]
            local_norm = get_total_norm(local_grads, norm_type=norm_type)
            norm_values.append(local_norm.to(self.device))
        assert_finite(torch.stack(norm_values), "gradient norm")

        if moe_params:
            clip_grads_with_norm_(moe_params, max_norm, total_norm)
        if params:
            clip_grads_with_norm_(params, max_norm, total_norm)
        if ignored_params:
            clip_grads_with_norm_(ignored_params, max_norm, total_norm)

        return total_norm

    def prepare_inputs(self, model: nn.Module, inputs: Any) -> Any:
        """Pad the global batch with the model's SP padding rules.

        ``cu_seq_lens_q/k`` keep describing the valid prefix, so attention and
        MoE statistics exclude the dummy media and SP alignment tails.
        """
        if not isinstance(inputs, Mapping):
            return inputs

        sp_size = self.get_sp_info()["sp_size"]
        if sp_size <= 1:
            return inputs
        return ulysses_pad_inputs(
            inputs, model.get_sequence_parallel_input_slicing_spec(), sp_size
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if (
            inputs.get("position_ids", None) is None
            and inputs.get("mm_token_type_ids") is None
        ):
            # Text batches default to contiguous coordinates. Multimodal
            # batches keep position_ids absent so the model derives mrope
            # grid coordinates from the media tokens instead of receiving
            # linear positions that silently corrupt image RoPE.
            inputs["position_ids"] = torch.arange(
                inputs["input_ids"].size(1), device=inputs["input_ids"].device
            ).unsqueeze(0)
        inputs["use_cache"] = False

        labels = inputs.pop("labels", None)
        assert labels is not None, "'labels' should not be None in inputs"

        num_items_in_batch = kwargs.get("num_items_in_batch", None)
        sp_size = self.get_sp_info()["sp_size"]
        sp_group = self.get_sp_info()["sp_group"]
        sp_rank = self.get_sp_info()["sp_rank"]

        # assert num_items_in_batch is not None, "'num_items_in_batch' should not be None"

        # # Repeated calculation in SP group
        # num_items_in_batch = num_items_in_batch // sp_size

        use_chunked_loss = self.config.loss_chunk.enabled and model.training
        mtp_loss_scale = self.config.mtp.loss_scaling_factor
        outputs, chunk_hidden_states = self._forward_for_lm_loss(
            model,
            inputs,
            use_chunked_loss=use_chunked_loss,
        )
        dspark_loss_fn = getattr(outputs, "dspark_loss_fn", None)

        total_loss, component_losses = self._compute_lm_loss_with_sp(
            model=model,
            outputs=outputs,
            labels=labels,
            cu_seq_lens=inputs.get("cu_seq_lens_q"),
            chunk_hidden_states=chunk_hidden_states,
            sp_size=sp_size,
            sp_group=sp_group,
            sp_rank=sp_rank,
            num_items_in_batch=num_items_in_batch,
            mtp_loss_scale=mtp_loss_scale,
        )
        if dspark_loss_fn is not None:
            outputs.dspark_loss_fn = None
            if self.config.dspark.loss_scaling_factor == 0.0:
                # Keep metrics from the forward without connecting its router
                # auxiliary losses to the training graph.
                with torch.no_grad():
                    dspark_loss_fn(labels)
            else:
                dspark_loss = dspark_loss_fn(labels)
                # DSpark returns a global anchor mean. Match the LM loss's
                # microbatch normalization before adapters apply their SP scale.
                if num_items_in_batch is None:
                    dspark_loss = dspark_loss / sp_size
                else:
                    shifted_labels, normalizers = self._prepare_loss_labels(
                        [labels],
                        labels,
                        sp_size=sp_size,
                        sp_group=sp_group,
                        sp_rank=sp_rank,
                        num_items_in_batch=num_items_in_batch,
                        cu_seq_lens=inputs.get("cu_seq_lens_q"),
                    )
                    num_tokens = (shifted_labels[0] != -100).sum()
                    if sp_size > 1:
                        dist.all_reduce(num_tokens, group=sp_group)
                    dspark_loss = dspark_loss * num_tokens / normalizers[0]
                total_loss = (
                    total_loss + dspark_loss * self.config.dspark.loss_scaling_factor
                )
        mtp_losses = component_losses[1:]

        if self.config.mtp.log_loss:
            for index, mtp_loss in enumerate(mtp_losses):
                logger.info_rank0(
                    "MTP loss: sp_rank=%d index=%d raw_loss=%.6f",
                    sp_rank,
                    index,
                    mtp_loss.item(),
                )

        outputs.loss = total_loss
        outputs.main_loss = component_losses[0].detach()
        if mtp_losses:
            outputs.mtp_loss = [loss.detach() for loss in mtp_losses]
            outputs.mtp_loss_scaled = (
                sum(mtp_losses) * mtp_loss_scale
            ).detach()
        return (total_loss, outputs) if return_outputs else total_loss

    @staticmethod
    def _prepare_loss_labels(
        predictions: list[torch.Tensor],
        labels: torch.Tensor,
        *,
        sp_size: int,
        sp_group: Any,
        sp_rank: int,
        num_items_in_batch: int | torch.Tensor | None,
        cu_seq_lens: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> tuple[list[torch.Tensor], list[float | torch.Tensor]]:
        """Build shifted labels and normalizers for each LM prediction stream.

        When supplied, ``num_items_in_batch`` must be a positive global count of
        valid tokens. This is a caller precondition, not a value checked here
        for device tensors: a zero value can produce a zero denominator and a
        NaN loss. Device tensors remain on device to avoid a synchronizing branch.
        """
        use_num_items = num_items_in_batch is not None
        num_items_normalizer: float | torch.Tensor | None = None
        if use_num_items:
            world_size = dist.get_world_size()
            if torch.is_tensor(num_items_in_batch):
                if num_items_in_batch.numel() != 1:
                    raise ValueError(
                        "num_items_in_batch must be a scalar tensor, got shape "
                        f"{tuple(num_items_in_batch.shape)}"
                    )
                # The caller supplies a positive count. Preserve the exact
                # floor-division and SP/world-size scaling on the device.
                num_items_local = torch.div(
                    num_items_in_batch, sp_size, rounding_mode="floor"
                )
                num_items_normalizer = num_items_local * sp_size / world_size
            else:
                num_items_value = int(num_items_in_batch)
                # A fully masked microbatch can contribute zero. A zero
                # step-level count means the whole optimizer step has no
                # supervision and is invalid.
                if num_items_value <= 0:
                    raise ValueError(
                        "num_items_in_batch must be positive, got "
                        f"{num_items_value}"
                    )
                num_items_local = num_items_value // sp_size
                if num_items_local <= 0:
                    raise ValueError(
                        "num_items_in_batch yields 0 valid tokens per SP rank "
                        f"(num_items_in_batch={num_items_value}, "
                        f"sp_size={sp_size}); the loss has no denominator"
                    )
                num_items_normalizer = (
                    float(num_items_local * sp_size) / float(world_size)
                )

        sequence_ends = None
        if cu_seq_lens is not None:
            token_positions = torch.arange(labels.size(-1), device=labels.device)
            # Segment index per token; padding beyond the last boundary clamps
            # to it, and its positions always cross that boundary once shifted.
            segment_index = torch.bucketize(
                token_positions, cu_seq_lens[1:], right=True
            ).clamp(max=cu_seq_lens.numel() - 2)
            sequence_ends = cu_seq_lens[segment_index + 1]

        shifted_labels: list[torch.Tensor] = []
        normalizers: list[float | torch.Tensor] = []
        for shift_extra, prediction in enumerate(predictions):
            shift = 1 + shift_extra
            labels_i = nn.functional.pad(
                labels, (0, shift), value=ignore_index
            )
            labels_i = labels_i[..., shift:].contiguous()
            if sequence_ends is not None:
                # Every prediction stream must target a token in its own sample.
                crosses_boundary = token_positions + shift >= sequence_ends
                labels_i.masked_fill_(crosses_boundary, ignore_index)
            # Count before SP slicing because the denominator covers the full
            # sequence. A fully masked stream uses one as a neutral denominator.
            label_num = (
                None
                if use_num_items
                else (labels_i != ignore_index).sum().clamp(min=1)
            )
            if sp_size > 1:
                labels_i = slice_input_tensor(
                    labels_i,
                    dim=1,
                    group=sp_group,
                    rank=sp_rank,
                )
            shifted_labels.append(labels_i.to(prediction.device))
            normalizers.append(
                num_items_normalizer
                if use_num_items
                else label_num * sp_size
            )
        return shifted_labels, normalizers

    def _compute_lm_loss_with_sp(
        self,
        model: nn.Module,
        outputs: Any,
        labels: torch.Tensor,
        chunk_hidden_states: list[torch.Tensor] | None,
        sp_size: int,
        sp_group: Any,
        sp_rank: int,
        num_items_in_batch: int | torch.Tensor | None,
        mtp_loss_scale: float,
        cu_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Select the LM loss path and handle its SP-specific preparation.

        Returns the scaled total plus ``1 + num_mtp`` detached per-stream
        losses, main path first.
        """
        if chunk_hidden_states is not None:
            hidden_states = chunk_hidden_states

            shifted_labels, normalizers = self._prepare_loss_labels(
                hidden_states,
                labels,
                sp_size=sp_size,
                sp_group=sp_group,
                sp_rank=sp_rank,
                num_items_in_batch=num_items_in_batch,
                cu_seq_lens=cu_seq_lens,
            )
            loss_scales = [1.0] + [mtp_loss_scale] * (len(hidden_states) - 1)
            (
                lm_head_weight,
                hidden_states[0],
                loss_anchor,
            ) = self._prepare_chunk_loss_lm_head(model, hidden_states[0])
            total_loss, component_losses = chunk_loss(
                hidden_states,
                shifted_labels,
                lm_head=lm_head_weight,
                normalizers=normalizers,
                loss_scales=loss_scales,
                chunk_size=self.config.loss_chunk.chunk_size,
            )
            # The anchor is always zero. It only requires grad when the main
            # hidden state is detached, where the wrapped positional input
            # cannot trigger FSDP's callbacks (for example, head-only training).
            total_loss = total_loss + loss_anchor.to(total_loss.dtype)
        else:
            main_logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if main_logits is None:
                raise RuntimeError(
                    "model forward returned no logits on the regular CE path"
                )
            logits = [main_logits]
            mtp_logits = getattr(outputs, "mtp_logits", None)
            if self.config.mtp.enabled and mtp_logits:
                logits.extend(mtp_logits)

            shifted_labels, normalizers = self._prepare_loss_labels(
                logits,
                labels,
                sp_size=sp_size,
                sp_group=sp_group,
                sp_rank=sp_rank,
                num_items_in_batch=num_items_in_batch,
                cu_seq_lens=cu_seq_lens,
            )
            loss_scales = [1.0] + [mtp_loss_scale] * (len(logits) - 1)
            total_loss, component_losses = logits_loss(
                logits,
                shifted_labels,
                normalizers=normalizers,
                loss_scales=loss_scales,
            )

        if sp_size > 1 and sp_group is not None:
            total_loss = torch.distributed.nn.all_reduce(
                total_loss, op=dist.ReduceOp.SUM, group=sp_group
            )
            component_losses = component_losses.clone()
            dist.all_reduce(component_losses, op=dist.ReduceOp.SUM, group=sp_group)

        return total_loss, tuple(component_losses.unbind())

    def compute_diffusion_loss(self, model, inputs, ga: int = 1, return_outputs: bool = False, **kwargs):
        """Block-diffusion loss — unshifted masked CE over ``[noisy | clean]``.

        Relocated verbatim (numerically) from the former ``run_lf_llada2._llada_bd_loss``.
        Consumes the BD batch produced by ``BDDataset``:

        - ``input_ids``: ``[1, 2T]`` = ``cat([noisy, clean])``
        - ``attention_mask``: ``[1, 1, 2T, 2T]`` additive block mask
        - ``position_ids``: ``[1, 2T]`` (two independent ``arange(T)``)
        - ``labels``: ``[1, 2T]`` (noisy half supervised, clean half = -100)

        ``cap_lambda`` (CAP confidence-loss weight) comes from ``self.config.diffusion``,
        not a global getter. Under SP>1 the sequence is sliced to the local shard (block
        mask stays full); loss is summed locally, all-reduced across the SP group, then
        divided by the global valid count and by ``ga``. ``ga`` must include torchturbo's
        SP factor (``gradient_accumulation_steps`` is already multiplied by ``sp_size``
        upstream).
        """
        if self.config.loss_chunk.enabled:
            raise NotImplementedError(
                "chunked LM loss does not support diffusion training"
            )

        inputs = dict(self.prepare_inputs(model, inputs))
        inputs["use_cache"] = False
        labels = inputs.pop("labels")  # [1, 2T], clean half = -100

        cap_lambda = float(self.config.diffusion.cap_lambda)  # 0 = CAP disabled

        from torch.nn import CrossEntropyLoss
        ignore_index = CrossEntropyLoss().ignore_index  # -100

        sp_info = self.get_sp_info()
        sp_size = sp_info["sp_size"]
        sp_group = sp_info["sp_group"]
        sp_rank = sp_info["sp_rank"]

        block_mask = inputs["attention_mask"]

        if sp_size > 1 and sp_group is not None:
            # The model slices its inputs; loss labels use the same local shard.
            n_valid = int((labels != ignore_index).sum().item())
            labels_local = slice_input_tensor(labels, dim=1, group=sp_group, rank=sp_rank)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=block_mask,
                position_ids=inputs["position_ids"],
                use_cache=False,
            )
            loss_local = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels_local.view(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )
            # SP all-reduce (kept in autograd graph), then /global valid count → SP=1 mean.
            loss_sum = torch.distributed.nn.all_reduce(
                loss_local, op=dist.ReduceOp.SUM, group=sp_group
            )
            loss = loss_sum / float(n_valid) if n_valid > 0 else loss_sum * 0.0

            # --- CAP confidence term (global mean via all-reduce of sum + count) ---
            if cap_lambda > 0:
                ent_local, nc_local = _confidence_entropy_sum(
                    outputs.logits, labels_local, ignore_index=ignore_index
                )
                ent_sum = torch.distributed.nn.all_reduce(ent_local, op=dist.ReduceOp.SUM, group=sp_group)
                nc = nc_local.clone()
                dist.all_reduce(nc, op=dist.ReduceOp.SUM, group=sp_group)  # count: no grad
                loss = loss + cap_lambda * (ent_sum / nc.clamp_min(1))
        else:
            outputs = model(**inputs)
            T = labels.size(1)
            noisy_logits = outputs.logits[:, :T].contiguous()  # [1, T, V]
            noisy_labels = labels[:, :T].contiguous()
            loss = nn.functional.cross_entropy(
                noisy_logits.view(-1, noisy_logits.size(-1)),
                noisy_labels.view(-1),
                ignore_index=ignore_index,
            )  # default mean over masked positions (== SP>1 sum/n_valid)

            # --- CAP confidence term ---
            if cap_lambda > 0:
                ent_sum, nc = _confidence_entropy_sum(
                    noisy_logits, noisy_labels, ignore_index=ignore_index
                )
                loss = loss + cap_lambda * (ent_sum / nc.clamp_min(1))

        loss = loss / ga  # align with HF Trainer's grad-accum normalization
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    def _scale_moe_gradient(self, params):
        if len(params) < 1:
            return

        assert params[0].device_mesh == self._mesh["edp"]
        grads = [p.grad for p in params]
        torch._foreach_mul_(grads, self.gradient_scaling_factor)

    def __repr__(self) -> str:
        return (
            f"<FSDP2Backend(mp={self.mixed_precision.value}, "
            f"strategy={self.sharding_strategy.value}, "
            f"reshard={self.reshard_after_forward})>"
        )


# Auto-register on import
register_backend("fsdp2", FSDP2Backend)
