"""FSDP2 distributed backend — ``torch.distributed.fsdp.fully_shard``."""

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch.distributed as dist
import torch
import torch.nn as nn
try:
    from torch.nn.utils import clip_grads_with_norm_, get_total_norm
except ImportError:  # pragma: no cover - torch<2.9 compatibility
    from torch.nn.utils.clip_grad import (  # type: ignore[attr-defined]
        _clip_grads_with_norm_ as clip_grads_with_norm_,
        _get_total_norm as get_total_norm,
    )

from torchturbo.models.modules.attention.ulysses import (
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from torchturbo.models.modules.comm import all_gather_tensor
from torchturbo.models.base import SequenceParallelInputMode
from torchturbo.ops import ops
from torchturbo.parallel import (
    FSDPPlanSpec,
    FSDPParallelPlan,
    ParallelPlan,
)
from torchturbo.runtime import DeviceRuntime
from torchturbo.engine.utils import strip_peft_prefix
from torchturbo.utils.flash_attn_utils import get_unpad_data

from ...backend_registry import register_backend
from ...config import EngineConfig
from ...distributed import init_process_group_if_needed
from ....logging import get_logger
from ..base import BaseBackend
from .config import FSDPConfig
from .prefetch import FSDPPrefetchManager

logger = get_logger(__name__)


@dataclass
class _FSDPUnit:
    modules: list[nn.Module]
    mesh: Any
    reshard_after_forward: bool | None = None


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

        if self.fsdp_config.cpu_offload:
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

        # Data parallel process group.
        # exposed through :`get_dp_info`.
        self._dp_group: dist.ProcessGroup = None
        self._dp_rank: int = 0
        self._dp_size: int = 1
        self.use_transformer_engine = False
        self.gradient_scaling_factor = 1.0
        self.parallel_plan: FSDPParallelPlan = None

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
        self._world_mesh = init_device_mesh(device_name, (world_size,))
        self._mesh = init_device_mesh(
            device_name, (world_size // ep_size, ep_size), mesh_dim_names=("edp", "ep")
        )

        logger.info_rank0("Created DeviceMesh")

        self._create_sp_group(sp_size)
        self._create_dp_group()

    def _patch_transformers(self):
        # currently assume packing or normal attention mask
        import transformers.modeling_flash_attention_utils
        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

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

    def _register_no_shard_params_grad_sync(self, model: nn.Module) -> None:
        """Synchronize gradients for ``no_shard`` parameters.

        FSDP receives these parameters as ``ignored_params`` and therefore does
        not reduce their gradients. FSDP2 uses ReduceOp.AVG by default, so the
        manual synchronization uses AVG as well.
        """
        no_shard_params, _ = self._collect_no_shard(model, self.parallel_plan)

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

        # Initialize DeviceMesh and SP groups
        self._init_mesh(plan)

        # Phase 1: Module replacement (e.g., MoE layers).
        model = self._replace_modules(model, plan)

        # Phase 2: Apply LoRA (after module replacement, before sharding).
        if peft_config is not None:
            from ...peft import apply_peft

            logger.info_rank0("Applying PEFT LoRA to model")
            model = apply_peft(model, peft_config)

        if self.config.mega_moe:
            from ....optimizations.mega_moe.integration import install

            install(model, self.config, self.get_ep_info()["ep_group"])

        # Phase 3: Activation checkpointing must be applied before FSDP2 so
        # checkpoint recompute runs inside the FSDP unit, not around it. Targets
        # are explicit in the parallel plan and patch module.forward directly, so
        # FQNs stay stable for later sharding/prefetch/weight logic.
        model = self._apply_activation_checkpointing(model, plan)

        # Phase 4: FSDP2 sharding.
        model = self._shard_model(model, plan)
        self.prefetch_manager.setup(model)

        ep_size = self.get_ep_info()["ep_size"]
        if ep_size > 1:
            self.gradient_scaling_factor = 1 / ep_size

        self._patch_transformers()

        return model

    def load_model(
        self,
        model: nn.Module,
        model_name_or_path: str,
        parallel_context: dict[str, Any],
        load_weights: bool = True,
    ):
        if load_weights:
            from .weight_loader import DistParamLoader

            weight_converter = model.get_weight_converter(parallel_context)
            loader = DistParamLoader(
                model=model,
                path=model_name_or_path,
                rank=self.rank,
                mesh=self._mesh,
                world_mesh=self._world_mesh,
                device_runtime=self.device_runtime,
                is_cpu_offload=self.fsdp_config.cpu_offload,
                weight_converter=weight_converter,
            )
            loader.load_weights()
        else:
            device = torch.device("cpu") if self.fsdp_config.cpu_offload else self.device
            model.to_empty(device=device)
            if self.fsdp_config.cpu_offload:
                self._keep_non_offloaded_on_device(model)

        # Register gradient all-reduce hooks for no_shard params after the
        # model is loaded or materialized.
        self._register_no_shard_params_grad_sync(model)
        return model

    def _keep_non_offloaded_on_device(self, model: nn.Module) -> None:
        """Keep FSDP-unmanaged tensors on the compute device under CPU offload.

        CPUOffloadPolicy offloads only sharded (DTensor) params. Buffers and
        ignored no_shard params (plain tensors) are not managed by FSDP, so they
        must stay on the compute device for forward/backward to work.
        """
        target = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        for _, submodule in model.named_modules():
            for buf_name, buf in list(submodule.named_buffers(recurse=False)):
                if buf is None or buf.device.type == target.type:
                    continue
                persistent = buf_name not in submodule._non_persistent_buffers_set
                submodule.register_buffer(
                    buf_name, buf.detach().to(target), persistent=persistent
                )
            for param in submodule.parameters(recurse=False):
                # DTensor (device_mesh) params are FSDP-managed / offloaded; only
                # ignored no_shard plain params need moving back to the device.
                if (
                    param is None
                    or hasattr(param, "device_mesh")
                    or param.device.type == target.type
                ):
                    continue
                param.data = param.data.to(target)

    def save_model(
        self,
        model: nn.Module,
        model_name_or_path: str,
        output_dir: str,
        parallel_context: dict[str, Any],
        **kwargs: Any,
    ):
        from .weight_saver import DistParamSaver

        weight_converter = model.get_weight_converter(parallel_context)
        saver = DistParamSaver(
            model=model,
            rank=self.rank,
            mesh=self._mesh,
            path=model_name_or_path,
            processing_class=None,
            weight_converter=weight_converter,
            device_runtime=self.device_runtime,
        )
        saver.save_model(output_dir)

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
        elif self.mixed_precision.value == "fp16":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                cast_forward_inputs=True,
            )
        if mp_policy:
            fsdp_kwargs["mp_policy"] = mp_policy

        if hasattr(self, "reshard_after_forward"):
            fsdp_kwargs["reshard_after_forward"] = self.reshard_after_forward

        if ignored_params:
            fsdp_kwargs["ignored_params"] = ignored_params

        if self.fsdp_config.cpu_offload:
            from torch.distributed.fsdp import CPUOffloadPolicy

            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(
                pin_memory=self.fsdp_config.pin_memory
            )

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
        layout: carve input embeddings, merge the final norm with the output
        embedding tail, and keep root/tail params resident because their
        backward all-gather cannot be hidden behind useful compute.
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

        # Output embedding (turbo ``get_output_embeddings`` -> ``get_lm_head`` by
        # default, overridable per model; non-turbo models keep HF's API).
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

        # Final norm is merged with the tail unit when possible so the last
        # small parameters stay resident with lm_head/output embedding.
        final_norm = self._find_final_norm(model)

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
            if can_add_auto_unit(final_norm):
                tail_embed_unit.modules = [final_norm] + tail_embed_unit.modules
            tail_embed_unit.reshard_after_forward = False
        else:
            tail_modules = [
                module
                for module in (final_norm, tail_embed)
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
        self, model: nn.Module, plan: FSDPParallelPlan
    ) -> nn.Module:
        from contextlib import nullcontext
        from torch.utils.checkpoint import checkpoint

        target_modules: list[tuple[str, nn.Module]] = []
        for fqn, submodule in model.named_modules():
            spec = plan.match(strip_peft_prefix(fqn))
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
                self.prefetch_manager.suspend_expert_forward_prefetch(),
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
                        with suspend_prefetch():
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
        self, model: nn.Module, plan: FSDPParallelPlan
    ) -> tuple[set[nn.Parameter], list[nn.Module]]:
        """Identify ``no_shard`` params and module-level subtrees from *plan*.

        Two matching modes, both driven by ``FSDPPlanSpec.no_shard``:

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

        for fqn, submodule in model.named_modules():
            spec = plan.match(strip_peft_prefix(fqn))
            if not isinstance(spec, FSDPPlanSpec) or not spec.no_shard:
                continue
            no_shard_modules.append(submodule)
            for name, param in submodule.named_parameters(recurse=True):
                if param in no_shard_params:
                    continue
                no_shard_params.add(param)

        for name, param in model.named_parameters():
            if param in no_shard_params:
                continue
            spec = plan.match(strip_peft_prefix(name))
            if isinstance(spec, FSDPPlanSpec) and spec.no_shard:
                no_shard_params.add(param)

        return no_shard_params, no_shard_modules

    def _shard_model(self, model: nn.Module, plan: FSDPParallelPlan) -> nn.Module:
        named_modules = list(model.named_modules())

        # no_shard identification is shared with _register_no_shard_params_grad_sync
        # via _collect_no_shard so both stay in sync. Module-level matches replicate
        # a whole subtree: mark _fsdp_no_shard so the wrap passes below skip it.
        # Param-level matches (e.g. deepseek_v4 hc_*/attn_sink) keep their owning
        # module sharded and only route the matched params to ignored_params.
        no_shard_params, no_shard_modules = self._collect_no_shard(model, plan)
        for submodule in no_shard_modules:
            for child in submodule.modules():
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

        # Explicit sharding specs first, bottom-up.
        for fqn, submodule in reversed(named_modules):
            if fqn == "" or getattr(submodule, "_fsdp_no_shard", False):
                continue

            clean_fqn = strip_peft_prefix(fqn)
            spec = plan.match(clean_fqn)
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

        Returns ``{"sp_size": ..., "sp_group": ...}``.
        """
        if not dist.is_initialized():
            raise ValueError("Distributed not initialized")

        if self._sp_size and self._sp_group:
            return {
                "sp_size": self._sp_size,
                "sp_group": self._sp_group,
                "sp_rank": self._sp_rank,
            }
        else:
            return {"sp_size": 1, "sp_group": None, "sp_rank": 0}

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

    def prepare_optimizer(
        self, model: nn.Module, optimizer_cls: type, **optimizer_kwargs
    ) -> Any:
        """For FSDP2, create the optimizer."""
        params = []
        moe_params = []
        ignored_params = []
        edp = self._mesh["edp"]
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if hasattr(param, "device_mesh"):
                if param.device_mesh == edp:
                    moe_params.append(param)
                else:
                    params.append(param)
            else:
                ignored_params.append(param)

        optimizer_params = [
            {"params": params},
            {"params": moe_params},
            {"params": ignored_params},
        ]

        return optimizer_cls(optimizer_params, **optimizer_kwargs)

    @torch.no_grad()
    def _clip_grad_norm_cpu_offload(
        self, moe_params, params, ignored_params, max_norm, norm_type
    ):
        """Grad-norm + clip for the FSDP2 CPU-offload path — fully chunked on the
        compute device (peak GPU mem = one chunk, ``CLIP_CHUNK_BYTES``).
        """
        all_params = list(moe_params) + list(params) + list(ignored_params)
        if not all_params:
            return torch.tensor(0.0)

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
                return m if bool(m > acc) else acc
            return acc + (t.detach().float().abs() ** norm_type).sum()

        acc = (
            torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
            if norm_type == float("inf")
            else torch.zeros((), device=reduce_device, dtype=torch.float32)
        )
        for chunk in self._iter_offloaded_chunks(offloaded):
            chunk_dev = [l.to(reduce_device, non_blocking=True) for l in chunk]
            self.device_runtime.synchronize()  # H2D complete before reading chunk_dev
            for t in chunk_dev:
                acc = _accumulate(t, acc)
        for g in plain_grads:
            acc = _accumulate(g, acc)
        if norm_type == float("inf"):
            dist.all_reduce(acc, op=dist.ReduceOp.MAX, group=world_group)
            total_norm = acc
        else:
            dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=world_group)
            total_norm = acc ** (1.0 / norm_type)

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

        # FSDP2 CPU offload: grads are sharded DTensors on CPU and the
        # DTensor-collective fast path has no cpu backend on the (cuda) mesh —
        # use the gloo-based local-reduce path instead.
        if self.fsdp_config.cpu_offload:
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

        if not edp_total_norm_list and not grads and not ignored_grads:
            return torch.tensor(0.0)

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
        ref_device = all_norms[0].device
        total_norm = torch.linalg.vector_norm(
            torch.cat([norm.to(ref_device) for norm in all_norms]), norm_type
        )

        if moe_params:
            clip_grads_with_norm_(moe_params, max_norm, total_norm)
        if params:
            clip_grads_with_norm_(params, max_norm, total_norm)
        if ignored_params:
            clip_grads_with_norm_(ignored_params, max_norm, total_norm)

        return total_norm

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if inputs.get("position_ids", None) is None:
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

        input_mode = model.get_sequence_parallel_input_mode()
        if input_mode == SequenceParallelInputMode.POST_EMBEDDING:
            # vlm model's inputs will be sliced after embedding
            inputs["input_ids"], inputs["position_ids"], _, pad_size = ulysses_pad(
                inputs["input_ids"],
                position_ids_rmpad=inputs["position_ids"],
                sp_size=sp_size,
            )
        else:
            inputs["input_ids"], inputs["position_ids"], _, pad_size = (
                ulysses_pad_and_slice_inputs(
                    inputs["input_ids"],
                    position_ids_rmpad=inputs["position_ids"],
                    sp_size=sp_size,
                    sp_group=sp_group,
                    sp_rank=sp_rank,
                )
            )
        assert pad_size == 0, "must be"

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Main path loss
        loss = self._compute_ce_loss_sp(
            logits=logits,
            labels=labels,
            shift_extra=0,
            sp_size=sp_size,
            sp_group=sp_group,
            sp_rank=sp_rank,
            num_items_in_batch=num_items_in_batch,
        )

        # MTP path loss
        # use the same SP-aware CE helper as the main path.
        mtp_logits_list = getattr(outputs, "mtp_logits", None)
        mtp_losses: list[torch.Tensor] = []
        if self.config.mtp.enabled and mtp_logits_list:
            for i, mtp_logits in enumerate(mtp_logits_list):
                # i-th MTP head predicts token_{t + i + 2}, so shift labels
                # (i + 1) extra positions vs the main path.
                mtp_loss_i = self._compute_ce_loss_sp(
                    logits=mtp_logits,
                    labels=labels,
                    shift_extra=i + 1,
                    sp_size=sp_size,
                    sp_group=sp_group,
                    sp_rank=sp_rank,
                    num_items_in_batch=num_items_in_batch,
                )
                mtp_losses.append(mtp_loss_i)
                if int(os.environ.get('TORCH_TURBO_DEBUG_MTP', '0')):
                    rank = torch.distributed.get_rank()
                    with torch.no_grad():
                        logger.debug(
                            "[TORCH_TURBO_DEBUG_MTP_LOSS] rank=%d "
                            "sp_rank=%d head=%d raw_mtp_loss=%.6f",
                            rank,
                            sp_rank,
                            i,
                            mtp_loss_i.item(),
                        )

        scaling = float(self.config.mtp.loss_scaling_factor)
        scaled_mtp_total = None

        if mtp_losses and scaling > 0.0:
            scaled_mtp_total = sum(mtp_losses) * scaling
            total_loss = loss + scaled_mtp_total
        else:
            total_loss = loss

        # Stash diagnostics on outputs for LogMTPPatch to read.
        outputs.loss = total_loss
        outputs.main_loss = loss.detach()
        if mtp_losses:
            outputs.mtp_loss = [m.detach() for m in mtp_losses]
            outputs.mtp_loss_scaled = (
                scaled_mtp_total.detach()
                if scaled_mtp_total is not None
                else torch.zeros((), device=loss.device, dtype=loss.dtype)
            )
        return (total_loss, outputs) if return_outputs else total_loss


    def _compute_ce_loss_sp(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        shift_extra: int,
        sp_size: int,
        sp_group: Any,
        sp_rank: int,
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        """Shared SP-aware shifted-CE loss helper.

        Parameters
        ----------
        logits:
            ``(B, T/sp_size, V)`` — already SP-local.
        labels:
            ``(B, T)`` — full sequence labels (caller passes the same tensor
            for main and MTP paths).
        shift_extra:
            Extra positions to shift left.  Main path uses ``0`` (just the
            standard "predict next token" shift).  i-th MTP head uses
            ``i + 1`` (predict token at t+i+2 instead of t+1).
        """
        ignore_index = -100

        # Shift labels by (1 + shift_extra) positions to the left.
        labels = nn.functional.pad(labels, (0, 1 + shift_extra), value=ignore_index)
        labels = labels[..., 1 + shift_extra :].contiguous()
        label_num = (labels != ignore_index).sum()
        if sp_size > 1:
            labels = slice_input_tensor(
                labels,
                dim=1,
                group=sp_group,
                rank=sp_rank,
                padding=False,
                sp_mode="ulysses",
            )

        labels = labels.view(-1).to(logits.device)
        logits = logits.view(-1, logits.size(-1))
        loss = ops.cross_entropy_loss(
            logits,
            labels,
            reduction="sum",
            ignore_index=ignore_index,
        )

        if sp_size > 1 and sp_group is not None:
            # Keep SP loss reduction in the autograd graph.  A plain
            # dist.all_reduce updates the scalar value but drops remote loss
            # contributions during backward.
            loss = torch.distributed.nn.all_reduce(
                loss, op=dist.ReduceOp.SUM, group=sp_group
            )

        # Final loss — two paths matching HF Trainer convention:
        if num_items_in_batch:
            # New-style: use num_items_in_batch as denominator.
            # This is the total valid token count across all GA micro-batches,
            # passed down by HF Trainer for correct gradient accumulation.

            # Repeated calculation in SP group
            num_items_in_batch = num_items_in_batch // sp_size
            loss /= num_items_in_batch

            loss /= sp_size
            # Scale by world_size to match HF Trainer's
            # average_tokens_across_devices convention.
            world_size = dist.get_world_size()
            loss = loss * world_size
        else:
            # Old-style: local mean, no num_processes scaling.
            # HF Trainer will divide by gradient_accumulation_steps itself.
            loss /= label_num
            loss /= sp_size

        return loss

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
        inputs = dict(inputs)
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
            # SP: slice the 2T sequence to the local shard; block mask stays full.
            n_valid = int((labels != ignore_index).sum().item())
            full_local = slice_input_tensor(inputs["input_ids"], dim=1, group=sp_group, rank=sp_rank)
            pos_local = slice_input_tensor(inputs["position_ids"], dim=1, group=sp_group, rank=sp_rank)
            labels_local = slice_input_tensor(labels, dim=1, group=sp_group, rank=sp_rank)

            outputs = model(
                input_ids=full_local,
                attention_mask=block_mask,
                position_ids=pos_local,
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
