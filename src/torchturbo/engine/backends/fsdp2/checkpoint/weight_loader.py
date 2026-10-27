import gc
import json
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed._tensor import DTensor, distribute_tensor

from torchturbo.engine.checkpoint.distributed import collective_check
from torchturbo.engine.peft.weights import (
    compute_dora_magnitude,
    get_peft_init_config,
    init_lora_tensor,
)
from torchturbo.engine.utils import (
    is_dora_magnitude_param,
    is_peft_adapter_param,
    strip_peft_prefix,
)
from torchturbo.logging import get_logger
from torchturbo.models.modules.cpu_resident import (
    CPUResidentParameter,
)

logger = get_logger(__name__)


class _PendingTask:
    """Holds a broadcast before and after it is submitted to the backend.

    Before issue, ``handle`` is ``None`` and ``source``/``group`` describe the
    collective to create. After issue, ``handle`` is waited on and the full
    tensor is distributed into the model state dict.
    """
    __slots__ = (
        'handle', 'full_tensor', 'param', 'mesh', 'key', 'target_dtype',
        'group', 'source',
    )

    def __init__(
        self,
        handle,
        full_tensor,
        param,
        mesh,
        key,
        target_dtype,
        group,
        source,
    ):
        self.handle = handle
        self.full_tensor = full_tensor
        self.param = param
        self.mesh = mesh
        self.key = key
        self.target_dtype = target_dtype
        self.group = group
        self.source = source


class DistParamLoader:
    # Fraction of total GPU memory to reserve as headroom during broadcasts.
    # Controlled by TORCHTURBO_BROADCAST_MEM_RESERVE env var (default 0.1).
    _MEM_RESERVE_FRACTION = float(os.environ.get("TORCHTURBO_BROADCAST_MEM_RESERVE", "0.1"))
    # Max bytes for in-flight broadcast tensors (default 5 GB).
    # Env var TORCHTURBO_BROADCAST_MEM_BUDGET_CAP is in GB.
    _BROADCAST_MEM_BUDGET_CAP = int(float(os.environ.get("TORCHTURBO_BROADCAST_MEM_BUDGET_CAP", "5")) * 1024**3)

    def __init__(
        self,
        model,
        path,
        rank,
        mesh,
        world_mesh,
        device_runtime,
        is_cpu_offload,
        weight_converter,
    ):
        if weight_converter is None:
            raise ValueError("weight_converter must not be None")
        self.model = model
        self.hf_config = model.config
        self.is_cpu_offload = is_cpu_offload
        self.device_runtime = device_runtime
        self.device = device_runtime.device
        self.path = Path(path)
        self.mesh = mesh
        self.world_mesh = world_mesh
        self.rank = rank
        self.world_size = dist.get_world_size()

        # EP topology
        self.ep_group = mesh["ep"].get_group()
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.ep_world_size = dist.get_world_size(group=self.ep_group)

        # EDP topology (mirrors weight_saver)
        self.edp_group = mesh["edp"].get_group()
        self.edp_rank = mesh["edp"].get_local_rank()
        current_ep_idx = mesh.get_local_rank("ep")
        self.edp_ranks = mesh.mesh[:, current_ep_idx].tolist()

        self.param_converter = weight_converter
        self.moe_config = self.param_converter.moe_config

        # Assignment data structures (mirrors weight_saver)
        self.param_size_map = {}
        self.param_to_rank = {}
        self.rank_shard_bytes = {}
        self.rank_to_params = [[] for _ in range(self.world_size)]
        self.sharded_param_groups = []
        self.WEIGHT_SHARD_SIZE = 5 * (1024 ** 3)

        # log related
        self.start_time = time.perf_counter()
        self.prev_time = self.start_time

        # build meta (replaces _build_param_to_rank)
        self.param_to_filename = collective_check(self._load_index_json)
        self.local_params = self._build_meta_info()

    # ── Index & Meta Info ──────────────────────────────────────────

    def _load_index_json(self):
        index_path = self.path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    def _sync_log_rank0(self, content, update_prev=True):
        self.device_runtime.synchronize()
        dist.barrier()
        cur = time.perf_counter()
        if self.rank == 0:
            logger.info(f"{content} in {cur - self.prev_time:.2f}s (total {cur - self.start_time:.2f}s)")
        if update_prev:
            self.prev_time = cur

    # ── Assignment Helpers (mirrors weight_saver) ──────────────────

    def _update_shard_bytes(self, rank, num_bytes):
        if rank in self.rank_shard_bytes:
            self.rank_shard_bytes[rank] += num_bytes
        else:
            self.rank_shard_bytes[rank] = num_bytes

    def _select_low_pressure_rank(self):
        lowest = float('inf')
        result = 0
        for r in range(self.world_size):
            if r not in self.rank_shard_bytes:
                return r
            if lowest > self.rank_shard_bytes[r]:
                result = r
                lowest = self.rank_shard_bytes[r]
        return result

    # ── Build Meta Info ────────────────────────────────────────────

    def _build_meta_info(self):
        """Classify model params, compute sizes, and assign to ranks.

        This mirrors ShardedParamManager._build_meta_info in weight_saver.py.
        Each rank determines which HF parameter names it needs to read from disk.
        """
        start_time = time.perf_counter()
        logger.info_rank0("[load] building meta info...")

        common_params = []
        expert_param_groups = []  # list of (model_param_name, [local_hf_names], total_local_bytes)

        for name, param in self.model.named_parameters():
            if isinstance(param, CPUResidentParameter):
                continue
            clean_name = strip_peft_prefix(name)
            if is_peft_adapter_param(clean_name):
                continue

            common_keys, expert_group, size_map = (
                self.param_converter.get_param_load_entries(
                    clean_name, param.numel(),
                    param_to_filename=self.param_to_filename,
                )
            )
            # Register virtual key → filename mapping for fused expert format.
            # Must happen before the filtering step so that virtual keys in
            # size_map pass the param_to_filename check.
            virtual_mapping = self.param_converter.get_virtual_key_filenames(
                clean_name, self.param_to_filename
            )
            self.param_to_filename.update(virtual_mapping)

            # Loader filters by param_to_filename (checkpoint must contain the key)
            filtered_keys = [k for k in common_keys if k in self.param_to_filename]
            filtered_size_map = {k: v for k, v in size_map.items() if k in self.param_to_filename}
            self.param_size_map.update(filtered_size_map)
            common_params.extend(filtered_keys)
            if expert_group is not None:
                expert_param_groups.append(expert_group)

        for name, buf in self.model.named_buffers():
            clean_name = strip_peft_prefix(name)
            if self.param_converter.is_persist_buffer(clean_name):
                hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
                if hf_buffer_name in self.param_to_filename:
                    common_params.append(hf_buffer_name)
                    self.param_size_map[hf_buffer_name] = buf.numel() * buf.element_size()

        self._assign_param_to_ranks(expert_param_groups, self.edp_ranks, common_params)

        elapsed = time.perf_counter() - start_time
        num_files = len(set(
            self.param_to_filename[p]
            for p in self.rank_to_params[self.rank]
            if p in self.param_to_filename
        ))
        logger.info_rank0(
            "[load] meta info built in %.2fs; rank 0 will read %d files for %d params",
            elapsed, num_files, len(self.rank_to_params[self.rank]),
        )

        return set(self.rank_to_params[self.rank])

    def _assign_param_to_ranks(self, expert_param_groups, edp_ranks, common_params):
        """Two-phase param-to-rank assignment.

        Phase 1: Expert params → EDP ranks, shard-size balanced round-robin.
        Phase 2: Common params → all world ranks, file-grouped least-loaded.
        """
        # ── Phase 1: Expert params → EDP ranks ──
        num_edp_ranks = len(edp_ranks)
        accumulated_bytes = 0
        shard_params = []
        i = 0

        for _model_name, hf_names, group_bytes in expert_param_groups:
            # Flush current shard before adding this group if it would cross the boundary.
            # This prevents splitting a model param's HF names across ranks.
            if accumulated_bytes > 0 and accumulated_bytes + group_bytes >= self.WEIGHT_SHARD_SIZE:
                dst_rank = edp_ranks[i % num_edp_ranks]
                # Commit this shard
                shard_bytes = 0
                for p in shard_params:
                    self.param_to_rank[p] = dst_rank
                    shard_bytes += self.param_size_map.get(p, 0)
                self._update_shard_bytes(dst_rank, shard_bytes)
                if dst_rank == self.rank:
                    self.sharded_param_groups.append(shard_params)
                shard_params = []
                accumulated_bytes = 0
                i += 1

            shard_params.extend(hf_names)
            accumulated_bytes += group_bytes

        # Flush remaining
        if shard_params:
            dst_rank = edp_ranks[i % num_edp_ranks]
            shard_bytes = 0
            for p in shard_params:
                self.param_to_rank[p] = dst_rank
                shard_bytes += self.param_size_map.get(p, 0)
            self._update_shard_bytes(dst_rank, shard_bytes)
            if dst_rank == self.rank:
                self.sharded_param_groups.append(shard_params)

        # Merge param_to_rank across EP group (each EP rank assigns different HF names)
        param_to_rank_list = [None] * self.ep_world_size
        dist.all_gather_object(param_to_rank_list, self.param_to_rank, group=self.ep_group)
        for rank_map in param_to_rank_list:
            self.param_to_rank.update(rank_map)

        # Recompute rank_shard_bytes from full expert assignments across all EP ranks
        self.rank_shard_bytes = {}
        for param, r in self.param_to_rank.items():
            self._update_shard_bytes(r, self.param_size_map.get(param, 0))

        # ── Phase 2: Common params → all world ranks, file-grouped ──
        file_to_common_params = {}
        for param in common_params:
            filename = self.param_to_filename.get(param)
            if filename is not None:
                file_to_common_params.setdefault(filename, []).append(param)

        # Sort file groups by descending total bytes for better bin-packing
        sorted_file_groups = sorted(
            file_to_common_params.items(),
            key=lambda kv: sum(self.param_size_map.get(p, 0) for p in kv[1]),
            reverse=True,
        )

        for _filename, params in sorted_file_groups:
            dst_rank = self._select_low_pressure_rank()
            shard_bytes = sum(self.param_size_map.get(p, 0) for p in params)
            for p in params:
                self.param_to_rank[p] = dst_rank
            self._update_shard_bytes(dst_rank, shard_bytes)
            if dst_rank == self.rank:
                self.sharded_param_groups.append(params)

        # Build rank_to_params index
        for param, r in self.param_to_rank.items():
            self.rank_to_params[r].append(param)

    # ── Load Local Shards ──────────────────────────────────────────

    def _load_local_shards(self):
        # Group local params by source file for efficient loading
        files_needed = {}
        for param_name in self.local_params:
            filename = self.param_to_filename.get(param_name)
            if filename is not None:
                files_needed.setdefault(filename, []).append(param_name)

        local_sd = {}
        if files_needed:
            files = [self.path / f for f in sorted(files_needed.keys())]
            logger.debug_rank0(
                "[load][rank %d] Loading %d files with %d parameters...",
                self.rank, len(files), len(self.local_params),
            )
            t0 = time.perf_counter()
            for f in files:
                file_params = load_file(f.as_posix(), device="cpu")
                needed_in_file = set(files_needed[f.name])

                # Separate virtual keys from physical keys.
                # Virtual keys (fused expert per-EP slices) need slicing
                # from the physical tensor rather than direct loading.
                virtual_needed = {p for p in needed_in_file
                                 if self.param_converter.is_virtual_key(p)}
                physical_needed = needed_in_file - virtual_needed

                # Physical keys: load directly (unchanged)
                for param_name, tensor in file_params.items():
                    if param_name in physical_needed:
                        local_sd[param_name] = tensor.clone()

                # Virtual keys: look up physical tensor, slice by ep_rank
                # encoded in the virtual key name, store under virtual key.
                # Slicing happens on CPU (file_params is CPU tensors),
                # so no GPU memory pressure.
                for virtual_key in virtual_needed:
                    physical_key = self.param_converter.get_physical_key(virtual_key)
                    if physical_key in file_params:
                        local_sd[virtual_key] = self.param_converter.slice_virtual_key(
                            virtual_key, file_params[physical_key]
                        )

            logger.debug_rank0(
                "[load][rank %d] Loaded %d params in %.2fs",
                self.rank, len(local_sd), time.perf_counter() - t0,
            )
        else:
            logger.debug_rank0("[load][rank %d] No parameters to load", self.rank)

        return local_sd

    # ── Async Broadcast Helpers ────────────────────────────────────

    def _prepare_tensor(self, name, shape=None, broadcast_dtype=None, group=None):
        """Prepare a broadcast tensor and return (tensor, mesh, group, source).

        Broadcasts in broadcast_dtype (defaults to checkpoint dtype). Owner
        keeps the loaded dtype without conversion; non-owners allocate in
        broadcast_dtype. group=None means the world group. Dtype conversion
        to param.dtype is deferred to the pending-task flush.
        """
        owner = self.param_to_rank[name]
        if self.rank == owner:
            assert self.local_sd[name].dtype == broadcast_dtype, "must be"
            tensor = self.local_sd[name].to(self.device)
        else:
            tensor = torch.empty(shape, device=self.device, dtype=broadcast_dtype)
        return tensor, self.world_mesh, group, owner

    def _prepare_expert_weight(self, name, param):
        """Prepare expert weight1 or weight2 for an EDP-group broadcast.

        The converter handles format-specific assembly (per-expert vs fused)
        and weight1 vs weight2 dispatch. Owner rank assembles the tensor;
        non-owners allocate empty. Dtype conversion is deferred to the
        pending-task flush.
        """
        hf_key = self.param_converter.get_expert_broadcast_key(name)
        owner = self.param_to_rank[hf_key]
        hf_dtype = self.param_converter.get_default_hf_dtype(hf_key)

        # Pre-allocate in checkpoint dtype
        tensor = torch.empty(param.shape, device=self.device, dtype=hf_dtype)
        if self.rank == owner:
            self.param_converter.assemble_expert_weight_for_broadcast(
                tensor, self.local_sd, name
            )

        return tensor, self.mesh["edp"], self.edp_group, owner

    def _prepare_lora(self, clean_name, param):
        """Prepare a LoRA adapter tensor for broadcast.

        Returns (tensor, mesh, group, source).
        """
        is_a = (".lora_A." in clean_name
                or ".lora_A2." in clean_name
                or ".lora_embedding_A." in clean_name)
        is_expert = (
            ".experts.lora_A2." in clean_name
            or ".experts.lora_B2." in clean_name
            or ".experts.lora_A." in clean_name
            or ".experts.lora_B." in clean_name
        )
        init_lora_weights, r = get_peft_init_config(self.model)

        if is_expert:
            edp_mesh = self.mesh["edp"]
            edp_group = edp_mesh.get_group()
            edp_rank = edp_mesh.get_local_rank()
            edp_src_global = dist.get_global_rank(edp_group, 0)

            if edp_rank == 0:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)
                init_lora_tensor(full, is_a, init_lora_weights, r)
            else:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)

            return full, edp_mesh, edp_group, edp_src_global
        else:
            if self.rank == 0:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)
                init_lora_tensor(full, is_a, init_lora_weights, r)
            else:
                full = torch.empty(param.shape, device=self.device, dtype=param.dtype)

            return full, self.world_mesh, None, 0

    # ── Memory Budget Sync ─────────────────────────────────────────

    def _sync_memory_budget(self):
        """Compute a synchronized memory budget across all ranks.

        Each rank computes its local budget (free memory - reserve),
        then all-reduce (MIN) ensures every rank uses the same conservative
        value so broadcast collectives never hang due to mismatched flush timing.
        """
        mem_free, mem_total = self.device_runtime.memory_info()
        reserve_bytes = int(mem_total * self._MEM_RESERVE_FRACTION)
        local_budget = max(mem_free - reserve_bytes, 0)

        budget_tensor = torch.tensor([local_budget], dtype=torch.float, device=self.device)
        dist.all_reduce(budget_tensor, op=dist.ReduceOp.MIN)
        budget_bytes = min(int(budget_tensor.item()), self._BROADCAST_MEM_BUDGET_CAP)

        logger.info_rank0(
            "[load] broadcast memory budget: %.2f GB (local free: %.2f GB, reserve: %.2f GB)",
            budget_bytes / 1024**3, mem_free / 1024**3, reserve_bytes / 1024**3,
        )
        return budget_bytes

    # ── Flush Pending Tasks ────────────────────────────────────────

    def _flush_pending_tasks(self, tasks, sharded_sd):
        """Issue async broadcasts in a stable order, then store their tensors.

        For DTensor (FSDP-managed) params: ``distribute_tensor`` the broadcast
        full tensor, ``to_local()`` the local shard, then rebuild a standalone
        DTensor with ``from_local()``. ``to_local()`` is a view into the
        just-broadcast full buffer, so the shard is materialised as an
        independent tensor (``clone`` on device, ``.to("cpu")`` under offload)
        before ``from_local``. That releases ``task.full_tensor`` every
        iteration instead of pinning each full buffer in ``sharded_sd``, so
        memory stays flat as the state dict accumulates.

        Plain (no_shard / replicated) params are not FSDP-managed: the broadcast
        tensor is stored directly and stays on the compute device even under
        offload.
        """
        # torch_npu/HCCL can stop advancing a large varied-size async-broadcast
        # window when its source changes repeatedly. Keep EDP work first and
        # sort world work by source while retaining the original preparation
        # order (and therefore LoRA initialization order) within each source.
        tasks.sort(
            key=lambda task: (
                0 if task.group is not None else 1,
                task.source,
            )
        )

        for task in tasks:
            task.handle = dist.broadcast(
                task.full_tensor,
                src=task.source,
                group=task.group,
                async_op=True,
            )

        for task in tasks:
            task.handle.wait()

            # Convert broadcast dtype → target dtype (e.g., bf16 → fp32)
            if task.target_dtype is not None and task.full_tensor.dtype != task.target_dtype:
                task.full_tensor = task.full_tensor.to(task.target_dtype)

            if isinstance(task.param, DTensor):
                sharded = distribute_tensor(
                    task.full_tensor, task.mesh, task.param.placements, src_data_rank=None
                )
                # Materialise a standalone local shard so the full broadcast
                # buffer can be freed this iteration (the view into it is gone).
                local = sharded.to_local()
                if not self.is_cpu_offload:
                    local = local.clone()
                del sharded, task.full_tensor
                # run_check=False skips the all-gather verification — the shard
                # came from distribute_tensor, so placements are already correct.
                # With run_check=False, from_local cannot infer the global shape
                # from a sharded local shard, so pass the param's global
                # shape/stride explicitly.
                sharded_sd[task.key] = DTensor.from_local(
                    local,
                    task.mesh,
                    task.param.placements,
                    run_check=False,
                    shape=task.param.shape,
                    stride=task.param.stride(),
                )
                if self.is_cpu_offload:
                    sharded_sd[task.key] = sharded_sd[task.key].to("cpu")
            else:
                # Only FSDP-managed (DTensor) params are CPU-offloaded. Ignored
                # no_shard params are plain tensors, not managed by CPUOffloadPolicy,
                # so they must stay on the compute device.
                sharded_sd[task.key] = task.full_tensor
                del task.full_tensor

        del tasks

        # Memory cleanup after each batch
        mem_free, mem_total = self.device_runtime.memory_info()
        mem_used = mem_total - mem_free
        utilization = mem_used / mem_total
        if utilization > (1 - self._MEM_RESERVE_FRACTION):
            before_free = mem_free
            self.device_runtime.empty_cache()
            gc.collect()
            new_free, _ = self.device_runtime.memory_info()
            freed_gb = (new_free - before_free) / 1024 ** 3
            logger.debug_rank0(
                "[load][rank %d] Cleaned %.2f GB GPU memory (utilization: %.1f%% -> %.1f%%)",
                self.rank, freed_gb, utilization * 100,
                (mem_total - new_free) / mem_total * 100,
            )

    # ── Buffer Handling ────────────────────────────────────────────

    def _broadcast_buffers(self, sharded_sd):
        """Broadcast persistent buffers synchronously into the state dict."""
        for name, buf in self.model.named_buffers():
            clean_name = strip_peft_prefix(name)
            if self.param_converter.is_persist_buffer(clean_name):
                hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
                if hf_buffer_name not in self.param_to_filename:
                    logger.debug_rank0(
                        "[load] SKIP buffer (not in checkpoint): %s (original: %s)",
                        clean_name, name,
                    )
                    continue
                if hf_buffer_name not in self.param_to_rank:
                    logger.debug_rank0(
                        "[load] SKIP buffer (not assigned): %s (original: %s)",
                        clean_name, name,
                    )
                    continue
                owner = self.param_to_rank[hf_buffer_name]
                hf_dtype = self.param_converter.get_default_hf_dtype(hf_buffer_name)
                if self.rank == owner:
                    full = self.local_sd[hf_buffer_name].to(self.device, dtype=hf_dtype)
                else:
                    full = torch.empty(buf.shape, device=self.device, dtype=hf_dtype)
                dist.broadcast(full, src=owner, async_op=False)
                sharded_sd[name] = full

    # ── Broadcast All Params ───────────────────────────────────────

    def _broadcast_params(self, sharded_sd):
        pending = []
        pending_bytes = 0
        num_params = len(list(self.model.parameters()))
        processed = 0

        # Dynamic memory budget: reserve a fraction of total GPU memory,
        # use the remaining free memory for in-flight broadcast tensors.
        # Must synchronize across ranks so all ranks flush at the same time,
        # otherwise broadcast collectives will hang.
        budget_bytes = self._sync_memory_budget()

        for name, param in self.model.named_parameters():
            if isinstance(param, CPUResidentParameter):
                continue
            clean_name = strip_peft_prefix(name)
            target_dtype = param.dtype

            if is_dora_magnitude_param(clean_name):
                processed += 1
                continue
            if is_peft_adapter_param(clean_name):
                full, mesh, group, source = self._prepare_lora(clean_name, param)
            elif self.param_converter.is_expert_weight(clean_name):
                full, mesh, group, source = self._prepare_expert_weight(clean_name, param)
            else:
                hf_name = self.param_converter.to_hf_name(clean_name)
                if hf_name not in self.param_to_filename:
                    logger.debug_rank0(
                        "[load] SKIP (not in checkpoint): %s (original: %s)",
                        clean_name, name,
                    )
                    continue
                if hf_name not in self.param_to_rank:
                    logger.debug_rank0(
                        "[load] SKIP (not assigned): %s (original: %s)",
                        clean_name, name,
                    )
                    continue
                # Broadcast in checkpoint dtype; dtype conversion deferred to flush
                hf_dtype = self.param_converter.get_default_hf_dtype(hf_name)
                full, mesh, group, source = self._prepare_tensor(
                    hf_name, param.shape, hf_dtype, group=None
                )

            tensor_bytes = full.numel() * full.element_size()

            # Flush existing pending if adding this tensor would exceed the memory budget
            if pending and pending_bytes + tensor_bytes > budget_bytes:
                self._flush_pending_tasks(pending, sharded_sd)
                pending = []
                pending_bytes = 0
                logger.info_rank0(
                    "[load] broadcast flushed %d/%d params", processed, num_params
                )

            pending.append(
                _PendingTask(
                    None, full, param, mesh, name, target_dtype, group, source
                )
            )
            pending_bytes += tensor_bytes
            processed += 1

        if pending:
            self._flush_pending_tasks(pending, sharded_sd)

        self._sync_log_rank0(f"[load] broadcast {processed}/{num_params} params done")

    def _initialize_dora_magnitudes(self, *, assign: bool) -> None:
        """Initialize DoRA magnitude parameters from the loaded base and LoRA state."""
        state_dict = {}
        marker = ".lora_magnitude_vector."
        for name, param in self.model.named_parameters():
            clean_name = strip_peft_prefix(name)
            if not is_dora_magnitude_param(clean_name):
                continue

            module_name, adapter_weight = name.split(marker, maxsplit=1)
            adapter_name, suffix = adapter_weight.rsplit(".", maxsplit=1)
            if suffix != "weight":
                raise ValueError(f"Unexpected DoRA magnitude parameter {name!r}.")
            lora_module = self.model.get_submodule(module_name)
            magnitude = compute_dora_magnitude(lora_module, adapter_name)
            magnitude = magnitude.to(self.device, dtype=param.dtype)

            if isinstance(param, DTensor):
                sharded = distribute_tensor(
                    magnitude,
                    param.device_mesh,
                    param.placements,
                    src_data_rank=None,
                )
                local = sharded.to_local().clone()
                magnitude = DTensor.from_local(
                    local,
                    param.device_mesh,
                    param.placements,
                    run_check=False,
                    shape=param.shape,
                    stride=param.stride(),
                )
                if self.is_cpu_offload:
                    magnitude = magnitude.to("cpu")
            state_dict[name] = magnitude

        if state_dict:
            self.model.load_state_dict(state_dict, assign=assign, strict=False)

    def _load_cpu_resident_weights(self) -> None:
        """Read checkpoint shards into each parameter's CPU storage."""
        def describe(name: str, parameter: CPUResidentParameter):
            converter = self.param_converter
            clean_name = strip_peft_prefix(name)
            source = converter.get_omitted_tied_source(
                clean_name, converter.to_hf_name(clean_name)
            )
            if source is not None:
                keys = [source]
            else:
                keys, _, _ = converter.get_param_load_entries(
                    clean_name, parameter.numel(), self.param_to_filename
                )
            if not keys:
                raise ValueError(f"CPU parameter {name} has no checkpoint source tensors")
            tensor_infos = converter.checkpoint_info.get_tensor_infos()
            dtype = tensor_infos[keys[0]][1]
            sizes = []
            for key in keys:
                shape, shard_dtype = tensor_infos[key]
                if len(shape) != parameter.ndim or shape[1:] != tuple(parameter.shape[1:]):
                    raise ValueError(f"Invalid CPU checkpoint shard shape for {key}: {shape}")
                if shard_dtype != dtype:
                    raise ValueError(f"Mixed CPU checkpoint shard dtypes for {key}")
                sizes.append(shape[0] if shape else 1)
            parameter.validate_shard_sizes(sizes)
            files = [self.path / self.param_to_filename[key] for key in keys]
            return keys, sizes, dtype, files

        for name, parameter in self.model.named_parameters():
            if not isinstance(parameter, CPUResidentParameter):
                continue
            keys, sizes, dtype, files = collective_check(partial(describe, name, parameter))

            def read(index: int, target: torch.Tensor) -> None:
                with safe_open(files[index], framework="pt", device="cpu") as handle:
                    target.copy_(handle.get_tensor(keys[index]))

            parameter.load_(sizes, read, dtype=dtype)

    # ── Cleanup ────────────────────────────────────────────────────

    def _cleanup(self):
        self.local_sd = None
        gc.collect()
        self.device_runtime.synchronize()
        dist.barrier()
        self.device_runtime.empty_cache()

    # ── Public API ─────────────────────────────────────────────────

    def load_weights(self, *, assign: bool = False) -> None:
        """Load weights, assigning storage only during initial meta loading."""
        self._load_cpu_resident_weights()
        # The timing barrier follows after all ranks report local read errors.
        self.local_sd = collective_check(self._load_local_shards)
        self._sync_log_rank0("[load] local shards loaded")

        sharded_sd = {}

        # buffers (small, kept synchronous)
        self._broadcast_buffers(sharded_sd)
        self._sync_log_rank0("[load] broadcast buffers")

        # parameters (pipelined async broadcasts)
        self._broadcast_params(sharded_sd)
        self.model.load_state_dict(sharded_sd, assign=assign, strict=False)
        if assign:
            self.model.tie_weights()
        self._initialize_dora_magnitudes(assign=assign)
        del sharded_sd
        self._sync_log_rank0("[load] broadcast/sharding done")

        self._cleanup()
