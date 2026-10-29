import gc
import json
import math
import os
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor
from transformers.utils import CONFIG_NAME

from torchturbo.engine.peft.weights import apply_lora_delta, build_lora_delta_map
from torchturbo.engine.utils import is_peft_adapter_param, strip_peft_prefix
from torchturbo.logging import get_logger
from torchturbo.models.modules.cpu_resident import CPUResidentParameter

from ....checkpoint.distributed import collective_check, rank_zero_call
from ....checkpoint.layout import HF_INDEX_FILE, save_json, sync_directory

logger = get_logger(__name__)

# Serialized locations of the MTP layer count. Composite configs nest it under
# their language config (llm_config is an alias of text_config), and Qwen3.5
# configs name it mtp_num_hidden_layers.
_MTP_LAYER_COUNT_PATHS = (
    ("num_nextn_predict_layers",),
    ("mtp_num_hidden_layers",),
    ("text_config", "num_nextn_predict_layers"),
    ("text_config", "mtp_num_hidden_layers"),
    ("llm_config", "num_nextn_predict_layers"),
)


def _sync_mtp_layer_count(config_path: Path, runtime_config: dict[str, Any]) -> None:
    """Record the trained MTP layer count in a copied source ``config.json``.

    Training with MTP disabled strips the MTP layers, so the export holds no
    MTP weights and its config must not declare any. Every other field keeps
    its source value, and the file stays byte-identical when the MTP layer
    counts it declares match the runtime config.

    Args:
        config_path: ``config.json`` copied from the source model.
        runtime_config: ``to_dict()`` of the model config used for training.
            It serializes the MTP layer count at the same locations as the
            source file.
    """
    with open(config_path) as config_file:
        data = json.load(config_file)
    changed = False
    for *parents, key in _MTP_LAYER_COUNT_PATHS:
        source_section = data
        runtime_section = runtime_config
        for parent in parents:
            source_section = source_section.get(parent, {})
            runtime_section = runtime_section.get(parent, {})
        if key not in source_section:
            continue
        if source_section[key] != runtime_section[key]:
            source_section[key] = runtime_section[key]
            changed = True
    if changed:
        save_json(data, str(config_path))


class ShardedParamManager:
    def __init__(
        self,
        model,
        rank,
        mesh,
        weight_converter,
        device_runtime,
        path=None,
        processing_class=None,
        is_cpu_offload: bool = False,
    ):
        if weight_converter is None:
            raise ValueError("weight_converter must not be None")
        self.move_to_cpu = True
        self.model = model
        self.device_runtime = device_runtime
        self.is_cpu_offload = is_cpu_offload

        self.rank = rank
        self.mesh = mesh
        self.ep_mesh = mesh["ep"]
        self.ep_group = self.ep_mesh.get_group(0)
        self.ep_rank = dist.get_rank(group=self.ep_group)
        self.ep_size = dist.get_world_size(group=self.ep_group)
        config = model.config
        self.model_type = config.model_type if hasattr(config, "model_type") else None
        if hasattr(self.model.config, "vision_config"):
            self.config = self.model.config.text_config
        else:
            self.config = self.model.config

        self.world_size = dist.get_world_size()
        current_ep_idx = self.mesh.get_local_rank("ep")
        self.edp_ranks = self.mesh.mesh[:, current_ep_idx].tolist()

        self.processing_class = processing_class
        self.src_path = Path(path) if path else None
        self.model_type = model.config.model_type
        self.param_converter = weight_converter
        self.param_size_map = {}
        self.param_to_rank = {}
        self.rank_shard_bytes = {}
        self.rank_to_params = [[] for _ in range(self.world_size)]
        self.sharded_param_groups = []
        self.WEIGHT_SHARD_SIZE = 5 * (1024 ** 3)
        # Load checkpoint index for format detection (fused vs per-expert)
        # and metadata preservation. Cached so the file is only read once.
        self._src_weight_map, self._src_metadata = (
            collective_check(lambda: self._load_index_json(self.src_path))
            if self.src_path is not None else ({}, {})
        )
        self.param_to_filename = self._src_weight_map
        self._cpu_keys = {}
        self.local_params = self._build_meta_info()
        self._lora_delta_map = build_lora_delta_map(model)
        if self._lora_delta_map:
            logger.info_rank0(
                "[LoRA] Detected %d base parameters with LoRA deltas to merge during save",
                len(self._lora_delta_map),
            )

    def _update_per_shard(self, shard_params, dst_rank):
        shard_bytes = 0
        for p in shard_params:
            self.param_to_rank[p] = dst_rank
            shard_bytes += self.param_size_map[p]
        self._update_shard_sizes(dst_rank, shard_bytes)
        if dst_rank == self.rank:
            self.sharded_param_groups.append(shard_params)
        return [], 0

    def _update_shard_sizes(self, rank, num_bytes):
        if rank in self.rank_shard_bytes:
            self.rank_shard_bytes[rank] += num_bytes
        else:
            self.rank_shard_bytes[rank] = num_bytes

    def _select_low_pressure_rank(self):
        lowest = float('inf')
        result = 0
        for rank in range(self.world_size):
            if rank not in self.rank_shard_bytes:
                # means has not assign a param
                return rank
            if lowest > self.rank_shard_bytes[rank]:
                result = rank
                lowest = self.rank_shard_bytes[rank]
        return result

    def _assign_param_to_ranks(self, ep_params, edp_ranks, common_params):
        i = 0
        num_edp_ranks = len(edp_ranks)
        accumulated_bytes = 0
        shard_params = []
        dst_rank = edp_ranks[0]
        for param in ep_params:
            accumulated_bytes += self.param_size_map[param]
            shard_params.append(param)
            if accumulated_bytes >= self.WEIGHT_SHARD_SIZE:
                dst_rank = edp_ranks[i % num_edp_ranks]
                shard_params, accumulated_bytes = self._update_per_shard(shard_params, dst_rank)
                i += 1
        if shard_params:
            dst_rank = edp_ranks[i % num_edp_ranks]
            shard_params, accumulated_bytes = self._update_per_shard(shard_params, dst_rank)
        param_to_rank_list = [None] * self.ep_size
        dist.all_gather_object(param_to_rank_list, self.param_to_rank, group=self.ep_group)
        for rank_map in param_to_rank_list:
            self.param_to_rank.update(rank_map)

        # Recompute rank_shard_bytes from full expert assignments across all EP ranks
        self.rank_shard_bytes = {}
        for param, rank in self.param_to_rank.items():
            self._update_shard_sizes(rank, self.param_size_map[param])

        shard_params = []
        accumulated_bytes = 0
        for param in common_params:
            accumulated_bytes += self.param_size_map[param]
            shard_params.append(param)
            if accumulated_bytes >= self.WEIGHT_SHARD_SIZE:
                dst_rank = self._select_low_pressure_rank()
                shard_params, accumulated_bytes = self._update_per_shard(shard_params, dst_rank)
        if shard_params:
            dst_rank = self._select_low_pressure_rank()
            shard_params, accumulated_bytes = self._update_per_shard(shard_params, dst_rank)

        for param, rank in self.param_to_rank.items():
            self.rank_to_params[rank].append(param)

    def _build_meta_info(self):
        """Build state_dict containing only the parameters this rank needs to save."""
        start_time = time.perf_counter()
        logger.info_rank0("[save] building meta info...")

        common_params = []
        expert_params = []
        for name, param in self.model.named_parameters():
            # Strip PEFT prefixes so names match the HF checkpoint format.
            # Skip LoRA adapter params (lora_A, lora_B) — they are not part
            # of the base model checkpoint.
            clean_name = strip_peft_prefix(name)
            if is_peft_adapter_param(clean_name):
                continue

            if isinstance(param, CPUResidentParameter):
                collective_check(
                    partial(self._register_cpu_keys, name, clean_name, param)
                )
                continue

            common_keys, expert_keys, size_map = (
                self.param_converter.get_param_save_entries(
                    clean_name, param.numel(),
                    param_to_filename=self.param_to_filename,
                )
            )
            self.param_size_map.update(size_map)
            common_params.extend(common_keys)
            expert_params.extend(expert_keys)

        for name, buf in self._persistent_buffers_to_save().items():
            clean_name = strip_peft_prefix(name)
            hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
            common_params.append(hf_buffer_name)
            self.param_size_map[hf_buffer_name] = buf.numel() * buf.element_size()

        self._assign_param_to_ranks(expert_params, self.edp_ranks, common_params)
        parameters = dict(self.model.named_parameters())
        for name, keys in self._cpu_keys.items():
            param = parameters[name]
            # Each parameter assigns its source shards to saving ranks.
            for index, key in enumerate(keys):
                owner = param.shard_owner(index)
                self._update_per_shard([key], owner)
                self.rank_to_params[owner].append(key)

        elapsed = time.perf_counter() - start_time
        logger.info_rank0("[save] meta info built in %.2fs", elapsed)

        return set(self.rank_to_params[self.rank])

    def _register_cpu_keys(
        self, name: str, clean_name: str, param: CPUResidentParameter
    ) -> None:
        """Resolve the HF keys of a CPU parameter from its loaded source layout."""
        if param.shard_sizes is None:
            raise RuntimeError(
                f"CPU parameter {name} has no shard layout; call load_ before HF export"
            )
        source = self.param_converter.get_omitted_tied_source(
            clean_name, self.param_converter.to_hf_name(clean_name)
        )
        if source is not None:
            keys = [source]
        else:
            keys, _, _ = self.param_converter.get_param_save_entries(
                clean_name, param.numel(), self.param_to_filename
            )
        if len(keys) != len(param.shard_sizes):
            raise ValueError(
                f"CPU parameter {name} shard layout does not match its HF keys"
            )
        row_elements = math.prod(param.shape[1:])
        for key, rows in zip(keys, param.shard_sizes, strict=True):
            dtype = self.param_converter.get_default_hf_dtype(key)
            element_size = torch.empty((), dtype=dtype, device="meta").element_size()
            self.param_size_map[key] = rows * row_elements * element_size
        self._cpu_keys[name] = keys

    def _persistent_buffers_to_save(self) -> dict[str, torch.Tensor]:
        """Return buffers selected by the model's checkpoint converter."""
        return {
            name: buffer
            for name, buffer in self.model.named_buffers()
            if self.param_converter.is_persist_buffer(strip_peft_prefix(name))
        }

    def _load_index_json(self, path):
        """Load the model.safetensors.index.json file."""
        index_path = Path(path) / HF_INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)
        return index.get("weight_map", {}), index.get("metadata", {})

    def _copy_non_safetensor_files(self, src_path, dst_path):
        """Copy all non-safetensor files from source to destination."""
        dst_path = Path(dst_path)
        dst_path.mkdir(parents=True, exist_ok=True)

        for item in Path(src_path).iterdir():
            if item.is_file():
                # Skip safetensors files and index json
                if item.suffix == '.safetensors' or item.name == HF_INDEX_FILE:
                    continue
                shutil.copy2(item, dst_path / item.name)
            elif item.is_dir():
                # Copy directories recursively
                shutil.copytree(item, dst_path / item.name, dirs_exist_ok=True)

    def _save_expert_weight1_to_hf(self, name, full_tensor, local_params, full_sd):
        w1_sd = self.param_converter.get_local_ep_w1_hf_state_dict(name, full_tensor)
        for w1_name, w1_tensor in w1_sd.items():
            if w1_name in local_params:
                full_sd[w1_name] = self._normalize_tensor(w1_name, w1_tensor)

    def _save_expert_weight2_to_hf(self, name, full_tensor, local_params, full_sd):
        w2_sd = self.param_converter.get_local_ep_w2_hf_state_dict(name, full_tensor)
        for w2_name, w2_tensor in w2_sd.items():
            if w2_name in local_params:
                full_sd[w2_name] = self._normalize_tensor(w2_name, w2_tensor)

    def _save_router_param_to_hf(self, name, full_tensor, local_params, full_sd):
        hf_param_name = self.param_converter.convert_router_param_to_hf(name)
        if hf_param_name in local_params:
            full_sd[hf_param_name] = self._normalize_tensor(hf_param_name, full_tensor)

    def _gc_cleanup(self):
        gpu_before = self._get_gpu_memory_usage_percent()
        cpu_before = self._get_cpu_memory_usage_percent()
        self.device_runtime.empty_cache()
        gc.collect()
        gpu_after = self._get_gpu_memory_usage_percent()
        cpu_after = self._get_cpu_memory_usage_percent()
        logger.warning_rank0(
            "Memory alert - GPU: %.1f%% -> %.1f%%, CPU: %.1f%% -> %.1f%%",
            gpu_before, gpu_after, cpu_before, cpu_after,
        )

    def _gather_full_tensor(self, param):
        """Return ``param``'s full (replicated) tensor for saving, CPU-offload aware.

        Stage the local shard onto the compute device first so
        the all-gather runs on the device backend.
        """
        if not isinstance(param, DTensor):
            return param
        if not self.is_cpu_offload:
            return param.full_tensor()

        compute_device = self.device_runtime.device
        local = param.to_local()
        if local.device.type == compute_device.type:
            return param.full_tensor()

        staged = DTensor.from_local(
            local.to(compute_device),
            param.device_mesh,
            param.placements,
            run_check=False,
            shape=param.shape,
            stride=param.stride(),
        )
        return staged.full_tensor()

    def _collect_cpu_parameter(
        self, name: str, param: CPUResidentParameter, state: dict[str, torch.Tensor]
    ) -> None:
        keys = self._cpu_keys[name]

        def collect(index: int, tensor: torch.Tensor) -> None:
            key = keys[index]
            if key in self.local_params:
                state[key] = self._normalize_tensor(key, tensor)

        param.save(collect)

    def _get_local_hf_state_dict(self, local_params):
        """Build state_dict containing only the parameters this rank needs to save."""
        start_time = time.perf_counter()
        logger.info_rank0("[save] building local HF state dict...")

        full_sd = {}
        for name, param in self.model.named_parameters():
            # Strip PEFT prefixes and skip LoRA adapter params.
            clean_name = strip_peft_prefix(name)
            if is_peft_adapter_param(clean_name):
                continue

            # Per-param trace is DEBUG-only (it was a large share of the sample
            # log because it fired on every rank × every parameter). Only mention
            # the original name when it actually differs from the clean name.
            if clean_name != name:
                logger.debug_rank0("[saving] param: %s (original: %s)", clean_name, name)
            else:
                logger.debug_rank0("[saving] param: %s", clean_name)
            if isinstance(param, CPUResidentParameter):
                self._collect_cpu_parameter(name, param, full_sd)
                continue
            full_tensor = self._gather_full_tensor(param)

            # Merge LoRA deltas into the base weight without modifying the model.
            full_tensor = apply_lora_delta(
                full_tensor, clean_name, self._lora_delta_map
            )

            if self.param_converter.is_expert_weight1(clean_name):
                self._save_expert_weight1_to_hf(clean_name, full_tensor, local_params, full_sd)
            elif self.param_converter.is_expert_weight2(clean_name):
                self._save_expert_weight2_to_hf(clean_name, full_tensor, local_params, full_sd)
            elif self.param_converter.is_router_param(clean_name):
                self._save_router_param_to_hf(clean_name, full_tensor, local_params, full_sd)
            else:
                if clean_name in local_params:
                    full_sd[clean_name] = self._normalize_tensor(clean_name, full_tensor)
            del full_tensor

            if self._get_cpu_memory_usage_percent() >= 80 or self._get_gpu_memory_usage_percent() >= 80:
                self._gc_cleanup()

        for name, buf in self._persistent_buffers_to_save().items():
            clean_name = strip_peft_prefix(name)
            hf_buffer_name = self.param_converter.convert_persist_buffer(clean_name)
            logger.debug_rank0("[saving] buffer: %s -> %s", clean_name, hf_buffer_name)
            buf = self._gather_full_tensor(buf)
            if hf_buffer_name in local_params:
                value = self._normalize_tensor(hf_buffer_name, buf)
                full_sd[hf_buffer_name] = value
            del buf
            if self._get_cpu_memory_usage_percent() >= 80 or self._get_gpu_memory_usage_percent() >= 80:
                self._gc_cleanup()

        # Final sync before returning
        missing_params = local_params - set(full_sd.keys())
        extra_params = set(full_sd.keys()) - local_params
        if len(full_sd.keys()) != len(local_params):
            logger.warning_rank0(
                "[save] param count mismatch on rank %d — missing=%d, extra=%d",
                self.rank, len(missing_params), len(extra_params),
            )
            logger.warning_rank0("[save] missing params: %s", missing_params)
            logger.warning_rank0("[save] extra params: %s", extra_params)

        elapsed = time.perf_counter() - start_time
        logger.info_rank0("[save] local HF state dict built in %.2fs", elapsed)

        return full_sd

    def _normalize_tensor(self, name, tensor):
        dtype = self.param_converter.get_default_hf_dtype(name)
        tensor = tensor.to(dtype).contiguous()
        if self.move_to_cpu:
            tensor = tensor.cpu()
        return tensor

    def _get_gpu_memory_usage_percent(self):
        mem_free, mem_total = self.device_runtime.memory_info()
        mem_used = mem_total - mem_free
        utilization = (mem_used / mem_total) * 100.0
        return utilization

    def _get_cpu_memory_usage_percent(self):
        return psutil.virtual_memory().percent

    def save_hf(self, state_dict, save_path):
        # step 1, assign file index
        cur_num_shards = len(self.sharded_param_groups)
        num_shards_list = [0] * self.world_size
        dist.all_gather_object(num_shards_list, cur_num_shards)
        start_index = 0
        cur_start_index = 0
        for i in range(self.world_size):
            if i == self.rank:
                cur_start_index = start_index
            start_index += num_shards_list[i]
        num_total_files = start_index

        def write_files() -> tuple[dict[str, str], int]:
            file_index = cur_start_index
            weight_map = {}
            total_size = 0
            for cur_param_group in self.sharded_param_groups:
                shard_dict = {}
                if cur_param_group:
                    file_name = f"model-{file_index:05d}-of-{num_total_files:05d}.safetensors"
                    file_index += 1
                for param in cur_param_group:
                    assert param in self.local_params, f"param '{param}' not in local_params"
                    assert param in state_dict, f"param '{param}' not in state_dict"
                    shard_dict[param] = state_dict[param]
                    tensor = shard_dict[param]
                    total_size += tensor.numel() * tensor.element_size()
                    weight_map[param] = file_name
                if shard_dict:
                    logger.debug(
                        "[save][rank %d] Saving %s with %d params...",
                        self.rank, file_name, len(shard_dict),
                    )
                    file_start = time.perf_counter()
                    file_path = save_path / file_name
                    save_file(shard_dict, file_path.as_posix())
                    with open(file_path, "rb") as shard_file:
                        os.fsync(shard_file.fileno())
                    file_elapsed = time.perf_counter() - file_start
                    logger.info(
                        "[save][rank %d] Saved %s in %.2fs",
                        self.rank, file_name, file_elapsed,
                    )

            return weight_map, total_size

        saved_info = collective_check(write_files)

        def write_metadata() -> None:
            self._copy_non_safetensor_files(self.src_path, save_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(save_path)
            # The export keeps the source config; other training-time
            # changes such as use_cache are not written back.
            _sync_mtp_layer_count(save_path / CONFIG_NAME, self.model.config.to_dict())
            # All shard writers have finished, including atomic renames. Sync
            # config/tokenizer files before publishing the index.
            for directory, _, files in os.walk(save_path):
                for name in files:
                    if name.endswith(".safetensors"):
                        continue
                    with open(os.path.join(directory, name), "rb") as file:
                        os.fsync(file.fileno())
                sync_directory(directory)

        rank_zero_call(write_metadata)
        # step 3, deal with non-tensor metas
        saved_info_list = [None] * self.world_size
        dist.all_gather_object(saved_info_list, saved_info)
        final_weight_map = {}
        total_size = 0
        for cur_map, cur_size in saved_info_list:
            final_weight_map.update(cur_map)
            total_size += cur_size

        # Every writer has synced its shards and rank zero has synced the
        # required metadata. The index is published only after that boundary.
        rank_zero_call(lambda: save_json(
            {"metadata": {**self._src_metadata, "total_size": total_size}, "weight_map": final_weight_map},
            str(save_path / HF_INDEX_FILE),
        ))
        logger.info_rank0("[save] HF model saved to %s", save_path.absolute())

    def get_local_hf_state_dict(self):
        return self._get_local_hf_state_dict(self.local_params)

    def save_model(self, save_path):
        save_path = Path(save_path)

        save_path.mkdir(parents=True, exist_ok=True)

        torch.distributed.barrier()  # Ensure directory is created

        assert self.src_path is not None, "must be"
        # Each rank only collects the parameters it needs to save
        state_dict = self._get_local_hf_state_dict(self.local_params)

        # Save shards assigned to this rank
        self.save_hf(state_dict, save_path)


# for backward compatiblity
DistParamSaver=ShardedParamManager
