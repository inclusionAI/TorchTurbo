import json
import struct
from pathlib import Path

from safetensors.torch import _TYPES
import torch
import torch.distributed as dist

from torchturbo.models.modules.moe.moe_layer import BaseMoELayer


class CheckpointInfo:
    """Collect checkpoint tensor shapes and dtypes for weight conversion."""

    def __init__(self, model_path):
        """Initialize ``self.tensor_infos``: {HF tensor name: (shape, torch.dtype)}.

        E.g. {"model.layers.0.self_attn.q_proj.weight": ((2048, 2048), torch.bfloat16)}.
        """
        self.tensor_infos = self._load_checkpoint_tensor_infos(model_path)

    def get_tensor_infos(self):
        return self.tensor_infos

    def _load_checkpoint_tensor_infos(self, model_path):
        """Read tensor metadata from safetensors checkpoints."""
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with index_path.open() as index_file:
            filenames = set(json.load(index_file)["weight_map"].values())

        tensor_infos = {}
        for filename in filenames:
            with (model_path / filename).open("rb") as shard:
                header_size = struct.unpack("<Q", shard.read(8))[0]
                header = json.loads(shard.read(header_size))
            tensor_infos.update({
                name: (tensor_metadata["shape"], tensor_metadata["dtype"])
                for name, tensor_metadata in header.items()
                if name != "__metadata__"
            })
        tensor_infos = {
            name: (tuple(shape), _TYPES[dtype])
            for name, (shape, dtype) in tensor_infos.items()
        }
        return tensor_infos


class BaseWeightConverter:
    def __init__(self, model, parallel_context, model_path):
        config = model.config
        ep_info = parallel_context["ep"]
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.ep_group = ep_info["ep_group"]
        self.ep_rank = ep_info["ep_rank"]
        self.ep_size = ep_info["ep_size"]
        self.moe_config = self.get_moe_config(model)
        self.config = config
        self.model_type = config.model_type
        self.checkpoint_info = CheckpointInfo(model_path)
        # Transformers 5.x populates this fully-expanded {target: source}
        # mapping in PreTrainedModel.post_init(). Keep that authoritative
        # snapshot instead of re-expanding names after Turbo/FSDP rewrites.
        self.tied_weight_keys = dict(model.all_tied_weights_keys)

    def get_omitted_tied_source(self, model_name, hf_name):
        """Return the present checkpoint source for an omitted tied target.

        Only a target explicitly declared by Transformers may be absent, and
        its source must actually exist in the checkpoint. Any other missing
        tensor continues through the normal path and raises a hard error.
        """
        source_name = self.tied_weight_keys.get(model_name)
        if source_name is None:
            return None

        tensor_infos = self.checkpoint_info.get_tensor_infos()
        hf_source_name = self.to_hf_name(source_name)
        if hf_name not in tensor_infos and hf_source_name in tensor_infos:
            return hf_source_name
        return None

    def get_moe_config(self, model):
        for _, module in model.named_modules():
            if isinstance(module, BaseMoELayer):
                return module.config
        return None

    def get_experts_name(self):
        return 'experts'

    def get_expert_gate_name(self):
        return 'gate_proj'

    def get_expert_up_name(self):
        return 'up_proj'

    def get_experts_weight1_name(self):
        return 'weight1'

    def get_expert_down_name(self):
        return 'down_proj'

    def get_experts_weight2_name(self):
        return 'weight2'

    def get_moe_expert_mlp_name(self):
        return 'mlp'

    def get_hf_expert_mlp_name(self):
        return 'mlp'

    def get_moe_router_name(self):
        return 'router'

    def get_hf_gate_name(self):
        return 'gate'
    
    def get_router_bias_name(self):
        return "router.expert_bias"
    
    def get_hf_gate_bias_name(self):
        return "gate.e_score_correction_bias"

    def get_default_hf_dtype(self, param_name):
        if self.is_virtual_key(param_name):
            param_name = self.get_physical_key(param_name)
        return self.checkpoint_info.get_tensor_infos()[param_name][1]

    def _get_element_size(self, name):
        """Return element size in bytes for the checkpoint dtype of this param."""
        return torch.tensor([], dtype=self.get_default_hf_dtype(name)).element_size()

    def is_expert(self, name):
        return self.get_experts_name() in name

    def is_expert_weight1(self, name):
        return self.is_expert(name) and self.get_experts_weight1_name() in name

    def is_expert_weight2(self, name):
        return self.is_expert(name) and self.get_experts_weight2_name() in name

    def is_expert_weight(self, name):
        """Return True if name is either expert weight1 or weight2."""
        return self.is_expert_weight1(name) or self.is_expert_weight2(name)
    
    def is_router_param(self, name):
        return self.get_moe_router_name() in name
    
    def is_persist_buffer(self, name):
        return self.get_router_bias_name() in name

    def convert_persist_buffer(self, name):
        if self.get_router_bias_name() in name:
            return name.replace(self.get_router_bias_name(), self.get_hf_gate_bias_name())
        return name

    def convert_expert_weight1_to_hf(self, name, only_local_experts = False):
        assert self.is_expert_weight1(name), "must be"
        assert self.moe_config != None, "must be"
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        gates = []
        ups = []
        for i in range(self.ep_size):
            if only_local_experts and i != self.ep_rank:
                continue
            num_local_experts = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local_experts):
                index = i * num_local_experts + j
                gate =f"{prefix}.{index}.{self.get_expert_gate_name()}.weight"
                up = f"{prefix}.{index}.{self.get_expert_up_name()}.weight"
                gates.append(gate)
                ups.append(up)
        return gates, ups

    def get_gate_weight_shape(self):
        return (self.config.moe_intermediate_size, self.config.hidden_size)

    def get_gate_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self.config.hidden_size * element_size

    def get_up_weight_shape(self):
        return (self.config.moe_intermediate_size, self.config.hidden_size)

    def get_up_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self.config.hidden_size * element_size

    def get_down_weight_shape(self):
        return (self.config.hidden_size, self.config.moe_intermediate_size)

    def get_down_weight_bytes(self, element_size):
        return self.config.moe_intermediate_size * self.config.hidden_size * element_size

    def get_local_ep_w1_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight1(name), "must be"
        assert self.moe_config != None, "must be"
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        experts = torch.split(tensor, self.config.hidden_size)
        start_index = self.ep_rank * num_local_experts
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        local_ep_sd = {}
        for i in range(num_local_experts):
            index = start_index + i
            gate =f"{prefix}.{index}.{self.get_expert_gate_name()}.weight"
            up = f"{prefix}.{index}.{self.get_expert_up_name()}.weight"
            expert2 = torch.split(experts[i].t(), self.config.moe_intermediate_size)
            gate_tensor = expert2[0]
            up_tensor = expert2[1]
            local_ep_sd[gate] = gate_tensor
            local_ep_sd[up] = up_tensor
        return local_ep_sd

    def convert_expert_weight2_to_hf(self, name, only_local_experts = False):
        assert self.is_expert_weight2(name), "must be"
        assert self.moe_config != None, "must be"
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        downs = []
        for i in range(self.ep_size):
            if only_local_experts and i != self.ep_rank:
                continue
            num_local_experts = self.moe_config.num_moe_experts // self.ep_size
            for j in range(num_local_experts):
                index = i * num_local_experts + j
                down =f"{prefix}.{index}.{self.get_expert_down_name()}.weight"
                downs.append(down)
        return downs

    def get_local_ep_w2_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight2(name), "must be"
        assert self.moe_config != None, "must be"
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        start_index = self.ep_rank * num_local_experts
        prefix = name[:-8].replace(self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name())
        experts = torch.split(tensor, self.config.moe_intermediate_size)
        local_ep_sd = {}
        for i in range(num_local_experts):
            index = start_index + i
            down =f"{prefix}.{index}.{self.get_expert_down_name()}.weight"
            down_tensor = experts[i].t()
            local_ep_sd[down] = down_tensor
        return local_ep_sd

    def convert_router_param_to_hf(self, name):
        assert self.get_moe_router_name() in name, "must be"
        return name.replace(self.get_moe_router_name(), self.get_hf_gate_name())

    def to_hf_name(self, name):
        """Convert a non-expert internal param name to its HF checkpoint name.

        Only handles router and common params (NOT expert weights).
        For router params, applies the router→gate rename.
        For other common params, returns the name unchanged.
        """
        if self.is_router_param(name):
            return self.convert_router_param_to_hf(name)
        return name

    def get_param_save_entries(self, name, numel, param_to_filename=None):
        """Classify a param and return (common_keys, expert_keys, size_map).

        Handles expert weights, router params, and common params uniformly.
        numel: number of elements in the param (used for common/router size).
        param_to_filename: optional checkpoint index; used by FusedExpertWeightConverter
            to detect whether an expert weight is in fused or per-expert format.
        """
        expert_result = self.get_expert_save_entries(name, param_to_filename=param_to_filename)
        if expert_result is not None:
            return expert_result
        hf_name = self.to_hf_name(name)
        element_size = self._get_element_size(hf_name)
        param_bytes = numel * element_size
        if self.is_router_param(name):
            return [hf_name], [], {hf_name: param_bytes}
        return [name], [], {name: param_bytes}

    def get_param_load_entries(self, name, numel, param_to_filename=None):
        """Classify a param and return (common_keys, expert_group, size_map).

        Handles expert weights, router params, and common params uniformly.
        numel: number of elements in the param (used for common/router size).
        expert_group: (model_name, local_names, total_local_bytes) or None.
        param_to_filename: optional checkpoint index; used by FusedExpertWeightConverter
            to detect whether an expert weight is in fused or per-expert format.
        """
        expert_result = self.get_expert_load_entries(name, param_to_filename=param_to_filename)
        if expert_result is not None:
            return expert_result
        hf_name = self.to_hf_name(name)
        if self.get_omitted_tied_source(name, hf_name) is not None:
            return [], None, {}

        element_size = self._get_element_size(hf_name)
        param_bytes = numel * element_size
        if self.is_router_param(name):
            return [hf_name], None, {hf_name: param_bytes}
        return [name], None, {name: param_bytes}

    def get_expert_save_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_keys, size_map) if name is an expert weight, else None.

        Dispatches to get_weight1_save_entries or get_weight2_save_entries.
        """
        if self.is_expert_weight1(name):
            return self.get_weight1_save_entries(name, param_to_filename=param_to_filename)
        elif self.is_expert_weight2(name):
            return self.get_weight2_save_entries(name, param_to_filename=param_to_filename)
        return None

    def get_weight1_save_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_keys, size_map) for weight1 save allocation."""
        ep_gates, ep_ups = self.convert_expert_weight1_to_hf(name, only_local_experts=True)
        gates, ups = self.convert_expert_weight1_to_hf(name)
        element_size = self._get_element_size(gates[0])
        gate_size = self.get_gate_weight_bytes(element_size)
        up_size = self.get_up_weight_bytes(element_size)
        local_keys = list(ep_gates) + list(ep_ups)
        size_map = {g: gate_size for g in gates}
        size_map.update({u: up_size for u in ups})
        return [], local_keys, size_map

    def get_weight2_save_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_keys, size_map) for weight2 save allocation."""
        ep_downs = self.convert_expert_weight2_to_hf(name, only_local_experts=True)
        downs = self.convert_expert_weight2_to_hf(name)
        element_size = self._get_element_size(downs[0])
        down_size = self.get_down_weight_bytes(element_size)
        local_keys = list(ep_downs)
        size_map = {d: down_size for d in downs}
        return [], local_keys, size_map

    def get_expert_load_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_group, size_map) if name is an expert weight, else None.

        Dispatches to get_weight1_load_entries or get_weight2_load_entries.
        """
        if self.is_expert_weight1(name):
            return self.get_weight1_load_entries(name, param_to_filename=param_to_filename)
        elif self.is_expert_weight2(name):
            return self.get_weight2_load_entries(name, param_to_filename=param_to_filename)
        return None

    def get_weight1_load_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_group, size_map) for weight1 load allocation.

        common_keys: list of HF param names to assign as common params.
        expert_group: (model_name, local_names, total_local_bytes) or None.
        size_map: dict mapping HF param name → byte size.
        """
        ep_gates, ep_ups = self.convert_expert_weight1_to_hf(name, only_local_experts=True)
        local_names = list(ep_gates) + list(ep_ups)
        all_gates, all_ups = self.convert_expert_weight1_to_hf(name)
        element_size = self._get_element_size(all_gates[0])
        gate_size = self.get_gate_weight_bytes(element_size)
        up_size = self.get_up_weight_bytes(element_size)
        size_map = {g: gate_size for g in all_gates}
        size_map.update({u: up_size for u in all_ups})
        total_local_bytes = len(ep_gates) * gate_size + len(ep_ups) * up_size
        return [], (name, local_names, total_local_bytes), size_map

    def get_weight2_load_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_group, size_map) for weight2 load allocation.

        common_keys: list of HF param names to assign as common params.
        expert_group: (model_name, local_names, total_local_bytes) or None.
        size_map: dict mapping HF param name → byte size.
        """
        ep_downs = self.convert_expert_weight2_to_hf(name, only_local_experts=True)
        local_names = list(ep_downs)
        all_downs = self.convert_expert_weight2_to_hf(name)
        element_size = self._get_element_size(all_downs[0])
        down_size = self.get_down_weight_bytes(element_size)
        size_map = {d: down_size for d in all_downs}
        total_local_bytes = len(ep_downs) * down_size
        return [], (name, local_names, total_local_bytes), size_map

    def get_expert_broadcast_key(self, name):
        """Return the HF key used to look up the owner rank for expert weight broadcast.

        Dispatches to get_weight1_broadcast_key or get_weight2_broadcast_key.
        """
        if self.is_expert_weight1(name):
            return self.get_weight1_broadcast_key(name)
        elif self.is_expert_weight2(name):
            return self.get_weight2_broadcast_key(name)
        raise ValueError(f"Not an expert weight: {name}")

    def get_weight1_broadcast_key(self, name):
        """Return the HF key used to look up the owner rank for weight1 broadcast."""
        gates, _ = self.convert_expert_weight1_to_hf(name, only_local_experts=True)
        return gates[0]

    def get_weight2_broadcast_key(self, name):
        """Return the HF key used to look up the owner rank for weight2 broadcast."""
        downs = self.convert_expert_weight2_to_hf(name, only_local_experts=True)
        return downs[0]

    def assemble_expert_weight_for_broadcast(self, tensor, local_sd, name):
        """Assemble expert weight data from checkpoint into the pre-allocated tensor.

        Dispatches to assemble_weight1_for_broadcast or assemble_weight2_for_broadcast.
        Only the owner rank should call this. Copies data into tensor in-place.
        Pops consumed keys from local_sd.
        """
        if self.is_expert_weight1(name):
            self.assemble_weight1_for_broadcast(tensor, local_sd, name)
        elif self.is_expert_weight2(name):
            self.assemble_weight2_for_broadcast(tensor, local_sd, name)
        else:
            raise ValueError(f"Not an expert weight: {name}")

    def assemble_weight1_for_broadcast(self, tensor, local_sd, name):
        """Assemble weight1 data from per-expert HF tensors into tensor in-place.

        Only the owner rank should call this.
        Pops consumed keys from local_sd.
        """
        device, dtype = tensor.device, tensor.dtype
        gates, ups = self.convert_expert_weight1_to_hf(name, only_local_experts=True)
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        rows_per_expert = tensor.shape[0] // num_local_experts
        for j in range(num_local_experts):
            gate = local_sd.pop(gates[j]).to(device).to(dtype)
            up = local_sd.pop(ups[j]).to(device).to(dtype)
            # cat(gate,up) → permute(1,0) gives each expert's final (rows, cols) block
            combined = torch.cat([gate, up]).permute(1, 0)
            offset = j * rows_per_expert
            tensor[offset:offset + rows_per_expert].copy_(combined)
            del gate, up, combined

    def assemble_weight2_for_broadcast(self, tensor, local_sd, name):
        """Assemble weight2 data from per-expert HF tensors into tensor in-place.

        Only the owner rank should call this.
        Pops consumed keys from local_sd.
        """
        device, dtype = tensor.device, tensor.dtype
        downs = self.convert_expert_weight2_to_hf(name, only_local_experts=True)
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        rows_per_expert = tensor.shape[0] // num_local_experts
        for j in range(num_local_experts):
            down = local_sd.pop(downs[j]).to(device).to(dtype)
            offset = j * rows_per_expert
            tensor[offset:offset + rows_per_expert].copy_(down.t())
            del down

    def is_virtual_key(self, key):
        """Return True if key was synthesized for per-EP assignment of fused experts."""
        return False

    def get_physical_key(self, virtual_key):
        """Return the checkpoint (physical) key that a virtual key maps to."""
        raise ValueError(f"Not a virtual key: {virtual_key}")

    def slice_virtual_key(self, virtual_key, physical_tensor):
        """Slice physical_tensor for the EP rank encoded in virtual_key."""
        raise NotImplementedError

    def get_virtual_key_filenames(self, name, param_to_filename):
        """Return {virtual_key: filename} entries to register in param_to_filename.

        Per-expert format returns {} (all keys exist in the checkpoint).
        Fused format returns one entry per EP rank.
        """
        return {}

    @property
    def fused_experts(self):
        return False


class FusedExpertWeightConverter(BaseWeightConverter):
    """Weight converter for models with fused expert tensors.

    Handles checkpoints where expert weights are stored as 3D fused tensors
    (e.g. gate_up_proj [E, 2*I, H]) rather than per-expert 2D tensors.
    When the fused key is not in the checkpoint (e.g. MTP layers in Qwen3.5
    use per-expert format), falls back to BaseWeightConverter's per-expert logic.
    """

    VIRTUAL_KEY_DELIMITER = "__ep"

    def __init__(self, model, parallel_context, model_path):
        super().__init__(model, parallel_context, model_path)
        # Model param names where the checkpoint uses per-expert format
        # (not fused). Filled by get_weight1/2_load_entries when the
        # fused key is absent from param_to_filename.
        self._per_expert_fallback_names = set()

    @property
    def fused_experts(self):
        return True

    def _make_virtual_key(self, fused_name):
        """Create a virtual per-EP key for the current EP rank: '<fused_name>__ep<N>'."""
        return f"{fused_name}{self.VIRTUAL_KEY_DELIMITER}{self.ep_rank}"

    def _make_virtual_key_for_ep(self, fused_name, ep_rank):
        """Create a virtual key for a specific EP rank (used in size_map)."""
        return f"{fused_name}{self.VIRTUAL_KEY_DELIMITER}{ep_rank}"

    def is_virtual_key(self, key):
        """Return True if key follows the virtual pattern '<prefix>__ep<int>'."""
        parts = key.rsplit(self.VIRTUAL_KEY_DELIMITER, 1)
        if len(parts) != 2:
            return False
        try:
            int(parts[1])
            return True
        except ValueError:
            return False

    def get_physical_key(self, virtual_key):
        """Strip the __ep<N> suffix to recover the physical fused key."""
        return virtual_key.rsplit(self.VIRTUAL_KEY_DELIMITER, 1)[0]

    def slice_virtual_key(self, virtual_key, physical_tensor):
        """Slice a fused [E, ...] tensor for the ep_rank encoded in virtual_key.

        The ep_rank is parsed from the virtual key name, NOT from self.ep_rank,
        because a rank may own a virtual key whose ep_rank differs from its own.
        """
        _, ep_rank_str = virtual_key.rsplit(self.VIRTUAL_KEY_DELIMITER, 1)
        ep_rank = int(ep_rank_str)
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        start = ep_rank * num_local_experts
        end = start + num_local_experts
        return physical_tensor[start:end].clone()

    def get_virtual_key_filenames(self, name, param_to_filename):
        """Return {virtual_key: filename} for all EP ranks' virtual keys."""
        # Per-expert format (MTP) doesn't use virtual keys
        if name in self._per_expert_fallback_names:
            return {}
        if self.is_expert_weight1(name):
            fused_name = self.convert_expert_weight1_to_hf(name)
        elif self.is_expert_weight2(name):
            fused_name = self.convert_expert_weight2_to_hf(name)
        else:
            return {}
        if fused_name not in param_to_filename:
            return {}
        filename = param_to_filename[fused_name]
        return {
            self._make_virtual_key_for_ep(fused_name, ep): filename
            for ep in range(self.ep_size)
        }

    def get_fused_w1_name(self):
        return 'gate_up_proj'

    def get_fused_w2_name(self):
        return 'down_proj'

    def _expert_prefix(self, name):
        return name.rsplit('.', 1)[0]

    def convert_expert_weight1_to_hf(self, name, only_local_experts=False):
        # Fallback: per-expert format (e.g. MTP layers in Qwen3.5)
        if name in self._per_expert_fallback_names:
            return BaseWeightConverter.convert_expert_weight1_to_hf(
                self, name, only_local_experts=only_local_experts
            )
        prefix = self._expert_prefix(name)
        hf_prefix = prefix.replace(
            self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name()
        )
        fused_name = f"{hf_prefix}.{self.get_fused_w1_name()}"
        return fused_name

    def convert_expert_weight2_to_hf(self, name, only_local_experts=False):
        # Fallback: per-expert format (e.g. MTP layers in Qwen3.5)
        if name in self._per_expert_fallback_names:
            return BaseWeightConverter.convert_expert_weight2_to_hf(
                self, name, only_local_experts=only_local_experts
            )
        prefix = self._expert_prefix(name)
        hf_prefix = prefix.replace(
            self.get_moe_expert_mlp_name(), self.get_hf_expert_mlp_name()
        )
        fused_name = f"{hf_prefix}.{self.get_fused_w2_name()}"
        return fused_name

    def get_fused_w1_shape(self):
        num_experts = self.moe_config.num_moe_experts
        return (num_experts, 2 * self.config.moe_intermediate_size, self.config.hidden_size)

    def get_fused_w2_shape(self):
        num_experts = self.moe_config.num_moe_experts
        return (num_experts, self.config.hidden_size, self.config.moe_intermediate_size)

    def get_fused_w1_bytes(self, element_size):
        s = self.get_fused_w1_shape()
        return s[0] * s[1] * s[2] * element_size

    def get_fused_w2_bytes(self, element_size):
        s = self.get_fused_w2_shape()
        return s[0] * s[1] * s[2] * element_size

    def get_local_ep_w1_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight1(name), "must be"
        # Fallback: per-expert format (e.g. MTP layers in Qwen3.5)
        if name in self._per_expert_fallback_names:
            return super().get_local_ep_w1_hf_state_dict(name, tensor)
        assert self.moe_config is not None, "must be"
        hf_name = self.convert_expert_weight1_to_hf(name)
        tensor = tensor.to(self.get_default_hf_dtype(hf_name))
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        w1 = tensor.reshape(num_local_experts, self.config.hidden_size, -1)
        w1 = w1.permute(0, 2, 1).contiguous()  # [num_local, 2I, H]
        # All-gather directly into output tensor, avoiding intermediate list + cat
        full_w1 = w1.new_zeros(self.moe_config.num_moe_experts, *w1.shape[1:])
        dist.all_gather_into_tensor(full_w1, w1, group=self.ep_group)
        return {hf_name: full_w1}

    def get_local_ep_w2_hf_state_dict(self, name, tensor):
        assert self.is_expert_weight2(name), "must be"
        # Fallback: per-expert format (e.g. MTP layers in Qwen3.5)
        if name in self._per_expert_fallback_names:
            return super().get_local_ep_w2_hf_state_dict(name, tensor)
        assert self.moe_config is not None, "must be"
        hf_name = self.convert_expert_weight2_to_hf(name)
        tensor = tensor.to(self.get_default_hf_dtype(hf_name))
        num_local_experts = self.moe_config.num_moe_experts // self.ep_size
        w2 = tensor.reshape(num_local_experts, self.config.moe_intermediate_size, self.config.hidden_size)
        w2 = w2.permute(0, 2, 1).contiguous()  # [num_local, H, I]
        # All-gather directly into output tensor, avoiding intermediate list + cat
        full_w2 = w2.new_zeros(self.moe_config.num_moe_experts, *w2.shape[1:])
        dist.all_gather_into_tensor(full_w2, w2, group=self.ep_group)
        return {hf_name: full_w2}

    def get_weight1_save_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_keys, size_map) for fused weight1 save.

        Fused expert keys are treated as common params.
        Falls back to BaseWeightConverter per-expert logic when the fused key
        is not in param_to_filename (e.g. MTP layers in Qwen3.5).
        """
        if name in self._per_expert_fallback_names:
            return super().get_weight1_save_entries(name)
        fused_name = self.convert_expert_weight1_to_hf(name)
        # Fallback: fused key not in checkpoint → per-expert format (e.g. MTP)
        if param_to_filename is not None and fused_name not in param_to_filename:
            self._per_expert_fallback_names.add(name)
            return super().get_weight1_save_entries(name)
        fused_size = self.get_fused_w1_bytes(self._get_element_size(fused_name))
        return [fused_name], [], {fused_name: fused_size}

    def get_weight2_save_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_keys, size_map) for fused weight2 save.

        Fused expert keys are treated as common params.
        Falls back to BaseWeightConverter per-expert logic when the fused key
        is not in param_to_filename (e.g. MTP layers in Qwen3.5).
        """
        if name in self._per_expert_fallback_names:
            return super().get_weight2_save_entries(name)
        fused_name = self.convert_expert_weight2_to_hf(name)
        # Fallback: fused key not in checkpoint → per-expert format (e.g. MTP)
        if param_to_filename is not None and fused_name not in param_to_filename:
            self._per_expert_fallback_names.add(name)
            return super().get_weight2_save_entries(name)
        fused_size = self.get_fused_w2_bytes(self._get_element_size(fused_name))
        return [fused_name], [], {fused_name: fused_size}

    def get_weight1_load_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_group, size_map) for fused weight1 load.

        Uses per-EP virtual keys so that fused expert params are assigned to
        EDP ranks in Phase 1 (like per-expert format), guaranteeing the
        broadcast owner is in the local EDP group.

        Falls back to BaseWeightConverter per-expert logic when the fused key
        is not in param_to_filename (e.g. MTP layers in Qwen3.5).
        """
        fused_name = self.convert_expert_weight1_to_hf(name)
        # Fallback: fused key not in checkpoint → per-expert format (e.g. MTP)
        if param_to_filename is not None and fused_name not in param_to_filename:
            self._per_expert_fallback_names.add(name)
            return super().get_weight1_load_entries(name)
        local_bytes = self.get_fused_w1_bytes(self._get_element_size(fused_name)) // self.ep_size
        virtual_key = self._make_virtual_key(fused_name)
        # Include sizes for ALL EP ranks' virtual keys so that after EP all-gather,
        # param_size_map has entries for every virtual key and rank_shard_bytes
        # is recomputed correctly (mirrors per-expert format's all_gates + all_ups).
        size_map = {
            self._make_virtual_key_for_ep(fused_name, ep): local_bytes
            for ep in range(self.ep_size)
        }
        return [], (name, [virtual_key], local_bytes), size_map

    def get_weight2_load_entries(self, name, param_to_filename=None):
        """Return (common_keys, expert_group, size_map) for fused weight2 load.

        Uses per-EP virtual keys so that fused expert params are assigned to
        EDP ranks in Phase 1 (like per-expert format), guaranteeing the
        broadcast owner is in the local EDP group.

        Falls back to BaseWeightConverter per-expert logic when the fused key
        is not in param_to_filename (e.g. MTP layers in Qwen3.5).
        """
        fused_name = self.convert_expert_weight2_to_hf(name)
        # Fallback: fused key not in checkpoint → per-expert format (e.g. MTP)
        if param_to_filename is not None and fused_name not in param_to_filename:
            self._per_expert_fallback_names.add(name)
            return super().get_weight2_load_entries(name)
        local_bytes = self.get_fused_w2_bytes(self._get_element_size(fused_name)) // self.ep_size
        virtual_key = self._make_virtual_key(fused_name)
        size_map = {
            self._make_virtual_key_for_ep(fused_name, ep): local_bytes
            for ep in range(self.ep_size)
        }
        return [], (name, [virtual_key], local_bytes), size_map

    def get_weight1_broadcast_key(self, name):
        """Return the key used to look up the owner rank for weight1 broadcast.

        Uses the virtual per-EP key for fused format, or falls back to
        BaseWeightConverter per-expert logic for MTP layers.
        """
        if name in self._per_expert_fallback_names:
            return super().get_weight1_broadcast_key(name)
        fused_name = self.convert_expert_weight1_to_hf(name)
        return self._make_virtual_key(fused_name)

    def get_weight2_broadcast_key(self, name):
        """Return the key used to look up the owner rank for weight2 broadcast.

        Uses the virtual per-EP key for fused format, or falls back to
        BaseWeightConverter per-expert logic for MTP layers.
        """
        if name in self._per_expert_fallback_names:
            return super().get_weight2_broadcast_key(name)
        fused_name = self.convert_expert_weight2_to_hf(name)
        return self._make_virtual_key(fused_name)

    def assemble_weight1_for_broadcast(self, tensor, local_sd, name):
        """Assemble weight1 data into tensor in-place.

        For fused format: pops pre-sliced virtual key data from local_sd.
        For per-expert format (MTP): falls back to BaseWeightConverter logic.
        Only the owner rank should call this.
        """
        if name in self._per_expert_fallback_names:
            return super().assemble_weight1_for_broadcast(tensor, local_sd, name)
        device, dtype = tensor.device, tensor.dtype
        fused_name = self.convert_expert_weight1_to_hf(name)
        virtual_key = self._make_virtual_key(fused_name)
        # Data was already sliced in _load_local_shards.
        # Shape: [num_local_experts, 2I, H] -> permute -> [num_local, H, 2I] -> reshape -> [num_local*H, 2I]
        sliced = local_sd.pop(virtual_key).to(device).to(dtype)
        tensor.copy_(sliced.permute(0, 2, 1).reshape(-1, sliced.shape[1]))
        del sliced

    def assemble_weight2_for_broadcast(self, tensor, local_sd, name):
        """Assemble weight2 data into tensor in-place.

        For fused format: pops pre-sliced virtual key data from local_sd.
        For per-expert format (MTP): falls back to BaseWeightConverter logic.
        Only the owner rank should call this.
        """
        if name in self._per_expert_fallback_names:
            return super().assemble_weight2_for_broadcast(tensor, local_sd, name)
        device, dtype = tensor.device, tensor.dtype
        fused_name = self.convert_expert_weight2_to_hf(name)
        virtual_key = self._make_virtual_key(fused_name)
        sliced = local_sd.pop(virtual_key).to(device).to(dtype)
        permuted = sliced.permute(0, 2, 1)
        tensor.copy_(permuted.reshape(-1, permuted.shape[-1]))
        del sliced
