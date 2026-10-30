from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from torchturbo.logging import get_logger
from torchturbo.runtime import DeviceRuntime, TurboDevice

from .base import OpImpl, OpImplKind, OpPolicy, OpSpec
from .moe import MoEPermutation, get_expert_indices
from .packed_params import PackedParams, get_packed_params
from .providers import iter_provider_modules
from .registry import OpRegistry

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Candidate:
    impl: OpImpl
    spec: OpSpec
    fallback: bool


class OpDispatcher:
    def __init__(
        self,
        *,
        device_runtime: DeviceRuntime,
        registry: OpRegistry,
        default_policy: OpPolicy = OpPolicy.AUTO,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(default_policy, OpPolicy):
            raise TypeError("default_policy must be an OpPolicy.")
        self.registry = registry
        self._device_runtime = device_runtime
        self._default_policy = default_policy
        self._overrides = dict(overrides or {})
        if any(
            not isinstance(op_name, str) or not isinstance(impl_name, str)
            for op_name, impl_name in self._overrides.items()
        ):
            raise TypeError(
                "overrides must map op names to implementation names."
            )
        self._impls: dict[str, OpImpl] = {}
        self._impl_classes: dict[str, type[OpImpl]] = {}
        self._candidate_impl_cache: dict[
            tuple[str, str, str | None], _Candidate
        ] = {}
        self._unresolved_ops: dict[str, str] = {}
        self._configure()

    @property
    def device_runtime(self) -> DeviceRuntime:
        return self._device_runtime

    @property
    def default_policy(self) -> OpPolicy:
        return self._default_policy

    @property
    def overrides(self) -> dict[str, str]:
        return dict(self._overrides)

    def _index_provider_module(
        self,
        module: ModuleType,
        impl_classes: dict[str, type[OpImpl]],
    ) -> None:
        for value in module.__dict__.values():
            if not isinstance(value, type):
                continue
            if value.__module__ != module.__name__:
                continue
            if not issubclass(value, OpImpl) or value is OpImpl:
                continue

            try:
                spec = value.op_spec()
            except RuntimeError:
                continue

            existing = impl_classes.get(spec.impl_name)
            if existing is not None and existing is not value:
                raise ValueError(
                    f"Op implementation {spec.impl_name!r} is defined by both "
                    f"{existing.__module__}.{existing.__qualname__} and "
                    f"{value.__module__}.{value.__qualname__}."
                )
            impl_classes[spec.impl_name] = value

    def _load_provider_impl_classes(
        self, device: TurboDevice
    ) -> dict[str, type[OpImpl]]:
        impl_classes: dict[str, type[OpImpl]] = {}
        for module_name in iter_provider_modules(device):
            module = importlib.import_module(module_name)
            self._index_provider_module(module, impl_classes)
        return impl_classes

    def _configure(self) -> None:
        device = self._device_runtime.turbo_device

        self._impl_classes = self._load_provider_impl_classes(device)
        specs = self.registry.list(
            spec_filter=lambda spec: not spec.devices or device in spec.devices
        )
        op_names = {spec.op_name for spec in specs}
        op_names.update(self._overrides)
        for op_name in sorted(op_names):
            override = self._overrides.get(op_name)
            try:
                self._candidate_impl(op_name, device, impl=override)
            except RuntimeError as exc:
                if override is not None:
                    raise RuntimeError(
                        f"Invalid override for op {op_name!r}: "
                        f"{override!r}. {exc}"
                    ) from exc
                self._unresolved_ops[op_name] = str(exc)

    def _candidate_key(
        self,
        op_name: str,
        device: TurboDevice,
        impl: str | None,
    ) -> tuple[str, str, str | None]:
        return (op_name, device.value, impl)

    def _candidate_impl(
        self,
        op_name: str,
        device: TurboDevice,
        impl: str | None = None,
    ) -> OpImpl:
        key = self._candidate_key(op_name, device, impl)
        cached = self._candidate_impl_cache.get(key)
        if cached is not None:
            return cached.impl

        if impl is not None:
            spec = self.registry.get_by_impl_name(impl)
            if spec.op_name != op_name:
                raise RuntimeError(
                    f"Implementation {impl!r} is registered for op "
                    f"{spec.op_name!r}, not {op_name!r}."
                )
            if spec.devices and device not in spec.devices:
                raise RuntimeError(
                    f"Implementation {impl!r} for op {op_name!r} does not support "
                    f"device={device.value}."
                )
            specs = [spec]
        else:
            def matches(spec: OpSpec) -> bool:
                if spec.op_name != op_name:
                    return False
                if spec.devices and device not in spec.devices:
                    return False
                if self._default_policy == OpPolicy.NATIVE:
                    return spec.kind == OpImplKind.NATIVE
                if self._default_policy == OpPolicy.CUSTOM:
                    return spec.kind == OpImplKind.CUSTOM
                return True

            specs = self.registry.list(spec_filter=matches)

        unavailable: list[str] = []
        for spec in specs:
            impl_obj = self._load_impl(spec)
            if not impl_obj.available():
                unavailable.append(spec.impl_name)
                continue
            self._candidate_impl_cache[key] = _Candidate(
                impl=impl_obj,
                spec=spec,
                fallback=bool(unavailable),
            )
            return impl_obj

        detail_msg = f" (unavailable={unavailable})" if unavailable else ""
        selected_policy = (
            f"policy={self._default_policy.value}"
            if impl is None
            else f"impl={impl}"
        )
        raise RuntimeError(
            f"No implementation for op={op_name!r}, device={device.value}, "
            f"{selected_policy}{detail_msg}."
        )

    def _load_impl(self, spec: OpSpec) -> OpImpl:
        impl = self._impls.get(spec.impl_name)
        if impl is not None:
            return impl

        impl_cls = self._impl_classes.get(spec.impl_name)
        if impl_cls is None:
            raise RuntimeError(
                f"No OpImpl class registered for impl_name {spec.impl_name!r}."
            )
        impl = impl_cls()
        self._impls[spec.impl_name] = impl
        return impl

    def log_ops_selection(self) -> None:
        """Log the selected ops for this dispatcher."""
        selected_candidates = {
            op_name: candidate
            for (op_name, _, impl_name), candidate in self._candidate_impl_cache.items()
            if impl_name == self._overrides.get(op_name)
        }
        op_names = sorted(set(selected_candidates) | set(self._unresolved_ops))
        rows: list[tuple[str, str, str, str, str]] = []
        for op_name in op_names:
            candidate = selected_candidates.get(op_name)
            if candidate is None:
                rows.append((op_name, "-", "-", "-", "unavailable"))
                continue

            if op_name in self._overrides:
                status = "overridden"
            else:
                status = "fallback" if candidate.fallback else "selected"

            rows.append(
                (
                    op_name,
                    candidate.spec.impl_name,
                    candidate.spec.kind.value,
                    str(candidate.spec.priority),
                    status,
                )
            )

        if not rows:
            rows.append(("<no registered ops>", "-", "-", "-", "-"))

        headers = ("Op", "Implementation", "Kind", "Priority", "Status")
        widths = [
            max(len(header), *(len(row[index]) for row in rows)) + 2
            for index, header in enumerate(headers)
        ]
        header = " ".join(
            f"{column:<{width}}" for column, width in zip(headers, widths)
        ).rstrip()
        separator = "-" * len(header)
        lines = [separator, header, separator]
        lines.extend(
            " ".join(
                f"{value:<{width}}" for value, width in zip(row, widths)
            ).rstrip()
            for row in rows
        )
        lines.append(separator)

        unavailable = [
            f"  {op_name}: {self._unresolved_ops[op_name]}"
            for op_name in op_names
            if op_name in self._unresolved_ops
        ]
        if unavailable:
            lines.extend(["", "Unavailable details:", *unavailable])

        logger.info_rank0(
            "TorchTurbo ops selection (device=%s, policy=%s):\n%s",
            self._device_runtime.device.type,
            self._default_policy.value,
            "\n".join(lines),
        )

    def _dispatch(
        self,
        op_name: str,
        *args: Any,
        impl: str | None = None,
        **kwargs: Any,
    ):
        device = self._device_runtime.turbo_device
        selected_impl = (
            impl
            if impl is not None
            else self._overrides.get(op_name)
        )
        impl_obj = self._candidate_impl(op_name, device, impl=selected_impl)
        return impl_obj(*args, **kwargs)

    def rms_norm(
        self,
        hidden_states,
        weight,
        eps,
        *,
        impl: str | None = None,
    ):
        return self._dispatch(
            "rms_norm",
            hidden_states,
            weight,
            eps,
            impl=impl,
        )

    def grouped_matmul(
        self,
        x,
        weight,
        group_list,
        *,
        trans_b: bool = False,
        impl: str | None = None,
    ):
        return self._dispatch(
            "grouped_matmul",
            x,
            weight,
            group_list,
            trans_b=trans_b,
            impl=impl,
        )

    def matmul_add(
        self,
        accumulator,
        lhs,
        rhs,
        *,
        transpose_lhs: bool = False,
        impl: str | None = None,
    ):
        return self._dispatch(
            "matmul_add",
            accumulator,
            lhs,
            rhs,
            transpose_lhs=transpose_lhs,
            impl=impl,
        )

    def gated_activation(
        self,
        x,
        *,
        activation_func,
        limit: float = 0.0,
        clamp_gate_after_activation: bool = False,
        impl: str | None = None,
    ):
        return self._dispatch(
            "gated_activation",
            x,
            activation_func=activation_func,
            limit=limit,
            clamp_gate_after_activation=clamp_gate_after_activation,
            impl=impl,
        )

    def swiglu(
        self,
        x,
        *,
        impl: str | None = None,
    ):
        return self._dispatch(
            "swiglu",
            x,
            impl=impl,
        )

    def gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        initial_state=None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        packed_params=None,
        save_backward_intermediates: bool | None = None,
        impl: str | None = None,
    ):
        if cu_seqlens is not None:
            if packed_params is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            packed_params = get_packed_params(cu_seqlens, q.size(1), device=q.device)
        return self._dispatch(
            "gated_delta_rule",
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            packed_params=packed_params,
            save_backward_intermediates=save_backward_intermediates,
            impl=impl,
        )

    def kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        use_beta_sigmoid_in_kernel: bool = False,
        disable_recompute: bool = False,
        head_major_qkv: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        packed_params: PackedParams | None = None,
        impl: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is not None:
            if packed_params is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            packed_params = get_packed_params(cu_seqlens, q.size(1), device=q.device)
        return self._dispatch(
            "kda",
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            disable_recompute=disable_recompute,
            head_major_qkv=head_major_qkv,
            packed_params=packed_params,
            impl=impl,
        )

    def causal_conv1d(
        self,
        x,
        weight,
        bias=None,
        *,
        activation=None,
        initial_state=None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        packed_params=None,
        head_num=None,
        impl: str | None = None,
    ):
        if cu_seqlens is not None:
            if packed_params is not None:
                raise ValueError("pass either cu_seqlens or packed_params, not both")
            packed_params = get_packed_params(cu_seqlens, x.size(1), device=x.device)
        return self._dispatch(
            "causal_conv1d",
            x,
            weight,
            bias,
            activation=activation,
            initial_state=initial_state,
            output_final_state=output_final_state,
            packed_params=packed_params,
            head_num=head_num,
            impl=impl,
        )

    def sparse_flash_mla(
        self,
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        cmp_residual_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_win_left,
        ori_win_right,
        *,
        cu_seq_lens_q=None,
        cu_seq_lens_ori_kv=None,
        cu_seq_lens_cmp_kv=None,
        impl: str | None = None,
    ):
        return self._dispatch(
            "sparse_flash_mla",
            query,
            ori_kv,
            cmp_kv,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            softmax_scale,
            cmp_ratio,
            ori_win_left,
            ori_win_right,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_ori_kv=cu_seq_lens_ori_kv,
            cu_seq_lens_cmp_kv=cu_seq_lens_cmp_kv,
            impl=impl,
        )

    def lightning_indexer(self, *args, impl: str | None = None, **kwargs):
        return self._dispatch("lightning_indexer", *args, impl=impl, **kwargs)

    def lightning_indexer_metadata(
        self, *args, impl: str | None = None, **kwargs
    ):
        return self._dispatch(
            "lightning_indexer_metadata", *args, impl=impl, **kwargs
        )

    def moe_permutation(
        self,
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor | None = None,
        num_out_tokens: int | None = None,
        expert_indices: torch.Tensor | None = None,
        *,
        impl: str | None = None,
    ) -> MoEPermutation:
        """Duplicate and group token assignments by expert without weighting.

        Every source token selects the same number of distinct experts. Each
        assignment produces one output row, ordered by expert ID, then source
        token ID. Selected routes are retained even when their weight is zero.

        Args:
            tokens: Source features of shape [num_tokens, hidden_size].
            routing_map: Boolean assignment mask of shape
                [num_tokens, num_experts] on the token device. Each row must
                contain exactly topk selected experts; token dropping is
                unsupported.
            probs: Optional dense routing weights of shape
                [num_tokens, num_experts] on the token device. Selected weights
                are gathered into packed row order, retaining their dtype.
                They are not multiplied into token features by this operator.
            num_out_tokens: Optional output row count. Must equal
                num_tokens * topk; when omitted, derive it from the assignments.
            expert_indices: Optional int64 tensor of shape [num_tokens, topk]
                on the token device. Each row must contain the selected expert
                IDs in ascending order and match routing_map. Providing it
                avoids deriving the IDs from the mask. For empty inputs,
                pass shape [0, topk] to retain the known topk; otherwise the
                inferred index shape is [0, 0].
            impl: Optional explicit implementation name. When omitted, use
                the configured operator override or dispatch policy.

        Returns:
            MoEPermutation containing packed features, optional packed weights,
            and provider-independent row/expert indices for unpermutation.

        Raises:
            ValueError: If deriving indices reveals nonuniform topk, or
                num_out_tokens differs from the number of assignments.
        """
        if expert_indices is None:
            expert_indices = get_expert_indices(routing_map)
        expected_tokens = expert_indices.numel()
        if num_out_tokens is not None and num_out_tokens != expected_tokens:
            raise ValueError(
                "num_out_tokens must equal num_tokens * topk; token dropping is unsupported."
            )
        return self._dispatch(
            "moe_permutation",
            tokens,
            routing_map,
            probs=probs,
            num_out_tokens=expected_tokens,
            expert_indices=expert_indices,
            impl=impl,
        )

    def moe_unpermutation(
        self,
        permutation: MoEPermutation,
        restore_shape: torch.Size,
        probs: torch.Tensor | None = None,
        *,
        impl: str | None = None,
    ) -> torch.Tensor:
        """Sum expert output rows back into source-token order.

        Source token i receives the sum of
        ``permutation.tokens[permutation.row_indices[i, j]]`` over all slots j.
        If probs is supplied, multiply each contribution by
        ``probs[i, permutation.expert_indices[i, j]]`` before summing.
        Permuting and immediately unpermuting without weighting therefore
        returns topk times the source features.

        Args:
            permutation: Shared operator data containing expert output features
                in the original packed row order and the corresponding inverse
                indices. It may originate from any compatible provider.
                The permutation.probs field is not used for merging.
            restore_shape: Output shape [num_tokens, hidden_size], matching
                the source-token count in row_indices and feature width in
                permutation.tokens. The caller restores any batch/sequence
                dimensions separately.
            probs: Optional dense merging weights of shape
                [num_tokens, num_experts] on the feature device. Expert IDs
                select the weight for each contribution. Omit when expert
                computation has already applied routing weights; None means
                sum the contributions without additional weighting.
            impl: Optional explicit implementation name, selected independently
                of the permutation implementation. When omitted, use the
                configured operator override or dispatch policy.

        Returns:
            Combined features with restore_shape and the dtype and device of
            permutation.tokens.
        """
        return self._dispatch(
            "moe_unpermutation",
            permutation,
            restore_shape,
            probs=probs,
            impl=impl,
        )

    def moe_sort_chunks(
        self,
        input: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor | None = None,
        *,
        impl: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reorder chunks using metadata on the input device."""
        if split_sizes.device != input.device or sorted_idxs.device != input.device:
            raise ValueError(
                "Chunk sorting requires split_sizes and sorted_idxs on the input device."
            )
        return self._dispatch(
            "moe_sort_chunks",
            input,
            split_sizes,
            sorted_idxs,
            probs=probs,
            impl=impl,
        )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if not self.registry.list(spec_filter=lambda spec: spec.op_name == name):
            raise AttributeError(name)

        def facade(*args: Any, impl: str | None = None, **kwargs: Any):
            return self._dispatch(name, *args, impl=impl, **kwargs)

        return facade


__all__ = ["OpDispatcher"]
