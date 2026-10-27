"""Node-local shared parameters and source-shard I/O coordination."""

from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Self

import torch
import torch.distributed as dist

from .base import CPUResidentParameter


class NodeSharedCPUResidentParameter(CPUResidentParameter):
    """A CPU parameter shared by the processes on each torchrun node.

    Construction is local and needs no distributed initialization. The first
    materialize_ call in a process creates native Gloo subgroups collectively
    across WORLD using homogeneous torchrun LOCAL_WORLD_SIZE, so every rank must
    reach that call in the same order. Later calls reuse the cached node group,
    which is shared for the lifetime of the training job. Materialization,
    loading, and saving are WORLD collective and must occur in matching order.
    Real initial values come from node rank zero; meta data allocates
    uninitialized storage. Multiprocess sharing requires frozen parameters
    until an optimizer ownership protocol exists.

    Native deepcopy creates independent values and reuses the cached node group.
    Native dtype conversions require explicit rematerialization before shared
    use. No module conversion hooks or checkpoint format handling are installed.
    Source shards are assigned to node ranks by shard index modulo node size.

    Args:
        data: Initial values, or a meta tensor declaring shape and dtype.
        requires_grad: Whether autograd records operations on the parameter.
        directory: Node-visible directory used for shared allocations.
    """

    _node_group: ClassVar[dist.ProcessGroup | None] = None
    directory: Path
    _mapping: (
        tuple[
            weakref.ReferenceType[torch.UntypedStorage],
            str,
            torch.Size,
            torch.dtype,
        ]
        | None
    )

    def __new__(
        cls,
        data: torch.Tensor | None = None,
        requires_grad: bool = True,
        *,
        directory: Path | str = "/dev/shm",
    ) -> Self:
        if data is None:
            data = torch.empty(0, device="cpu")
        elif not data.is_meta:
            data = data.cpu()
        result = super().__new__(cls, data, requires_grad)
        result.directory = Path(directory)
        result.shard_sizes = None
        result._mapping = None
        return result

    @property
    def group(self) -> dist.ProcessGroup:
        """Node-local group used to coordinate this parameter's storage and I/O.

        The group exists once any shared parameter has been materialized in
        this process.
        """
        if self._node_group is None:
            raise RuntimeError("Call materialize_ before using the node group")
        return self._node_group

    @classmethod
    def _ensure_node_group(cls) -> dist.ProcessGroup:
        """Return the cached node group, creating it collectively across WORLD."""
        if cls._node_group is not None:
            return cls._node_group
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before shared materialization"
            )
        world = dist.group.WORLD
        error = None
        try:
            num_local_ranks = int(os.environ["LOCAL_WORLD_SIZE"])
            if num_local_ranks <= 0 or dist.get_world_size() % num_local_ranks:
                raise ValueError(
                    "LOCAL_WORLD_SIZE must be positive and divide world size"
                )
        except (KeyError, ValueError) as exc:
            error = ValueError(f"Invalid LOCAL_WORLD_SIZE: {exc}")
        cls._check_errors(error, world)
        sizes = [None] * dist.get_world_size()
        dist.all_gather_object(sizes, num_local_ranks, group=world)
        if any(size != num_local_ranks for size in sizes):
            raise RuntimeError("LOCAL_WORLD_SIZE must agree across all ranks")
        cls._node_group, _ = dist.new_subgroups(
            group_size=num_local_ranks, backend="gloo"
        )
        return cls._node_group

    def __deepcopy__(self, memo: dict) -> Self:
        copied = type(self)(
            self.detach().clone(), self.requires_grad, directory=self.directory
        )
        copied.shard_sizes = self.shard_sizes
        memo[id(self)] = copied
        return copied

    def materialize_(self, *, dtype: torch.dtype | None = None) -> Self:
        """Establish node-shared CPU storage and agree on success across WORLD.

        The backing file is unlinked after all peers have mapped it.

        Args:
            dtype: Requested storage dtype. None keeps the current dtype.

        Returns:
            This parameter, preserving its identity. An unchanged common mapping
            is reused when every group member agrees on its allocation.
        """
        self._ensure_node_group()
        error = None
        try:
            self._materialize_node(dtype=dtype)
        except Exception as exc:  # noqa: BLE001 - Every node receives failures.
            error = exc
        self._check_errors(error, dist.group.WORLD)
        return self

    def _materialize_node(self, *, dtype: torch.dtype | None) -> None:
        group = self.group
        num_ranks = dist.get_world_size(group)
        token = None
        allocation_dtype = self.dtype if dtype is None else dtype
        if self._mapping_is_current() and allocation_dtype == self.dtype:
            token = self._mapping[1]
        description = (
            self.shape,
            allocation_dtype,
            self.layout,
            self.requires_grad,
            token,
        )
        descriptions = [None] * num_ranks
        dist.all_gather_object(descriptions, description, group=group)
        if any(item[:2] != description[:2] for item in descriptions):
            raise RuntimeError(
                "Shared parameter metadata must agree on shape and dtype"
            )
        if any(item[2] != torch.strided for item in descriptions):
            raise RuntimeError("Shared parameters require strided tensor storage")
        if num_ranks > 1 and any(item[3] for item in descriptions):
            raise RuntimeError(
                "Multiprocess node-shared trainable parameters require an optimizer "
                "ownership protocol; freeze them before shared materialization"
            )
        if token is not None and all(item[4] == token for item in descriptions):
            return
        data, token = self._allocate_shared(group, allocation_dtype)
        error = None
        try:
            gradient = self.grad
            self._bind(data)
            if gradient is not None:
                self.grad = gradient.to(device="cpu", dtype=allocation_dtype)
        except Exception as exc:  # noqa: BLE001 - All peers must receive failures.
            error = exc
        self._check_errors(error, group)
        self._mapping = (
            weakref.ref(self.untyped_storage()),
            token,
            self.shape,
            self.dtype,
        )

    def load_(
        self,
        shard_sizes: Sequence[int],
        reader: Callable[[int, torch.Tensor], None],
        *,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Load node-assigned source shards and agree on success across WORLD.

        Args:
            shard_sizes: Source row counts covering the parameter; scalars use [1].
            reader: Fills reader(shard_index, detached_target_view).
            dtype: Storage dtype, or None to preserve the current dtype.

        Returns:
            This parameter after every node succeeds. Failed reads preserve the
            previous source layout; partially written data is not rolled back.
        """
        self.materialize_(dtype=dtype)
        error = None
        try:
            sizes = self._run_shards(shard_sizes, reader)
        except Exception as exc:  # noqa: BLE001 - Other nodes receive reader failures.
            error = exc
        self._check_errors(error, dist.group.WORLD)
        self.shard_sizes = sizes
        return self

    def save(
        self,
        writer: Callable[[int, torch.Tensor], None],
    ) -> None:
        """Write existing shards once, using the node containing global rank 0.

        The remembered source layout is used when present; otherwise, rows are
        balanced across the saving node without recording that temporary layout.

        Args:
            writer: Called as writer(shard_index, detached_view) for this rank's
                assigned shards on the selected node. The writer owns output
                formats and file names.

        Every WORLD rank participates. Writer failures and completion are
        agreed across WORLD before return.
        """
        error = None
        try:
            _ = self.group
        except Exception as exc:  # noqa: BLE001 - All ranks validate before I/O.
            error = exc
        self._check_errors(error, dist.group.WORLD)
        error = None
        try:
            self._run_shards(self.shard_sizes, writer, for_save=True)
        except Exception as exc:  # noqa: BLE001 - Other nodes receive writer failures.
            error = exc
        self._check_errors(error, dist.group.WORLD)

    def shard_owner(self, index: int) -> int:
        """Return the node rank assigned to source shard ``index``.

        On the saving node, node ranks equal their global ranks.
        """
        return index % dist.get_world_size(self.group)

    def local_shards(
        self,
        shard_sizes: Sequence[int] | None = None,
        *,
        for_save: bool = False,
    ) -> list[tuple[int, int, torch.Tensor]]:
        """Return existing shared-storage views assigned to this node rank.

        This is a local operation; materialize_ must have established the node
        group and shared mapping beforehand.

        Args:
            shard_sizes: Explicit source row counts. None uses the remembered
                layout, or balanced row intervals across the node when unknown.
            for_save: Return no shards outside the node containing global rank 0.

        Returns:
            Source-index, row-offset, detached-view triples, preserving empty
            source intervals and source order.
        """
        if not self._mapping_is_current():
            raise RuntimeError("Call materialize_ before shared shard I/O")
        group = self.group
        if shard_sizes is None:
            shard_sizes = self.shard_sizes
        if shard_sizes is None:
            rows = self.shape[0] if self.ndim else 1
            num_shards = max(1, min(rows, dist.get_world_size(group)))
            rows_per_shard, extra_rows = divmod(rows, num_shards)
            shard_sizes = tuple(
                rows_per_shard + (index < extra_rows) for index in range(num_shards)
            )
        sizes = self.validate_shard_sizes(shard_sizes)
        if for_save and dist.get_global_rank(group, 0) != 0:
            return []
        rank = dist.get_rank(group)
        data = self.detach()
        shards = []
        start = 0
        for index, size in enumerate(sizes):
            stop = start + size
            if self.shard_owner(index) == rank:
                view = data if self.ndim == 0 else data[start:stop]
                shards.append((index, start, view))
            start = stop
        return shards

    def _run_shards(
        self,
        shard_sizes: Sequence[int] | None,
        operation: Callable[[int, torch.Tensor], None],
        *,
        for_save: bool = False,
    ) -> tuple[int, ...] | None:
        group = self.group
        error = None
        try:
            sizes = None if for_save and shard_sizes is None else tuple(shard_sizes)
            shards = self.local_shards(sizes, for_save=for_save)
        except Exception as exc:  # noqa: BLE001 - All peers validate before callbacks.
            error = exc
        self._check_errors(error, group)
        partitions = [None] * dist.get_world_size(group)
        dist.all_gather_object(partitions, sizes, group=group)
        if any(partition != sizes for partition in partitions):
            raise RuntimeError("Source shard sizes must agree within the node")
        error = None
        try:
            for index, _, view in shards:
                operation(index, view)
        except Exception as exc:  # noqa: BLE001 - Peers must receive callback failures.
            error = exc
        self._check_errors(error, group)
        return sizes

    def validate_shard_sizes(self, shard_sizes: Sequence[int]) -> tuple[int, ...]:
        sizes = tuple(shard_sizes)
        if any(not isinstance(size, int) or size < 0 for size in sizes):
            raise ValueError("Source shard sizes must be nonnegative integers")
        if self.ndim == 0:
            if sizes != (1,):
                raise ValueError("A scalar requires exactly one source shard of size 1")
        elif sum(sizes) != self.shape[0]:
            raise ValueError("Source shard sizes must cover the leading dimension")
        return sizes

    def _bind(self, data: torch.Tensor) -> None:
        if self.device == data.device:
            self.data = data
        else:
            backward_hooks = self._backward_hooks
            post_accumulate_hooks = self._post_accumulate_grad_hooks
            replacement = type(self)(data, requires_grad=self.requires_grad)
            replacement.__dict__.update(self.__dict__)
            torch.utils.swap_tensors(self, replacement)
            # Native setters preserve the original removable hook handles.
            self._backward_hooks = backward_hooks
            self._post_accumulate_grad_hooks = post_accumulate_hooks

    def _mapping_is_current(self) -> bool:
        if self._mapping is None:
            return False
        storage, _, shape, dtype = self._mapping
        return (
            storage() is self.untyped_storage()
            and self.shape == shape
            and self.dtype == dtype
            and self.is_contiguous()
            and self.storage_offset() == 0
        )

    @staticmethod
    def _check_errors(error: Exception | None, group: dist.ProcessGroup) -> None:
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(
            errors, None if error is None else str(error), group=group
        )
        for rank, message in enumerate(errors):
            if message is not None:
                raise RuntimeError(
                    f"Shared parameter operation failed on group rank {rank}: {message}"
                )

    def _allocate_shared(
        self,
        group: dist.ProcessGroup,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, str]:
        directory = self.directory
        rank = dist.get_rank(group)
        owned_path = None
        paths = [None]
        num_elements = self.numel()
        primary_error = None
        try:
            error = None
            if rank == 0:
                try:
                    num_bytes = (
                        num_elements
                        * torch.empty((), dtype=dtype, device="meta").element_size()
                    )
                    if shutil.disk_usage(directory).free < num_bytes:
                        raise OSError(
                            f"Insufficient space for shared CPU storage in {directory}"
                        )
                    descriptor, owned_path = tempfile.mkstemp(
                        prefix="torchturbo-cpu-resident-", dir=directory
                    )
                    try:
                        os.ftruncate(descriptor, num_bytes)
                    finally:
                        os.close(descriptor)
                    paths[0] = owned_path
                except Exception as exc:  # noqa: BLE001 - All peers must receive failures.
                    error = exc
            self._check_errors(error, group)
            dist.broadcast_object_list(
                paths, src=dist.get_global_rank(group, 0), group=group
            )
            path = paths[0]
            error = None
            try:
                # Peers must map the existing leader file, never create their own.
                Path(path).stat()
                data = torch.from_file(
                    path, shared=True, size=num_elements, dtype=dtype, device="cpu"
                ).view(self.shape)
            except Exception as exc:  # noqa: BLE001 - All peers must receive failures.
                error = exc
            self._check_errors(error, group)
            error = None
            if rank == 0 and not self.is_meta:
                try:
                    data.copy_(self.detach())
                except Exception as exc:  # noqa: BLE001 - All peers must receive failures.
                    error = exc
            self._check_errors(error, group)
            return data, path
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error = None
            try:
                if owned_path is not None:
                    Path(owned_path).unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - Cleanup must also agree across peers.
                cleanup_error = exc
            try:
                self._check_errors(cleanup_error, group)
            except Exception:
                if primary_error is None:
                    raise
