import torch
import torch.distributed as dist


def all_gather_tensor(local_tensor, group):
    """gather tensors"""
    world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group)

    return output


def all_gather_tensor_chunked(local_tensor, group, chunk_dim=0, num_chunks=4):
    """gather tensors with chunk"""
    world_size = dist.get_world_size(group=group)

    local_chunks = torch.tensor_split(local_tensor, num_chunks, dim=chunk_dim)

    cpu_results = []

    for chunk in local_chunks:
        chunk_output_shape = list(chunk.shape)
        chunk_output_shape[0] = chunk_output_shape[0] * world_size
        temp_gpu_output = torch.empty(
            chunk_output_shape, dtype=chunk.dtype, device=chunk.device
        )
        dist.all_gather_into_tensor(temp_gpu_output, chunk, group=group)
        cpu_results.append(temp_gpu_output.cpu())
        del temp_gpu_output

    reshaped_parts = []
    for t in cpu_results:
        reshaped_parts.append(t.view(world_size, -1, *t.shape[1:]))

    output_cpu_view = torch.cat(reshaped_parts, dim=1)
    output_cpu = output_cpu_view.flatten(0, 1)

    return output_cpu


def _gather_along_first_dim(
    input_, group=None, output_split_sizes=None, use_global_buffer=False
):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed.all_gather_into_tensor(
            output, input_.contiguous(), group=group
        )
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def _reduce_scatter_along_first_dim(
    input_, group=None, input_split_sizes=None, use_global_buffer=False
):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed.reduce_scatter_tensor(
            output, input_.contiguous(), group=group
        )
    else:
        rank = torch.distributed.get_rank(group)
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))
        output = torch.empty_like(input_tensor_list[rank])
        torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
    return output


def _split_along_first_dim(input_, group=None):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = torch.distributed.get_rank(group)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=input.device,
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(
                ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes
            ),
            None,
            None,
        )


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(
        graph,
        input_,
        tensor_parallel_output_grad=True,
        group=None,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Symbolic function for tracing."""
        return _gather_along_first_dim(
            input_, group, output_split_sizes, use_global_buffer
        )

    @staticmethod
    def forward(
        ctx,
        input_,
        tensor_parallel_output_grad=True,
        group=None,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Forward function."""
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(
            input_, group, output_split_sizes, use_global_buffer
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return (
                _reduce_scatter_along_first_dim(
                    grad_output,
                    ctx.group,
                    ctx.output_split_sizes,
                    ctx.use_global_buffer,
                ),
                None,
                None,
                None,
                None,
            )
        else:
            assert ctx.output_split_sizes is None
            return (
                _split_along_first_dim(grad_output, ctx.group),
                None,
                None,
                None,
                None,
            )


class _LazyA2ATensor(torch.Tensor):
    __slots__ = ["_elem", "_work", "_completed"]

    @staticmethod
    def __new__(cls, elem, work):
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            dtype=elem.dtype,
            device=elem.device,
            requires_grad=False,
        )
        result._elem = elem
        result._work = work
        result._completed = False
        return result

    def __tensor_flatten__(self):
        return ["_elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        if meta is not None:
            raise AssertionError("meta must be None for _LazyA2ATensor")
        return inner_tensors["_elem"]

    def trigger_wait(self):
        if not self._completed:
            self._work.wait()
            self._completed = True
        return self._elem

    def __del__(self):
        if not self._completed:
            self.trigger_wait()

    def __repr__(self):
        return f"_LazyA2ATensor({self.trigger_wait()!r})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(tensor):
            return tensor.trigger_wait()

        unwrapped_args = torch.utils._pytree.tree_map_only(
            _LazyA2ATensor, unwrap, args
        )
        unwrapped_kwargs = torch.utils._pytree.tree_map_only(
            _LazyA2ATensor, unwrap, (kwargs or {})
        )
        return func(*unwrapped_args, **unwrapped_kwargs)


class _AllToAllLazyWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=input.device,
            )
        work = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        return _LazyA2ATensor(output, work)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        if ctx.output_split_sizes is None:
            output = torch.empty_like(grad_output)
        else:
            output = grad_output.new_empty(
                size=[sum(ctx.input_split_sizes)] + list(grad_output.size()[1:]),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
        work = dist.all_to_all_single(
            output,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
            async_op=True,
        )
        grad_input = _LazyA2ATensor(output, work).trigger_wait()
        return None, grad_input, None, None


# all_to_all
def all_to_all(
    group, input_, output_split_sizes_=None, input_split_sizes=None,
    async_comm=False,
):
    """Wrapper for autograd function"""
    if async_comm and torch.distributed.get_world_size(group=group) > 1:
        return _AllToAllLazyWait.apply(
            group, input_.contiguous(), output_split_sizes_, input_split_sizes
        )
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_global_buffer=False,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _GatherFromSequenceParallelRegion.apply(
        input_,
        tensor_parallel_output_grad,
        group,
        output_split_sizes,
        use_global_buffer,
    )
