# Autograd wrappers for npu_alltoallv_gmm / npu_gmm_alltoallv.
#
# Adapted from MindSpeed-MM (Apache-2.0, Huawei)
#   mindspeed_mm/fsdp/ops/moe_ops/gemm_mc2.py
# Changes vs the MindSpeed-MM originals:
#  1. send_counts / recv_counts are accepted as host-side lists, pre-staged by
#     MoEAlltoAllTokenDispatcher's async dtoh pipeline at its existing
#     "before_ep_alltoall" sync point -> the fused path adds ZERO new device
#     syncs (the originals' .tolist() would be a blocking sync twice per call).
#  2. tokens-per-local-expert (group_list, device int64 [num_local_experts])
#     is passed in from preprocess instead of being re-derived here.
# Layout contract (verified by gmmfold_proto/toy_gmmfold_align.py, EP=8,
# bitwise-identical fwd/bwd vs the unfused chain):
#   send_counts[e]           = tokens this rank sends to global expert e,
#                              len num_experts (= routing_map.sum(dim=0))
#   recv_counts[s*e_loc + j] = tokens received from src rank s for local
#                              expert j, len num_experts
#   weights                  = [num_local_experts, K, N], non-transposed
#   inputs                   = [sum(send_counts), K], global-expert-major order
#                              (output of moe_token_permutation / perm1)

import torch
from torch.distributed._tensor import DTensor

try:
    import torch_npu
except ImportError:  # CPU-only import (static checks)
    torch_npu = None


def _as_counts_list(counts, name):
    if isinstance(counts, torch.Tensor):
        if counts.device.type != "cpu":
            raise RuntimeError(
                f"TT_GMM_FOLD: {name} must be pre-staged to host "
                "(dispatcher dtoh pipeline); refusing a blocking .tolist()")
        return counts.tolist()
    return list(counts)


def _pad_counts_host(counts):
    """Replace zero edge counts with one dummy row.

    The vendor fused collectives hang on zero-count send/recv edges, so every
    zero edge carries one padding row instead. Returns the padded counts, the
    row offsets of the inserted dummies in the padded tensor, and the padded
    total. All inputs are host-side lists, so this adds no device sync.
    """
    padded = []
    dummy_offsets = []
    offset = 0
    for count in counts:
        count = int(count)
        if count == 0:
            count = 1
            dummy_offsets.append(offset)
        padded.append(count)
        offset += count
    return padded, dummy_offsets, offset


def _pad_rows(inputs, dummy_offsets):
    """Insert one zero row at each padded offset; identity without dummies."""
    if not dummy_offsets:
        return inputs
    total = inputs.size(0) + len(dummy_offsets)
    padded = inputs.new_zeros((total, *inputs.shape[1:]))
    keep = torch.ones(total, dtype=torch.bool, device=inputs.device)
    keep[torch.tensor(dummy_offsets, device=inputs.device)] = False
    padded[keep] = inputs
    return padded


def _drop_rows(inputs, dummy_offsets):
    """Remove the rows inserted by _pad_rows; identity without dummies."""
    if not dummy_offsets:
        return inputs
    keep = torch.ones(inputs.size(0), dtype=torch.bool, device=inputs.device)
    keep[torch.tensor(dummy_offsets, device=inputs.device)] = False
    return inputs[keep]


def _plain(t):
    """Unwrap DTensor -> its current local tensor.

    Under torchturbo FSDP2 the expert weights stay DTensor objects whose
    underlying storage during fwd/bwd windows holds the full (unsharded)
    weight — the stock mindspeed gmm path relies on the same fact (its pybind
    kernel reads the underlying TensorImpl directly). torch_npu dispatcher
    ops instead reject mixed Tensor/DTensor inputs, so the fused ops must be
    fed the plain local tensor. Autograd still records against the DTensor
    Function input, matching the proven stock GMMFunction pattern.
    """
    if isinstance(t, DTensor):
        return t._local_tensor
    return t


def _get_hcomm(group):
    rank = torch.distributed.get_rank(group)
    global_rank = torch.distributed.get_global_rank(group, rank)
    return group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)


def _pad_group_list(group_list, recv_counts, num_local):
    """Add the per-expert dummy rows implied by the padded recv edge counts."""
    zeros_per_expert = [0] * num_local
    for index, count in enumerate(recv_counts):
        if int(count) == 0:
            zeros_per_expert[index % num_local] += 1
    if not any(zeros_per_expert):
        return group_list
    return group_list + torch.tensor(
        zeros_per_expert, dtype=group_list.dtype, device=group_list.device
    )


def padded_total(counts):
    """Total row count after zero-count edges gain one dummy row each."""
    return sum(max(int(count), 1) for count in counts)


class AllToAllGroupedMatmul(torch.autograd.Function):
    """dispatch side: alltoallv + recv-side expert-major permute + GMM (fc1).

    Zero-count send/recv edges hang the vendor fused collective, so each one
    carries one zero dummy row; the dummy rows are dropped from the returned
    gradient. Padding rules are identical on every rank, keeping the collective
    edges consistent without extra communication.
    """

    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts,
                group_list, shared_inputs, shared_weight):
        hcomm = _get_hcomm(group)
        ep_world_size = torch.distributed.get_world_size(group)
        send_counts = _as_counts_list(send_counts, "send_counts")
        recv_counts = _as_counts_list(recv_counts, "recv_counts")
        group_list = _pad_group_list(
            group_list, recv_counts, weights.shape[0]
        )
        send_counts, send_dummies, _ = _pad_counts_host(send_counts)
        recv_counts, _, _ = _pad_counts_host(recv_counts)

        weights_p = _plain(weights)
        output, shared_output, permute_output = torch_npu.npu_alltoallv_gmm(
            _pad_rows(_plain(inputs), send_dummies), weights_p, hcomm,
            ep_world_size, send_counts,
            recv_counts, mm_x=_plain(shared_inputs),
            mm_weight=_plain(shared_weight), permute_out_flag=True)

        ctx.save_for_backward(weights_p, _plain(shared_inputs),
                              _plain(shared_weight), permute_output)
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.send_dummies = send_dummies
        ctx.group_list = group_list.clone()
        return output, shared_output

    @staticmethod
    def backward(ctx, output_grad, shared_output_grad):
        weights, shared_inputs, shared_weight, permute_output = ctx.saved_tensors

        inputs_grad, shared_inputs_grad = torch_npu.npu_gmm_alltoallv(
            output_grad, weights, ctx.hcomm, ctx.ep_world_size,
            ctx.recv_counts, ctx.send_counts, mm_x=shared_output_grad,
            mm_weight=shared_weight, trans_gmm_weight=True)
        inputs_grad = _drop_rows(inputs_grad, ctx.send_dummies)

        weights_grad = torch_npu.npu_grouped_matmul(
            [permute_output.T], [output_grad], bias=None,
            group_list=ctx.group_list,
            split_item=3, group_type=2, group_list_type=1)[0]

        shared_weight_grad = None if shared_inputs is None else torch.matmul(
            shared_inputs.T, shared_output_grad)
        return (inputs_grad, weights_grad, None, None, None, None,
                shared_inputs_grad, shared_weight_grad)


class GroupedMatmulAllToAll(torch.autograd.Function):
    """combine side: GMM (fc2) + inverse permute + alltoallv back.

    send_counts/recv_counts are from the COMBINE point of view (= dispatch
    side's recv_counts/send_counts). Output rows are in the sender's pre-a2a
    order (dispatch perm1 output order), so the stock unperm1
    (moe_token_unpermutation with the reversed perm1 mapping + probs)
    applies unchanged.
    """

    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts,
                group_list, shared_inputs, shared_weight):
        hcomm = _get_hcomm(group)
        ep_world_size = torch.distributed.get_world_size(group)
        send_counts = _as_counts_list(send_counts, "send_counts")
        recv_counts = _as_counts_list(recv_counts, "recv_counts")
        group_list = _pad_group_list(
            group_list, send_counts, weights.shape[0]
        )
        send_counts, _, _ = _pad_counts_host(send_counts)
        recv_counts, recv_dummies, _ = _pad_counts_host(recv_counts)

        output, shared_output = torch_npu.npu_gmm_alltoallv(
            _plain(inputs), _plain(weights), hcomm, ep_world_size, send_counts,
            recv_counts, mm_x=_plain(shared_inputs),
            mm_weight=_plain(shared_weight))
        # Rows return in the sender's padded pre-a2a order; drop the dummies.
        output = _drop_rows(output, recv_dummies)

        ctx.save_for_backward(_plain(inputs), _plain(weights),
                              _plain(shared_inputs), _plain(shared_weight))
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.recv_dummies = recv_dummies
        ctx.group_list = group_list.clone()
        return output, shared_output

    @staticmethod
    def backward(ctx, output_grad, shared_output_grad):
        inputs, weights, shared_inputs, shared_weight = ctx.saved_tensors

        # Grad rows for the dropped dummies are zero by construction.
        output_grad = _pad_rows(output_grad, ctx.recv_dummies)
        inputs_grad, shared_inputs_grad, permute_grad = torch_npu.npu_alltoallv_gmm(
            output_grad, weights, ctx.hcomm, ctx.ep_world_size,
            ctx.recv_counts, ctx.send_counts, mm_x=shared_output_grad,
            mm_weight=shared_weight, permute_out_flag=True, trans_gmm_weight=True)

        weights_grad = torch_npu.npu_grouped_matmul(
            [inputs.T], [permute_grad], bias=None, group_list=ctx.group_list,
            split_item=3, group_type=2, group_list_type=1)[0]

        shared_weight_grad = None if shared_inputs is None else torch.matmul(
            shared_inputs.T, shared_output_grad)
        return (inputs_grad, weights_grad, None, None, None, None,
                shared_inputs_grad, shared_weight_grad)


def all2all_grouped_matmul(inputs, weights, group, send_counts, recv_counts,
                           group_list, shared_inputs=None, shared_weight=None):
    output, shared_output = AllToAllGroupedMatmul.apply(
        inputs, weights, group, send_counts, recv_counts, group_list,
        shared_inputs, shared_weight)
    if shared_inputs is not None:
        return output, shared_output
    return output


def grouped_matmul_all2all(inputs, weights, group, send_counts, recv_counts,
                           group_list, shared_inputs=None, shared_weight=None):
    output, shared_output = GroupedMatmulAllToAll.apply(
        inputs, weights, group, send_counts, recv_counts, group_list,
        shared_inputs, shared_weight)
    if shared_inputs is not None:
        return output, shared_output
    return output
