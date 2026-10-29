from types import SimpleNamespace

import torch

from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.runtime import TurboDevice


def make_dispatcher(enabled=True):
    dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = SimpleNamespace(moe_gmm_fold=enabled)
    dispatcher.num_experts = 512
    dispatcher.num_local_experts = 4
    dispatcher.tp_size = 1
    dispatcher.ep_size = 128
    dispatcher.device_runtime = SimpleNamespace(turbo_device=TurboDevice.NPU)
    dispatcher._gf_Q = None
    return dispatcher


def test_gmm_fold_supports_ep128_with_half_calls():
    dispatcher = make_dispatcher()

    assert dispatcher.gmm_fold_active is True


def test_gmm_fold_half_call_permutation_is_inverse():
    dispatcher = make_dispatcher()

    q_indices, p_indices = dispatcher._gf_perm_tables(torch.device("cpu"))

    torch.testing.assert_close(q_indices[p_indices], torch.arange(512))


def test_gmm_fold_is_disabled_by_default():
    dispatcher = make_dispatcher(enabled=False)

    assert dispatcher.gmm_fold_active is False


def test_pad_counts_host_replaces_zero_edges():
    from torchturbo.models.modules.moe.gmm_fold import _pad_counts_host

    padded, dummies, total = _pad_counts_host([3, 0, 2, 0, 1])

    assert padded == [3, 1, 2, 1, 1]
    assert dummies == [3, 6]
    assert total == 8


def test_pad_rows_and_drop_rows_are_inverse():
    from torchturbo.models.modules.moe.gmm_fold import (
        _drop_rows,
        _pad_counts_host,
        _pad_rows,
    )

    inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    _, dummies, _ = _pad_counts_host([2, 0, 3, 0, 1])

    padded = _pad_rows(inputs, dummies)
    assert padded.shape == (8, 2)
    assert padded[2].abs().sum().item() == 0
    assert padded[6].abs().sum().item() == 0
    torch.testing.assert_close(_drop_rows(padded, dummies), inputs)

    torch.testing.assert_close(_pad_rows(inputs, []), inputs)
    torch.testing.assert_close(_drop_rows(inputs, []), inputs)


def test_pad_group_list_counts_zero_edges_per_expert():
    from torchturbo.models.modules.moe.gmm_fold import _pad_group_list

    # recv layout [ep_size=2, num_local=2]: expert 0 has one zero edge,
    # expert 1 has two.
    group_list = torch.tensor([5, 7])
    padded = _pad_group_list(group_list, [5, 0, 0, 0], 2)
    torch.testing.assert_close(padded, torch.tensor([6, 9]))

    untouched = _pad_group_list(group_list, [5, 1, 2, 7], 2)
    assert untouched is group_list


def test_padded_total():
    from torchturbo.models.modules.moe.gmm_fold import padded_total

    assert padded_total([3, 0, 2, 0, 1]) == 8
    assert padded_total([3, 2, 1]) == 6
