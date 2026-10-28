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
