"""Folding GQA heads into varlen queries preserves selection and gradients."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F
import transformers.modeling_flash_attention_utils as flash_utils
from torch.utils.checkpoint import checkpoint

from torchturbo.models.qwen4_exp import qsa

from ..flash_attention_fakes import lazy_flash_attention_kernels
from .test_model import _token_ids_to_mask
from .test_qsa_attention import _cpu_varlen_kernel, _flash_kwargs


@pytest.mark.parametrize("heads_per_kv", [1, 4, 12])
@pytest.mark.parametrize("padding_only", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("checkpoint_mode", [None, False, True])
def test_folded_varlen_matches_independent_dense_reference(
    monkeypatch, heads_per_kv, padding_only, batch_size, checkpoint_mode
):
    kernels = lazy_flash_attention_kernels(None, _cpu_varlen_kernel)
    monkeypatch.setattr(flash_utils, "lazy_import_flash_attention", lambda *a, **k: (kernels, _flash_kwargs))
    monkeypatch.setattr(qsa, "_QSA_ATTENTION_MAX_KV_BYTES", batch_size * 5 * 2 * 2 * 8 * 8 * 3)
    torch.manual_seed(104)
    inputs = [torch.randn(batch_size, length, heads, 8, dtype=torch.float64, requires_grad=True)
              for length, heads in [(9, 2 * heads_per_kv), (17, 2), (17, 2)]]
    ids = torch.full((batch_size, 9, 5), -1, dtype=torch.int32)
    if not padding_only:
        for batch in range(batch_size):
            for row in range(7):
                ids[batch, row, :3] = torch.randperm(17)[:3].int()
    reference = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    function = partial(qsa.qsa_attention_forward, softmax_scale=0.23)
    actual = (
        function(*inputs, ids)
        if checkpoint_mode is None else checkpoint(
            function, *inputs, ids, use_reentrant=checkpoint_mode
        )
    )
    query, key, value = reference
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.repeat_interleave(heads_per_kv, 2).transpose(1, 2),
        value.repeat_interleave(heads_per_kv, 2).transpose(1, 2),
        attn_mask=_token_ids_to_mask(ids, 17), scale=0.23,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    for tensor, ref in zip(inputs, reference, strict=True):
        torch.testing.assert_close(tensor.grad, ref.grad)
