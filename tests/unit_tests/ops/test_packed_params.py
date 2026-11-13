import pytest
import torch

from torchturbo.ops.packed_params import get_packed_params


def test_padding_metadata_preserves_prefix_and_reuses_host_offsets():
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    packed = get_packed_params(cu, 6)

    assert packed.cu_seqlens_list == [0, 2, 5, 6]
    assert packed.chunk_indices_list["64"] == [0, 0, 1, 0, 2, 0]
    assert cu.tolist() == [0, 2, 5]
    assert get_packed_params(cu, 6) is packed
    assert get_packed_params(cu, 6, device=torch.device("cpu")) is packed
    assert get_packed_params(cu, 5).cu_seqlens_list == [0, 2, 5]


def test_cached_padding_metadata_tracks_boundary_mutations():
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    before = get_packed_params(cu, 6)

    cu[1] = 3
    after = get_packed_params(cu, 6)

    assert after is not before
    assert before.cu_seqlens_list == [0, 2, 5, 6]
    assert after.cu_seqlens_list == [0, 3, 5, 6]


def test_invalid_boundaries_do_not_replace_cached_metadata():
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    packed = get_packed_params(cu, 6)
    invalid_cu = torch.tensor([0, 2, 5], dtype=torch.float32)

    for _ in range(2):
        with pytest.raises(ValueError, match="cu_seqlens must be int32/int64"):
            get_packed_params(invalid_cu, 6)

    assert get_packed_params(cu, 6) is packed


def test_missing_boundaries_do_not_build_metadata():
    assert get_packed_params(None, 6) is None
    assert get_packed_params(None, 6, device=torch.device("cpu")) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_boundaries_share_metadata_on_the_target_device_and_track_mutation():
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    device = torch.device("cuda", torch.cuda.current_device())
    cpu_packed = get_packed_params(cu, 6)
    packed = get_packed_params(cu, 6, device=device)

    assert packed is not cpu_packed
    assert get_packed_params(cu, 6, device=device) is packed
    assert packed.cu_seqlens.device == device
    assert packed.cu_seqlens.cpu().tolist() == [0, 2, 5, 6]
    assert packed.cu_seqlens_list == [0, 2, 5, 6]
    for key, indices in packed.chunk_indices.items():
        assert indices.device == device
        torch.testing.assert_close(indices.cpu(), cpu_packed.chunk_indices[key])
    assert cu.device.type == "cpu"
    assert cu.tolist() == [0, 2, 5]

    cu[1] = 3
    updated = get_packed_params(cu, 6, device=device)
    assert updated is not packed
    assert get_packed_params(cu, 6, device=device) is updated
    assert packed.cu_seqlens.cpu().tolist() == [0, 2, 5, 6]
    assert updated.cu_seqlens.cpu().tolist() == [0, 3, 5, 6]
    assert updated.cu_seqlens_list == [0, 3, 5, 6]

    returned_to_cpu = get_packed_params(cu, 6)
    assert returned_to_cpu is not updated
    assert returned_to_cpu.cu_seqlens.device.type == "cpu"
    assert returned_to_cpu.cu_seqlens.tolist() == [0, 3, 5, 6]
    assert returned_to_cpu.cu_seqlens_list == updated.cu_seqlens_list


def test_inference_boundaries_do_not_require_a_mutation_counter():
    with torch.inference_mode():
        cu = torch.tensor([0, 2, 5], dtype=torch.int32)
        assert get_packed_params(cu, 6).cu_seqlens_list == [0, 2, 5, 6]
        cu[1] = 3
        assert get_packed_params(cu, 6).cu_seqlens_list == [0, 3, 5, 6]
