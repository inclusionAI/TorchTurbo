import torch

from torchturbo.models.modules.comm.comm import _LazyA2ATensor, all_to_all


def test_async_all_to_all_waits_at_consumers_and_materializes_backward(monkeypatch):
    waits = []

    class Work:
        def wait(self):
            waits.append(1)

    def fake_all_to_all_single(output, input, **kwargs):
        assert kwargs["async_op"] is True
        output.copy_(input)
        return Work()

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed, "all_to_all_single", fake_all_to_all_single
    )

    input = torch.randn(4, 3, requires_grad=True)
    output = all_to_all(object(), input, async_comm=True)

    assert isinstance(output, _LazyA2ATensor)
    assert waits == []
    output_view = output.view(output.shape)
    assert waits == [1]
    output_view.sum().backward()

    assert waits == [1, 1]
    torch.testing.assert_close(input.grad, torch.ones_like(input))


def test_stock_all_to_all_does_not_use_async_collective(monkeypatch):
    def fake_all_to_all_single(output, input, **kwargs):
        assert "async_op" not in kwargs
        output.copy_(input)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed, "all_to_all_single", fake_all_to_all_single
    )

    input = torch.randn(4, 3, requires_grad=True)
    output = all_to_all(object(), input)

    assert not isinstance(output, _LazyA2ATensor)
    output.sum().backward()
    torch.testing.assert_close(input.grad, torch.ones_like(input))

