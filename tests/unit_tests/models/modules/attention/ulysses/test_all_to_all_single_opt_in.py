from contextlib import contextmanager

import torch

from torchturbo.models.modules.attention.ulysses import ulysses


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _received_blocks():
    return [
        torch.full((1, 3, 2, 2), 10.0),
        torch.full((1, 3, 2, 2), 20.0),
    ]


def test_all_to_all_single_falls_back_for_batch_greater_than_one(monkeypatch):
    calls = []
    received = [
        torch.full((2, 3, 2, 2), 10.0),
        torch.full((2, 3, 2, 2), 20.0),
    ]

    def fake_all_to_all(output_list, input_list, **kwargs):
        calls.append("list")
        assert kwargs["async_op"] is True
        for output, value in zip(output_list, received):
            output.copy_(value)
        return _Work()

    def unexpected_all_to_all_single(*args, **kwargs):
        raise AssertionError("default path must not call all_to_all_single")

    monkeypatch.setattr(ulysses.dist, "all_to_all", fake_all_to_all)
    monkeypatch.setattr(
        ulysses.dist, "all_to_all_single", unexpected_all_to_all_single
    )

    output = ulysses.all_to_all_tensor(
        torch.zeros(2, 3, 4, 2),
        scatter_dim=2,
        gather_dim=1,
        sp_group=object(),
        sp_size=2,
    )

    assert calls == ["list"]
    torch.testing.assert_close(output, torch.cat(received, dim=1))


def test_batch_one_all_to_all_single_is_selected_automatically(monkeypatch):
    calls = []
    received = _received_blocks()

    def unexpected_all_to_all(*args, **kwargs):
        raise AssertionError("opt-in path must not call list all_to_all")

    def fake_all_to_all_single(output, input, **kwargs):
        calls.append("single")
        assert kwargs["async_op"] is True
        output.copy_(torch.stack(received))
        return _Work()

    monkeypatch.setattr(ulysses.dist, "all_to_all", unexpected_all_to_all)
    monkeypatch.setattr(ulysses.dist, "all_to_all_single", fake_all_to_all_single)

    output = ulysses.all_to_all_tensor(
        torch.zeros(1, 3, 4, 2),
        scatter_dim=2,
        gather_dim=1,
        sp_group=object(),
        sp_size=2,
    )

    assert calls == ["single"]
    torch.testing.assert_close(output, torch.cat(received, dim=1))


def test_batch_one_all_to_all_single_waits(monkeypatch):
    work = _Work()
    received = _received_blocks()

    def fake_all_to_all_single(output, input, **kwargs):
        assert kwargs["async_op"] is True
        output.copy_(torch.stack(received))
        return work

    monkeypatch.setattr(ulysses.dist, "all_to_all_single", fake_all_to_all_single)

    output = ulysses.all_to_all_tensor(
        torch.zeros(1, 3, 4, 2),
        scatter_dim=2,
        gather_dim=1,
        sp_group=object(),
        sp_size=2,
    )

    assert work.waited
    torch.testing.assert_close(output, torch.cat(received, dim=1))


def test_batch_one_all_to_all_single_uses_side_stream(monkeypatch):
    events = []
    received = _received_blocks()

    class _Stream:
        def wait_stream(self, stream):
            events.append(("wait_stream", stream))

    class _Runtime:
        def current_stream(self):
            return "current"

        @contextmanager
        def stream(self, stream):
            events.append(("enter", stream))
            yield
            events.append(("exit", stream))

    stream = _Stream()

    def fake_all_to_all_single(output, input, **kwargs):
        events.append(("collective", stream))
        output.copy_(torch.stack(received))
        return _Work()

    def fake_record_stream(tensor, recorded_stream):
        events.append(("record_stream", recorded_stream))

    monkeypatch.setattr(ulysses, "get_device_runtime", _Runtime)
    monkeypatch.setattr(ulysses.dist, "all_to_all_single", fake_all_to_all_single)
    monkeypatch.setattr(torch.Tensor, "record_stream", fake_record_stream)

    output = ulysses.all_to_all_tensor(
        torch.zeros(1, 3, 4, 2),
        scatter_dim=2,
        gather_dim=1,
        sp_group=object(),
        sp_size=2,
        a2a_stream=stream,
    )

    assert events == [
        ("wait_stream", "current"),
        ("enter", stream),
        ("collective", stream),
        ("exit", stream),
        ("record_stream", stream),
        ("record_stream", stream),
    ]
    torch.testing.assert_close(output, torch.cat(received, dim=1))


def test_all_to_all_single_supports_reverse_dimensions(monkeypatch):
    received = [
        torch.full((1, 2, 3, 2), 10.0),
        torch.full((1, 2, 3, 2), 20.0),
    ]

    def fake_all_to_all_single(output, input, **kwargs):
        output.copy_(torch.stack(received))
        return _Work()

    monkeypatch.setattr(ulysses.dist, "all_to_all_single", fake_all_to_all_single)

    output = ulysses.all_to_all_tensor(
        torch.zeros(1, 4, 3, 2),
        scatter_dim=1,
        gather_dim=2,
        sp_group=object(),
        sp_size=2,
    )

    torch.testing.assert_close(output, torch.cat(received, dim=2))


def test_seq_all_to_all_routes_backward_through_same_wrapper(monkeypatch):
    calls = []

    def fake_all_to_all_tensor(value, scatter_dim, gather_dim, *args, **kwargs):
        calls.append((scatter_dim, gather_dim))
        return value.clone()

    monkeypatch.setattr(ulysses, "all_to_all_tensor", fake_all_to_all_tensor)
    x = torch.randn(1, 2, 2, requires_grad=True)

    output = ulysses.SeqAllToAll.apply(x, object(), 2, 2, 1, None)
    output.sum().backward()

    assert calls == [(2, 1), (1, 2)]
    torch.testing.assert_close(x.grad, torch.ones_like(x))
