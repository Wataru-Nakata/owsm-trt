"""A failed reader/callback must not leave work using the next request's buffers."""

from types import SimpleNamespace

import pytest
import torch

from owsm_trt.runtime import GreedySession


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("priority", [False, True])
@pytest.mark.parametrize("phase", ["launch", "meanwhile", "finished"])
def test_failed_stream_drains_pending_work_and_can_be_reused(device, priority, phase):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("compute allocation required")
    cuda = device == "cuda"
    events = []
    # Keep the fake encoder/decoder buffers alive to distinguish unfinished work from allocation bugs.
    buffers = []
    error = OSError("injected streaming read/callback failure")
    session = SimpleNamespace(
        device=torch.device(device), windows=(4,), capacity=16, rules=None, rows=1,
        _stage=[(), ()], _draft_rule_mode=lambda primers: None,
    )

    def pending_work():
        if cuda:
            value = torch.zeros(1, device=device)
            torch.cuda._sleep(30_000_000)
            value.fill_(7)
            event = torch.cuda.Event()
            event.record()
            events.append(event)
            buffers.append(value)

    def batches():
        if phase != "launch":
            yield None
            yield None
        pending_work()  # encoder-side work before a reader throws
        raise error

    def finished(index, tokens):
        raise error

    def drafted(rows, refill, bucket, limit, insertions, give_up, meanwhile, callback, concurrent):
        pending_work()  # decoder-side work before meanwhile/callback throws
        if phase == "meanwhile":
            meanwhile()
        else:
            callback(0, [])
        raise AssertionError("expected the original exception")

    session._drafted = drafted
    try:
        with pytest.raises(OSError) as caught:
            GreedySession.decode_stream(session, [2], batches(), lambda ids: [], 8,
                                       priority=priority, finished=finished)
        assert caught.value is error
        assert all(event.query() for event in events), "GPU work outlived the failed request"
        assert all(value.item() == 7 for value in buffers)
        session._drafted = lambda *args: ({}, 0)
        assert GreedySession.decode_stream(session, [2], iter(()), lambda ids: [], 8,
                                           priority=priority) == ([], 0)
    finally:
        if cuda:
            torch.cuda.synchronize()
