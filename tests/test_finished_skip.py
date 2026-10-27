"""A uniform completed-session guard must preserve active computations and survive graph reuse."""

from types import SimpleNamespace

import pytest
import torch
from test_beam import CFG

from owsm_trt import megakernel, megakernel_ring
from owsm_trt.runtime import GreedySession, capture_graph


class ScriptedStopBackend:
    supports_finished_stop = True

    def __init__(self, length):
        self.length = length
        self.computed = self.skipped = 0

    def step_fn(self, tokens, offset, k, v, ck, cv, logits, new_kv, kind="decode", stop=None):
        def run():
            if kind == "prefill":
                self.end = tokens.shape[1] + self.length - 1
                self.computed = self.skipped = 0
                eos = self.length == 0
            else:
                if stop is not None and bool(stop):
                    self.skipped += 1
                    return
                self.computed += 1
                eos = int(offset[0]) >= self.end
            logits.fill_(-10.)
            logits[..., 0 if eos else 1] = 0.
            new_kv.zero_()
        run.skips_finished = stop is not None
        return run


@pytest.mark.parametrize("length", [0, 1, 2, 7])
@pytest.mark.parametrize("lazy", [False, True])
def test_plain_stop_keeps_eos_rules_and_reuse(length, lazy):
    backends = [ScriptedStopBackend(length) for _ in range(2)]
    sessions = [GreedySession(be, CFG, 1, 16, 0, torch.float32, torch.device("cpu"),
                             use_cuda_graph=False, buckets=[4, 8, 16], lazy_plain=lazy, skip_finished=bool(i))
                for i, be in enumerate(backends)]
    assert not sessions[0].skip_finished and sessions[1].skip_finished
    assert sessions[1].done is sessions[1]._plain_stop
    for primer in ([2, 3, 4], [2, 3, 4, 5, 6]):
        for limit in (10, 1, 7):
            tokens = torch.tensor([primer])
            expected = sessions[0].decode(tokens, limit)
            actual = sessions[1].decode(tokens, limit)
            assert actual == expected
            assert torch.equal(sessions[0].pos, sessions[1].pos)
            assert backends[0].computed == backends[1].computed + backends[1].skipped
            if length == 0 and limit > 1:
                assert backends[1].computed == 0 and backends[1].skipped > 0


@pytest.mark.parametrize("supported,batch", [(False, 1), (True, 3)])
def test_plain_stop_falls_back_without_single_row_capability(supported, batch):
    be = ScriptedStopBackend(2)
    be.supports_finished_stop = supported
    session = GreedySession(be, CFG, batch, 16, 0, torch.float32, torch.device("cpu"),
                            use_cuda_graph=False, buckets=[4, 8, 16], skip_finished=True)
    assert not session.skip_finished
    session.decode(torch.tensor([[2, 3, 4]] * batch), 10)
    assert be.skipped == 0


@pytest.mark.parametrize("rows,window,dtype,count", [(2, 1, torch.bool, 1), (1, 2, torch.bool, 1),
                                                      (1, 1, torch.int32, 1), (1, 1, torch.bool, 2)])
def test_ring_rejects_nonuniform_stop_contract(rows, window, dtype, count):
    ring = megakernel_ring.RingDecoder.__new__(megakernel_ring.RingDecoder)
    ring.m = SimpleNamespace()
    with pytest.raises(ValueError, match="finished-stop"):
        ring.step_fn(torch.zeros(rows, window, dtype=torch.long), None, None, None, None, None, None, None,
                     stop=torch.zeros(count, dtype=dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute GPU required")
@pytest.mark.parametrize("argmax", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_ring_stop_all_ctas_and_restart_exact(argmax, graphs):
    from test_megakernel import CFG as BIG
    from test_megakernel import _model

    if not megakernel_ring.available(torch.device("cuda")):
        pytest.skip("Hopper Gluon required")
    network = _model(BIG)
    ring = megakernel_ring.RingDecoder(megakernel.MegaDecoder(network.decoder, torch.device("cuda")))
    stop = torch.zeros(1, dtype=torch.bool, device="cuda")
    shape = (BIG.dec_layers, 1, BIG.dec_heads, 64, BIG.d_model // BIG.dec_heads)
    k0 = torch.randn(shape, device="cuda", dtype=torch.float16)
    v0 = torch.randn_like(k0)
    cross = (BIG.dec_layers, 1, BIG.dec_heads, BIG.enc_frames, BIG.d_model // BIG.dec_heads)
    ck = torch.randn(cross, device="cuda", dtype=torch.float16)
    cv = torch.randn_like(ck)
    tok = torch.tensor([[4]], device="cuda")
    offset = torch.tensor([3], device="cuda")
    runs, owners, buffers = [], [], []
    for guarded in (False, True):
        k, v = k0.clone(), v0.clone()
        logits = torch.zeros(1, 1, BIG.vocab_size, device="cuda")
        kv = torch.zeros(BIG.dec_layers, 2, 1, BIG.dec_heads, 1, BIG.d_model // BIG.dec_heads,
                         device="cuda", dtype=torch.float16)
        pred = torch.zeros(1, 1, dtype=torch.long, device="cuda") if argmax else None
        run = ring.step_fn(tok, offset, k, v, ck, cv, logits, kv, pred=pred, stop=stop if guarded else None)
        run()
        owners.append(run)  # Keep context/workspaces bound into a graph alive through every replay.
        buffers.append([k, v, logits, kv] + ([pred] if pred is not None else []))
        runs.append(capture_graph(run).replay if graphs else run)
    for run, values in zip(runs, buffers):
        values[0].copy_(k0)
        values[1].copy_(v0)
        run()
    expected = [v.clone() for v in buffers[0]]
    for left, right in zip(buffers[1], expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for _ in range(3):
        stop.fill_(True)
        watched = buffers[1] + list(owners[1].buffers)
        previous = [v.clone() for v in watched]
        runs[1]()
        for left, right in zip(watched, previous):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        stop.zero_()
        buffers[1][0].copy_(k0)
        buffers[1][1].copy_(v0)
        runs[1]()
        for left, right in zip(buffers[1], expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    ring.check()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_public_factory_forwards_finished_skip_and_rollback(enabled, batch):
    from owsm_trt.pipeline import OWSM

    tokenizer = SimpleNamespace(first_time=None, notime=None, eos=0, punctuation=frozenset())
    model = OWSM(CFG, tokenizer, None, ScriptedStopBackend(2),
                 SimpleNamespace(dtype="float32", capacity=16), torch.device("cpu"),
                 use_cuda_graph=False, ctc_draft=False, greedy_kernels=enabled)
    session = model._session(batch)
    assert session.skip_finished is (enabled and batch == 1)
    assert model._session(batch) is session
    assert session.decode(torch.tensor([[2, 3, 4]] * batch), 10) == [[1, 1]] * batch
    assert all(bool(getattr(fn, "skips_finished", False)) is (enabled and batch == 1)
               for fn in session._steps.values())
