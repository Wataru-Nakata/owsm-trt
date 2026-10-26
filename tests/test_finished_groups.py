"""Completed-group skipping must be uniform before persistent-kernel barriers start."""

from types import SimpleNamespace

import pytest
import torch
from test_beam import CFG

from owsm_trt import megakernel, megakernel_gluon
from owsm_trt.runtime import GreedySession, capture_graph


class GroupBackend:
    supports_finished_stop = False

    def supports_finished_groups(self, rows):
        return rows in (3, 8)

    def step_fn(self, tokens, offset, k, v, ck, cv, logits, new_kv, kind="decode", stop=None):
        def run():
            lengths = torch.tensor([0, 2, 7] + [3] * (tokens.shape[0] - 3))
            if kind == "prefill":
                self.ends = tokens.shape[1] + lengths - 1
                self.computed = self.skipped = self.partial = 0
                ended = lengths == 0
            else:
                if stop is not None and bool(stop.all()):
                    self.skipped += 1
                    return
                self.computed += 1
                if stop is not None and bool(stop.any()):
                    self.partial += 1
                ended = offset >= self.ends
            logits.fill_(-10.)
            logits[:, :, 1] = 0.
            logits[ended, :, 0] = 1.
            new_kv.zero_()
        run.skips_finished = stop is not None
        return run


@pytest.mark.parametrize("batch", [3, 8])
@pytest.mark.parametrize("lazy", [False, True])
def test_partial_rows_are_computed_until_whole_group_ends(batch, lazy):
    backends = [GroupBackend(), GroupBackend()]
    sessions = [GreedySession(be, CFG, batch, 16, 0, torch.float32, torch.device("cpu"),
                             use_cuda_graph=False, buckets=[4, 8, 16], lazy_plain=lazy,
                             skip_finished=True, skip_finished_groups=bool(i))
                for i, be in enumerate(backends)]
    assert not sessions[0].skip_finished and sessions[1].skip_finished
    skipped = 0
    for primer in ([2, 3, 4], [2, 3, 4, 5, 6]):
        for limit in (10, 1, 7):
            tokens = torch.tensor([primer] * batch)
            assert sessions[0].decode(tokens, limit) == sessions[1].decode(tokens, limit)
            assert torch.equal(sessions[0].pos, sessions[1].pos)
            assert backends[0].computed == backends[1].computed + backends[1].skipped
            if limit == 10:
                assert backends[1].partial > 0
                skipped += backends[1].skipped
    # A bucket boundary may align the host check exactly with EOS for one primer.
    assert skipped > 0


@pytest.mark.parametrize("rows,failed,expected", [(1, False, False), (3, False, True),
                                                 (8, False, True), (9, False, False), (3, True, False)])
def test_group_capability_uses_actual_routing_limits(rows, failed, expected):
    kernel = SimpleNamespace(supports_finished_groups=True)
    backend = megakernel.MegaBackend.__new__(megakernel.MegaBackend)
    backend.rows_kernel = kernel, 16, 8
    backend.failed = {id(kernel)} if failed else set()
    assert backend.supports_finished_groups(rows) is expected


@pytest.mark.parametrize("rows,window,dtype,count", [(1, 1, torch.bool, 1), (3, 2, torch.bool, 3),
                                                      (3, 1, torch.int32, 3), (3, 1, torch.bool, 1)])
def test_gluon_rejects_invalid_group_stop(rows, window, dtype, count):
    kernel = megakernel_gluon.GluonDecoder.__new__(megakernel_gluon.GluonDecoder)
    kernel.m = SimpleNamespace()
    with pytest.raises(ValueError, match="finished-group"):
        kernel.step_fn(torch.zeros(rows, window, dtype=torch.long), None, None, None, None, None, None, None,
                       stop=torch.zeros(count, dtype=dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute GPU required")
@pytest.mark.parametrize("rows", [3, 4, 8])
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_gluon_partial_full_done_and_restart_exact(rows, graphs):
    from test_megakernel import CFG as BIG
    from test_megakernel import _model

    if not megakernel_gluon.available(torch.device("cuda")):
        pytest.skip("Hopper Gluon required")
    model = _model(BIG)
    kernel = megakernel_gluon.GluonDecoder(megakernel.MegaDecoder(model.decoder, torch.device("cuda")))
    stop = torch.zeros(rows, dtype=torch.bool, device="cuda")
    shape = (BIG.dec_layers, rows, BIG.dec_heads, 64, BIG.d_model // BIG.dec_heads)
    k0 = torch.randn(shape, device="cuda", dtype=torch.float16)
    v0 = torch.randn_like(k0)
    cross = (BIG.dec_layers, rows, BIG.dec_heads, BIG.enc_frames, BIG.d_model // BIG.dec_heads)
    ck = torch.randn(cross, device="cuda", dtype=torch.float16)
    cv = torch.randn_like(ck)
    tok = torch.full((rows, 1), 4, device="cuda")
    offset = torch.full((rows,), 3, device="cuda")
    runs, owners, buffers = [], [], []
    for guarded in (False, True):
        k, v = k0.clone(), v0.clone()
        logits = torch.zeros(rows, 1, BIG.vocab_size, device="cuda")
        kv = torch.zeros(BIG.dec_layers, 2, rows, BIG.dec_heads, 1, BIG.d_model // BIG.dec_heads,
                         device="cuda", dtype=torch.float16)
        run = kernel.step_fn(tok, offset, k, v, ck, cv, logits, kv, stop=stop if guarded else None)
        run()
        owners.append(run)
        buffers.append([k, v, logits, kv])
        runs.append(capture_graph(run).replay if graphs else run)
    expected = [v.clone() for v in buffers[0]]
    for flags in ([False]*rows, [True]+[False]*(rows-1), [True]*rows, [False]*rows,
                  [False]+[True]*(rows-1), [True]*rows, [False]*rows):
        buffers[1][0].copy_(k0)
        buffers[1][1].copy_(v0)
        stop.copy_(torch.tensor(flags, device="cuda"))
        watched = buffers[1] + list(owners[1].buffers)
        before = [v.clone() for v in watched] if all(flags) else None
        runs[1]()
        actual, reference = (watched, before) if all(flags) else (buffers[1], expected)
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    kernel.check()
