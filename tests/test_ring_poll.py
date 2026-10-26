"""Relaxed polling keeps an acquire before consumers use cross-CTA payloads."""

import pytest
import torch
from test_megakernel import CFG, _model

from owsm_trt import megakernel, megakernel_gluon, megakernel_ring
from owsm_trt.runtime import capture_graph

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not megakernel_gluon.available(torch.device("cuda")),
    reason="needs Hopper + Gluon",
)

if megakernel_gluon.gluon is not None:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    from owsm_trt.megakernel_gluon import _garrive, _gexit, _gwait

    @gluon.jit
    def _exchange(payload, bar, done, err, bad, P: gl.constexpr, RELAXED: gl.constexpr):
        pid = gl.program_id(0)
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
        ids = gl.arange(0, P, layout=layout)
        for step in range(64):
            gl.store(payload + pid, step * P + pid + 1)
            _garrive(bar)
            _gwait(bar, err, (2 * step + 1) * P, 1 << 20, RELAXED)
            values = gl.load(payload + ids, cache_modifier=".cg")
            total = gl.sum(values, 0)
            if total != P * (step * P + 1) + P * (P - 1) // 2:
                gl.atomic_add(bad, 1)
            # No CTA may overwrite the payload before every consumer finished reading it.
            _garrive(bar)
            _gwait(bar, err, (2 * step + 2) * P, 1 << 20, RELAXED)
        _gexit(bar, done, P)

    @gluon.jit
    def _timeout(bar, err, RELAXED: gl.constexpr):
        _gwait(bar, err, 1, 16, RELAXED)


@pytest.mark.parametrize("relaxed", [False, True])
def test_polling_timeout_still_reports_error(relaxed):
    bar, err = [torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(2)]
    _timeout[(1,)](bar, err, relaxed)
    assert int(err.item()) == 1


@pytest.mark.parametrize("programs", [4, 32, 64])
@pytest.mark.parametrize("relaxed", [False, True])
def test_polling_publication_across_ctas_and_replays(programs, relaxed):
    if torch.cuda.get_device_properties("cuda").multi_processor_count < programs:
        pytest.skip("all persistent CTAs must be resident")
    payload = torch.zeros(programs, device="cuda", dtype=torch.int32)
    bar, done, err, bad = [torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(4)]
    graph = capture_graph(lambda: _exchange[(programs,)](payload, bar, done, err, bad, programs, relaxed))
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    assert (int(bar.item()), int(done.item()), int(err.item()), int(bad.item())) == (0, 0, 0, 0)


@pytest.fixture(scope="module")
def decoder():
    return _model(CFG).decoder


@pytest.mark.parametrize("window", [1, 8, 32])
@pytest.mark.parametrize("solo", [False, True])
def test_ring_poll_preserves_logits_cache_and_graph(decoder, window, solo):
    dev = torch.device("cuda")
    layers, heads, size, cache, frames = CFG.dec_layers, CFG.dec_heads, 64, 128, CFG.enc_frames
    torch.manual_seed(91 + window)
    kw = dict(device=dev, dtype=torch.float16)
    tokens = torch.randint(CFG.vocab_size, (1, window), device=dev)
    offset = torch.tensor([7], device=dev)
    initial = [torch.randn(layers, 1, heads, cache, size, **kw) for _ in range(2)]
    cross = [torch.randn(layers, 1, heads, frames, size, **kw) for _ in range(2)]
    mega = megakernel.MegaDecoder(decoder, dev)
    expected = None
    for relaxed in (False, True):
        ring = megakernel_ring.RingDecoder(mega, relaxed_poll=relaxed)
        k, v = [x.clone() for x in initial]
        logits = torch.zeros(1, window, CFG.vocab_size, device=dev)
        new = torch.zeros(layers, 2, 1, heads, window, size, **kw)
        pred = torch.full((1, window), -1, dtype=torch.long, device=dev)
        step = ring.step_fn(tokens, offset, k, v, *cross, logits, new, pred=pred, solo=solo)
        assert step.const["RELAXED_POLL"] == relaxed
        graph = capture_graph(step)
        actual = []
        for pos in (0, 7, 63, 7):
            offset.fill_(pos)
            k.copy_(initial[0])
            v.copy_(initial[1])
            for _ in range(3):
                graph.replay()
            actual.append([x.clone() for x in (logits, new, k, v, pred)])
        ring.check()
        if expected is None:
            expected = actual
        else:
            for values, reference in zip(actual, expected):
                for a, b in zip(values, reference):
                    assert torch.equal(a, b)
