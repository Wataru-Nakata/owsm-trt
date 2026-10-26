"""The decoder megakernel against Decoder.step and in the decoding sessions (CUDA + Triton only)."""

import copy
import dataclasses

import pytest
import torch
from test_model import TINY

from owsm_trt import megakernel, megakernel_gluon, megakernel_ring
from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import GreedySession, TorchBackend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not megakernel.available(torch.device("cuda")),
    reason="needs CUDA + Triton",
)

# head size 64 as in every OWSM model (tl.dot needs >= 16)
CFG = dataclasses.replace(
    TINY, d_model=256, n_heads=4, dec_heads=4, dec_ffn=512, dec_layers=3, vocab_size=1000
)


# a width that is not a power of 2 (as owsm_v4_base / small: 384 / 768)
CFG384 = dataclasses.replace(CFG, d_model=384, n_heads=6, dec_heads=6, dec_ffn=768, dec_layers=2)


def _model(cfg) -> OWSMModel:
    torch.manual_seed(0)
    m = OWSMModel(cfg).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    for name, p in m.decoder.named_parameters():  # non-trivial LayerNorms and biases
        if "norm" in name:
            p.add_(0.2 * torch.randn_like(p))
        elif name.endswith("bias"):
            p.copy_(0.2 * torch.randn_like(p))
    m.decoder.output_layer.weight.mul_(8.0)  # clear argmax margins
    m.fuse_projections()
    return m.cuda()


@pytest.fixture(scope="module")
def model() -> OWSMModel:
    return _model(CFG)


@pytest.fixture(scope="module")
def model384() -> OWSMModel:
    return _model(CFG384)


@pytest.mark.parametrize("rows,window", [(1, 1), (2, 5), (1, 32), (4, 16), (16, 16), (16, 1)])
def test_step_matches_decoder(model, rows, window):
    dec = model.decoder
    L, H, dk, C, T = CFG.dec_layers, CFG.dec_heads, CFG.d_model // CFG.dec_heads, 64, CFG.enc_frames
    torch.manual_seed(rows * 100 + window)
    dev = torch.device("cuda")
    tokens = torch.randint(0, CFG.vocab_size, (rows, window), device=dev)
    offset = torch.randint(0, C - window, (rows,), device=dev)
    offset[0] = 0  # an empty cache
    f16 = {"device": dev, "dtype": torch.float16}
    k, v = torch.randn(L, rows, H, C, dk, **f16), torch.randn(L, rows, H, C, dk, **f16)
    ck, cv = torch.randn(L, rows, H, T, dk, **f16), torch.randn(L, rows, H, T, dk, **f16)
    ref, ref_kv = dec.step(
        tokens, offset, list(k.float()), list(v.float()), list(ck.float()), list(cv.float())
    )
    logits = torch.zeros(rows, window, CFG.vocab_size, device=dev)
    new_kv = torch.zeros(L, 2, rows, H, window, dk, **f16)
    k2, v2 = k.clone(), v.clone()
    mk = megakernel.MegaDecoder(dec, dev)
    run = mk.step_fn(tokens, offset, k2, v2, ck, cv, logits, new_kv)
    assert run.writes_cache
    run()
    mk.check()
    scale = ref.abs().max().item()
    assert (logits - ref).abs().max().item() < 0.02 * scale
    assert (logits.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.95
    torch.testing.assert_close(new_kv.float(), ref_kv, atol=0.05, rtol=0.02)
    slots = (offset[:, None] + torch.arange(window, device=dev))[None, :, None, :, None].expand(
        L, rows, H, window, dk
    )
    k.scatter_(3, slots, new_kv[:, 0])
    v.scatter_(3, slots, new_kv[:, 1])
    assert torch.equal(k2, k) and torch.equal(v2, v)  # the new keys/values at each row's offset, nothing else


def _session(backend, batch, windows=()):
    return GreedySession(
        backend, CFG, batch, 64, eos=1, dtype=torch.float16, device=torch.device("cuda"), buckets=[16, 32, 64],
        draft_windows=windows,
    )  # fmt: skip


@pytest.mark.parametrize("drafted", [False, True], ids=["steps", "drafted"])
def test_greedy_session_matches_torch(model, drafted):
    """Greedy decoding with the megakernel for decoder calls gives the PyTorch backend's output."""
    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    torch_be = TorchBackend(m16)
    mega_be = megakernel.MegaBackend(
        TorchBackend(m16), megakernel.MegaDecoder(model.decoder, torch.device("cuda"))
    )
    batch = 3
    feats = m16.frontend(torch.randn(batch, CFG.num_samples, device="cuda"))
    primer = torch.tensor([[2, 3, 4]] * batch, device="cuda")
    outs = []
    for be in (torch_be, mega_be):
        sess = _session(be, batch, (2, 4, 8) if drafted else ())
        sess.encode(feats)
        if drafted:
            drafts = [[5, 6, 7, 8, 9, 10], [11, 12], []]
            out, _ = sess.decode_drafted(primer, drafts, 20)
        else:
            out = sess.decode(primer, 20)
        outs.append(out)
    assert outs[0] == outs[1]


gluon_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not megakernel_gluon.available(torch.device("cuda")),
    reason="needs Hopper + Gluon",
)


@gluon_only
@pytest.mark.parametrize(
    "rows,window", [(1, 1), (1, 8), (2, 3), (8, 1), (8, 4), (1, 16), (1, 32), (4, 8), (2, 16)]
)
def test_gluon_step_matches_decoder(model, rows, window, kind="gluon", programs=None, chunk=2, warps=4, cfg=CFG,
                                    solo=False):  # fmt: skip
    dec = model.decoder
    L, H, dk, C, T = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads, 128, cfg.enc_frames
    torch.manual_seed(rows * 100 + window)
    dev = torch.device("cuda")
    tokens = torch.randint(0, cfg.vocab_size, (rows, window), device=dev)
    offset = torch.randint(0, C - window, (rows,), device=dev)
    offset[0] = 0  # an empty cache
    f16 = {"device": dev, "dtype": torch.float16}
    k, v = torch.randn(L, rows, H, C, dk, **f16), torch.randn(L, rows, H, C, dk, **f16)
    ck, cv = torch.randn(L, rows, H, T, dk, **f16), torch.randn(L, rows, H, T, dk, **f16)
    ref, ref_kv = dec.step(
        tokens, offset, list(k.float()), list(v.float()), list(ck.float()), list(cv.float())
    )
    logits = torch.zeros(rows, window, cfg.vocab_size, device=dev)
    new_kv = torch.zeros(L, 2, rows, H, window, dk, **f16)
    k2, v2 = k.clone(), v.clone()
    mega = megakernel.MegaDecoder(dec, dev)
    if kind == "ring":
        gk = megakernel_ring.RingDecoder(
            mega, programs=programs, row_programs=programs, chunk=chunk, num_warps=warps
        )
    else:
        gk = megakernel_gluon.GluonDecoder(mega)
    pred = torch.full((rows, window), -1, dtype=torch.long, device=dev)
    if kind == "ring":  # with the argmax
        run = gk.step_fn(tokens, offset, k2, v2, ck, cv, logits, new_kv, pred=pred, solo=solo)
        assert run.writes_pred
        assert run.programs == (gk.P_rows if rows > 1 else gk.P_solo if solo else gk.P)
    else:
        run = gk.step_fn(tokens, offset, k2, v2, ck, cv, logits, new_kv)
    run()
    gk.check()
    if kind == "ring":
        assert torch.equal(pred, logits.argmax(-1))
    assert (logits - ref).abs().max().item() < 0.02 * ref.abs().max().item()
    assert (logits.argmax(-1) == ref.argmax(-1)).float().mean().item() >= 0.95
    torch.testing.assert_close(new_kv.float(), ref_kv, atol=0.05, rtol=0.02)
    slots = (offset[:, None] + torch.arange(window, device=dev))[None, :, None, :, None].expand(
        L, rows, H, window, dk
    )
    k.scatter_(3, slots, new_kv[:, 0])
    v.scatter_(3, slots, new_kv[:, 1])
    assert torch.equal(k2, k) and torch.equal(v2, v)


@gluon_only
@pytest.mark.parametrize("rows,window", [(1, 1), (1, 8), (2, 3), (8, 1), (1, 16), (1, 32), (4, 8), (2, 16)])
@pytest.mark.parametrize(
    "programs,chunk,warps", [(None, 2, 4), (3, 2, 4), (None, 1, 4), (3, 1, 8), (None, 2, 8)]
)
def test_ring_step_matches_decoder(model, rows, window, programs, chunk, warps):
    """The ring kernel, with as many programs as the largest stage has units or with fewer (several units per
    program and stage), ring slots of 1 or 2 weight tiles, units of 64 or 128 rows (4 or 8 warps)."""
    test_gluon_step_matches_decoder(model, rows, window, "ring", programs, chunk, warps)


@gluon_only
@pytest.mark.parametrize("rows,window", [(1, 1), (1, 8), (1, 32), (2, 3),
                                         (3, 1), (5, 1), (8, 1), (8, 4)])
@pytest.mark.parametrize("solo", [False, True])
def test_ring_step_other_widths(model384, rows, window, solo):
    """A model width that is not a power of 2 (statistics in 6 parts, 6 K tiles per unit) and the solo calls
    (more programs)."""
    test_gluon_step_matches_decoder(model384, rows, window, "ring", cfg=CFG384, solo=solo)


@gluon_only
@pytest.mark.parametrize("window", [1, 8, 32])
def test_ring_solo_is_bitwise_the_same(model, window):
    """A single-row call gives the same logits with or without the encoder alongside (more programs then): the
    outputs do not depend on timing."""
    dec, dev = model.decoder, torch.device("cuda")
    L, H, dk, C, T = CFG.dec_layers, CFG.dec_heads, CFG.d_model // CFG.dec_heads, 128, CFG.enc_frames
    torch.manual_seed(window)
    f16 = {"device": dev, "dtype": torch.float16}
    tokens = torch.randint(0, CFG.vocab_size, (1, window), device=dev)
    offset = torch.tensor([7], device=dev)
    ck, cv = torch.randn(L, 1, H, T, dk, **f16), torch.randn(L, 1, H, T, dk, **f16)
    ring = megakernel_ring.RingDecoder(megakernel.MegaDecoder(dec, dev))
    outs = []
    for solo in (False, True):
        k, v = torch.zeros(L, 1, H, C, dk, **f16), torch.zeros(L, 1, H, C, dk, **f16)
        logits = torch.zeros(1, window, CFG.vocab_size, device=dev)
        new_kv = torch.zeros(L, 2, 1, H, window, dk, **f16)
        run = ring.step_fn(tokens, offset, k, v, ck, cv, logits, new_kv, solo=solo)
        run()
        outs.append(logits)
    assert ring.P != ring.P_solo
    assert torch.equal(outs[0], outs[1])


@gluon_only
def test_backend_trial_falls_back(model):
    """A kernel whose call fails (here: raises when run) leaves the wrapped backend's call, for that call and the
    later ones; a kernel that does not fit the model is refused when made."""

    class Broken:
        name, argmax = "broken", False

        def step_fn(self, *args, **kw):
            def run():
                raise RuntimeError("compile error")

            return run

        def check(self):
            pass

    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    inner = TorchBackend(m16)
    be = megakernel.MegaBackend(inner, Broken(), 32, 8, trial=True)
    sess = _session(be, 1, (2, 4))
    sess.encode(m16.frontend(torch.randn(1, CFG.num_samples, device="cuda")))
    out, _ = sess.decode_drafted(torch.tensor([[2, 3, 4]], device="cuda"), [[5, 6]], 8)
    assert be.failed and len(out) == 1
    with pytest.raises(ValueError):
        megakernel_gluon.GluonDecoder(megakernel.MegaDecoder(_model(CFG384).decoder, torch.device("cuda")))


@gluon_only
def test_ring_stream_matches_torch(model):
    """b1 decode_stream on the ring / Gluon backend (single-row calls with the encoder alongside, then without)
    gives the PyTorch backend's tokens."""
    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    mega = megakernel.MegaDecoder(model.decoder, torch.device("cuda"))
    ring = megakernel_ring.RingDecoder(mega)
    assert ring.solo_differs
    be = megakernel.MegaBackend(TorchBackend(m16), ring, 32, 8, rows_kernel=(megakernel_gluon.GluonDecoder(mega), 16, 8),
                                trial=True)  # fmt: skip
    torch.manual_seed(3)
    n = 4
    feats = m16.frontend(torch.randn(n, CFG.num_samples, device="cuda"))
    drafts = [[5, 6, 7, 8, 9, 10], [11, 12], [], [13, 14, 15]]
    outs = []
    for b in (TorchBackend(m16), be):
        sess = _session(b, 1, (2, 4, 8))
        if b is be:
            assert sess._verify_solo
        given = iter(drafts)
        batches = (feats[i : i + 1] for i in range(n))
        out, _ = sess.decode_stream([2, 3, 4], batches, lambda paths: [next(given) for _ in paths], 20)
        outs.append(out)
    assert not be.failed
    assert outs[0] == outs[1]


@gluon_only
@pytest.mark.parametrize("drafted", [False, True], ids=["steps", "drafted"])
@pytest.mark.parametrize("kind,batch", [("gluon", 3), ("ring", 3), ("hybrid", 1), ("hybrid", 3)])
def test_gluon_session_matches_torch(model, drafted, kind, batch):
    """Greedy decoding with a Gluon kernel for calls of up to 32 tokens (the PyTorch backend for the rest);
    hybrid: the ring kernel for single-row calls, the Gluon kernel for the others (as pipeline's "ring")."""
    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    mega = megakernel.MegaDecoder(model.decoder, torch.device("cuda"))
    if kind == "gluon":
        mega_be = megakernel.MegaBackend(
            TorchBackend(m16), megakernel_gluon.GluonDecoder(mega), megakernel_gluon.MAX_TOKENS
        )
    else:
        rows = (megakernel_gluon.GluonDecoder(mega), 16, 8) if kind == "hybrid" else None
        mega_be = megakernel.MegaBackend(TorchBackend(m16), megakernel_ring.RingDecoder(mega), megakernel_ring.MAX_TOKENS,
                                         rows_kernel=rows)  # fmt: skip
    feats = m16.frontend(torch.randn(batch, CFG.num_samples, device="cuda"))
    primer = torch.tensor([[2, 3, 4]] * batch, device="cuda")
    drafts = [[5, 6, 7, 8, 9, 10], [11, 12], []][:batch]
    outs = []
    for be in (TorchBackend(m16), mega_be):
        sess = _session(be, batch, (2, 4, 8) if drafted else ())
        sess.encode(feats)
        if drafted:
            out, _ = sess.decode_drafted(primer, drafts, 20)
        else:
            out = sess.decode(primer, 20)
        outs.append(out)
    assert outs[0] == outs[1]


@gluon_only
def test_decoder_kernel_falls_back(model, monkeypatch):
    """pipeline._decoder_kernel leaves the wrapped backend when making the kernels fails in any way (here: out
    of memory) and, in auto mode, when a kernel's check call does not match the decoder; else the ring / Gluon
    backend."""
    from owsm_trt import pipeline

    dev, runner = torch.device("cuda"), object()
    dec = copy.deepcopy(model.decoder).float().cpu()
    monkeypatch.setattr(pipeline, "_load_decoder", lambda cfg, ckpt: dec)
    be = pipeline._decoder_kernel(runner, CFG, None, dev, "auto")
    assert isinstance(be, megakernel.MegaBackend) and be.trial and be.rows_kernel[0] is not None

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory")

    with monkeypatch.context() as mp:
        mp.setattr(megakernel, "MegaDecoder", oom)
        assert pipeline._decoder_kernel(runner, CFG, None, dev, "auto") is runner
        assert pipeline._decoder_kernel(runner, CFG, None, dev, "ring") is runner
    orig = megakernel_ring.RingDecoder.step_fn

    def wrong(self, *a, **k):  # logits off
        run = orig(self, *a, **k)

        def bad():
            run()
            a[6].mul_(-1.0)

        return bad

    with monkeypatch.context() as mp:
        mp.setattr(megakernel_ring.RingDecoder, "step_fn", wrong)
        assert pipeline._decoder_kernel(runner, CFG, None, dev, "auto") is runner


@gluon_only
def test_ring_argmax_without_timestamps(model):
    """The kernel's argmax leaves out the timestamp tokens in the rows flagged <notimestamps>."""
    dec, dev = model.decoder, torch.device("cuda")
    L, H, dk, C, T = CFG.dec_layers, CFG.dec_heads, CFG.d_model // CFG.dec_heads, 64, CFG.enc_frames
    torch.manual_seed(11)
    f16 = {"device": dev, "dtype": torch.float16}
    rows, window, ft, lt = 2, 4, 500, 990
    tokens = torch.randint(0, 400, (rows, window), device=dev)
    offset = torch.tensor([3, 5], device=dev)
    k, v = torch.randn(L, rows, H, C, dk, **f16), torch.randn(L, rows, H, C, dk, **f16)
    ck, cv = torch.randn(L, rows, H, T, dk, **f16), torch.randn(L, rows, H, T, dk, **f16)
    ring = megakernel_ring.RingDecoder(megakernel.MegaDecoder(dec, dev))
    logits = torch.zeros(rows, window, CFG.vocab_size, device=dev)
    new_kv = torch.zeros(L, 2, rows, H, window, dk, **f16)
    pred = torch.full((rows, window), -1, dtype=torch.long, device=dev)
    flags = torch.tensor([True, False], device=dev)
    ring.step_fn(tokens, offset, k, v, ck, cv, logits, new_kv, pred=pred, notime=(flags, ft, lt))()
    masked = logits.clone()
    masked[0, :, ft : lt + 1] = float("-inf")
    assert torch.equal(pred[0], masked[0].argmax(-1))
    assert torch.equal(
        pred[1], logits[1].argmax(-1)
    )  # (not final for an unflagged row: kernels.choose's then)


@gluon_only
@pytest.mark.parametrize("primer", [[2, 3, 999], [2, 3, 4]], ids=["notimestamps", "timestamps"])
def test_ring_session_with_rules(model, primer):
    """Greedy decoding under the timestamp rules (the kernel's argmax for <notimestamps>, kernels.choose for the
    rest) gives the PyTorch backend's output."""
    from owsm_trt.runtime import TimestampRules

    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    mega = megakernel.MegaDecoder(model.decoder, torch.device("cuda"))
    be = megakernel.MegaBackend(TorchBackend(m16), megakernel_ring.RingDecoder(mega), 32, 8, trial=True)
    rules = TimestampRules(999, 900, 998, 1, CFG.vocab_size, torch.device("cuda"))
    feats = m16.frontend(torch.randn(1, CFG.num_samples, device="cuda"))
    outs = []
    for b in (TorchBackend(m16), be):
        sess = GreedySession(b, CFG, 1, 64, eos=1, dtype=torch.float16, device=torch.device("cuda"),
                             buckets=[16, 32, 64], draft_windows=(2, 4, 8), rules=rules)  # fmt: skip
        sess.encode(feats)
        out, _ = sess.decode_drafted(torch.tensor([primer], device="cuda"), [[5, 6, 7, 8]], 20)
        outs.append(out)
    assert not be.failed
    assert outs[0] == outs[1]


@gluon_only
@pytest.mark.parametrize("programs", [16, 48, 96])
def test_ring_program_budget_matches_decoder(model, programs):
    """Existing SM-budget settings: logits, argmax, new KV and cache writes including idle CTAs."""
    if torch.cuda.get_device_properties("cuda").multi_processor_count < programs:
        pytest.skip("requires enough SMs for the persistent grid")
    test_gluon_step_matches_decoder(model, 1, 8, "ring", programs=programs)


@gluon_only
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("start_plain", [False, True])
@pytest.mark.parametrize("cached_flags", [False, True])
@torch.inference_mode()
def test_notime_reuse_modes_long_prefill_and_graph_lifetimes(model, rows, start_plain, cached_flags):
    from owsm_trt.runtime import TimestampRules

    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    dev = torch.device("cuda")
    backend = megakernel.MegaBackend(
        TorchBackend(m16),
        megakernel_ring.RingDecoder(megakernel.MegaDecoder(model.decoder, dev)),
        32,
        8,
        trial=True,
    )
    rules = TimestampRules(999, 900, 998, 1, CFG.vocab_size, dev)
    torch.manual_seed(27)
    cross = [
        torch.randn(CFG.dec_layers, rows, CFG.dec_heads, CFG.enc_frames, 64, device=dev, dtype=torch.float16)
        for _ in range(2)
    ]
    plain, timed = [2, 3, 999], [2, 3, 4]
    long_plain = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 999]
    patterns = [
        [plain if start_plain else timed] * rows,
        [long_plain] * rows,
        [plain] * rows,
        [plain if i % 2 else timed for i in range(rows)],
        [timed] * rows,
        [plain] * rows,
    ]
    feats = [m16.frontend(torch.randn(rows, CFG.num_samples, device=dev)) for _ in range(2)]
    reference = None
    for reuse in (False, True):
        sess = GreedySession(
            backend,
            CFG,
            rows,
            64,
            eos=1,
            dtype=torch.float16,
            device=dev,
            buckets=[16, 32, 64],
            draft_windows=(2, 4, 8),
            rules=rules,
            lazy_drafts=True,
            split_choose=True,
            fuse_accept=True,
            reuse_notime=reuse,
            cache_draft_flags=reuse and cached_flags,
        )
        sess.cross_k.copy_(cross[0])
        sess.cross_v.copy_(cross[1])
        outs, graph_ids = [], None
        for index, primers in enumerate(patterns):
            outs.append(sess.decode_drafted(primers, [[5, 6, 7, 8]] * rows, 36)[0])
            bounds = [*sess._verify.values(), *sess._verify_solo.values()]
            assert all(b.use_notime == (reuse and all(999 in p for p in primers)) for b in bounds)
            if index == 0:
                if reuse and start_plain:
                    assert all(b.graph is None and b.notime_graph is not None for b in bounds)
                for bound in bounds:
                    bound.bufs.host[:, bound.bufs.w] = 60  # deliberately unsafe for cold small-bucket replay
            if reuse and index == 1:
                graph_ids = [id(b.notime_graph) for b in bounds]
                assert all(b.notime_graph is not None for b in bounds)
            if reuse and index > 1:
                assert graph_ids == [id(b.notime_graph) for b in bounds]
            if index == 2:
                outs.append(sess.decode(torch.tensor([timed] * rows, device=dev), 10))
                outs.append(sess.decode_drafted([plain] * rows, [[5, 6, 7]] * rows, 20)[0])
                assert sess.t_notime.all()
        # The streaming path uses the same request mode before launching its encoder.
        # Fixed synthetic draft factory; no audio/reference model work on login nodes.
        outputs, _ = sess.decode_stream(plain, iter(feats), lambda ids: [[5, 6, 7]] * len(ids), 20)
        assert len(outputs) == 2 * rows
        outs.append(outputs)
        if reference is None:
            reference = outs
        else:
            assert outs == reference
        assert not backend.failed


@gluon_only
@pytest.mark.parametrize("window", [1, 8, 32])
@pytest.mark.parametrize("solo", [False, True])
def test_ring_prefetch_preserves_logits_and_cache(model, window, solo):
    """Weight prefetch must leave exact arithmetic, cache writes and repeated graph results intact."""
    dev = torch.device("cuda")
    L, H, dk, C, T = CFG.dec_layers, CFG.dec_heads, 64, 128, CFG.enc_frames
    torch.manual_seed(71 + window)
    kw = dict(device=dev, dtype=torch.float16)
    tok = torch.randint(CFG.vocab_size, (1, window), device=dev)
    off = torch.tensor([7], device=dev)
    initial = [torch.randn(L, 1, H, C, dk, **kw) for _ in range(2)]
    ck, cv = [torch.randn(L, 1, H, T, dk, **kw) for _ in range(2)]
    mega = megakernel.MegaDecoder(model.decoder, dev)
    expected = None
    for ahead in (0, 1, 2):
        ring = megakernel_ring.RingDecoder(mega, ahead=ahead)
        k, v = [x.clone() for x in initial]
        logits = torch.zeros(1, window, CFG.vocab_size, device=dev)
        new = torch.zeros(L, 2, 1, H, window, dk, **kw)
        pred = torch.full((1, window), -1, dtype=torch.long, device=dev)
        step = ring.step_fn(tok, off, k, v, ck, cv, logits, new, pred=pred, solo=solo)
        from owsm_trt.runtime import capture_graph

        graph = capture_graph(step)
        results = []
        for offset in (7, 63):
            off.fill_(offset)
            k.copy_(initial[0])
            v.copy_(initial[1])
            graph.replay()
            results.append([x.clone() for x in (logits, new, k, v, pred)])
        ring.check()
        if expected is None:
            expected = results
        else:
            for actuals, refs in zip(results, expected):
                for actual, ref in zip(actuals, refs):
                    assert torch.equal(actual, ref)


@gluon_only
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("graphs", [False, True])
def test_deferred_verify_preserves_prefill_bucket_and_mode_changes(model, rows, graphs):
    from owsm_trt.runtime import TimestampRules

    m16 = copy.deepcopy(model).half()
    m16.frontend.float()
    dev = torch.device("cuda")
    backend = megakernel.MegaBackend(
        TorchBackend(m16), megakernel_ring.RingDecoder(megakernel.MegaDecoder(model.decoder, dev)),
        32, 8, trial=True,
    )
    rules = TimestampRules(999, 900, 998, 1, CFG.vocab_size, dev)
    torch.manual_seed(198)
    cross = [
        torch.randn(CFG.dec_layers, rows, CFG.dec_heads, CFG.enc_frames, 64, device=dev, dtype=torch.float16)
        for _ in range(2)
    ]
    timed, plain = [2, 3, 4], [2, 3, 999]
    long_plain = [2, 3] + list(range(5, 19)) + [999]
    reference = None
    for deferred in (False, True):
        sess = GreedySession(
            backend, CFG, rows, 64, -1, torch.float16, dev, use_cuda_graph=graphs,
            buckets=[16, 32, 64], draft_windows=(2, 4, 8), rules=rules, lazy_drafts=True,
            split_choose=True, fuse_accept=True, reuse_notime=True, lazy_verify_graphs=deferred,
        )
        sess.cross_k.copy_(cross[0])
        sess.cross_v.copy_(cross[1])
        outputs = []
        for primers in ([long_plain] * rows, [timed] * rows,
                        [plain if r % 2 else timed for r in range(rows)], [plain] * rows):
            outputs.append(sess.decode_drafted(primers, [[5, 6, 7, 8]] * rows, 38)[0])
            assert all(len(row) == 38 for row in outputs[-1])
        if reference is None:
            reference = outputs
        else:
            assert outputs == reference
            assert any(b.step is None for b in sess._verify.values())
            previous = [(id(b.graph), id(b.notime_graph), id(b.step)) for b in sess._verify.values()]
            assert sess.decode_drafted([plain] * rows, [[5, 6, 7, 8]] * rows, 38)[0] == outputs[-1]
            assert previous == [(id(b.graph), id(b.notime_graph), id(b.step)) for b in sess._verify.values()]
        assert not backend.failed
