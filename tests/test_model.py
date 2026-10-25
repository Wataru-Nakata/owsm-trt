import math

import pytest
import torch

from owsm_trt.audio import resample
from owsm_trt.config import OWSMConfig
from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import BeamSession, GreedySession, TimestampRules, TorchBackend

TINY = OWSMConfig(
    vocab_size=50, d_model=32, n_heads=4, enc_layers=2, enc_ffn=64, cgmlp_units=64, cgmlp_kernel=5,
    merge_kernel=3, subsample="conv2d8", dec_layers=2, dec_ffn=64, dec_heads=4, n_mels=16, n_fft=64,
    win_length=48, hop_length=32, sample_rate=1600, speech_seconds=2.0, time_resolution=0.02,
)  # fmt: skip


@pytest.fixture(scope="module")
def model() -> OWSMModel:
    torch.manual_seed(0)
    m = OWSMModel(TINY).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    return m


def _session(model: OWSMModel, batch: int, capacity: int = 32, eos: int = 1, buckets=None) -> GreedySession:
    return GreedySession(
        TorchBackend(model), TINY, batch, capacity, eos=eos, dtype=torch.float32,
        device=torch.device("cpu"), use_cuda_graph=False, buckets=buckets,
    )  # fmt: skip


def test_shapes(model):
    assert TINY.num_frames == 101 and TINY.enc_frames == 11
    kv = model.encode(torch.randn(3, TINY.num_samples))
    assert kv.shape == (2, 2, 3, 4, 11, 8)


def test_prefill_matches_incremental(model):
    sess = _session(model, batch=2)
    sess.encode(model.frontend(torch.randn(2, TINY.num_samples)))
    tokens = torch.randint(2, 50, (2, 6))
    sk, sv = list(sess.self_k), list(sess.self_v)
    ck, cv = list(sess.cross_k), list(sess.cross_v)
    full, _ = model.decoder.step(tokens, torch.zeros(2, dtype=torch.long), sk, sv, ck, cv)
    for t in range(6):
        logits, new = model.decoder.step(tokens[:, t : t + 1], torch.full((2,), t), sk, sv, ck, cv)
        sess.self_k[:, :, :, t] = new[:, 0, :, :, 0]
        sess.self_v[:, :, :, t] = new[:, 1, :, :, 0]
    torch.testing.assert_close(logits[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)


def test_greedy_session_matches_naive(model):
    sess = _session(model, batch=2)
    feats = model.frontend(torch.randn(2, TINY.num_samples))
    sess.encode(feats)
    primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
    out = sess.decode(primer, max_new_tokens=10)

    # naive: re-run the whole prefix every step with an empty cache
    empty = torch.zeros_like(sess.self_k)
    for b in range(2):
        seq = primer[b : b + 1]
        naive = []
        for _ in range(10):
            logits, _ = model.decoder.step(
                seq, torch.zeros(1, dtype=torch.long), list(empty[:, b : b + 1]), list(empty[:, b : b + 1]),
                list(sess.cross_k[:, b : b + 1]), list(sess.cross_v[:, b : b + 1]),
            )  # fmt: skip
            nxt = int(logits[0, -1].argmax(-1))
            if nxt == 1:
                break
            naive.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
        assert out[b] == naive


def test_resample_preserves_tone():
    sr_in, sr_out, f = 44100, 16000, 440.0
    t = torch.arange(sr_in) / sr_in
    y = resample(torch.sin(2 * math.pi * f * t), sr_in, sr_out)
    assert y.shape[0] == sr_out
    ref = torch.sin(2 * math.pi * f * torch.arange(sr_out) / sr_out)
    mid = slice(1000, -1000)  # ignore edge effects
    assert (y[mid] - ref[mid]).abs().max() < 1e-2


def test_fused_projections_match(model):
    import copy

    fused = copy.deepcopy(model)
    fused.fuse_projections()
    speech = torch.randn(2, TINY.num_samples)
    torch.testing.assert_close(fused.encode(speech), model.encode(speech), rtol=1e-4, atol=1e-5)
    s1, s2 = _session(model, batch=2), _session(fused, batch=2)
    feats = model.frontend(speech)
    s1.encode(feats)
    s2.encode(feats)
    primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
    assert s1.decode(primer, 12) == s2.decode(primer, 12)
    torch.testing.assert_close(s1.logits, s2.logits, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("split", [True, False])
def test_bucket_switching_matches_single_cache(model, split):
    """Growing through tiny cache buckets gives the same tokens and logits as one big cache."""
    model.decoder.split_attention = split
    try:
        feats = model.frontend(torch.randn(2, TINY.num_samples))
        primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
        outs, logits = [], []
        for buckets in ([4, 8, 16, 32], None):
            sess = _session(model, batch=2, eos=-1, buckets=buckets)  # eos never fires: always 20 steps
            sess.encode(feats)
            outs.append(sess.decode(primer, 20))
            logits.append(sess.logits.clone())
        assert outs[0] == outs[1] and len(outs[0][0]) == 20
        torch.testing.assert_close(logits[0], logits[1], rtol=1e-4, atol=1e-4)
    finally:
        model.decoder.split_attention = True


def test_buildable_drops_plugins_that_cannot_be_built(monkeypatch):
    """Engines built without the plugins get options of their own (another engine directory), so they are rebuilt
    once the plugins can be built; with every plugin buildable the options (and directory) stay the same."""
    import dataclasses

    from owsm_trt import attn_plugin, build, plugins

    opts = build.BuildOptions()
    monkeypatch.setattr(plugins, "available", lambda: True)
    monkeypatch.setattr(attn_plugin, "unavailable", lambda: None)
    assert build.buildable(opts, TINY) == opts
    monkeypatch.setattr(plugins, "available", lambda: False)
    assert build.buildable(opts, TINY) == dataclasses.replace(
        opts, dwconv_plugin=False, conv1_plugin=False, ctc_plugins=False
    )
    monkeypatch.setattr(attn_plugin, "unavailable", lambda: "no Triton")
    off = build.buildable(opts, TINY)
    assert not (off.attn_plugin or off.attn_plugin_steps or off.dwconv_plugin)
    ctc = dataclasses.replace(TINY, ctc_only=True)  # (no decoder: its options stay)
    assert build.buildable(opts, ctc).attn_plugin


def test_decoder_profile_shapes_are_consistent():
    from owsm_trt.build import BuildOptions, decoder_input_names, decoder_profile_shapes, decoder_profiles

    opts = BuildOptions(max_batch=4, capacity=256, max_prefill=32)
    profiles = decoder_profiles(opts.max_batch)
    assert [p["kind"] for p in profiles] == ["decode", "decode", "verify", "verify", "prefill"]
    for p in profiles:
        shapes = decoder_profile_shapes(TINY, opts, p)
        assert set(shapes) == set(decoder_input_names(TINY.dec_layers))
        for name, (lo, opt, hi) in shapes.items():
            assert len(lo) == len(opt) == len(hi), name
            assert all(a <= b <= c for a, b, c in zip(lo, opt, hi)), (name, lo, opt, hi)
    # every batch size 1..max_batch is covered by exactly one decode and one verify profile
    for kind in ("decode", "verify"):
        for b in range(1, opts.max_batch + 1):
            assert sum(p["kind"] == kind and p["batch"][0] <= b <= p["batch"][2] for p in profiles) == 1


def _end_detect(ended, i, M=3, D_end=-10.0):
    """espnet2.legacy.nets.e2e_asr_common.end_detect on (score, yseq) pairs."""
    if not ended:
        return False
    best = max(score for score, _ in ended)
    count = 0
    for m in range(M):
        same = [score for score, seq in ended if len(seq) == i - m]
        if same and max(same) - best < D_end:
            count += 1
    return count == M


# token layout for the rules tests: <eos>=1, <notimestamps>=5, timestamps 10..20
NOTIME, FIRST_T, LAST_T = 5, 10, 20


def _score_filter(y: list[int], n_primer: int, eos: int) -> torch.Tensor:
    """espnet2.bin.s2t_inference.ScoreFilter.score (first step: nothing generated yet)."""
    score = torch.zeros(TINY.vocab_size)
    if NOTIME in y:
        score[FIRST_T : LAST_T + 1] = float("-inf")
    elif len(y) == n_primer:
        score[:FIRST_T] = float("-inf")
        score[LAST_T + 1 :] = float("-inf")
    else:
        prev = [t for t in y if FIRST_T <= t <= LAST_T]
        if len(prev) % 2 == 1:
            score[eos] = float("-inf")
            score[FIRST_T : prev[-1] + 1] = float("-inf")
        elif FIRST_T <= y[-1] <= LAST_T:
            score[: y[-1]] = float("-inf")
            score[LAST_T + 1 :] = float("-inf")
            score[eos] = 0.0
        else:
            score[:] = float("-inf")
    return score


def _naive_beam(model, cross_k, cross_v, primer, beam, eos, maxlen, rules=False):
    """ESPnet's single-utterance beam search, re-running the decoder over the whole prefix."""
    L, _, H, _, dk = cross_k.shape
    empty = [torch.zeros(1, H, 1, dk)] * L
    running, ended = [(0.0, list(primer))], []
    for i in range(maxlen):
        cands = []
        for score, seq in running:
            logits, _ = model.decoder.step(
                torch.tensor([seq]),
                torch.zeros(1, dtype=torch.long),
                empty,
                empty,
                list(cross_k),
                list(cross_v),
            )
            logp = torch.log_softmax(logits[0, -1], -1)
            if rules:
                logp = logp + _score_filter(seq, len(primer), eos)
            cands += [(score + float(lp), seq + [t]) for t, lp in enumerate(logp)]
        best = sorted(cands, key=lambda x: x[0], reverse=True)[:beam]
        running = []
        for score, seq in best:
            if i == maxlen - 1:
                ended.append((score, seq + [eos]))
            elif seq[-1] == eos:
                ended.append((score, seq))
            else:
                running.append((score, seq))
        if i == maxlen - 1 or not running or _end_detect(ended, i):
            break
    score, seq = sorted(ended, key=lambda x: x[0], reverse=True)[0]
    out = seq[len(primer) :]
    return out[: out.index(eos)] if eos in out else out


# (eos id, eos logit boost): hits the length limit / end_detect early / end_detect late / plain <eos>
@pytest.mark.parametrize("eos, eos_boost", [(1, 0.3), (1, 2.0), (11, 0.2), (7, 0.3), (30, 0.0)])
@pytest.mark.parametrize("upper_bound", [False, True])
def test_beam_session_matches_naive_espnet(model, eos, eos_boost, upper_bound):
    """Vectorised beam search over a batch == ESPnet's per-utterance search (incl. end_detect)."""
    import copy

    m = copy.deepcopy(model)
    m.decoder.output_layer.bias[eos] += eos_boost  # make hypotheses end at varied lengths
    torch.manual_seed(1)
    speech = torch.randn(3, TINY.num_samples)
    primer = torch.tensor([[2, 3, 4]] * 3)
    sess = BeamSession(
        TorchBackend(m), TINY, 3, 4, 32, eos=eos, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False, buckets=[4, 8, 16, 32], upper_bound=upper_bound,
    )  # fmt: skip
    sess.encode(m.frontend(speech))
    out = sess.decode(primer, max_new_tokens=24)
    for b in range(3):
        ref = _naive_beam(m, sess.cross_kb[:, b : b + 1], sess.cross_vb[:, b : b + 1], [2, 3, 4], 4, eos, 24)
        assert out[b] == ref, (b, out[b], ref)


def test_beam_of_one_is_greedy(model):
    feats = model.frontend(torch.randn(2, TINY.num_samples))
    primer = torch.tensor([[2, 3, 4], [2, 5, 4]])
    greedy = _session(model, batch=2)
    greedy.encode(feats)
    beam = BeamSession(
        TorchBackend(model), TINY, 2, 1, 32, eos=1, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False,
    )  # fmt: skip
    beam.encode(feats)
    assert beam.decode(primer, 20) == greedy.decode(primer, 20)


def test_drafted_decoding_is_greedy(model):
    """Checking drafted tokens several at a time gives the greedy output, whatever the draft."""
    import random

    feats = model.frontend(torch.randn(1, TINY.num_samples))
    primer = torch.tensor([[2, 3, 4]])
    sess = GreedySession(
        TorchBackend(model), TINY, 1, 32, eos=1, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False, buckets=[8, 16, 32], draft_windows=(2, 4, 8),
    )  # fmt: skip
    sess.encode(feats)
    greedy = sess.decode(primer, 25)[0]
    assert len(greedy) == 25  # the random model never emits <eos>: windows cross bucket boundaries

    rng = random.Random(0)
    corrupted = list(greedy)
    corrupted[3] = 7  # substitution
    del corrupted[9]  # deletion
    corrupted.insert(15, 5)  # insertion
    drafts = {
        "perfect": greedy,
        "corrupted": corrupted,
        "random": [rng.randrange(2, 50) for _ in range(30)],
        "empty": [],
    }
    for name, draft in drafts.items():
        (out,), calls = sess.decode_drafted(primer, [draft], 25)
        assert out == greedy, name
        if name == "perfect":
            assert calls < 10, calls  # prefill + ceil(24 / 7) windows of 8


def _rules(eos: int = 1) -> TimestampRules:
    return TimestampRules(NOTIME, FIRST_T, LAST_T, eos, TINY.vocab_size, torch.device("cpu"))


@pytest.mark.parametrize("primer", [[2, 3, 4], [2, 3, 4, NOTIME]], ids=["timestamps", "notimestamps"])
@pytest.mark.parametrize("upper_bound", [False, True])
def test_timestamp_rules_greedy_and_beam(model, primer, upper_bound):
    """Greedy and beam search under the rules == ESPnet's searches with its ScoreFilter."""
    torch.manual_seed(2)
    feats = model.frontend(torch.randn(2, TINY.num_samples))
    greedy = GreedySession(
        TorchBackend(model), TINY, 2, 32, eos=1, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False, rules=_rules(),
    )  # fmt: skip
    greedy.encode(feats)
    out = greedy.decode(torch.tensor([primer] * 2), 24)
    beam = BeamSession(
        TorchBackend(model), TINY, 2, 3, 32, eos=1, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False, buckets=[8, 16, 32], rules=_rules(), upper_bound=upper_bound,
    )  # fmt: skip
    beam.encode(feats)
    out_beam = beam.decode(torch.tensor([primer] * 2), 24)
    for b in range(2):
        ck, cv = greedy.cross_k[:, b : b + 1], greedy.cross_v[:, b : b + 1]
        assert out[b] == _naive_beam(model, ck, cv, primer, 1, 1, 24, rules=True)
        assert out_beam[b] == _naive_beam(model, ck, cv, primer, 3, 1, 24, rules=True)
        times = [t for t in out[b] if FIRST_T <= t <= LAST_T]
        if NOTIME in primer:
            assert not times
        else:
            assert out[b][0] == times[0] and times == sorted(times)  # starts with a timestamp, monotonic


@pytest.mark.parametrize("primer", [[2, 3, 4], [2, 3, 4, NOTIME]], ids=["timestamps", "notimestamps"])
def test_drafted_decoding_follows_rules(model, primer):
    import random

    feats = model.frontend(torch.randn(1, TINY.num_samples))
    sess = GreedySession(
        TorchBackend(model), TINY, 1, 32, eos=1, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False, buckets=[8, 16, 32], draft_windows=(2, 4, 8), rules=_rules(),
    )  # fmt: skip
    sess.encode(feats)
    pr = torch.tensor([primer])
    greedy = sess.decode(pr, 25)[0]
    rng = random.Random(0)
    text = [t for t in greedy if not FIRST_T <= t <= LAST_T]  # a draft without timestamps, like CTC's
    insertions = set(range(FIRST_T, LAST_T + 1))
    for draft in (greedy, text, [rng.randrange(2, 50) for _ in range(30)]):
        (out,), _ = sess.decode_drafted(pr, [draft], 25, insertions)
        assert out == greedy


@pytest.mark.parametrize("subsample", ["conv2d", "conv2d2"])
def test_transformer_encoder_model(subsample):
    """OWSM v1-v3 / OWLS layout: Transformer encoder layers, x4 or x2 subsampling."""
    import dataclasses

    cfg = dataclasses.replace(TINY, encoder="transformer", subsample=subsample)
    m = OWSMModel(cfg).eval().requires_grad_(False)
    assert not hasattr(m.encoder.encoders[0], "cgmlp")
    kv = m.encode(torch.randn(2, cfg.num_samples))
    assert kv.shape == (2, 2, 2, 4, cfg.enc_frames, 8)
    assert cfg.enc_frames == {"conv2d": 24, "conv2d2": 48}[subsample]


def test_ctc_model_shapes():
    """OWSM-CTC: prompt encoder + prefix + cross-attention + interCTC conditioning -> best path."""
    import dataclasses

    from owsm_trt.modeling import OWSMCTCModel
    from owsm_trt.runtime import CTCSession, TorchCTCBackend

    cfg = dataclasses.replace(
        TINY, dec_layers=0, dec_ffn=0, dec_heads=0, ctc_only=True, prompt_d=16, prompt_heads=2,
        prompt_ffn=32, prompt_layers=2, interctc_layers=(1,), cross_attn_layers=(1,),
    )  # fmt: skip
    m = OWSMCTCModel(cfg).eval().requires_grad_(False)
    m.fuse_projections()
    feats = m.frontend(torch.randn(3, cfg.num_samples))
    ids = m(feats, torch.tensor([[2]] * 3), torch.tensor([[4, 5]] * 3))
    assert ids.shape == (3, cfg.n_prefix + cfg.enc_frames) and ids.dtype == torch.int32
    sess = CTCSession(TorchCTCBackend(m), cfg, 3, 1, torch.device("cpu"), use_cuda_graph=False)
    assert torch.equal(sess(feats, torch.tensor([[2]] * 3), torch.tensor([[4, 5]] * 3)), ids)
    # the encoder attends to the encoded prompt
    prefix = m.embed_proj(m.embed(torch.tensor([[4, 5]] * 3)))
    a = m.encoder(feats, prefix, torch.zeros(3, 1, cfg.d_model), m.ctc)
    b = m.encoder(feats, prefix, torch.ones(3, 1, cfg.d_model), m.ctc)
    assert not torch.allclose(a, b)
    # features trimmed after the audio + the padding frame: the same encoder output
    from owsm_trt.modeling import frames_needed, padding_frame

    speech = torch.zeros(3, cfg.num_samples)
    speech[:, :1000] = torch.randn(3, 1000)
    feats = m.frontend(speech)
    memory = torch.randn(3, 1, cfg.d_model)
    ref = m.encoder(feats, prefix, memory, m.ctc)
    m.encoder.fill = padding_frame(m)
    trimmed = m.encoder(feats[:, : frames_needed(cfg, 1000)], prefix, memory, m.ctc)
    torch.testing.assert_close(trimmed, ref, rtol=1e-4, atol=1e-4)


def test_drafted_decoding_recovers_from_draft_errors(model):
    """Host-side drafting logic against an oracle decoder that always emits ``target``: a draft with
    a few CTC-like errors costs about one extra call per error, a useless draft is abandoned."""
    import random

    from owsm_trt.runtime import _accepted

    rng = random.Random(0)
    target = [rng.randrange(6, 50) for _ in range(60)]
    p = 3
    sess = GreedySession(
        TorchBackend(model), TINY, 1, 128, eos=1, dtype=torch.float32, device=torch.device("cpu"),
        use_cuda_graph=False, buckets=[128], draft_windows=(2, 4, 8, 16, 32),
    )  # fmt: skip

    def prefill(primer, cross_k, cross_v):
        logits = torch.zeros(1, TINY.vocab_size)
        logits[0, target[0]] = 1.0
        return logits, torch.zeros(TINY.dec_layers, 2, 1, TINY.dec_heads, p, 8), 128

    def oracle(w):
        def verify(
            inp, meanwhile=None
        ):  # one row: [w tokens | offset | #timestamps | last one | first position]
            base = inp[w] - p  # index of the window's first token in target (negative: in the primer)
            pred = [target[base + i + 1] if 0 <= base + i + 1 < len(target) else 1 for i in range(w)]
            return [list(_accepted(pred, inp[:w], max(inp[w + 3], 0), eos=1))]

        return verify

    sess._prefill = prefill
    sess._verify = {key: oracle(key[1]) for key in sess._verify}
    t = target
    cases = {  # draft: most calls allowed
        "perfect": (t, 4),
        "deletion": (t[:10] + t[11:], 5),
        "insertion": (t[:10] + [7] + t[10:], 5),
        "substitution": (t[:10] + [7] + t[11:], 6),
        "3 missing": (t[:10] + t[13:], 8),
        "3 errors": (t[:10] + [7] + t[11:30] + t[31:45] + [5] + t[45:], 8),
        "random": ([rng.randrange(6, 50) for _ in range(60)], 61),
    }
    for name, (draft, most) in cases.items():
        (out,), calls = sess.decode_drafted(torch.tensor([[2, 3, 4]]), [draft], 100)
        assert out == target, name
        assert calls <= most, (name, calls)


def test_explicit_attention_matches_sdpa():
    """The export's written-out attention (scale on q) == scaled_dot_product_attention."""
    from owsm_trt.modeling import MultiHeadedAttention

    torch.manual_seed(0)
    att = MultiHeadedAttention(32, 4)
    x, memory = torch.randn(2, 7, 32), torch.randn(2, 5, 32)
    ref = att(x), att.cross(x, memory)
    MultiHeadedAttention.explicit = True
    try:
        torch.testing.assert_close(att(x), ref[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(att.cross(x, memory), ref[1], rtol=1e-5, atol=1e-5)
    finally:
        MultiHeadedAttention.explicit = False


def test_resync_rules():
    """Where the draft continues after the decoder emitted something else than draft[j]."""
    from owsm_trt.runtime import _resync

    draft, comma, time = [10, 11, 12, 13, 14, 15], 3, 40
    punct = frozenset({comma, 4})
    assert _resync(draft, 2, [10, 11, 13], set(), punct) == 4  # draft[2] was dropped: anchor on 13
    assert _resync(draft, 2, [10, 11, 30], set(), punct) == 3  # unknown word: it replaces draft[2]
    assert _resync(draft, 2, [10, 11, comma], set(), punct) == 2  # punctuation the draft lacks: inserted
    assert _resync(draft, 2, [10, 11, time], {time}, punct) == 2  # never drafted (timestamps): inserted
    with_commas = [10, comma, 11, 12, 13, comma, 14]
    # a lone comma is no anchor (the next one in the draft is not this one)...
    assert _resync(with_commas, 3, [10, comma, 11, comma], set(), punct) == 3
    # ...but a comma after an anchored word is
    assert _resync(with_commas, 4, [11, 12, 13, comma], set(), punct) == 6


def _resync_scan(draft, j, out, insertions, punctuation, reach=8):
    """_resync's rules, as a plain scan of every anchor (the reference for its shortcuts)."""
    if out[-1] in insertions:
        return j
    for k in (3, 2, 1):
        tail = out[-k:]
        if len(tail) < k or any(t in insertions for t in tail) or (k == 1 and tail[0] in punctuation):
            continue
        hits = [
            i
            for i in range(max(0, j - reach), min(len(draft) - k, j + reach) + 1)
            if draft[i : i + k] == tail
        ]
        if hits:
            return min(hits, key=lambda i: abs(i + k - 1 - j)) + k
    if j < len(draft) and not (out[-1] in punctuation and draft[j] not in punctuation):
        return j + 1
    return j


def test_resync_matches_scan():
    import random

    from owsm_trt.runtime import _resync

    rng = random.Random(0)
    insertions, punct = {9}, frozenset({7, 8})
    for _ in range(20000):
        vocab = rng.choice([4, 6, 12])  # small vocabularies: many repeats, ties and anchors far from j
        draft = [rng.randrange(vocab) for _ in range(rng.randrange(0, 30))]
        out = [rng.randrange(vocab) for _ in range(rng.randrange(1, 6))]
        if draft and rng.random() < 0.5:  # the decoder followed the draft up to j
            j = rng.randrange(len(draft) + 1)
            out = (out + draft[max(0, j - 3) : j])[-rng.randrange(1, 5) :]
            out += [draft[j] if j < len(draft) and rng.random() < 0.7 else rng.randrange(vocab)]
        j = rng.randrange(len(draft) + 3)
        want = _resync_scan(draft, j, out, insertions, punct)
        assert _resync(draft, j, out, insertions, punct) == want, (draft, j, out)
        assert _resync(draft, j, out[-3:], insertions, punct) == want  # all it reads of the output


def test_accept_matches_reference():
    """_Verify.accept (what a call returns) == the per-row rule, wherever the head ends."""
    import random

    from owsm_trt.runtime import _accepted, _Verify

    rng = random.Random(1)
    eos, R = 3, 40
    for w in (1, 2, 4, 8):
        bufs = _Verify(TINY, R, w, {"device": torch.device("cpu"), "dtype": torch.float32}, None, None, eos)
        rows, preds = [], []
        for _ in range(R):
            tok = [rng.randrange(6) for _ in range(w)]
            pred = [tok[i + 1] if i + 1 < w and rng.random() < 0.8 else rng.randrange(6) for i in range(w)]
            rows.append(tok + [0, 0, -1, rng.randrange(-1, w)])
            preds.append(pred)
        bufs.inp.copy_(torch.tensor(rows))
        bufs.pred.copy_(torch.tensor(preds))
        want = [list(_accepted(p, r[:w], max(r[w + 3], 0), eos)) for p, r in zip(preds, rows)]
        assert bufs.accept().tolist() == want


def test_ctc_collapse_rows():
    import random

    from owsm_trt.pipeline import ctc_collapse, ctc_collapse_rows

    rng = random.Random(2)
    paths = torch.tensor(
        [[rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(50)] for _ in range(8)], dtype=torch.int32
    )
    paths[0] = 0
    assert ctc_collapse_rows(paths, 0) == [ctc_collapse(p, 0) for p in paths.tolist()]


@pytest.mark.parametrize("seconds", [0.3, 1.0, 2.0])
def test_encoder_graph_ctc_on_computed_frames(seconds):
    """The exported encoder's CTC ids: the full input's best path on the subsampled frames computed from
    trimmed features, a blank, and the window's last frame (blank too when every frame is computed)."""
    import dataclasses

    from owsm_trt.build import _EncoderGraph
    from owsm_trt.modeling import frames_needed, padding_frame

    cfg = dataclasses.replace(TINY, normalize="global_mvn")
    torch.manual_seed(8)
    m = OWSMModel(cfg).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    m.frontend.mean.normal_()
    n = int(seconds * cfg.sample_rate)
    speech = torch.zeros(2, cfg.num_samples)
    speech[:, :n] = torch.randn(2, n)
    feats = m.frontend(speech)
    full = m.ctc(m.encoder(feats))
    t = frames_needed(cfg, n)
    m.encoder.fill = padding_frame(m)
    try:
        k, v, ids = _EncoderGraph(m)(feats[:, :t])
    finally:
        m.encoder.fill = None
    t = m.encoder.embed(feats[:, :t]).shape[1]  # subsampled frames computed
    assert ids.shape[1] == t + 2 and (t < cfg.enc_frames) == (seconds < 1.9)
    assert torch.equal(ids[:, :t], full[:, :t]) and not ids[:, t].any()
    assert torch.equal(ids[:, t + 1], full[:, -1] if t < cfg.enc_frames else torch.zeros_like(full[:, -1]))


@pytest.mark.parametrize("split", [True, False])
def test_decoder_step_with_per_row_offsets(model, split):
    """Rows at different cache offsets in one call == each row on its own."""
    model.decoder.split_attention = split
    try:
        torch.manual_seed(3)
        L, H, dk, T = TINY.dec_layers, TINY.dec_heads, 8, TINY.enc_frames
        sk, sv = torch.randn(L, 2, H, 16, dk), torch.randn(L, 2, H, 16, dk)
        ck, cv = torch.randn(L, 2, H, T, dk), torch.randn(L, 2, H, T, dk)
        tokens, offset = torch.randint(2, 50, (2, 3)), torch.tensor([3, 7])
        logits, new_kv = model.decoder.step(tokens, offset, list(sk), list(sv), list(ck), list(cv))
        for b in range(2):
            rb = slice(b, b + 1)
            lb, kvb = model.decoder.step(
                tokens[rb], offset[rb], list(sk[:, rb]), list(sv[:, rb]), list(ck[:, rb]), list(cv[:, rb])
            )
            torch.testing.assert_close(logits[rb], lb, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(new_kv[:, :, rb], kvb, rtol=1e-4, atol=1e-5)
    finally:
        model.decoder.split_attention = True


@pytest.mark.parametrize("primer", [[2, 3, 4], [2, 3, 4, NOTIME]], ids=["timestamps", "notimestamps"])
@pytest.mark.parametrize("rules", [False, True], ids=["free", "rules"])
def test_batched_drafted_decoding_is_greedy(model, primer, rules):
    """Rows with different drafts advance by different amounts; each row still gets its greedy output."""
    torch.manual_seed(4)
    feats = model.frontend(torch.randn(3, TINY.num_samples))
    kw = {
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "use_cuda_graph": False,
        "buckets": [8, 16, 32],
    }
    kw["rules"] = _rules() if rules else None
    greedy = GreedySession(TorchBackend(model), TINY, 3, 32, eos=1, **kw)
    greedy.encode(feats)
    pr = torch.tensor([primer] * 3)
    ref = greedy.decode(pr, 20)
    sess = GreedySession(TorchBackend(model), TINY, 3, 32, eos=1, draft_windows=(2, 4, 8), **kw)
    sess.encode(feats)
    corrupted = list(ref[1])
    corrupted[2:4] = [7, 8, 9]
    insertions = set(range(FIRST_T, LAST_T + 1)) if rules else set()
    out, _ = sess.decode_drafted(pr, [ref[0], corrupted, []], 20, insertions)
    assert out == ref
    # a batch runs until its slowest row is done: with every draft right, a few calls suffice
    out, calls = sess.decode_drafted(pr, ref, 20, insertions)
    assert out == ref and calls <= 5, calls


@pytest.mark.parametrize("rules", [False, True], ids=["free", "rules"])
def test_drafted_decoding_with_a_long_primer(model, rules):
    """A primer longer than the largest window is prefilled on its own; then drafts as usual."""
    torch.manual_seed(7)
    feats = model.frontend(torch.randn(2, TINY.num_samples))
    kw = {"dtype": torch.float32, "device": torch.device("cpu"), "use_cuda_graph": False, "buckets": [16, 32]}
    kw["rules"] = _rules() if rules else None
    pr = torch.tensor([[2, 3, 30, 31, 32, 33, 34, 35, 4]] * 2)  # e.g. sop + prompt + sos lang task
    greedy = GreedySession(_DistinctBackend(model), TINY, 2, 32, eos=1, **kw)
    greedy.encode(feats)
    ref = greedy.decode(pr, 20)
    assert min(len(r) for r in ref) > 10
    sess = GreedySession(_DistinctBackend(model), TINY, 2, 32, eos=1, draft_windows=(2, 4, 8), **kw)
    sess.encode(feats)
    bad = list(ref[0])
    bad[2:4] = [7, 8, 9]
    insertions = set(range(FIRST_T, LAST_T + 1)) if rules else set()
    for drafts in ([ref[0], ref[1]], [bad, []]):
        out, _ = sess.decode_drafted(pr, drafts, 20, insertions)
        assert out == ref


class _DistinctBackend(TorchBackend):
    """Cross-attention values that differ a lot between utterances (the tiny random model hardly
    listens to its input), so that a row decoding with another row's encoder output shows."""

    def encode(self, feats, cross_k, cross_v, ctc_ids):
        paths = super().encode(feats, cross_k, cross_v, ctc_ids)
        for b, f in enumerate(feats):
            g = torch.Generator().manual_seed(int(f.abs().sum() * 100))
            cross_v[:, b] += 10 * torch.randn(cross_v[:, b].shape, generator=g)
        return paths


@pytest.mark.parametrize("primer", [[2, 3, 4], [2, 3, 4, NOTIME]], ids=["timestamps", "notimestamps"])
@pytest.mark.parametrize("rules", [False, True], ids=["free", "rules"])
def test_streamed_drafted_decoding_is_greedy(model, primer, rules):
    """decode_stream refills a row as soon as its utterance is done: whatever the order of the
    utterances and of the refills, each one gets its greedy output, in fewer calls than fixed batches."""
    import random

    torch.manual_seed(5)
    n, rows, backend = 8, 3, _DistinctBackend(model)
    feats = model.frontend(torch.randn(n, TINY.num_samples))
    pr = torch.tensor([primer] * n)
    insertions = set(range(FIRST_T, LAST_T + 1)) if rules else set()

    def session(batch: int, eos: int, **kw) -> GreedySession:
        return GreedySession(
            backend, TINY, batch, 32, eos=eos, dtype=torch.float32, device=torch.device("cpu"),
            use_cuda_graph=False, buckets=[8, 16, 32], rules=_rules(eos) if rules else None, **kw,
        )  # fmt: skip

    full = session(n, eos=1)
    full.encode(feats)
    eos = max(full.decode(pr, 20), key=len)[4]  # an <eos> some utterances emit early (rules permitting)
    greedy = session(n, eos)
    greedy.encode(feats)
    ref = greedy.decode(pr, 20)
    assert len({tuple(r) for r in ref}) > n // 2

    rng = random.Random(0)
    drafts = []
    for i, r in enumerate(ref):
        bad = list(r)
        bad[1:3] = [7, 8, 9]
        drafts.append([r, bad, [], [rng.randrange(2, 50) for _ in range(20)]][i % 4])
    sess = session(rows, eos, draft_windows=(2, 4, 8))
    for seed in range(3):
        order = list(range(n))
        random.Random(seed).shuffle(order)
        given = iter([drafts[j] for j in order])
        batches = (feats[order[i : i + rows]] for i in range(0, n, rows))
        drafts_of = lambda paths: [next(given) for _ in paths]  # noqa: E731
        out, calls = sess.decode_stream(primer, batches, drafts_of, 20, insertions)
        assert out == [ref[j] for j in order], seed
        fixed = 0
        for i in range(0, n, rows):
            idx = order[i : i + rows]
            batch = session(len(idx), eos, draft_windows=(2, 4, 8))
            batch.encode(feats[idx])
            got, c = batch.decode_drafted(pr[idx], [drafts[j] for j in idx], 20, insertions)
            assert got == [ref[j] for j in idx]
            fixed += c
        assert calls <= fixed, (calls, fixed)


@pytest.mark.parametrize("seconds", [0.3, 1.0, 1.9, 2.0])
def test_encoder_on_trimmed_features(model, seconds):
    """Features cut after frames_needed, the rest of the subsampled frames filled with padding_frame,
    give the encoder output of the whole padded input."""
    from owsm_trt.modeling import frames_needed, padding_frame

    torch.manual_seed(6)
    n = int(seconds * TINY.sample_rate)
    speech = torch.zeros(2, TINY.num_samples)
    speech[:, :n] = torch.randn(2, n)
    feats = model.frontend(speech)
    ref = model.encoder(feats)
    t = frames_needed(TINY, n)
    assert t < TINY.num_frames or seconds >= 1.9
    model.encoder.fill = padding_frame(model)
    try:
        torch.testing.assert_close(model.encoder(feats[:, :t]), ref, rtol=1e-4, atol=1e-4)
    finally:
        model.encoder.fill = None


def test_fp8_encoder_quantization():
    """Calibrated FP8 fake quantization keeps the encoder output close and swaps in the FP8 layers."""
    import copy

    from owsm_trt import fp8

    torch.manual_seed(0)
    m = OWSMModel(TINY).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    feats = m.frontend(torch.randn(2, TINY.num_samples))
    ref = m.encoder(feats)
    q = copy.deepcopy(m)
    amax = fp8.calibrate(q.encoder, q.encoder, [feats])
    assert fp8.quantize(q.encoder, amax) == len(amax) > 0
    assert any(isinstance(mod, fp8.QConv2d) for mod in q.encoder.modules())
    out = q.encoder(feats)
    assert float((out - ref).norm() / ref.norm()) < 0.1


@pytest.mark.parametrize("plugins", [{"self", "cross"}, {"self"}, {"cross"}])
@pytest.mark.parametrize("fused", [False, True], ids=["qkv", "fused_qkv"])
def test_decoder_step_with_attention_ops(model, plugins, fused):
    """The decoder's export path with attn_plugin ops (their PyTorch implementations) == the plain step."""
    import copy

    m = copy.deepcopy(model)
    if fused:
        m.fuse_projections()
    torch.manual_seed(5)
    L, H, dk, T = TINY.dec_layers, TINY.dec_heads, 8, TINY.enc_frames
    sk, sv = torch.randn(L, 3, H, 16, dk), torch.randn(L, 3, H, 16, dk)  # garbage past the offsets
    ck, cv = torch.randn(L, 3, H, T, dk), torch.randn(L, 3, H, T, dk)
    tokens, offset = torch.randint(2, 50, (3, 4)), torch.tensor([0, 5, 12])
    ref = m.decoder.step(tokens, offset, list(sk), list(sv), list(ck), list(cv))
    m.decoder.plugins = frozenset(plugins)
    logits, new_kv = m.decoder.step(tokens, offset, list(sk), list(sv), list(ck), list(cv))
    torch.testing.assert_close(logits, ref[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(new_kv, ref[1], rtol=1e-5, atol=1e-5)


def test_decoder_export_with_attention_plugins(model, tmp_path):
    """Each layer's attention becomes one node the TensorRT ONNX parser maps to an attn_plugin plugin."""
    onnx = pytest.importorskip("onnx")
    from owsm_trt.build import BuildOptions, export_decoder_onnx

    opts = BuildOptions(dtype="float32", max_batch=4, capacity=32, max_prefill=8, max_verify=4)
    path = export_decoder_onnx(model, tmp_path / "decoder.onnx", opts, plugins=frozenset({"self", "cross"}))
    assert model.decoder.plugins == frozenset()
    graph = onnx.load(str(path), load_external_data=False).graph
    nodes = [n for n in graph.node if n.domain == "owsm"]
    L = TINY.dec_layers
    assert sorted(n.op_type for n in nodes) == ["cross_attn"] * L + ["self_attn"] * L
    for n in nodes:
        assert {a.name: a.s for a in n.attribute} == {"plugin_namespace": b"owsm", "plugin_version": b"1"}
    self_nodes = [n for n in nodes if n.op_type == "self_attn"]
    assert [list(n.input[1:]) for n in self_nodes] == [
        [f"self_k_{i}", f"self_v_{i}", "offset"] for i in range(L)
    ]
    cross = [n for n in nodes if n.op_type == "cross_attn"]
    assert [list(n.input[1:]) for n in cross] == [[f"cross_k_{i}", f"cross_v_{i}"] for i in range(L)]


@pytest.mark.parametrize("kind", ["aed", "ctc"])
def test_export_without_trimming_runs_at_num_frames(tmp_path, kind):
    """Utterance-MVN models (e.g. OWSM-CTC v3.2) get whole windows: their encoder export has a static
    time dimension, which must be num_frames (the engine profiles' length), and computes the model."""
    import dataclasses

    pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    from owsm_trt.build import BuildOptions, export_ctc_onnx, export_onnx, trims_features
    from owsm_trt.modeling import OWSMCTCModel

    cfg = dataclasses.replace(TINY, normalize="utterance_mvn")
    if kind == "ctc":
        cfg = dataclasses.replace(
            cfg, dec_layers=0, dec_ffn=0, dec_heads=0, ctc_only=True, prompt_d=16, prompt_heads=2,
            prompt_ffn=32, prompt_layers=2, interctc_layers=(1,), cross_attn_layers=(1,),
        )  # fmt: skip
    assert not trims_features(cfg)
    torch.manual_seed(9)
    m = (OWSMCTCModel if kind == "ctc" else OWSMModel)(cfg).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    opts = BuildOptions(
        dtype="float32", max_batch=4, attn_plugin=False, attn_plugin_steps=False, dwconv_plugin=False
    )
    feats = m.frontend(torch.randn(2, cfg.num_samples))
    assert feats.shape[1] == cfg.num_frames
    if kind == "ctc":
        path = export_ctc_onnx(m, tmp_path, opts)
        prompt, prefix = torch.tensor([[2, 9, 10]] * 2), torch.tensor([[4, 5]] * 2)
        (ids,) = ReferenceEvaluator(str(path)).run(
            None, {"feats": feats.numpy(), "prompt": prompt.numpy(), "prefix": prefix.numpy()}
        )
        assert torch.equal(torch.from_numpy(ids), m(feats, prompt, prefix))
        return
    path, _ = export_onnx(m, tmp_path, opts)
    k, v, ids = ReferenceEvaluator(str(path)).run(None, {"feats": feats.numpy()})
    enc = m.encoder(feats)
    kv = m.decoder.cross_kv(enc)
    torch.testing.assert_close(torch.from_numpy(k), kv[:, 0], rtol=1e-4, atol=1e-4)
    full = m.ctc(enc)  # every frame computed: then two blanks (see build._EncoderGraph)
    assert torch.equal(torch.from_numpy(ids), torch.cat([full, torch.zeros_like(full[:, :2])], 1))


def test_fp8_scale_files(tmp_path, monkeypatch):
    """Scale files: per-recording calibration, a file keyed by the checkpoint's sha256, found by it,
    margin and fp16 layers when quantizing."""
    import copy
    import hashlib

    from owsm_trt import fp8

    torch.manual_seed(0)
    m = OWSMModel(TINY).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    feats = m.frontend(torch.randn(3, TINY.num_samples))
    rows = fp8.calibrate_rows(m.encoder, m.encoder, [feats[:2], feats[2:]])
    amax = fp8.calibrate(m.encoder, m.encoder, [feats])
    assert all(len(v) == 3 for v in rows.values())
    assert {k: max(v) for k, v in rows.items()} == pytest.approx(amax)
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"weights")
    sha = fp8.checkpoint_sha256(ckpt, tmp_path / "ids.json")
    assert sha == hashlib.sha256(b"weights").hexdigest() and (tmp_path / "ids.json").exists()
    blob = tmp_path / "blobs" / ("a" * 64)
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    (tmp_path / "link.pth").symlink_to(blob)
    assert fp8.checkpoint_sha256(tmp_path / "link.pth") == "a" * 64  # a Hugging Face cache blob: its name
    monkeypatch.setattr(fp8, "SCALES_DIR", tmp_path / "scales")
    (tmp_path / "scales").mkdir()
    assert fp8.packaged_scales(sha) is None
    fp8.save_scales(tmp_path / "scales" / "tiny.json", amax, "tiny", sha, {"recordings": 3})
    assert fp8.packaged_scales(sha) == tmp_path / "scales" / "tiny.json"
    assert fp8.load_scales(fp8.packaged_scales(sha))["amax"] == pytest.approx(amax)
    q = copy.deepcopy(m)
    n = fp8.quantize(q.encoder, amax, margin=2.0, skip=r"feed_forward\.w_2")
    assert 0 < n < len(amax)
    w1 = q.encoder.encoders[0].feed_forward.w_1
    assert isinstance(w1, fp8.QLinear) and not isinstance(q.encoder.encoders[0].feed_forward.w_2, fp8.QLinear)
    key = "encoders.0.feed_forward.w_1"
    assert float(w1.x_scale) == pytest.approx(2.0 * amax[key] / fp8.E4M3_MAX, rel=1e-3)


def test_packaged_fp8_scales(tmp_path, monkeypatch, caplog):
    """The scale files the package ships load, name encoder layers, and are found only for their checkpoint:
    from_pretrained(fp8=True) of another checkpoint warns and stays fp16."""
    import logging

    from owsm_trt import fp8, hub
    from owsm_trt.pipeline import _fp8_scales

    files = sorted(fp8.SCALES_DIR.glob("*.json"))
    assert {f.stem for f in files} >= {"owsm_v4_medium_1B", "owsm_ctc_v4_1B"}
    for f in files:
        d = fp8.load_scales(f)
        assert len(d["checkpoint_sha256"]) == 64 and d["model"].startswith("espnet/")
        assert all(k.startswith(("embed.", "encoders.", "conditioning_layer")) for k in d["amax"])
        assert all(v > 0 for v in d["amax"].values())
        assert fp8.packaged_scales(d["checkpoint_sha256"]) == f
    ckpt = tmp_path / "other.pth"
    ckpt.write_bytes(b"not a packaged checkpoint")
    monkeypatch.setattr(hub, "cache_dir", lambda: tmp_path / "cache")
    model = hub.ModelFiles(name="someone/else", root=tmp_path, config=tmp_path / "c.yaml", checkpoint=ckpt)
    with caplog.at_level(logging.WARNING):
        assert _fp8_scales(model, True) is None
    assert "no FP8 scales shipped" in caplog.text
    assert _fp8_scales(model, str(files[0])) == files[0]  # a given file (with a warning: other checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (streams)")
def test_streamed_decoding_with_encoder_lanes(model, monkeypatch):
    """decode_stream with one row: its two staging slots encoded on two streams at the same time give the
    outputs of one encoder stream."""
    import copy

    import owsm_trt.runtime as runtime

    torch.manual_seed(6)
    m = copy.deepcopy(model).cuda()
    dev = torch.device("cuda")
    n = 7
    feats = m.frontend(torch.randn(n, TINY.num_samples, device=dev))
    sess = GreedySession(
        TorchBackend(m), TINY, 1, 32, eos=1, dtype=torch.float32, device=dev, use_cuda_graph=True,
        buckets=[8, 16, 32], draft_windows=(2, 4, 8),
    )  # fmt: skip

    def run(lanes: bool) -> list[list[int]]:
        monkeypatch.setattr(runtime, "ENCODER_LANES", lanes)
        batches = (feats[i : i + 1] for i in range(n))
        out, _ = sess.decode_stream([2, 3, 4], batches, lambda paths: [[5, 6, 7]] * len(paths), 20)
        return out

    ref = run(False)
    assert len(ref) == n
    for _ in range(2):
        assert run(True) == ref
