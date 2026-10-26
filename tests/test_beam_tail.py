"""Single remaining beam group: routing/state tests and GPU-only strided-cache checks."""

import pytest
import torch
from test_beam import CFG, ScriptedBackend, session
from test_beam_cross import gluon_only

from owsm_trt import megakernel, megakernel_gluon
from owsm_trt.runtime import BeamSession


class GroupBackend(ScriptedBackend):
    """Cross KV carries a per-utterance target end position, so every group ends separately."""

    name = "synthetic-groups"

    def supports_shared_cross(self, rows):
        return rows <= 5

    def step_fn(self, tokens, offset, k, v, ck, cv, logits, new_kv, **kw):
        def run():
            target = ck[0, :, 0, 0, 0].repeat_interleave(tokens.shape[0] // ck.shape[1])
            ended = offset >= target
            logits.fill_(-torch.inf)
            logits[:, :, 1] = torch.where(ended, -torch.inf, 0.).unsqueeze(1)
            logits[:, :, 0] = torch.where(ended, 0., -torch.inf).unsqueeze(1)
            new_kv.fill_(1)
        return run


@pytest.mark.parametrize("last_group", [0, 1, 2])
@pytest.mark.parametrize("chunk", [1, 4])
def test_tail_reuse_partial_end_bucket_and_capacity(last_group, chunk):
    backend = GroupBackend()
    ref = session(batch=3, beam=3, backend=backend, chunk=chunk, diagnostics=True)
    fast = session(batch=3, beam=3, backend=backend, chunk=chunk, diagnostics=True, compact_tail=True)
    for targets, primer_len, limit in (([4, 5, 13], 3, 13), ([3, 3, 3], 3, 10), ([5, 6, 99], 5, 11)):
        targets[last_group], targets[2] = targets[2], targets[last_group]
        primer = torch.full((3, primer_len), 2)
        for s in (ref, fast):
            for b, target in enumerate(targets):
                s.cross_kb[:, b].fill_(target)
            s._expand_cross()
        assert fast.decode(primer, limit) == ref.decode(primer, limit)
        torch.testing.assert_close(fast.best_score, ref.best_score, rtol=0, atol=0)
        assert fast.get_diagnostics()['decoder_calls'] == ref.get_diagnostics()['decoder_calls']
        if max(targets) > 6:
            assert fast._tail_started[0] == last_group
            assert fast.get_diagnostics()['decoder_row_calls'] < ref.get_diagnostics()['decoder_row_calls']
            d = fast.get_diagnostics()
            assert sum(u['executed_decoder_calls'] for u in d['utterances']) * fast.beam == d['decoder_row_calls']
            assert all(u['extra_decoder_calls'] >= 0 for u in d['utterances'])
        else:
            assert fast._tail_started is None


def test_tail_unsupported_backend_is_explicit():
    s = session(batch=3, compact_tail=True)
    assert not s.compact_tail and s.compact_tail_fallback
    assert not s._tail_steps and not s._tail_graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compute allocation required')
@pytest.mark.parametrize('last_group', [0, 1, 2])
@pytest.mark.parametrize('clear_once', [False, True])
@torch.inference_mode()
def test_tail_graph_replay_reset_and_bucket_growth(last_group, clear_once):
    device = torch.device('cuda')
    args = (GroupBackend(), CFG, 3, 3, 32, 0, torch.float32, device)
    ref = BeamSession(*args, buckets=[4, 8, 16, 32], diagnostics=True)
    fast = BeamSession(*args, buckets=[4, 8, 16, 32], diagnostics=True, compact_tail=True,
                       kv_clear_once=clear_once)
    for targets in ([4, 5, 27], [20, 7, 6], [5, 5, 5]):
        targets[last_group], targets[2] = targets[2], targets[last_group]
        for s in (ref, fast):
            for b, target in enumerate(targets):
                s.cross_kb[:, b].fill_(target)
            s._expand_cross()
        if clear_once:
            for pair in fast.caches.values():
                for cache in pair:
                    cache.fill_(torch.nan)
        primer = torch.full((3, 3), 2, device=device)
        assert fast.decode(primer, 29) == ref.decode(primer, 29)
        torch.testing.assert_close(fast.best_score, ref.best_score, rtol=0, atol=0)
        assert fast.get_diagnostics()['decoder_calls'] == ref.get_diagnostics()['decoder_calls']


@gluon_only
@pytest.mark.parametrize('group', [0, 1, 2])
@pytest.mark.parametrize('prefetch', [False, True])
@torch.inference_mode()
def test_gluon_strided_group_cache_matches_contiguous(group, prefetch):
    from test_megakernel import CFG as config
    from test_megakernel import _model

    torch.manual_seed(21)
    dev = torch.device('cuda')
    model = _model(config)
    kernel = megakernel_gluon.GluonDecoder(megakernel.MegaDecoder(model.decoder, dev),
                                         prefetch_attention=prefetch)
    L, H, dk, R, C, T = config.dec_layers, config.dec_heads, config.d_model // config.dec_heads, 5, 32, config.enc_frames
    large = [torch.randn(L, R * 3, H, C, dk, device=dev, dtype=torch.float16) for _ in range(2)]
    cross = [torch.randn(L, 3, H, T, dk, device=dev, dtype=torch.float16) for _ in range(2)]
    views = [a[:, group * R:(group + 1) * R] for a in large]
    cv = [a[:, group:group + 1] for a in cross]
    copies = [a.clone() for a in views + cv]
    before = [a.clone() for a in large]
    tok = torch.randint(0, config.vocab_size, (R, 1), device=dev)
    offsets = torch.full((R,), 7, dtype=torch.long, device=dev)
    runs, owners, outputs = [], [], []
    for k, v, ck, cval in ((*views, *cv), copies):
        logits = torch.empty(R, 1, config.vocab_size, device=dev)
        kv = torch.empty(L, 2, R, H, 1, dk, device=dev, dtype=torch.float16)
        run = kernel.step_fn(tok, offsets, k, v, ck, cval, logits, kv)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        runs.append(graph)
        owners.append(run)  # Retain both externally allocated workspaces through graph replay.
        outputs.append((logits, kv))
    for _ in range(3):
        for graph in runs:
            graph.replay()
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(views, copies):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(large, before):
        for other in set(range(3)) - {group}:
            torch.testing.assert_close(a[:, other * R:(other + 1) * R], b[:, other * R:(other + 1) * R], rtol=0, atol=0)
    kernel.check()


def test_tail_trial_failure_refreshes_cross_for_each_utterance():
    class Broken:
        shared_cross_kv = True
        name = 'broken-tail'

        def check(self):
            pass

        def step_fn(self, *args, **kwargs):
            raise ValueError('deliberate tail trial failure')

    backend = megakernel.MegaBackend(GroupBackend(), Broken(), 16, 5, trial=True)
    fast = session(batch=3, beam=3, backend=backend, compact_tail=True, chunk=1)
    assert fast.compact_tail and backend.failed
    assert all(getattr(fn, 'prepare_cross', None) is not None for fn in fast._tail_steps.values())
    for targets in ([4, 5, 13], [12, 4, 5]):
        ref = session(batch=3, beam=3, backend=GroupBackend(), chunk=1)
        for s in (ref, fast):
            for b, target in enumerate(targets):
                s.cross_kb[:, b].fill_(target)
            s._expand_cross()
        primer = torch.full((3, 3), 2)
        assert fast.decode(primer, 13) == ref.decode(primer, 13)
