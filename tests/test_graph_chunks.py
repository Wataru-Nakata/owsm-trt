"""Preserve every original host check and bucket transition; GPU state comparisons run on compute."""

from types import SimpleNamespace

import pytest
import torch
from graph_chunk_experiment import prepare, run_chunks
from test_beam import CFG

from owsm_trt.runtime import BeamSession, GreedySession, TimestampRules, _Session


@pytest.mark.parametrize('start,steps,stop', [(2, 0, 50), (2, 1, 0), (2, 27, 50), (3, 20, 5), (7, 8, 0), (14, 12, 22)])
def test_same_host_checks_and_bucket_transitions(start, steps, stop):
    def fake(grouped):
        state = SimpleNamespace(chunk=4, _chunk_width=4, _chunk_graphs={}, events=[], pos=start,
                                calls=0, checks=[], buckets=[4, 8, 16, 32])

        def graph(n):
            def replay():
                state.events.append(n)
                state.pos += n
                state.calls += n
            return SimpleNamespace(replay=replay)

        state.graphs = {c: graph(1) for c in state.buckets}
        if grouped:
            state._chunk_graphs = {c: graph(4) for c in (16, 32)}

        def grow(c, p):
            state.events.append(('grow', c, p))
            return next(x for x in state.buckets if x >= p)

        def done(_):
            state.checks.append(state.pos)
            return state.pos >= stop

        state._grow, state._all_done = grow, done
        return state

    reference, candidate = fake(False), fake(True)
    c = next(x for x in reference.buckets if x > start)
    _Session._run(reference, c, start, steps, None)
    run_chunks(candidate, c, start, steps, None)
    assert reference.calls == candidate.calls
    assert reference.pos == candidate.pos
    assert reference.checks == candidate.checks
    assert [x for x in reference.events if isinstance(x, tuple)] == [x for x in candidate.events if isinstance(x, tuple)]
    assert len(candidate.events) <= len(reference.events)


class DependentBackend:
    def __init__(self, end):
        self.end = end

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, **kw):
        vocab = torch.arange(CFG.vocab_size, device=logits.device)
        rows = torch.arange(tokens.shape[0], device=logits.device)

        def run():
            target = (tokens + offset[:, None] + rows[:, None]) % 7 + 1
            logits.copy_(-(vocab - target[..., None]).abs().float())
            logits[..., 0] = torch.where(offset[:, None] >= self.end + rows[:, None] % 3, 20., -20.)
            new_kv.copy_((tokens + offset[:, None]).view(1, 1, tokens.shape[0], 1, tokens.shape[1], 1))
        return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compute allocation required')
@pytest.mark.parametrize('beam', [1, 5])
@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('timestamps', [False, True])
@pytest.mark.parametrize('end', [3, 11, 50])
@pytest.mark.parametrize('fused', [False, True])
@torch.inference_mode()
def test_four_steps_match_single_graph_state_buckets_limits_and_reuse(beam, batch, timestamps, end, fused):
    device = torch.device('cuda')
    rules = TimestampRules(7, 5, 6, 0, 8, device)
    options = dict(backend=DependentBackend(end), cfg=CFG, batch=batch, capacity=32, eos=0,
                   dtype=torch.float32, device=device, use_cuda_graph=True, chunk=4,
                   buckets=[4, 8, 16, 32], rules=rules)
    if beam == 1:
        reference, candidate = (GreedySession(**options, fused_select=fused) for _ in range(2))
    else:
        reference, candidate = (BeamSession(**options, beam=beam, fused_rank=fused, fused_finish=fused, fused_state=fused) for _ in range(2))
    assert prepare(candidate)['captures'] == [16, 32]
    for length in (3, 7, 14, 29):
        primer = torch.tensor([[2]*(length-1) + [3 if timestamps else 7]]*batch, device=device)
        for limit in (1, min(5, 32-length), 32-length):
            assert candidate.decode(primer, limit) == reference.decode(primer, limit)
            for key in ('pos', 'done', 't_notime', 't_first', 't_count', 't_last', 'tok'):
                torch.testing.assert_close(getattr(candidate, key), getattr(reference, key), rtol=0, atol=0)
            position = int(reference.pos)
            torch.testing.assert_close(candidate.seq[:, :position+1], reference.seq[:, :position+1], rtol=0, atol=0)
            bucket = reference._bucket(max(length+1, position))
            for left, right in zip(candidate.caches[bucket], reference.caches[bucket]):
                torch.testing.assert_close(left[..., :position, :], right[..., :position, :], rtol=0, atol=0)
            if beam > 1:
                for key in ('step', 'score', 'active', 'best_score', 'best_end', 'numerical_error'):
                    torch.testing.assert_close(getattr(candidate, key), getattr(reference, key), rtol=0, atol=0)
                for b, end_at in enumerate(reference.best_end.tolist()):
                    torch.testing.assert_close(candidate.best_seq[b, :end_at], reference.best_seq[b, :end_at], rtol=0, atol=0)
