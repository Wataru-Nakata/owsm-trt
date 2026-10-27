"""Opt-in N normalization: preserve masks/errors and full-vocabulary log probabilities."""

import pytest
import torch
from test_beam import ready, session

from owsm_trt import beam_kernels
from owsm_trt.runtime import TimestampRules, capture_graph


@pytest.fixture
def device():
    if beam_kernels.triton is None or not torch.cuda.is_available():
        pytest.skip('CUDA required; large vocabulary tests belong on a compute node')
    return torch.device('cuda')


@pytest.mark.parametrize('vocab', [8, 2053, 50002])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('kind', ['finite', 'near_tie', 'sparse', 'bad'])
def test_normalizer_mask_and_error_contract(device, vocab, shared, kind):
    s = session(batch=3, beam=5, device=device, fused_rank=True, fused_finish=False)
    ready(s)
    s.fused_lse = True
    s.logits = torch.empty(s.rows, 1, vocab, device=device)
    s._candidates = torch.empty(s.rows, vocab, device=device)
    blocks = (vocab + 1023) // 1024
    s._bad_blocks = torch.empty(s.rows, blocks, dtype=torch.bool, device=device)
    s._lse_partials = torch.empty(s.rows, 2, blocks, device=device)
    s.rules = TimestampRules(vocab - 1, vocab - 3, vocab - 2, 0, vocab, device)
    s.active.fill_(True)
    s.active[9] = False
    s.done[2] = True
    s.t_first[:5] = True
    s.t_notime[5:] = True
    s.score.copy_(torch.arange(s.rows, device=device) * -.3)
    rows = s.batch if shared else s.rows
    storage = torch.zeros(rows, 37, vocab, device=device)
    raw = storage[:, -1]  # Strided prefill must use source-row strides.
    raw.copy_(-torch.arange(vocab, device=device)[None, :].float() * .001)
    raw[:, 1] = 100.  # Forbidden for first timestamps, still dominates the full-vocabulary normalizer.
    if kind == 'near_tie':
        raw.zero_()
        raw[:, -3] = 1.
        raw[:, -2] = torch.nextafter(torch.tensor(1., device=device), torch.tensor(torch.inf, device=device))
    elif kind == 'sparse':
        raw[0].fill_(-torch.inf)
        raw[0, -3] = 0.  # Fewer permitted candidates than K.
        raw[1].fill_(-torch.inf)  # Exhausted without a numerical error.
    elif kind == 'bad':
        raw[0, 1] = torch.nan
        raw[1, 1] = torch.inf
        raw[-1, 1] = torch.nan  # Done source cannot set a numerical-error flag.
    if shared:
        s.active.zero_()
        s.active[::s.beam] = True
        s.logits.fill_(torch.nan)  # Never read the stale expanded prefill logits.
        source = raw
    else:
        s.logits[:, -1].copy_(raw)
        source = s.logits[:, -1]
    reference = beam_kernels.candidates(s, torch.log_softmax(source, -1), raw if shared else None).clone()
    error = s.numerical_error.clone()
    s.numerical_error.zero_()
    actual = beam_kernels.candidates(s, None, raw if shared else None)
    torch.testing.assert_close(torch.isfinite(actual), torch.isfinite(reference), rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(s.numerical_error, error, rtol=0, atol=0)
    # Exact index equality is deliberately not required for fp32 near ties; global top-K remains unchanged.
    torch.testing.assert_close(actual.view(3, -1).topk(5).values, reference.view(3, -1).topk(5).values,
                               rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('timestamps', [False, True])
@pytest.mark.parametrize('shared', [False, True])
def test_normalizer_decode_graph_buckets_limits_and_reuse(device, batch, timestamps, shared):
    rules = TimestampRules(7, 5, 6, 0, 8, device)
    options = dict(batch=batch, beam=5, device=device, rules=rules, diagnostics=True,
                   use_cuda_graph=True, upper_bound=True, prefill_shared=shared)
    reference = session(**options)
    candidate = session(**options, fused_lse=True)
    assert candidate.fused_lse
    for length in (3, 7):
        primer = torch.tensor([[2] * (length - 1) + [3 if timestamps else 7]] * batch, device=device)
        for limit in (16 - length, 5, 1):
            assert candidate.decode(primer, limit) == reference.decode(primer, limit)
            for key in ('active', 'done', 'best_end', 'best_seq', 'seq'):
                torch.testing.assert_close(getattr(candidate, key), getattr(reference, key), rtol=0, atol=0)
            for key in ('score', 'best_score'):
                torch.testing.assert_close(getattr(candidate, key), getattr(reference, key), rtol=2e-6, atol=2e-5)


def test_normalizer_capture_replays_updated_full_vocabulary(device):
    s = session(batch=1, beam=5, device=device, fused_lse=True)
    ready(s)
    s.active.fill_(True)
    graph = capture_graph(lambda: beam_kernels.candidates(s, None))
    for value in (0., 100., -100.):
        s.logits.fill_(value)
        s.logits[0, 0, 2] = value + 2
        graph.replay()
        actual = s._candidates.clone()
        reference = beam_kernels.candidates(s, torch.log_softmax(s.logits[:, -1], -1)).clone()
        torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-5)


def test_normalizer_cpu_fallback_allocates_no_scratch():
    s = session(fused_lse=True)
    assert not s.fused_lse and s._lse_partials is None
