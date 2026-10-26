"""Unused plain-decoder setup is deferred without capturing over a live prefix."""

import pytest
import torch
from test_model import TINY

from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import GreedySession, TimestampRules, TorchBackend


def test_plain_setup_is_absent_until_needed_and_idempotent():
    class Backend:
        def __init__(self):
            self.calls = 0

        def step_fn(self, tok, offs, *args, **kwargs):
            self.calls += 1
            assert not tok.any() and not offs.any()
            return lambda: None

    backend = Backend()
    sess = GreedySession(
        backend,
        TINY,
        1,
        32,
        1,
        torch.float32,
        torch.device("cpu"),
        use_cuda_graph=False,
        buckets=[8, 16, 32],
        lazy_drafts=True,
        lazy_plain=True,
    )
    assert backend.calls == 0 and not sess._steps and not sess._plain_ready
    sess.pos.fill_(1000)
    sess.offs.fill_(1000)
    sess.tok.fill_(-1)
    sess.done.fill_(True)
    sess._prepare_plain()
    steps = dict(sess._steps)
    assert backend.calls == 3 and sess._plain_ready and not sess.done.any()
    sess._prepare_plain()
    assert backend.calls == 3 and sess._steps == steps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("timed", [False, True])
@torch.inference_mode()
def test_deferred_plain_after_drafts_prefill_and_repeated_switches(rows, graphs, timed):
    torch.manual_seed(82)
    dev = torch.device("cuda")
    model = OWSMModel(TINY).eval().requires_grad_(False).to(dev)
    rules = TimestampRules(6, 30, 40, 1, TINY.vocab_size, dev) if timed else None
    cross = [
        torch.randn(TINY.dec_layers, rows, TINY.dec_heads, TINY.enc_frames, 8, device=dev) for _ in range(2)
    ]
    primers = [[2, 3, 6] if not timed or r % 2 else [2, 3, 4] for r in range(rows)]
    long_primers = [[2, 3] + list(range(7, 24)) + [p[-1]] for p in primers]
    insertions = set(range(30, 41)) if timed else set()
    reference = None
    for lazy in (False, True):
        sess = GreedySession(
            TorchBackend(model),
            TINY,
            rows,
            64,
            -1,
            torch.float32,
            dev,
            use_cuda_graph=graphs,
            buckets=[16, 32, 64],
            draft_windows=(2, 4, 8),
            rules=rules,
            lazy_drafts=True,
            lazy_verify_graphs=True,
            cache_draft_flags=True,
            lazy_plain=lazy,
        )
        sess.cross_k.copy_(cross[0])
        sess.cross_v.copy_(cross[1])
        outputs = []
        for primer, count in ((primers, 40), (long_primers, 34)):
            outputs.append(sess.decode_drafted(primer, [[] for _ in range(rows)], count, insertions)[0])
            assert bool(sess._steps) != lazy
            assert len(sess.graphs) == (3 if graphs and not lazy else 0)
        # Language-ID-style prefill alone must not construct plain token-by-token steps.
        logits, _, _ = sess._prefill(torch.tensor(primers, device=dev), sess.cross_k, sess.cross_v)
        prefill_logits = logits.clone()
        assert bool(sess._steps) != lazy
        plain_steps = None
        for primer, count in ((primers, 40), (long_primers, 34), (primers, 40)):
            outputs.append(sess.decode(torch.tensor(primer, device=dev), count))
            assert sess._plain_ready and len(sess._steps) == 3
            if plain_steps is None:
                plain_steps = {k: id(v) for k, v in sess._steps.items()}
            else:
                assert plain_steps == {k: id(v) for k, v in sess._steps.items()}
            outputs.append(sess.decode_drafted(primers, [[] for _ in range(rows)], 40, insertions)[0])
        if reference is None:
            reference = (outputs, prefill_logits)
        else:
            assert outputs == reference[0]
            assert torch.equal(prefill_logits, reference[1])
