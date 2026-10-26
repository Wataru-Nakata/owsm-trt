"""Deferred verification setup must precede prefill and preserve repeated requests."""

import pytest
import torch
from test_model import TINY

from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import GreedySession, TimestampRules, TorchBackend


def test_unused_drafts_and_idempotent_setup():
    class Backend:
        def __init__(self):
            self.calls = 0

        def step_fn(self, *args, **kwargs):
            self.calls += 1
            return lambda: None

    backend = Backend()
    session = GreedySession(backend, TINY, 1, 32, 1, torch.float32, torch.device("cpu"),
                            use_cuda_graph=False, buckets=[8, 16, 32], draft_windows=(2, 4, 8),
                            lazy_drafts=True)
    assert backend.calls == 3 and not session._verify
    session._prepare_drafts()
    assert backend.calls == 15 and len(session._verify) == 12
    session._prepare_drafts()
    assert backend.calls == 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("timestamp", [False, True])
@torch.inference_mode()
def test_lazy_capture_before_long_prefill_and_reuse(timestamp):
    torch.manual_seed(72)
    model = OWSMModel(TINY).eval().cuda()
    rules = TimestampRules(6, 30, 40, 1, TINY.vocab_size, torch.device("cuda")) if timestamp else None
    cross = torch.randn(2, 1, 4, TINY.enc_frames, 8, device="cuda")
    reference = None
    for lazy in (False, True):
        sess = GreedySession(TorchBackend(model), TINY, 1, 32, 1, torch.float32, torch.device("cuda"),
                             buckets=[16, 32], draft_windows=(2, 4, 8), rules=rules, lazy_drafts=lazy)
        sess.cross_k.copy_(cross)
        sess.cross_v.copy_(cross)
        assert bool(sess._verify) != lazy
        # Start with plain decoding, then a prompt longer than every verification window.
        plain = torch.tensor([[2, 3, 4, 6]], device="cuda")
        short = [2, 3, 4] if timestamp else [2, 3, 4, 6]
        primer = [2, 3, 7, 8, 9, 10, 11, 12, 4] + ([] if timestamp else [6])
        out = [sess.decode(plain, 12)]
        assert bool(sess._verify) != lazy
        insertions = set(range(30, 41)) if timestamp else set()
        out.append(sess.decode_drafted([primer], [[5, 8, 11]], 20, insertions)[0])
        verify_ids = {key: id(ver) for key, ver in sess._verify.items()}
        out.append(sess.decode_drafted([short], [[]], 24, insertions)[0])
        out.append(sess.decode(plain, 12))
        assert verify_ids == {key: id(ver) for key, ver in sess._verify.items()}
        if reference is None:
            reference = out
        else:
            assert out == reference
