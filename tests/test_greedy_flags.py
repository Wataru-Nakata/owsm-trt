"""Draft-only flag caching must survive mixed primers and plain-decode state replacement."""

import torch
from test_model import TINY

from owsm_trt.runtime import GreedySession, TimestampRules


def test_mixed_flags_and_plain_state_invalidation():
    class Backend:
        def step_fn(self, *args, **kwargs):
            return lambda: None

    rules = TimestampRules(49, 30, 40, 1, 50, torch.device("cpu"))
    sess = GreedySession(
        Backend(),
        TINY,
        2,
        32,
        1,
        torch.float32,
        torch.device("cpu"),
        use_cuda_graph=False,
        rules=rules,
        cache_draft_flags=True,
    )
    primers = [[2, 49], [2, 3]]
    sess._draft_flags_start(primers)
    assert sess.t_notime.tolist() == [True, False]
    sess._draft_flags_start(primers)
    assert sess.t_notime.tolist() == [True, False]
    # Plain decode overwrites the shared flags. Returning to the same draft primers must restore them.
    sess._rules_start(torch.tensor([[2, 3], [2, 3]]))
    assert sess.t_notime.tolist() == [False, False]
    sess._draft_flags_start(primers)
    assert sess.t_notime.tolist() == [True, False]
    sess._draft_flags_start(primers[::-1])
    assert sess.t_notime.tolist() == [False, True]
    sess._draft_flags_start([[49], [49]])
    assert sess.t_notime.tolist() == [True, True]
