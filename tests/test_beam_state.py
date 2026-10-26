"""Exact beam metadata fusion, including duplicate/cyclic parents and graph reuse."""

import pytest
import torch
from test_beam import session
from test_beam_kernels import device as device

from owsm_trt import beam_kernels
from owsm_trt.runtime import TimestampRules


@pytest.mark.parametrize("mapping", [[0, 1, 2, 3, 4], [2, 2, 2, 2, 2], [4, 0, 1, 2, 3], [3, 1, 3, 0, 2]])
@pytest.mark.parametrize("capacity,pos", [(16, 0), (16, 15), (512, 511)])
@pytest.mark.parametrize("rules", [False, True])
def test_reorder_state_matches_torch(device, mapping, capacity, pos, rules):
    s = session(batch=3, beam=5, device=device)
    s.capacity = capacity
    s.seq = (torch.arange(s.rows * (capacity + 2), device=device).reshape(s.rows, capacity + 2) % 8)
    s.pos.fill_(pos)
    s.rules = TimestampRules(7, 5, 6, 0, 8, device) if rules else None
    s.t_notime.copy_(torch.arange(s.rows, device=device) % 2 == 0)
    s.t_first.fill_(True)
    s.t_count.copy_(torch.arange(s.rows, device=device))
    s.t_last.copy_(torch.arange(s.rows, device=device) % 8)
    s.active.fill_(True)
    s.active[2] = False
    s.done[2] = True
    s._selection_state = (torch.empty(s.rows, dtype=torch.long, device=device),
                          torch.empty(s.rows, dtype=torch.long, device=device),
                          torch.empty(s.rows, dtype=torch.bool, device=device))
    parent = torch.tensor([p + b * 5 for b in range(3) for p in mapping], device=device)
    token = torch.tensor([0, 1, 5, 6, 7] * 3, device=device)
    idx = ((parent % 5) * 8 + token).reshape(3, 5)
    top = -torch.arange(s.rows, dtype=torch.float32, device=device).reshape(3, 5)
    top[1, 3:] = -torch.inf
    top[2] = -torch.inf
    for _ in range(3):
        seq = s.seq.index_select(0, parent)
        seq[:, pos] = token
        notime = s.t_notime.index_select(0, parent) if rules else s.t_notime.clone()
        count = s.t_count.index_select(0, parent) if rules else s.t_count.clone()
        last = s.t_last.index_select(0, parent) if rules else s.t_last.clone()
        first = torch.zeros_like(s.t_first) if rules else s.t_first.clone()
        if rules:
            is_time = (token >= 5) & (token <= 6)
            count += is_time
            last = torch.where(is_time, token, last)
        valid = s.active[parent] & ~s.done.repeat_interleave(5) & torch.isfinite(top.flatten())
        actual = beam_kernels.reorder_state(s, top, idx)
        for a, b in zip(actual, (parent, token, valid)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip((s.seq, s.score, s.t_notime, s.t_first, s.t_count, s.t_last),
                        (seq, top.flatten(), notime, first, count, last)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_state_fusion_cpu_fallback():
    s = session(fused_state=True)
    assert not s.fused_state and s._selection_state is None


@pytest.mark.parametrize("fused_rank,fused_finish", [(True, True), (False, False)])
def test_state_fusion_graphs_and_bucket_growth(device, fused_rank, fused_finish):
    if device.type != "cuda":
        pytest.skip("CUDA graph requires a compute allocation")
    rules = TimestampRules(7, 5, 6, 0, 8, device)
    kw = dict(batch=3, beam=3, device=device, rules=rules, diagnostics=True,
              use_cuda_graph=True, upper_bound=False, fused_rank=fused_rank, fused_finish=fused_finish)
    reference = session(**kw, fused_state=False)
    actual = session(**kw, fused_state=True)
    assert actual.fused_state
    for prefix, limit in (([2, 3, 7], 13), ([2, 3, 4, 3, 7], 7), ([2, 3, 7], 1)):
        primer = torch.tensor([prefix] * 3, device=device)
        assert actual.decode(primer, limit) == reference.decode(primer, limit)
        n = int(reference.step) + 1
        for key in ("scores", "state"):
            torch.testing.assert_close(actual._beam_history[key][:n], reference._beam_history[key][:n],
                                       rtol=0, atol=0)


@pytest.mark.parametrize("logit", [float("nan"), float("inf"), -float("inf")])
def test_state_fusion_invalid_live_logits(device, logit):
    from test_beam import ready

    refs = []
    for fused in (False, True):
        s = session(batch=1, beam=3, device=device, fused_rank=False, fused_finish=False)
        s.fused_state = fused  # Also exercise the interpreter on CPU.
        if fused:
            s._selection_state = (torch.empty(s.rows, dtype=torch.long, device=device),
                                  torch.empty(s.rows, dtype=torch.long, device=device),
                                  torch.empty(s.rows, dtype=torch.bool, device=device))
        ready(s)
        s.logits.fill_(logit)
        s._select(4)
        refs.append(s)
    for name in ("score", "seq", "active", "done", "best_seq", "best_score", "tok", "numerical_error"):
        torch.testing.assert_close(getattr(refs[0], name), getattr(refs[1], name), rtol=0, atol=0)
    for s in refs:
        with pytest.raises(RuntimeError):
            s._check()


@torch.inference_mode()
def test_state_fusion_real_decoder_graphs(device):
    if device.type != 'cuda':
        pytest.skip('real decoder requires a compute allocation')
    from test_model import TINY

    from owsm_trt.modeling import OWSMModel
    from owsm_trt.runtime import BeamSession, TorchBackend

    torch.manual_seed(29)
    model = OWSMModel(TINY).eval().to(device)
    backend = TorchBackend(model)
    args = (backend, TINY, 3, 3, 32, -1, torch.float32, device)
    ref = BeamSession(*args, buckets=[4, 8, 16, 32], diagnostics=True, fused_state=False)
    fast = BeamSession(*args, buckets=[4, 8, 16, 32], diagnostics=True, fused_state=True)
    for primer_len, limit in ((3, 29), (9, 7), (3, 20)):
        primer = torch.full((3, primer_len), 2, device=device)
        assert fast.decode(primer, limit) == ref.decode(primer, limit)
        torch.testing.assert_close(fast.logits, ref.logits, rtol=0, atol=0)
        n = int(ref.step) + 1
        for key in ('scores', 'state'):
            torch.testing.assert_close(fast._beam_history[key][:n], ref._beam_history[key][:n], rtol=0, atol=0)
