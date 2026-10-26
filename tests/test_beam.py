"""Small synthetic beam states; no audio processing or model inference."""

from types import SimpleNamespace

import pytest
import torch

from owsm_trt.config import OWSMConfig
from owsm_trt.runtime import BeamSession, TimestampRules

CFG = OWSMConfig(
    vocab_size=8, d_model=4, n_heads=1, enc_layers=1, enc_ffn=4, cgmlp_units=4,
    cgmlp_kernel=3, merge_kernel=3, subsample="conv2d8", dec_layers=1, dec_ffn=4,
    dec_heads=1, n_mels=4, n_fft=16, win_length=16, hop_length=8, sample_rate=80,
    speech_seconds=2, time_resolution=0.02,
)


class ScriptedBackend:
    def __init__(self, values=None):
        self.values = values if values is not None else [0., 2., 1., -1., -2., -3., -4., -5.]

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, **kw):
        values = torch.tensor(self.values, device=logits.device)

        def run():
            logits.copy_(values)
            new_kv.zero_()
        return run


def session(batch=1, beam=3, capacity=16, **kwargs):
    backend = kwargs.pop("backend", ScriptedBackend())
    device = kwargs.pop("device", torch.device("cpu"))
    graphs = kwargs.pop("use_cuda_graph", False)
    return BeamSession(backend, CFG, batch, beam, capacity, 0, torch.float32, device,
                       use_cuda_graph=graphs, buckets=[4, 8, 16][: {4: 1, 8: 2, 16: 3}[capacity]], **kwargs)


def ready(s):
    s.pos.fill_(3)
    s.step.zero_()
    s.last_step.fill_(10)
    s.best_score.fill_(-torch.inf)
    s.best_end.fill_(3)
    s.ended_by_len.fill_(-torch.inf)
    s.active.zero_()
    s.active[::s.beam] = True
    s.seq[:, :3] = torch.tensor([2, 3, 4], device=s.device)
    s.logits.fill_(-torch.inf)
    s.logits[::s.beam, 0, 1] = 0


def test_invalid_parents_do_not_revive():
    s = session()
    ready(s)
    s._select(4)
    assert s.active.sum() == 1
    assert s.best_score.isneginf().all()
    assert s.ended_by_len.isneginf().all()


def test_finite_candidate_below_old_sentinel_survives():
    s = session()
    ready(s)
    s.score[0] = -2e10
    s._select(4)
    assert s.active.sum() == 1
    assert s.score[s.active].item() == -2e10


@pytest.mark.parametrize("has_finished", [False, True])
def test_all_candidates_forbidden(has_finished):
    s = session()
    ready(s)
    s.logits.zero_()
    s._rules_mask = lambda scores: scores.fill_(-torch.inf)
    if has_finished:
        s.best_score.fill_(-4)
    s._select(4)
    assert s.done.all() and not s.active.any()
    if has_finished:
        s._check()
        assert s.best_score.item() == -4
    else:
        with pytest.raises(RuntimeError, match="no valid finished"):
            s._check()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_live_logits_raise(value):
    s = session()
    ready(s)
    s.logits[0, 0, 2] = value
    s._select(4)
    with pytest.raises(RuntimeError, match="NaN or positive infinity"):
        s._check()


def test_inactive_nonfinite_logits_are_ignored():
    s = session()
    ready(s)
    s.logits[1:] = float("nan")
    s.last_step.zero_()
    s._select(4)
    s._check()
    assert s.best_score.item() == 0


def test_all_negative_infinity_is_exhaustion():
    s = session(backend=ScriptedBackend([-torch.inf] * 8))
    with pytest.raises(RuntimeError, match="no valid finished"):
        s.decode(torch.tensor([[2, 3, 4]]), 5)


@pytest.mark.parametrize("force", [False, True])
def test_only_valid_eos_enters_history(force):
    s = session()
    ready(s)
    if force:
        s.last_step.zero_()
    else:
        s.logits[0, 0, 1] = -torch.inf
        s.logits[0, 0, 0] = 0
    s._select(4)
    assert s.done.all() and not s.active.any()
    assert s.best_score.item() == 0
    assert torch.isfinite(s.ended_by_len).sum() == 1
    assert s.ended_by_len[0, 5 if force else 4] == 0


def test_timestamp_mask_can_leave_fewer_than_beam_candidates():
    rules = TimestampRules(7, 5, 6, 0, 8, torch.device("cpu"))
    s = session(beam=3, rules=rules)
    ready(s)
    s.logits.zero_()
    s._rules_start(s.seq[:, :3])
    s._select(4)
    assert s.active.sum() == 2
    assert s.best_score.isneginf().all()  # EOS is forbidden on the first selection
    assert set(s.tok[s.active, 0].tolist()) == {5, 6}


def test_partial_batch_completion_is_sticky():
    s = session(batch=2)
    ready(s)
    s.logits[0, 0, 1] = -torch.inf
    s.logits[0, 0, 0] = 0
    s._select(4)
    assert s.done.tolist() == [True, False]
    s.step += 1
    s.pos += 1
    s.logits.zero_()
    s._select(8)
    assert not s.active[:3].any()
    assert s.best_score[0] == 0


@pytest.mark.parametrize("prompt_len, limit", [(3, 1), (7, 9), (15, 1)])
def test_forced_eos_and_capacity(prompt_len, limit):
    s = session(backend=ScriptedBackend([-torch.inf, 0., -torch.inf, -torch.inf,
                                        -torch.inf, -torch.inf, -torch.inf, -torch.inf]), diagnostics=True)
    assert s.decode(torch.tensor([[2] * prompt_len]), limit) == [[1] * limit]
    assert s.best_end.item() == prompt_len + limit
    d = s.get_diagnostics()["utterances"][0]
    assert d["best_end_with_primer"] == prompt_len + limit
    assert d["best_finished_length_with_primer"] == prompt_len + limit + 1
    with pytest.raises(ValueError, match="does not fit"):
        s.decode(torch.tensor([[2] * 16]), 1)


@pytest.mark.parametrize("eos_logit, expected", [(0., False), (1e-6, True), (-1e-6, False)])
def test_strict_bound_ties_and_near_ties(eos_logit, expected):
    s = session(beam=2, upper_bound=True, diagnostics=True)
    ready(s)
    s.logits[0, 0, 0] = eos_logit
    s._select(4)
    assert s.done.item() == expected
    assert bool(s.get_diagnostics()["utterances"][0]["history"][0]["bound"]) == expected


def test_no_finished_never_satisfies_bound():
    s = session(upper_bound=True, diagnostics=True)
    ready(s)
    s._select(4)
    assert not s.done.any()
    assert s.get_diagnostics()["utterances"][0]["first_bound_step"] is None


@pytest.mark.parametrize("diagnostics", [False, True])
def test_bound_toggle_preserves_tokens_and_reduces_steps(diagnostics):
    s = session(backend=ScriptedBackend([4., 1., 0., -1., -2., -3., -4., -5.]),
                diagnostics=diagnostics, upper_bound=False)
    primer = torch.tensor([[2, 3, 4]])
    off = s.decode(primer, 13)
    off_steps = int(s.step)
    s.set_upper_bound(True)
    assert s.decode(primer, 13) == off == [[]]
    assert int(s.step) < off_steps
    if diagnostics:
        d = s.get_diagnostics()
        row = d["utterances"][0]
        assert d["decoder_calls"] == 1  # first bucket boundary forces a check
        assert d["prefill_selections"] == 1 and d["selection_calls"] == 2
        assert row["logical_end_step"] == 0 and row["host_observed_done_step"] == 1
        assert row["first_eos_step"] == row["last_best_update_step"] == 0
        assert row["extra_decoder_calls"] == 1 and row["stop_mask"] == 8
        assert row["best_end_with_primer"] == 4
        assert s.decode(primer, 1) == [[]]
        assert s.get_diagnostics()["selection_calls"] == 1  # stale history not returned
    else:
        assert s.get_diagnostics() is None and s._beam_history is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_diagnostics_and_bound_toggle():
    s = session(backend=ScriptedBackend([4., 1., 0., -1., -2., -3., -4., -5.]),
                diagnostics=True, device=torch.device("cuda"), use_cuda_graph=True, upper_bound=False)
    primer = torch.tensor([[2, 3, 4]], device="cuda")
    off = s.decode(primer, 13)
    graph_ids = [id(g) for g in s.graphs.values()]
    s.set_upper_bound(True)
    assert s.decode(primer, 13) == off
    assert s.get_diagnostics()["utterances"][0]["logical_end_step"] == 0
    assert graph_ids == [id(g) for g in s.graphs.values()]


@pytest.mark.parametrize("batch, chunk, calls", [(1, None, 1), (3, None, 4), (1, 2, 2)])
def test_default_bound_and_host_check_interval(batch, chunk, calls):
    s = session(batch=batch, chunk=chunk, diagnostics=True,
                backend=ScriptedBackend([4., 1., 0., -1., -2., -3., -4., -5.]))
    assert s.decode(torch.tensor([[2, 3, 4, 5]] * batch), 12) == [[]] * batch
    d = s.get_diagnostics()
    assert d["decoder_calls"] == calls
    assert all(u["logical_end_step"] == 0 and u["stop_mask"] == 8 for u in d["utterances"])


@pytest.mark.parametrize("chunk", [0, -1])
def test_invalid_host_check_interval(chunk):
    with pytest.raises(ValueError, match="must be positive"):
        session(chunk=chunk)


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_forwards_beam_kernel_choice(monkeypatch, enabled):
    from owsm_trt import pipeline

    captured = {}

    def create(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, "BeamSession", create)
    model = pipeline.OWSM(
        CFG, SimpleNamespace(first_time=None, notime=None, eos=0), None, None,
        SimpleNamespace(dtype="float32", capacity=16), torch.device("cpu"),
        use_cuda_graph=False, beam_kernels=enabled,
    )
    s = model._session(1, 3)
    assert captured["fused_rank"] is enabled and captured["fused_finish"] is enabled
    assert captured["kv_reorder"] == ("prefix" if enabled else "torch")
    assert captured["share_cross_kv"] is enabled
    assert model._session(1, 3) is s
