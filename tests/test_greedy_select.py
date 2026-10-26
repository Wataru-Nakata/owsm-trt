"""Plain greedy selection must match PyTorch state updates, including nonfinite logits."""

from dataclasses import replace

import pytest
import torch
from test_model import TINY

from owsm_trt import kernels
from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import GreedySession, TimestampRules, TorchBackend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or kernels.triton is None, reason="needs CUDA + Triton"
)


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("vocab", [4097, 50002])
@pytest.mark.parametrize("rules_on", [False, True])
def test_plain_select_matches_torch_state_and_graph(rows, vocab, rules_on):
    class Backend:
        def step_fn(self, *args, **kwargs):
            return lambda: None

    dev = torch.device("cuda")
    cfg = replace(TINY, vocab_size=vocab)
    rules = TimestampRules(999, 2040, 2070, 2, vocab, dev) if rules_on else None
    sessions = [
        GreedySession(
            Backend(),
            cfg,
            rows,
            32,
            2,
            torch.float32,
            dev,
            use_cuda_graph=False,
            rules=rules,
            fused_select=fused,
        )
        for fused in (False, True)
    ]
    ref, fast = sessions
    assert fast.select_workspace is not None
    fast._select(32)  # JIT before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast._select(32)
    torch.manual_seed(32)
    modes = ("random", "tie", "near_tie", "invalid", "inf", "nan_text", "nan_time", "all_nan", "eos")
    for mode in modes:
        initial = {
            "t_notime": [True, False, False, False, False, True, False],
            "t_first": [False, True, False, False, False, False, False],
            "t_count": [1, 0, 1, 2, 2, 0, 3],
            "t_last": [2049, -1, 2049, 2051, 2067, -1, 2049],
            "tok": [100, 100, 100, 2051, 100, 100, 2048],
            "done": [False, False, False, False, False, True, False],
        }
        for sess in sessions:
            for key, values in initial.items():
                target = getattr(sess, key)
                target.copy_(torch.tensor(values[:rows], device=dev).view_as(target))
            sess.seq.copy_(torch.arange(rows * 33, device=dev).view(rows, 33))
        for pos in (0, 15, 32):  # includes the last legal output slot; state persists across replays
            logits = torch.randn(rows, 1, vocab, device=dev)
            if mode in ("tie", "near_tie", "inf"):
                logits.fill_(-2)
                logits[:, :, 17] = logits[:, :, 2049] = 1
                logits[:, :, 4096] = (
                    torch.nextafter(torch.tensor(1.0, device=dev), torch.tensor(float("inf"), device=dev))
                    if mode == "near_tie"
                    else 1
                )
                if mode == "inf":
                    logits[:, :, [17, 2049, 4096]] = float("inf")
            elif mode in ("invalid", "all_nan"):
                logits.fill_(float("-inf") if mode == "invalid" else float("nan"))
            elif mode == "nan_text":
                logits[:, :, [17, 4096]] = float("nan")
            elif mode == "nan_time":
                logits[:, :, [2048, 2060]] = float("nan")
            elif mode == "eos":
                logits[:, :, 2] = 100
            for sess in sessions:
                sess.logits.copy_(logits)
                sess.pos.fill_(pos)
            ref._select(32)
            graph.replay()
            for key in (*initial, "seq", "pos"):
                assert torch.equal(getattr(ref, key), getattr(fast, key)), (mode, pos, key)
            torch.testing.assert_close(fast.logits, logits, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_plain_select_real_decoder_prefill_and_bucket_parity(rows, graphs):
    dev = torch.device("cuda")
    cfg = replace(TINY, vocab_size=4097)
    torch.manual_seed(54)
    model = OWSMModel(cfg).eval().requires_grad_(False).to(dev)
    rules = TimestampRules(999, 2040, 2070, 1, cfg.vocab_size, dev)
    cross = [
        torch.randn(cfg.dec_layers, rows, cfg.dec_heads, cfg.enc_frames, 8, device=dev) for _ in range(2)
    ]
    reference = None
    for fused in (False, True):
        sess = GreedySession(
            TorchBackend(model),
            cfg,
            rows,
            64,
            -1,
            torch.float32,
            dev,
            use_cuda_graph=graphs,
            buckets=[16, 32, 64],
            rules=rules,
            fused_select=fused,
        )
        sess.cross_k.copy_(cross[0])
        sess.cross_v.copy_(cross[1])
        outputs = []
        for p, timed in ((3, False), (20, True), (3, True), (20, False)):
            primer = torch.arange(10, 10 + p, device=dev).repeat(rows, 1)
            if not timed:
                primer[:, -1] = 999
            outputs.append(sess.decode(primer, 25))
            assert all(len(x) == 25 for x in outputs[-1])
        if reference is None:
            reference = outputs
        else:
            assert outputs == reference
