"""Fixed-batch encoder staging: same order/shapes, reusable graphs and stream ownership."""

from types import SimpleNamespace

import pytest
import torch
from test_model import TINY

from owsm_trt import pipeline
from owsm_trt.build import BuildOptions
from owsm_trt.modeling import OWSMModel
from owsm_trt.runtime import TorchBackend


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("beam, task, drafts", [(5, "asr", True), (1, "st_eng", True), (1, "asr", False), (1, "asr", True)])
def test_lookahead_gate_preserves_order_and_drafted_paths(device, beam, task, drafts):
    model = pipeline.OWSM.__new__(pipeline.OWSM)
    model.cfg = SimpleNamespace(ctc_only=False, sample_rate=1)
    model.device = torch.device(device)
    model.ctc_draft, model._encoder_lookahead = drafts, True
    model._batch_size = lambda *_: 2
    model._primer = lambda *_: [2, 3, 4]
    model._draft_windows = lambda _: ()  # Long prompts/plain paths never enter decode_stream here.
    model._wave = lambda audio, _: audio
    model._result = lambda ids, offset: ids
    calls = []

    def staged(items, chunks, primer, beam, sr):
        calls.append(chunks)
        return [[len(items[j])] for chunk in chunks for j in chunk]

    model._decode_lookahead = staged
    model._decode = lambda waves, *_: [[len(w)] for w in waves]
    assert model.transcribe([torch.zeros(n) for n in (7, 2, 5)], task=task, beam_size=beam) == [[7], [2], [5]]
    enabled = device == "cuda" and (beam > 1 or task != "asr" or not drafts)
    assert calls == ([[[1, 2], [0]]] if enabled else [])
    calls.clear()
    assert model.transcribe([torch.zeros(4)], task=task, beam_size=beam) == [[4]]
    assert not calls  # A single batch keeps its direct path.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("beam", [1, 3])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_staged_encoder_matches_sequential_reuse_and_partial_batch(beam, batch, graphs):
    torch.manual_seed(43)
    dev = torch.device("cuda")
    network = OWSMModel(TINY).eval().requires_grad_(False).to(dev)
    tokenizer = SimpleNamespace(first_time=None, notime=None, eos=1, punctuation=frozenset())
    model = pipeline.OWSM(TINY, tokenizer, None, TorchBackend(network),
                          BuildOptions(dtype="float32", capacity=32), dev,
                          use_cuda_graph=graphs, ctc_draft=False, beam_kernels=False, greedy_kernels=False, encoder_lookahead=True)
    assert model._encoder_lookahead
    model._features = lambda rows: torch.stack(rows)
    model._wave = lambda value, _: value
    snapshots = []
    original_session = model._session

    def observed_session(rows, width=1):
        sess = original_session(rows, width)
        if not getattr(sess, "_observed_cross", False):
            original_decode = sess.decode

            def decode(*args):
                # Clones enqueue on the current stream without a host wait; retain overlap.
                buffers = (getattr(sess, "cross_kb", sess.cross_k), getattr(sess, "cross_vb", sess.cross_v),
                           sess.cross_k, sess.cross_v)
                snapshots.append(tuple(t.clone() for t in buffers))
                return original_decode(*args)

            sess.decode = decode
            sess._observed_cross = True
        return sess

    model._session = observed_session

    def same_encoder_values(expected):
        assert len(snapshots) == len(expected)
        for actual, prior in zip(snapshots, expected):
            for left, right in zip(actual, prior):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        snapshots.clear()

    items = [torch.randn(TINY.num_frames, TINY.n_mels, device=dev) for _ in range(7)]
    order = [6, 0, 4, 2, 1, 5, 3]
    chunks = [order[i:i+batch] for i in range(0, len(items), batch)]
    for primer in ([2, 3, 4], [2, 3, 4, 5, 6, 7, 8]):
        expected = [ids for chunk in chunks
                    for ids in model._decode([items[j] for j in chunk], [primer] * len(chunk), beam)]
        encoder_values = snapshots.copy()
        snapshots.clear()
        assert model._decode_lookahead(items, chunks, primer, beam, None) == expected
        same_encoder_values(encoder_values)
        sessions = list(model._sessions.values())
        captures = [len(s._enc_graphs) for s in sessions]
        buffers = [[t.data_ptr() for bufs in s._lookahead_stage.values() for t in bufs] for s in sessions]
        assert model._decode_lookahead(items, chunks, primer, beam, None) == expected
        same_encoder_values(encoder_values)
        assert captures == [len(s._enc_graphs) for s in sessions]
        assert buffers == [[t.data_ptr() for bufs in s._lookahead_stage.values() for t in bufs] for s in sessions]
        actual = [ids for chunk in chunks
                  for ids in model._decode([items[j] for j in chunk], [primer] * len(chunk), beam)]
        assert actual == expected  # Ordinary encode can safely follow asynchronous staging.
        same_encoder_values(encoder_values)
    if graphs and batch == 3 and beam == 3:
        original = model._features
        count = 0

        def fail_second(rows):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError("deliberate prefetch failure")
            return original(rows)

        model._features = fail_second
        with pytest.raises(ValueError, match="deliberate prefetch"):
            model._decode_lookahead(items, chunks, primer, beam, None)
        model._features = original
        assert model._decode_lookahead(items, chunks, primer, beam, None) == expected
        same_encoder_values(encoder_values)
