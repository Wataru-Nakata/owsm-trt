"""Fixed-batch prefetch preserves order, exclusions and failure cleanup, without model work."""

import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from owsm_trt import pipeline


def stub_model(monkeypatch, workers):
    model = pipeline.OWSM.__new__(pipeline.OWSM)
    model.cfg = SimpleNamespace(ctc_only=False, sample_rate=1)
    model.device = torch.device('cpu')
    model.ctc_draft, model._encoder_lookahead = True, False
    model._audio_prefetch_workers = workers
    model._batch_size = lambda *_: 2
    model._primer = lambda *_: [2, 3, 4, 5]
    model._draft_windows = lambda _: ()  # Force fixed batches, including the long-prompt ASR path.
    model._result = lambda ids, offset: ids
    calls, reads = [], []

    def value(x):
        return x if isinstance(x, int) else int(Path(x).stem)

    monkeypatch.setattr(pipeline, 'duration', lambda x, sr: value(x))

    def load(x, sr):
        reads.append((value(x), threading.current_thread()))
        return value(x)

    def decode(waves, primers, beam, draft):
        assert threading.current_thread() is threading.main_thread()
        assert primers == [[2, 3, 4, 5]] * len(waves)
        calls.append((waves, beam, draft))
        return [[w] for w in waves]

    model._wave, model._decode = load, decode
    return model, calls, reads


@pytest.mark.parametrize('workers', [0, 1, 2])
@pytest.mark.parametrize('beam,task', [(5, 'asr'), (1, 'st_eng'), (1, 'asr')])
def test_fixed_batches_preserve_sort_partial_batch_and_restore_order(monkeypatch, workers, beam, task):
    model, calls, reads = stub_model(monkeypatch, workers)
    inputs = [Path(f'{i}.wav') for i in (7, 2, 5, 1, 8)]
    assert model.transcribe(inputs, beam_size=beam, task=task, sr=1) == [[7], [2], [5], [1], [8]]
    assert calls == [([1, 2], beam, task == 'asr'), ([5, 7], beam, task == 'asr'), ([8], beam, task == 'asr')]
    assert sorted(v for v, _ in reads) == [1, 2, 5, 7, 8]
    assert all((t is not threading.main_thread()) == bool(workers) for _, t in reads)
    assert not any(t.name.startswith('owsm-audio') for t in threading.enumerate())


@pytest.mark.parametrize('kind', ['empty', 'single', 'single_batch', 'mixed', 'loaded', 'autocast'])
def test_exclusions_keep_loader_on_calling_thread(monkeypatch, kind):
    model, calls, reads = stub_model(monkeypatch, 2)
    inputs = {'empty': [], 'single': '7.wav', 'single_batch': ['7.wav', '2.wav'],
              'mixed': ['7.wav', 2, '5.wav'], 'loaded': [7, 2, 5],
              'autocast': ['7.wav', '2.wav', '5.wav']}[kind]
    with torch.autocast('cpu') if kind == 'autocast' else nullcontext():
        result = model.transcribe(inputs, beam_size=5)
    values = inputs if isinstance(inputs, list) else [inputs]
    expected = [[x if isinstance(x, int) else int(Path(x).stem)] for x in values]
    assert result == (expected[0] if kind == 'single' else expected)
    assert all(t is threading.main_thread() for _, t in reads)


@pytest.mark.parametrize('workers', [0, 1, 2])
@pytest.mark.parametrize('failure', ['read', 'decode'])
def test_failure_occurs_in_order_joins_readers_and_reuses_model(monkeypatch, workers, failure):
    model, calls, reads = stub_model(monkeypatch, workers)
    load, decode = model._wave, model._decode
    error = OSError('injected second batch failure')

    def fail_read(x, sr):
        if Path(x).stem == '5':
            raise error
        return load(x, sr)

    def fail_decode(waves, *args):
        if waves[0] == 5:
            raise error
        return decode(waves, *args)

    if failure == 'read':
        model._wave = fail_read
    else:
        model._decode = fail_decode
    paths = [f'{i}.wav' for i in (7, 2, 5, 1, 8)]
    with pytest.raises(OSError) as caught:
        model.transcribe(paths, beam_size=5)
    assert caught.value is error
    assert calls == [([1, 2], 5, True)]
    assert not any(t.name.startswith('owsm-audio') for t in threading.enumerate())
    model._wave, model._decode = load, decode
    assert model.transcribe(paths, beam_size=5) == [[7], [2], [5], [1], [8]]
