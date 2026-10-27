"""Thread/order/lifetime contracts using fake reads: no files or model execution."""

import os
import threading

import pytest
import torch

from owsm_trt.audio_prefetch import prefetch_batches


def audio_threads():
    return {t for t in threading.enumerate() if t.name.startswith("owsm-audio")}


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_order_inference_and_thread_cleanup(workers):
    before = audio_threads()
    main = threading.current_thread()
    calls = []

    def load(item):
        calls.append((item, threading.current_thread(), torch.is_inference_mode_enabled()))
        return item

    with prefetch_batches([[i, i + 1] for i in range(0, 8, 2)], load, workers) as batches:
        assert list(batches) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert audio_threads() == before
    assert len(calls) == 8
    assert all((t is not main and inference) if workers else t is main for _, t, inference in calls)


def test_two_batches_bounded_and_completed_out_of_order():
    before = audio_threads()
    second_done = threading.Event()
    visited = []

    def load(item):
        visited.append(item)
        if item == 0:
            assert second_done.wait(5)
        if item == 1:
            second_done.set()
        return item

    with prefetch_batches([[i] for i in range(8)], load, 2) as batches:
        assert next(batches) == [0]
        assert set(visited) == {0, 1}  # No third batch until the consumer advances.
        assert list(batches) == [[i] for i in range(1, 8)]
    assert audio_threads() == before


@pytest.mark.parametrize("workers", [1, 2])
def test_early_exit_joins_and_does_not_read_whole_input(workers):
    before = audio_threads()
    visited = []

    def load(item):
        visited.append(item)
        return item

    with prefetch_batches([[i] for i in range(30)], load, workers) as batches:
        assert next(batches) == [0]
    assert audio_threads() == before
    assert set(visited) <= {0, 1}


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_error_is_observed_after_preceding_batch_and_cleans_up(workers):
    before = audio_threads()

    def load(item):
        if item == 1:
            raise OSError("failed second file")
        return item

    with pytest.raises(OSError, match="failed second"):
        with prefetch_batches([[0], [1], [2]], load, workers) as batches:
            assert next(batches) == [0]
            next(batches)
    assert audio_threads() == before


def test_consumer_exception_joins_workers():
    before = audio_threads()
    with pytest.raises(ValueError, match="consumer"):
        with prefetch_batches([[0], [1]], lambda x: x, 2) as batches:
            next(batches)
            raise ValueError("consumer")
    assert audio_threads() == before


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_empty_and_unconsumed_source_is_lazy(workers):
    def source():
        raise AssertionError("should remain lazy")
        yield []

    before = audio_threads()
    with prefetch_batches(source(), lambda x: x, workers):
        pass
    with prefetch_batches([], lambda x: x, workers) as batches:
        assert list(batches) == []
    assert audio_threads() == before


def test_cpu_autocast_keeps_loading_on_calling_thread():
    main = threading.current_thread()

    def load(x):
        assert threading.current_thread() is main
        assert torch.is_autocast_enabled("cpu")
        return x

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with prefetch_batches([[0], [1]], load, 2) as batches:
            assert list(batches) == [[0], [1]]


@pytest.mark.parametrize("workers", [-1, 3, True, 1.5])
def test_invalid_workers(workers):
    with pytest.raises(ValueError, match="workers"):
        with prefetch_batches([], lambda x: x, workers):
            pass


@pytest.mark.skipif(not os.environ.get('PBS_JOBID'), reason='audio work requires compute allocation')
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('rate', [16000, 44100])
@pytest.mark.parametrize('channels', [1, 2])
def test_real_parallel_file_decode_and_resample(tmp_path, workers, rate, channels):
    import numpy as np
    import soundfile as sf

    from owsm_trt.audio import load_audio

    rng = np.random.default_rng(42)
    paths = []
    for i in range(5):
        path = tmp_path / f'{i}.wav'
        sf.write(path, rng.uniform(-0.5, 0.5, (rate // 20 + i, channels)), rate)
        paths.append(path)
    expected = [load_audio(p) for p in paths]
    with prefetch_batches([paths[:2], paths[2:4], paths[4:]], load_audio, workers) as batches:
        actual = [w for batch in batches for w in batch]
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not audio_threads()
