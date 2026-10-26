"""Frontend graph outputs have independent lifetimes across encoder streams."""

from dataclasses import replace

import pytest
import torch
from test_model import TINY

from owsm_trt.frontend_graph import FrontendReplay
from owsm_trt.modeling import Frontend


def test_invalid_capacity_and_cpu_fallback():
    with pytest.raises(ValueError):
        FrontendReplay(None, max_entries=0)
    replay = FrontendReplay(lambda x: x + 1)
    assert replay(torch.tensor([[1.0]])).item() == 2
    assert replay.info()["captures"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("normalize", ["global_mvn", "utterance_mvn"])
@torch.inference_mode()
def test_retained_outputs_cross_streams_and_cache_limit(fused, normalize):
    frontend = Frontend(replace(TINY, normalize=normalize)).eval().cuda()
    frontend.melmat.uniform_(0, 1)
    if frontend.global_mvn:
        frontend.mean.normal_()
        frontend.std.uniform_(0.1, 2.0)
    frontend.fused_power = frontend.fused_logmel = fused
    replay = FrontendReplay(frontend, max_entries=2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    waves = [torch.randn(1, n, device="cuda") for n in (1601, 3200, 1601, 2401, 3200, 1601)]
    expected = [frontend(w) for w in waves]
    torch.cuda.synchronize()
    actual = []
    for _ in range(3):
        for i, w in enumerate(waves):
            with torch.cuda.stream(streams[i % 2]):
                actual.append(replay(w))
    torch.cuda.synchronize()
    for i, v in enumerate(actual):
        torch.testing.assert_close(v, expected[i % len(waves)], rtol=0, atol=0)
    assert replay.info()["captures"] == 2 and replay.fallbacks == 3
    assert len({v.data_ptr() for v in actual}) == len(actual)
    # Larger batches retain eager execution, including per-utterance normalization/strides.
    two = torch.cat([waves[0], waves[0]])
    torch.testing.assert_close(replay(two), frontend(two), rtol=0, atol=0)
    assert replay.info()["captures"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_autocast_uses_eager_frontend_after_fp32_capture(fused):
    frontend = Frontend(TINY).eval().cuda()
    frontend.melmat.uniform_(0, 1)
    speech = torch.randn(1, 3200, device="cuda")
    replay = FrontendReplay(frontend)
    expected = frontend(speech)
    frontend.fused_power = frontend.fused_logmel = fused
    torch.testing.assert_close(replay(speech), expected, rtol=0, atol=0)
    for dtype in (torch.float16, torch.bfloat16):
        with torch.autocast("cuda", dtype=dtype):
            frontend.fused_power = frontend.fused_logmel = False
            amp_expected = frontend(speech)
            frontend.fused_power = frontend.fused_logmel = fused
            torch.testing.assert_close(replay(speech), amp_expected, rtol=0, atol=0)
    torch.testing.assert_close(replay(speech), expected, rtol=0, atol=0)
    assert replay.info()["captures"] == 1


def test_direct_input_cpu_padding_and_trim():
    replay = FrontendReplay(lambda x: x.clone())
    wave = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    torch.testing.assert_close(
        replay.from_wave(wave, 5, torch.device("cpu")), torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(replay.from_wave(wave, 2, torch.device("cpu")), torch.tensor([[1.0, 2.0]]))
    assert replay.info()["captures"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("normalize", ["global_mvn", "utterance_mvn"])
@pytest.mark.parametrize("source", ["cpu", "cuda"])
@torch.inference_mode()
def test_direct_input_matches_padded_graph_and_retains_outputs(normalize, source):
    frontend = Frontend(replace(TINY, normalize=normalize)).eval().cuda()
    frontend.melmat.uniform_(0, 1)
    replay = FrontendReplay(frontend, max_entries=2)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    specs = [(1601, 700), (3200, 4001), (1601, 0), (2401, 1601), (3200, 400)]
    waves = [torch.randn(n * 2, device=source)[::2] for _, n in specs]
    expected = []
    for (length, _), wave in zip(specs, waves):
        padded = torch.zeros(1, length, device="cuda")
        padded[0, : min(length, len(wave))].copy_(wave[:length])
        expected.append(frontend(padded))
    padded = torch.zeros(1, 1601, device="cuda")
    padded[0, :700].copy_(waves[0])
    torch.testing.assert_close(replay(padded), expected[0], rtol=0, atol=0)
    torch.cuda.synchronize()
    actual = []
    for _ in range(3):
        for i, ((length, _), wave) in enumerate(zip(specs, waves)):
            with torch.cuda.stream(streams[i % 2]):
                actual.append(replay.from_wave(wave, length, torch.device("cuda")))
    torch.cuda.synchronize()
    for i, out in enumerate(actual):
        torch.testing.assert_close(out, expected[i % len(waves)], rtol=0, atol=0)
    assert replay.info()["captures"] == 2 and replay.fallbacks == 3
    assert len({out.data_ptr() for out in actual}) == len(actual)
    # The eager AMP fallback still applies when populating the input directly.
    padded = torch.zeros(1, 1601, device="cuda")
    padded[0, :700].copy_(waves[0])
    with torch.autocast("cuda", dtype=torch.float16):
        torch.testing.assert_close(
            replay.from_wave(waves[0], 1601, torch.device("cuda")), frontend(padded), rtol=0, atol=0
        )
