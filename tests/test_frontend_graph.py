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
