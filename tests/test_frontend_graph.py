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
    with pytest.raises(ValueError):
        FrontendReplay(None, max_batch=0)
    replay = FrontendReplay(lambda x: x + 1)
    assert replay(torch.tensor([[1.0]])).item() == 2
    assert replay.info()["captures"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute allocation required")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("normalize", ["global_mvn", "utterance_mvn"])
@pytest.mark.parametrize("max_batch", [1, 8])
@pytest.mark.parametrize("reuse", [False, True])
@torch.inference_mode()
def test_retained_outputs_cross_streams_and_cache_limit(fused, normalize, max_batch, reuse):
    frontend = Frontend(replace(TINY, normalize=normalize)).eval().cuda()
    frontend.melmat.uniform_(0, 1)
    if frontend.global_mvn:
        frontend.mean.normal_()
        frontend.std.uniform_(0.1, 2.0)
    frontend.fused_power = frontend.fused_logmel = fused
    replay = FrontendReplay(frontend, max_entries=2, max_batch=max_batch, reuse_warmup_stream=reuse)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    batches = [1]*6 if max_batch == 1 else [3,8,3,8,8,3]
    waves = [torch.randn(b,n,device="cuda") for b,n in zip(batches,(1601,3200,1601,2401,3200,1601))]
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
    # Unmeasured larger batches retain eager execution, with the same per-utterance normalization.
    two = torch.randn(max_batch+1,1601,device="cuda")
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
