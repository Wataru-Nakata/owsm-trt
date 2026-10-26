"""Frontend numerical equivalence and stream lifetime; audio/FFT only in a compute allocation."""

from dataclasses import replace

import pytest
import torch
from test_model import TINY

from owsm_trt import frontend_kernels
from owsm_trt.modeling import Frontend


def test_cpu_fallback_available():
    assert not frontend_kernels.available(torch.device('cpu'))


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU compute allocation required')


@cuda_only
@pytest.mark.parametrize('layout', ['contiguous', 'fft', 'slice'])
@pytest.mark.parametrize('shape', [(1, 257, 101), (3, 33, 17), (16, 257, 1001)])
def test_power_exact(shape, layout):
    torch.manual_seed(31)
    b, f, t = shape
    spec = torch.randn(b, f, t, dtype=torch.complex64, device='cuda')
    if layout == 'fft':
        spec = spec.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == 'slice':
        spec = torch.randn(b * 2, f, t * 2, dtype=torch.complex64, device='cuda')[1::2, :, 1::2]
    expected = (spec.real**2 + spec.imag**2).transpose(1, 2)
    actual = frontend_kernels.power(spec)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


@cuda_only
@pytest.mark.parametrize('global_mvn', [False, True])
@pytest.mark.parametrize('mel_count', [16, 128])
def test_log_normalize_exact(global_mvn, mel_count):
    torch.manual_seed(32)
    mel = torch.exp(torch.randn(3, 101, mel_count, device='cuda') * 12)
    mel.flatten()[:8] = torch.tensor([0., -1., 1.e-10, 1.e-11, 1.e-9, float('inf'), float('nan'), 1.e-38], device='cuda')
    mean = torch.randn(mel_count, device='cuda') if global_mvn else None
    std = torch.rand(mel_count, device='cuda') + .01 if global_mvn else None
    expected = mel.clamp(min=1.e-10).log()
    if global_mvn:
        expected = (expected - mean) / std
    actual = frontend_kernels.log_mvn(mel, mean, std)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@cuda_only
@pytest.mark.parametrize('normalize', ['global_mvn', 'utterance_mvn'])
@pytest.mark.parametrize('variant', ['power', 'post', 'both'])
@torch.inference_mode()
def test_frontend_exact_graph_and_independent_streams(normalize, variant):
    cfg = replace(TINY, normalize=normalize)
    frontend = Frontend(cfg).eval().cuda()
    frontend.melmat.uniform_(0, 1)
    if frontend.global_mvn:
        frontend.mean.normal_()
        frontend.std.uniform_(.1, 2.)
    waves = [torch.randn(3, n, device='cuda') for n in (1601, 3200)]
    expected = [frontend(w) for w in waves]
    frontend.fused_power = variant in ('power', 'both')
    frontend.fused_logmel = variant in ('post', 'both')
    graphs, values = [], []
    streams = [torch.cuda.Stream() for _ in waves]
    for stream, w in zip(streams, waves):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            frontend(w)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                value = frontend(w)
        graphs.append(g)
        values.append(value)
    torch.cuda.synchronize()
    for _ in range(3):
        for stream, g in zip(streams, graphs):
            with torch.cuda.stream(stream):
                g.replay()
        torch.cuda.synchronize()
        for actual, want in zip(values, expected):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
