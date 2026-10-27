"""Compute-only regression for frontend capture versus pooled-stream graph destruction."""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from owsm_trt.frontend_graph import FrontendReplay
from owsm_trt.modeling import Frontend
from owsm_trt.runtime import capture_graph


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-frontend', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('requires a compute allocation')
    cls = FrontendReplay
    if args.baseline_frontend:
        spec = importlib.util.spec_from_file_location('owsm_trt._baseline_frontend', args.baseline_frontend)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.FrontendReplay
    args.out.mkdir(parents=True, exist_ok=True)

    def save(**values):
        (args.out / 'progress.json').write_text(json.dumps(values, indent=2) + '\n')

    torch.manual_seed(12)
    cfg = SimpleNamespace(n_fft=512, hop_length=160, win_length=400, n_mels=80, normalize='global_mvn')
    frontend = Frontend(cfg).eval().cuda()
    frontend.melmat.uniform_(0.01, 1)
    frontend.mean.normal_()
    frontend.std.uniform_(0.5, 2)
    waves = [torch.randn(1, n, device='cuda') for n in (16001, 32001, 48001)]
    expected = [frontend(w) for w in waves]
    torch.cuda.synchronize()
    replay = cls(frontend)
    for wave, ref in zip(waves, expected):
        torch.testing.assert_close(replay(wave), ref, rtol=0, atol=0)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in range(96)]
    collisions = [s for s in streams if s.cuda_stream == replay.capture_stream.cuda_stream]
    save(phase='captured', pooled_alias=bool(collisions), unique_pooled_streams=len({s.cuda_stream for s in streams}))
    if not args.baseline_frontend:
        assert not collisions
    # Deliberately destroy a different graph on every pool stream. The old frontend aliases
    # one of these, so destruction clears its externally cached warm-up workspace.
    x = torch.ones(16, device='cuda')
    for stream in {s.cuda_stream: s for s in streams}.values():
        graph = capture_graph(lambda: x.add_(1), stream)
        del graph
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    save(phase='foreign_graphs_destroyed', pooled_alias=bool(collisions))
    for _ in range(3):
        for wave, ref in zip(waves, expected):
            torch.testing.assert_close(replay(wave), ref, rtol=0, atol=0)
    torch.cuda.synchronize()
    release = getattr(replay, '_release', None)
    del replay
    gc.collect()
    if release is not None:
        assert not release.alive
    save(phase='complete', pooled_alias=bool(collisions), outputs_equal=True,
         finalizer_complete=release is not None and not release.alive)


if __name__ == '__main__':
    main()
