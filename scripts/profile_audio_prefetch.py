"""File-inclusive bounded audio prefetch ablation, with all GPU work on the caller thread."""

import argparse
import importlib.metadata
import json
import os
import threading
from pathlib import Path

import torch
from profile_beam import save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub, pipeline
from owsm_trt.audio import load_audio
from owsm_trt.audio_prefetch import prefetch_batches
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--fixed', action='store_true', help='fixed beam/translation batches, excluding encoder lookahead')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace, args.long_audio, args.variants = False, None, ['reference', 'worker1', 'worker2']
    items = json.loads(args.manifest.read_text())
    if len(items) < 9:
        ap.error('at least9 inputs required for B8 streaming')
    paths = [x['path'] for x in items]
    waves, first_load_ms = wall(lambda: [load_audio(path) for path in paths])
    for workers in (1, 2):
        with prefetch_batches([paths[i:i + 8] for i in range(0, len(paths), 8)], load_audio, workers) as batches:
            actual = [w for batch in batches for w in batch]
        assert len(actual) == len(waves)
        for a, b in zip(actual, waves):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    del actual
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
    save(args.out / 'environment.json', dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        gpu=torch.cuda.get_device_name(), sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
        cuda=torch.version.cuda, versions={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt', 'soundfile')},
        load_ms=load_ms, first_audio_load_ms=first_load_ms, checkpoint_sha256=sha256(files.checkpoint),
        engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')}, manifest_sha256=sha256(args.manifest),
        input_mode='File paths for every variant; current public mono loader; no resident-input speed comparison.',
        workers={'reference': 0, 'worker1': 1, 'worker2': 2}, submitted_batch_bound=2, waveforms_equal=True,
        frontend_graph_max_batch=model._frontend_graph_batch_limit, fixed_batches=args.fixed,
        scope='E/N intent: same model/engines/search/batch order, but CPU read timing may change concurrent decoder dispatch. '
              'All timed calls include thread startup/join, duration sorting and file reads; '
              'node-local warm OS cache after staging, not cold disk/Lustre/network. '
              'Reference waveform preloading is only correctness preparation, never a timed input.'))
    inputs = {v: paths for v in args.variants}
    summary = {}

    def configure(m, mode):
        m._audio_prefetch_workers = {'reference': 0, 'worker1': 1, 'worker2': 2}[mode]

    def observe(m, call):
        # This diagnostic pass is excluded from all wall-time samples.
        original = pipeline.load_audio
        lock = threading.Lock()
        threads = {}

        def counted(*args, **kwargs):
            result = original(*args, **kwargs)
            with lock:
                name = threading.current_thread().name
                threads[name] = threads.get(name, 0) + 1
            return result

        pipeline.load_audio = counted
        try:
            counts = count_decoder_dispatch(m, call)
        finally:
            pipeline.load_audio = original
        assert sum(threads.values()) == len(paths)
        assert all(name.startswith('owsm-audio') == bool(m._audio_prefetch_workers) for name in threads)
        counts['audio_loader_threads'] = threads
        counts['audio_prefetch_workers'] = m._audio_prefetch_workers
        return counts

    cases = ('stream_greedy_b1', 'stream_greedy_b8', 'stream_ts_b8')
    if args.fixed:
        CASES.update({
            'fixed_beam_ts_b1': (1, 5, True, False, True, 'asr', None, None),
            'fixed_beam_ts_b3': (3, 5, True, False, True, 'asr', None, None),
            'fixed_translation_b1': (1, 1, False, False, True, 'st_eng', None, None),
            'fixed_translation_b8': (8, 1, False, False, True, 'st_eng', None, None),
        })
        cases = tuple(k for k in CASES if k.startswith('fixed_'))
    for case in cases:
        summary[case] = run_case(model, args, case, items, waves, configure=configure,
            observe=observe, input_sources=inputs)
        save(args.out / 'summary.json', summary)
        assert not getattr(model.backend, 'failed', set())
        assert all(v['equal'] == v['total'] for mode in summary[case].values() for v in mode['agreement'].values())
        assert not any(t.name.startswith('owsm-audio') for t in threading.enumerate())


if __name__ == '__main__':
    main()
