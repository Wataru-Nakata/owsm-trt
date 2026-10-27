"""Compute-node public file-prefetch paths, exclusions, errors and model/session reuse."""

import argparse
import faulthandler
import gc
import importlib.util
import json
import os
import threading
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import save, sha256, wall

from owsm_trt import OWSM, hub, pipeline
from owsm_trt.audio import duration, load_audio, resample
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--long-audio', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, nargs='+', default=[0, 1, 2], choices=[0, 1, 2])
    ap.add_argument('--reference', type=Path, help='saved complete/partial output of worker0')
    ap.add_argument('--baseline-frontend', type=Path, help='diagnostic only: load a frozen frontend replay class')
    ap.add_argument('--frontend-eager', action='store_true', help='diagnostic only: disable frontend replay')
    ap.add_argument('--legacy-frontend-warmup', action='store_true', help='diagnostic only: restore fresh warmup streams')
    ap.add_argument('--sync-cases', action='store_true', help='diagnostic only: synchronize before/after each API case')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    if args.workers[0] != 0 and args.reference is None:
        ap.error('a reference is required when skipping worker0')
    import soundfile as sf

    if args.baseline_frontend:
        from owsm_trt import frontend_graph

        spec = importlib.util.spec_from_file_location('owsm_trt._baseline_frontend', args.baseline_frontend)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        frontend_graph.FrontendReplay = mod.FrontendReplay

    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    paths = [x['path'] for x in json.loads(args.manifest.read_text())[:10]]
    waves = [load_audio(path) for path in paths]
    args.out.mkdir(parents=True, exist_ok=True)
    local_audio = Path(os.environ['TMPDIR']) / 'prefetch_api_audio'
    local_audio.mkdir(exist_ok=True)
    torch.manual_seed(7)
    for name, wave, rate in [('silence', torch.zeros(16000), 16000),
                             ('noise', torch.randn(16000) * .01, 16000),
                             ('stereo44', resample(waves[0][:32000], 16000, 44100), 44100)]:
        path = local_audio / f'{name}.wav'
        data = wave.numpy() if name != 'stereo44' else torch.stack((wave, wave * .5), dim=1).numpy()
        sf.write(path, data, rate)
        paths.append(str(path))
    original_loader = pipeline.load_audio
    records, expected, engine_hashes = {}, None, None
    if args.reference is not None:
        previous = json.loads(args.reference.read_text())
        expected = previous.get('modes', previous)['0']['outputs']

    iteration = -1

    def progress(workers, case, phase):
        save(args.out / 'case_progress.json', dict(iteration=iteration, workers=workers, case=case, phase=phase))
        print(iteration, workers, case, phase, flush=True)
        if args.sync_cases:
            torch.cuda.synchronize()
    try:
        for iteration, workers in enumerate(args.workers):
            progress(workers, 'model_load', 'start')
            model, ms = wall(lambda: OWSM.from_pretrained(args.model, audio_prefetch_workers=workers, frontend_graph=not args.frontend_eager))
            model._frontend_reuse_warmup_stream = not args.legacy_frontend_warmup
            assert model._audio_prefetch_workers == workers
            files = hub.resolve(args.model)
            engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
            hashes = {p.name: sha256(p) for p in engines.glob('*.engine')}
            if engine_hashes is not None:
                assert hashes == engine_hashes
            engine_hashes = hashes
            outputs, observed = {}, {}
            mixed = paths[:9] + [waves[0]]
            cases = [
                ('single_file', paths[0], True, False, False, dict(timestamps=True)),
                ('stream_b1', paths, True, True, False, dict(batch_size=1)),
                ('stream_b8', paths, True, True, False, dict(batch_size=8)),
                ('stream_ts_b8', paths, True, True, False, dict(batch_size=8, timestamps=True)),
                ('mixed', mixed, True, False, False, dict(batch_size=8, timestamps=True)),
                ('loaded', waves, True, False, False, dict(batch_size=8, timestamps=True)),
                ('plain3', paths[:3], False, False, False, dict(batch_size=3, timestamps=True)),
                ('beam3', paths[:6], True, True, False, dict(batch_size=3, beam_size=5, timestamps=True)),
                ('translation8', paths[:10], True, True, False, dict(batch_size=8, task='st_eng')),
                ('long_prompt', paths[:2], True, True, False,
                 dict(batch_size=1, timestamps=True, prompt='Previous speech context. ' * 20)),
                ('cpu_autocast', [paths[-1], paths[0]], True, False, True, dict(batch_size=1, timestamps=True)),
                ('stream_again', paths, True, True, False, dict(batch_size=8, timestamps=True)),
                ('empty', [], True, False, False, dict(batch_size=8)),
            ]
            for name, audio, draft, eligible, autocast, kwargs in cases:
                progress(workers, name, 'start')
                if model.ctc_draft != draft:
                    model._sessions.clear()
                model.ctc_draft = draft
                counts, lock = {}, threading.Lock()

                def counted(*args, **kwargs):
                    value = original_loader(*args, **kwargs)
                    with lock:
                        thread = threading.current_thread().name
                        counts[thread] = counts.get(thread, 0) + 1
                    return value

                pipeline.load_audio = counted
                with torch.autocast('cpu', dtype=torch.bfloat16) if autocast else nullcontext():
                    value = model.transcribe(audio, **kwargs)
                assert all(t.startswith('owsm-audio') == (eligible and workers > 0) for t in counts)
                assert not any(t.name.startswith('owsm-audio') for t in threading.enumerate())
                outputs[name] = [asdict(x) for x in value] if isinstance(value, list) else asdict(value)
                observed[name] = counts
                save(args.out / 'cases_partial.json', dict(workers=workers, outputs=outputs, loader_threads=observed))
                progress(workers, name, 'complete')
            # Failure in the first sorted read must close the worker scope; the next request resets
            # normal session state. Inject after valid header sorting, before any batch is decoded.
            first = min(paths, key=lambda path: duration(path, 16000))

            def fail_first(audio, *args, **kwargs):
                if os.fspath(audio) == first:
                    raise OSError('injected first audio read failure')
                return original_loader(audio, *args, **kwargs)

            progress(workers, 'reader_error', 'start')
            pipeline.load_audio = fail_first
            try:
                model.transcribe(paths, batch_size=8, timestamps=True)
            except OSError as error:
                assert 'injected first' in str(error)
            else:
                raise AssertionError('expected injected reader failure')
            finally:
                pipeline.load_audio = original_loader
            assert not any(t.name.startswith('owsm-audio') for t in threading.enumerate())
            progress(workers, 'after_error', 'start')
            value = model.transcribe(paths, batch_size=8, timestamps=True)
            outputs['after_error'] = [asdict(x) for x in value]
            assert outputs['after_error'] == outputs['stream_ts_b8']
            progress(workers, 'after_error', 'complete')
            # A later fixed-batch read may already be running while the first batch decodes.
            # Observe its failure in order, close workers, then reuse the same beam session.
            failing = sorted(paths[:6], key=lambda path: duration(path, 16000))[3]

            def fail_fixed(audio, *args, **kwargs):
                if os.fspath(audio) == failing:
                    raise OSError('injected fixed audio read failure')
                return original_loader(audio, *args, **kwargs)

            progress(workers, 'fixed_reader_error', 'start')
            pipeline.load_audio = fail_fixed
            try:
                model.transcribe(paths[:6], batch_size=3, beam_size=5, timestamps=True)
            except OSError as error:
                assert 'injected fixed' in str(error)
            else:
                raise AssertionError('expected fixed reader failure')
            finally:
                pipeline.load_audio = original_loader
            assert not any(t.name.startswith('owsm-audio') for t in threading.enumerate())
            recovered = model.transcribe(paths[:6], batch_size=3, beam_size=5, timestamps=True)
            assert [asdict(x) for x in recovered] == outputs['beam3']
            progress(workers, 'fixed_reader_error', 'cleanup_and_reuse_complete')
            progress(workers, 'long', 'start')
            outputs['long'] = asdict(model.transcribe_long(args.long_audio))
            progress(workers, 'long', 'complete')
            assert not getattr(model.backend, 'failed', set())
            if expected is None:
                expected = outputs
            else:
                assert outputs == expected, workers
            records[str(workers)] = dict(load_ms=ms, outputs=outputs, loader_threads=observed,
                error_cleanup_and_reuse=True)
            save(args.out / 'partial.json', records)
            progress(workers, 'model_cleanup', 'start')
            torch.cuda.synchronize()
            model._sessions.clear()
            model._frontend_replay = None
            del model
            gc.collect()
            torch.cuda.empty_cache()
            progress(workers, 'model_cleanup', 'complete')
            print(workers, 'public prefetch checks and cleanup complete', flush=True)
        save(args.out / 'complete.json', dict(all_equal=True, cases=len(expected), modes=records,
            checkpoint_sha256=sha256(files.checkpoint), engine_sha256=engine_hashes,
            commit=os.environ.get('OWSM_PROFILE_COMMIT'), gpu=torch.cuda.get_device_name(),
            long_token_scope='long-form compares text/segments; joined token IDs unavailable',
            diagnostic_sync_cases=args.sync_cases, requested_workers=args.workers,
            frontend_graph=not args.frontend_eager, reuse_frontend_warmup=not args.legacy_frontend_warmup, cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
            baseline_frontend_sha256=sha256(args.baseline_frontend) if args.baseline_frontend else None,
            reference_sha256=sha256(args.reference) if args.reference else None))
    finally:
        pipeline.load_audio = original_loader
        for path in local_audio.iterdir():
            path.unlink()
        local_audio.rmdir()


if __name__ == '__main__':
    main()
