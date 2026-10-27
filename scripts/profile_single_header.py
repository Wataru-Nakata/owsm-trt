"""Isolate redundant singleton duration/header lookup using a frozen pre-change API method."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType

import torch
from profile_beam import save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import run_case

from owsm_trt import OWSM, hub, pipeline
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--baseline-pipeline', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    spec = importlib.util.spec_from_file_location('owsm_trt._header_baseline', args.baseline_pipeline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    args.trace, args.long_audio, args.variants = False, None, ['reference', 'no_header']
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    items = json.loads(args.manifest.read_text())
    if not items:
        ap.error('nonempty manifest required')
    paths = [x['path'] for x in items]
    waves, disk_ms = wall(lambda: [load_audio(p) for p in paths])
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
    save(args.out / 'environment.json', dict(
        commit=os.environ.get('OWSM_PROFILE_COMMIT'), gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
        cuda=torch.version.cuda, versions={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt', 'soundfile')},
        checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')},
        baseline_pipeline_sha256=sha256(args.baseline_pipeline), manifest_sha256=sha256(args.manifest),
        load_ms=load_ms, disk_load_ms=disk_ms,
        scope='E intent. Frozen pre-change transcribe method vs current method; same model/engines/loader/defaults. '
              'Every timed input is a singleton file list, OS cache warm/node local. '
              'No metadata/audio cache or altered batch ordering. All session pools resident.'))
    methods = {'reference': baseline.OWSM.transcribe, 'no_header': OWSM.transcribe}

    def configure(m, mode):
        m.transcribe = MethodType(methods[mode], m)

    def observe(m, call):
        counts = {'reference': 0, 'no_header': 0}
        original = [baseline.duration, pipeline.duration]

        def wrap(func, mode):
            def counted(*args, **kwargs):
                counts[mode] += 1
                return func(*args, **kwargs)
            return counted

        baseline.duration = wrap(original[0], 'reference')
        pipeline.duration = wrap(original[1], 'no_header')
        try:
            result = count_decoder_dispatch(m, call)
        finally:
            baseline.duration, pipeline.duration = original
        result['header_reads'] = counts
        return result

    cases = ('single_greedy',) if model.cfg.ctc_only else ('single_ts', 'beam_ts', 'translation')
    inputs, summary = {m: paths for m in args.variants}, {}
    for case in cases:
        summary[case] = run_case(model, args, case, items, waves, configure=configure,
                                 observe=observe, input_sources=inputs)
        save(args.out / 'summary.json', summary)
        assert all(v['equal'] == v['total'] for mode in summary[case].values() for v in mode['agreement'].values())
        assert not getattr(model.backend, 'failed', set())
        for mode in methods:
            counts = json.loads((args.out / case / f'actual_dispatch_{mode}.json').read_text())['header_reads']
            assert counts['no_header'] == 0
            assert (counts['reference'] > 0) == (mode == 'reference')
    # Scalar path, singleton list and loaded tensor keep API output shapes and loader errors.
    options = {} if model.cfg.ctc_only else dict(timestamps=True)
    outputs = {}
    for mode in methods:
        configure(model, mode)
        scalar = model.transcribe(paths[0], **options)
        single = model.transcribe([paths[0]], **options)
        loaded = model.transcribe(waves[0], **options)
        assert scalar == single[0] == loaded
        for audio in (str(Path(os.environ['TMPDIR']) / 'absent-header-regression.wav'), torch.zeros(2, 2, 2)):
            try:
                model.transcribe(audio, **options)
            except (RuntimeError, ValueError) as error:
                outputs.setdefault(mode, []).append(type(error).__name__)
            else:
                raise AssertionError('invalid input was accepted')
    assert outputs['reference'] == outputs['no_header']
    save(args.out / 'api_errors.json', dict(output_forms_equal=True, errors=outputs))


if __name__ == '__main__':
    main()
