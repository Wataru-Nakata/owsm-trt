"""Measure public beam opt-ins individually and together; never multiply isolated gains."""

import argparse
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import command, dispatch, save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag

PRESETS = {
    'reference': (False, False),
    'lse': (False, True),
    'grouped_tail': (True, False),
    'combined': (True, True),
}
PRESET_CASES = {
    'beam_single_ts': (1, 5, True, False, False, 'asr', None, None),
    'beam_stream_ts_b2': (2, 5, True, False, True, 'asr', None, None),
    'beam_stream_ts_b3': (3, 5, True, False, True, 'asr', None, None),
}
CASES.update(PRESET_CASES)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--long-audio', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace, args.variants = False, list(PRESETS)
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    gpu = torch.cuda.get_device_properties(model.device)
    if (model.cfg.dec_layers != 18 or model.cfg.d_model != 1024
            or gpu.name != 'NVIDIA H200' or gpu.multi_processor_count != 132):
        raise ValueError('this comparison targets the validated H200/1B grouped-row preset')
    items = json.loads(args.manifest.read_text())
    if len(items) < 3:
        ap.error('at least three utterances required')
    waves, disk_ms = wall(lambda: [load_audio(item['path']) for item in items])
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
    save(args.out / 'environment.json', dict(
        commit=os.environ.get('OWSM_PROFILE_COMMIT'), gpu=gpu.name, sm_count=gpu.multi_processor_count,
        cuda=torch.version.cuda, versions={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt')},
        nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap,memory.total', '--format=csv']),
        runtime_environment={k: v for k, v in os.environ.items()
                             if k.startswith(('OWSM_TRT_', 'CUDA_', 'TRITON_')) or k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')},
        manifest_sha256=sha256(args.manifest), load_ms=load_ms, disk_load_ms=disk_ms, presets=PRESETS,
        scope='N-class existing public opt-ins. Same engine/checkpoint and resident audio for every mode. '
              'B1 measures separate utterances; B2/B3 measure the whole corpus request. '
              'Grouped/tail is one previously validated preset; LSE is the new independent factor. '
              'Separate session pools; combined resident peak is not standalone deployment memory. '
              'No reference timestamp annotations or cold-disk timing.'))
    summary = {}

    def configure(m, mode):
        grouped, lse = PRESETS[mode]
        m.beam_grouped_cross = m.beam_compact_tail = grouped
        m.beam_fused_lse = lse

    def observe(m, call):
        observed = count_decoder_dispatch(m, call)
        observed['sessions'] = {
            str(key): dict(dispatch=dispatch(sess), fused_lse=sess.fused_lse,
                          compact_tail=sess.compact_tail, chunk=sess.chunk,
                          last_request_tail_start=sess._tail_started)
            for key, sess in m._sessions.items()
        }
        assert all(s.fused_lse == m.beam_fused_lse for s in m._sessions.values())
        assert not getattr(m._beam_backend, 'failed', set()), 'grouped decoder silently fell back'
        return observed

    for name in PRESET_CASES:
        summary[name] = run_case(model, args, name, items, waves, configure=configure, observe=observe)
        save(args.out / 'summary.json', summary)
        # Record all N-class output differences, while rejecting failed decoder trials and new warm captures.
        assert not getattr(model.backend, 'failed', set())
        assert not getattr(model._beam_backend, 'failed', set())

    generator = torch.Generator().manual_seed(0)
    silence = torch.zeros(model.cfg.sample_rate * 2)
    noise = torch.randn(model.cfg.sample_rate * 2, generator=generator) * .01
    window = waves[0].repeat(-(-model.cfg.num_samples // len(waves[0])))[:model.cfg.num_samples]
    cases = [('no_timestamps', [waves[0]], dict(timestamps=False)),
             ('translation', [waves[0]], dict(task='st_eng')),
             ('prompt', [waves[0]], dict(prompt='Previous speech context. ' * 20)),
             ('silence_noise', [silence, noise], {}), ('30s_repeated_speech', [window], {})]
    extra = {}
    for mode in PRESETS:
        configure(model, mode)
        model._sessions.clear()
        extra[mode] = {}
        for name, inputs, options in cases:
            kwargs = dict(lang=args.lang, beam_size=5, batch_size=3, timestamps=True)
            kwargs.update(options)
            value = model.transcribe(inputs, **kwargs)
            extra[mode][name] = [asdict(x) for x in value]
        if args.long_audio:
            extra[mode]['long_audio'] = asdict(model.transcribe_long(args.long_audio, lang=args.lang, beam_size=5))
        save(args.out / 'extra_outputs.json', extra)
        assert not getattr(model.backend, 'failed', set())
        assert not getattr(model._beam_backend, 'failed', set())
    save(args.out / 'extra_agreement.json', {
        mode: {name: values == extra['reference'][name] for name, values in rows.items()}
        for mode, rows in extra.items()
    })


if __name__ == '__main__':
    main()
