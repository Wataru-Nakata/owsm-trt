"""Opt-in N experiment using reviewed wider Ring numerical evidence; production routing is unchanged."""

import argparse
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import command, save, sha256, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag

WIDE_CASES = {
    'wide_beam_b2': (2, 5, True, False, True, 'asr', None, None),
    'wide_beam_b3': (3, 5, True, False, True, 'asr', None, None),
    'wide_translation_b10': (10, 1, False, False, True, 'st_eng', None, None),
    'wide_translation_b16': (16, 1, False, False, True, 'st_eng', None, None),
}
CASES.update(WIDE_CASES)


class WideBackend:
    """Only exact reviewed R/S/kind tuples use new routes; all other calls retain the base backend."""

    def __init__(self, base, routes):
        self.base, self.routes = base, routes

    def __getattr__(self, key):
        return getattr(self.base, key)

    @property
    def failed(self):
        return self.base.failed.union(*(backend.failed for backend in self.routes.values()))

    def step_fn(self, tokens, *args, kind='decode', **kwargs):
        route = self.routes.get(tokens.shape[0]) if tokens.shape[1] == 1 and kind == 'decode' else None
        return (route or self.base).step_fn(tokens, *args, kind=kind, **kwargs)


def validate_evidence(evidence, model, files, engines):
    """Known failed random gates remain failed; this authorizes an N study, never production parity."""
    gpu = torch.cuda.get_device_properties(model.device)
    if (evidence['checkpoint_sha256'] != sha256(files.checkpoint)
            or evidence['engine_sha256'] != {p.name: sha256(p) for p in engines.glob('*.engine')}
            or evidence['gpu'] != torch.cuda.get_device_name(model.device)
            or gpu.multi_processor_count != 132 or 'H200' not in gpu.name
            or evidence['versions'] != {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt')}
            or evidence['cuda'] != torch.version.cuda or not evidence['wide_one_token']):
        raise RuntimeError('numerical evidence does not match checkpoint/engines/GPU/software')
    package = Path(megakernel_ring.__file__).parent
    if any(sha256(package/name) != digest for name,digest in evidence['source_sha256'].items()):
        raise RuntimeError('numerical evidence is for different runtime/kernel source')
    if evidence['tests']['passed'] != 6 or len(evidence['cases']) != 9:
        raise RuntimeError('incomplete numerical tests')
    for name, case in evidence['cases'].items():
        ring = case['versus_cpu_fp32']['ring']
        if (case['status'] != 'complete' or not case['ring_repeat_exact'] or not case['ring_prefix_preserved']
                or not ring['finite'] or case['actual_ring_cross_splits'] != 1
                or case['actual_ring_programs'] != case['rows']*6):
            raise RuntimeError(f'invalid numerical state evidence: {name}')
        if case['real_prefill'] and (not ring['existing_check_pass'] or ring['argmax_agreement'] != 1):
            raise RuntimeError(f'real-prefill evidence failed: {name}')
    failures = {name for name,c in evidence['cases'].items()
                if not c['versus_cpu_fp32']['ring']['existing_check_pass']}
    if failures != {'random_t374_r10_s1', 'random_t374_r15_s1'}:
        raise RuntimeError('unexpected numerical gate failures require review')


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--numerics', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('compute allocation required')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace, args.long_audio, args.variants = False, None, ['reference', 'wide']
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    base = model.backend
    if (not isinstance(base, megakernel.MegaBackend) or not isinstance(base.kernel, megakernel_ring.RingDecoder)
            or (base.kernel.m.D, base.kernel.m.F, base.kernel.m.L, base.kernel.m.H) != (384, 1536, 6, 6)):
        raise RuntimeError('only the diagnosed102M Ring model is supported')
    evidence = json.loads(args.numerics.read_text())
    files = hub.resolve(args.model)
    engines = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(model.opts)
    validate_evidence(evidence, model, files, engines)
    routes = {rows: megakernel.MegaBackend(base.inner,
              megakernel_ring.RingDecoder(base.kernel.m, row_programs=rows*6),
              max_tokens=rows, max_rows=rows, trial=True) for rows in (10, 15, 16)}
    wide = WideBackend(base, routes)
    items = json.loads(args.manifest.read_text())
    if len(items) < 10:
        ap.error('at least ten inputs required')
    waves = [load_audio(item['path']) for item in items]
    save(args.out/'environment.json', dict(
        commit=os.environ.get('OWSM_PROFILE_COMMIT'), checkpoint_sha256=sha256(files.checkpoint),
        engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
        numerics_sha256=sha256(args.numerics), numerical_gate_failures_retained=['random_t374_r10_s1', 'random_t374_r15_s1'],
        gpu=torch.cuda.get_device_name(), sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
        versions={n:importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt')}, cuda=torch.version.cuda,
        nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap,memory.total', '--format=csv']),
        manifest_sha256=sha256(args.manifest), load_ms=load_ms,
        scope='N-class isolated wide Ring vs current auto. R10/15/16 S1 decode only; other calls unchanged. '
              'Same engines/weights, loaded inputs, separate session pools. All token/time/text/segments differences '
              'recorded; no reference translation or timestamp annotations. B1beam control first12individual calls; '
              'others whole-corpus fixed sorted batches. Japanese12 translationB16 is an unchanged R12control.'))

    def configure(m, mode):
        m.backend = base if mode == 'reference' else wide

    def observe(m, call):
        return count_decoder_dispatch(m, call)

    summary = {}
    for name in ('beam_ts', *WIDE_CASES):
        summary[name] = run_case(model, args, name, items, waves, configure=configure, observe=observe)
        save(args.out/'summary.json', summary)
        assert not wide.failed, 'silent decoder fallback in numerical experiment'

    generator = torch.Generator().manual_seed(0)
    silence = torch.zeros(model.cfg.sample_rate*2)
    noise = torch.randn(model.cfg.sample_rate*2, generator=generator)*.01
    window = waves[0].repeat(-(-model.cfg.num_samples//len(waves[0])))[:model.cfg.num_samples]
    cases = [('no_timestamps', waves[:3], dict(timestamps=False)),
             ('translation', waves[:3], dict(task='st_eng')),
             ('prompt', waves[:3], dict(prompt='Previous speech context. '*20)),
             ('silence_noise_speech', [silence,noise,waves[0]], {}),
             ('30s_mixed', [window,waves[1],waves[2]], {})]
    extra = {}
    for mode in args.variants:
        configure(model, mode)
        model._sessions.clear()
        extra[mode] = {}
        for name, inputs, options in cases:
            kwargs = dict(lang=args.lang, beam_size=5, batch_size=3, timestamps=True)
            kwargs.update(options)
            extra[mode][name] = [asdict(x) for x in model.transcribe(inputs, **kwargs)]
        save(args.out/'extra_outputs.json', extra)
        assert not wide.failed
    save(args.out/'extra_agreement.json', {name: extra['wide'][name]==extra['reference'][name] for name,_,_ in cases})


if __name__ == '__main__':
    main()
