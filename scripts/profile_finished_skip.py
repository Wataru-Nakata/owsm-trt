"""Same-engine plain-decoder EOS overshoot ablation; wall and GPU counter runs are separate."""

import argparse
import gc
import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from profile_beam import command, comparison, dispatch, quality, result_record, save, sha256, stats, wall

from owsm_trt import OWSM, hub, pipeline
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.runtime import GreedySession


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra-checks", action="store_true")
    ap.add_argument("--long-audio", type=Path)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        ap.error("audio/model work requires a compute allocation")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    items = json.loads(args.manifest.read_text())
    assert len({it['id'] for it in items}) == len(items)
    waves, disk_ms = wall(lambda: [load_audio(it['path']) for it in items])
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, ctc_draft=False))
    if not getattr(model.backend, "supports_finished_stop", False):
        raise RuntimeError("single-row decoder does not support finished-stop; no silent no-op benchmark")
    files = hub.resolve(args.model)
    directory = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    save(args.out / 'environment.json', dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
         checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name: sha256(p) for p in directory.glob('*.engine')},
         torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
         nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
         disk_ms=disk_ms, load_ms=load_ms,
         note='Fixed batches, no CTC drafts; same host polling and graph replay counts. Whole-list warmed wall time. '
              'Cold observations share unequal compiler caches; all variants coexist in process peak. '
              'Separate device counters count graph calls and calls entered after EOS; no per-step host reads.'))
    cases = dict(plain_b1=dict(batch_size=1), plain_ts_b1=dict(batch_size=1, timestamps=True),
                 translation_b1=dict(batch_size=1, task='st_eng'), translation_b3=dict(batch_size=3, task='st_eng'))
    summaries = {}
    for case, options in cases.items():
        caches = {mode: OrderedDict() for mode in ('reference', 'skip')}
        records, cold, samples = {}, {}, []
        torch.cuda.reset_peak_memory_stats()

        def call(mode):
            class Session(GreedySession):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **(kwargs | {'skip_finished': mode == 'skip'}))
            model._sessions = caches[mode]
            with patch.object(pipeline, 'GreedySession', Session):
                return model.transcribe(waves, lang=args.lang, **options)

        def record(outputs):
            return {it['id']: result_record(model, result, it, len(wave) / model.cfg.sample_rate)
                    for it, wave, result in zip(items, waves, outputs)}

        for mode in caches:
            first, cold[mode] = wall(lambda: call(mode))
            records[mode] = record(first)
            warm, _ = wall(lambda: call(mode))
            assert record(warm) == records[mode]
        save(args.out / case / 'tokens.json', records)
        agreement = comparison(records['reference'], records['skip'])
        save(args.out / case / 'agreement.json', agreement)
        assert records['reference'] == records['skip'], 'finished-stop changed full output'
        for repeat in range(args.repeat):
            for mode in (tuple(caches) if repeat % 2 == 0 else tuple(caches)[::-1]):
                outputs, elapsed = wall(lambda: call(mode))
                assert record(outputs) == records[mode]
                samples.append(dict(mode=mode, repeat=repeat, wall_ms=elapsed))
        save(args.out / case / 'wall.json', samples)
        summary = dict(agreement=agreement, cold_ms=cold, timing={}, quality={}, dispatch={})
        for mode in caches:
            values = [r['wall_ms'] for r in samples if r['mode'] == mode]
            summary['timing'][mode] = stats(values) | {'rtfx': sum(len(w) for w in waves) / model.cfg.sample_rate / (stats(values)['p50']/1000)}
            summary['quality'][mode] = (quality(items, records[mode], args.lang) if options.get('task', 'asr') == 'asr'
                                        else {'unmeasured': 'No reference English translations'})
            summary['dispatch'][mode] = {str(k): dict(calls=dispatch(s), skip_finished=s.skip_finished,
                call_guards={c: bool(getattr(fn, 'skips_finished', False)) for c, fn in s._steps.items()})
                for k,s in caches[mode].items()}
            for s in caches[mode].values():
                expected = mode == 'skip' and s.batch == 1
                assert s.skip_finished is expected
                assert all(bool(getattr(fn, 'skips_finished', False)) is expected for fn in s._steps.values())
        assert not getattr(model.backend, 'failed', set()), 'decoder trial fallback invalidates ablation'
        summary['shared_process_peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        summaries[case] = summary
        save(args.out / 'summary.json', summaries)

        diagnostics = {}
        for mode in caches:
            diagnostic = []

            class Observed(GreedySession):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **(kwargs | {'skip_finished': mode == 'skip'}))
                    assert not self._plain_ready, 'counter instrumentation requires lazy plain construction'
                    self.counts = torch.zeros(2, dtype=torch.int32, device=self.device)

                def _prepare_steps(self):
                    super()._prepare_steps()
                    for c, fn in self._steps.items():
                        def observed(fn=fn):
                            self.counts[0].add_(1)
                            self.counts[1].add_(self.done.all().to(torch.int32))
                            fn()
                        observed.__dict__.update(fn.__dict__)
                        self._steps[c] = observed

                def decode(self, primer, maximum):
                    self._prepare_plain()
                    self.counts.zero_()
                    tokens = super().decode(primer, maximum)
                    diagnostic.append(dict(calls=self.counts.cpu().tolist(), output_lengths=list(map(len, tokens)),
                                           skip_finished=self.skip_finished))
                    return tokens

            model._sessions = OrderedDict()
            with patch.object(pipeline, 'GreedySession', Observed):
                outputs = model.transcribe(waves, lang=args.lang, **options)
            assert record(outputs) == records[mode], 'counter instrumentation changed output'
            diagnostics[mode] = diagnostic
            save(args.out / case / f'{mode}_diagnostics.json', dict(batches=diagnostic,
                 sorted_ids=[items[i]['id'] for i in sorted(range(len(waves)), key=lambda i: len(waves[i]))]))
        assert [[b['calls'], b['output_lengths']] for b in diagnostics['reference']] == [
            [b['calls'], b['output_lengths']] for b in diagnostics['skip']], 'changed replay/EOS counts'
        summary['diagnostic_calls'] = {mode: dict(total=sum(b['calls'][0] for b in rows),
            after_eos=sum(b['calls'][1] for b in rows)) for mode, rows in diagnostics.items()}
        save(args.out / 'summary.json', summaries)
        if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as stream:
                stream.write(f'{args.out.name}: {case} finished-stop comparison complete\n')
        model._sessions = OrderedDict()
        caches.clear()
        gc.collect()
        torch.cuda.empty_cache()


    if args.extra_checks:
        generator = torch.Generator().manual_seed(73)
        synthetic = [torch.zeros(model.cfg.sample_rate * 2),
                     torch.randn(model.cfg.sample_rate * 2, generator=generator) * .01]
        window = waves[0].repeat(math.ceil(model.cfg.num_samples / len(waves[0])))[:model.cfg.num_samples]
        checks = [('silence_noise', synthetic, dict(timestamps=True), False),
                  ('long_prompt', waves[:1], dict(prompt='Previous speech context. ' * 20, timestamps=True), False),
                  ('full_window', [window], dict(timestamps=True), False),
                  ('drafted_b1', waves[:3], dict(batch_size=1, timestamps=True), True),
                  ('drafted_b3', waves[:3], dict(batch_size=3, timestamps=True), True),
                  ('lookahead', waves[:3], dict(task='st_eng'), False)]
        if args.long_audio:
            checks.append(('long_audio', [load_audio(args.long_audio)], {}, False))
        extra = {}
        for name, audio, options, drafted in checks:
            extra[name] = {}
            for mode in ('reference', 'skip'):
                class Session(GreedySession):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **(kwargs | {'skip_finished': mode == 'skip'}))
                model._sessions = OrderedDict()
                model.ctc_draft = drafted
                model._encoder_lookahead = name == 'lookahead'
                with patch.object(pipeline, 'GreedySession', Session):
                    fn = model.transcribe_long if name == 'long_audio' else model.transcribe
                    results = fn(audio, **(dict(lang=args.lang, batch_size=1) | options))
                extra[name][mode] = [asdict(r) for r in results]
                save(args.out / 'extra_checks.json', extra)
            assert extra[name]['reference'] == extra[name]['skip'], f'extra output changed: {name}'
            assert not getattr(model.backend, 'failed', set())
            if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
                with Path(path).open('a') as stream:
                    stream.write(f'{args.out.name}: {name} extra finished-stop check complete\n')
        # Long-form public output contains joined segments, not per-window raw token IDs.
        save(args.out / 'extra_scope.json', {'long_form_raw_token_parity': 'unmeasured',
             'long_form_checked': 'public joined text and segments only'})


if __name__ == '__main__':
    main()
