"""Public opt-in coverage beyond the K5 corpus timing benchmark; no speed claims."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import comparison, quality, result_record, save, sha256, wall

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.runtime import BeamSession


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('requires a compute allocation')
    items = json.loads(args.manifest.read_text())[:6]
    waves = [load_audio(x['path']) for x in items]
    records, expected, hashes = {}, None, None
    expected_records = {}
    for enabled in (False, True):
        model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, beam_fused_lse=enabled))
        assert model.beam_fused_lse is enabled
        files = hub.resolve(args.model)
        engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
        fingerprint = {p.name: sha256(p) for p in engines.glob('*.engine')}
        if hashes is not None:
            assert hashes == fingerprint
        hashes = fingerprint
        observed, outputs, metrics = {}, {}, {}
        original = model._session
        current = None

        def session(batch, beam=1):
            value = original(batch, beam)
            if isinstance(value, BeamSession):
                assert value.fused_lse == (enabled and model.beam_kernels)
            observed.setdefault(current, []).append(dict(batch=batch, beam=beam,
                fused_lse=getattr(value, 'fused_lse', None)))
            return value

        model._session = session
        cases = [
            ('beam2', 2, 3, True, True, {}), ('beam3', 3, 3, True, True, {}),
            ('beam5', 5, 3, True, True, {}), ('beam8', 8, 2, True, True, {}),
            ('no_timestamps', 5, 1, False, True, {}),
            ('translation', 5, 3, False, True, {'task': 'st_eng'}),
            ('prompt', 5, 1, True, True, {'prompt': 'Previous speech context. ' * 20}),
            ('greedy_plain', 1, 1, True, False, {}),
            ('greedy_stream', 1, 3, True, True, {}),
            ('greedy_translation', 1, 3, False, True, {'task': 'st_eng'}),
            ('beam5_again', 5, 3, True, True, {}),
            ('torch_selection', 5, 3, True, True, {}),
        ]
        for current, beam, batch, timestamps, draft, extra in cases:
            save(args.out / 'progress.json', dict(enabled=enabled, case=current, phase='start'))
            if model.ctc_draft != draft or current == 'torch_selection':
                model._sessions.clear()
            model.ctc_draft = draft
            model.beam_kernels = current != 'torch_selection'
            result = model.transcribe(waves, beam_size=beam, batch_size=batch, timestamps=timestamps, **extra)
            outputs[current] = [asdict(x) for x in result]
            detailed = {item['id']: result_record(model, value, item, len(wave) / model.cfg.sample_rate)
                        for item, wave, value in zip(items, waves, result)}
            if expected is None:
                expected_records[current] = detailed
            metrics[current] = dict(agreement=comparison(expected_records[current], detailed),
                quality=quality(items, detailed, 'eng') if extra.get('task', 'asr') == 'asr' else None,
                records=detailed)
            if expected is not None and (beam == 1 or current == 'torch_selection'):
                assert outputs[current] == expected[current], current
            save(args.out / 'partial.json', dict(enabled=enabled, outputs=outputs, dispatch=observed, metrics=metrics))
        assert outputs['beam5'] == outputs['beam5_again']
        assert not getattr(model.backend, 'failed', set())
        if expected is None:
            expected = outputs
        agreement = {name: value == expected[name] for name, value in outputs.items()}
        records[str(enabled)] = dict(outputs=outputs, agreement=agreement, dispatch=observed,
                                     metrics=metrics, load_ms=load_ms)
        torch.cuda.synchronize()
        model._session = original
        model._sessions.clear()
        model._frontend_replay = None
        original = model = None
        gc.collect()
        torch.cuda.empty_cache()
    save(args.out / 'complete.json', dict(modes=records, all_equal=all(records['True']['agreement'].values()),
        cases=len(cases), checkpoint_sha256=sha256(files.checkpoint), engine_sha256=hashes,
        gpu=torch.cuda.get_device_name(), commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        note='N-class beam differences are recorded; greedy/fallback regressions fail. Full K5 WER/CER and timestamp comparisons are in the separate corpus profiler.'))


if __name__ == '__main__':
    main()
