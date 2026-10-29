"""Check cache eviction/reuse across greedy, translation, prompt and beam requests on a compute node."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import tarfile
from contextlib import ExitStack
from pathlib import Path

from profile_expanded_greedy import select_inputs


def cases():
    prompt = 'Previous speech context. ' * 20
    # Each list preserves the same public request grouping across both policies.
    return [
        dict(name='draft_ts_b1', indices=[0], batch=1, beam=1, timestamps=True, draft=True),
        dict(name='draft_notime_stream_b3', indices=[1, 2, 3, 4, 5], batch=3, beam=1,
             timestamps=False, draft=True),
        dict(name='plain_ts_stream_b2', indices=[2, 3, 4, 5], batch=2, beam=1,
             timestamps=True, draft=False),
        dict(name='translation_b3', indices=[0, 1, 2, 3, 4], batch=3, beam=1,
             timestamps=False, draft=True, task='st_eng'),
        dict(name='prompt_greedy_b1', indices=[3], batch=1, beam=1, timestamps=True,
             draft=True, prompt=prompt),
        dict(name='beam_ts_b3', indices=[0, 1, 2, 3, 4], batch=3, beam=5, timestamps=True, draft=True),
        dict(name='prompt_beam_b1', indices=[1], batch=1, beam=5, timestamps=True,
             draft=True, prompt=prompt),
        dict(name='beam_notime_b1', indices=[0], batch=1, beam=5, timestamps=False, draft=True),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('model/audio work requires an allocated compute job')

    import soundfile as sf
    import torch
    from profile_beam import command, dispatch, result_record, save, sha256
    from profile_runtime import captures

    from owsm_trt import OWSM, hub
    from owsm_trt.audio import load_audio
    from owsm_trt.build import engine_tag

    with torch.inference_mode(), ExitStack() as stack:
        candidate = OWSM.from_pretrained(args.model, session_cache_size=3)
        reference = OWSM(candidate.cfg, candidate.tokenizer, candidate.frontend, candidate.backend,
                         candidate.opts, candidate.device, session_cache_size=2)
        models = dict(reference=reference, candidate=candidate)
        gpu = torch.cuda.get_device_properties(candidate.device)
        assert (candidate.cfg.dec_layers, candidate.cfg.d_model, gpu.name, gpu.multi_processor_count) == (
            18, 1024, 'NVIDIA H200', 132)
        selected = select_inputs(json.loads((args.data/'index.json').read_text()), 32,
                                 candidate.cfg.speech_seconds)
        selected = {lang: selected[lang][:6] for lang in ('eng', 'jpn')}
        save(args.out/'selection.json', selected)
        files = hub.resolve(args.model)
        engines = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(candidate.opts)
        env = dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'), job=os.environ['PBS_JOBID'],
                   checkpoint_sha256=sha256(files.checkpoint), index_sha256=sha256(args.data/'index.json'),
                   selection_sha256=sha256(args.out/'selection.json'),
                   engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')},
                   gpu=gpu.name, cuda=torch.version.cuda,
                   versions={name: importlib.metadata.version(name) for name in ('torch', 'triton', 'tensorrt')},
                   nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
                   session_limits={v: m.session_cache_size for v, m in models.items()}, cases=cases(),
                   scope='Compatibility, not timing. Same inputs/engine/precision/grouping, two policies. '
                         'Full records and differences retained. No reference translation/timestamp annotation. '
                         'Changing task or primer intentionally changes search inputs between named cases.')
        save(args.out/'environment.json', env)
        archives, expected, calls, differences = {}, {}, [], []
        for lang, rows in selected.items():
            waves = []
            for row in rows:
                if row['shard'] not in archives:
                    archives[row['shard']] = stack.enter_context(tarfile.open(args.data/row['shard']))
                data = archives[row['shard']].extractfile(row['audio_member']).read()
                assert hashlib.sha256(data).hexdigest() == row['audio_sha256']
                values, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
                mono = values[:, 0] if values.shape[1] == 1 else values.mean(axis=1)
                waves.append(load_audio(torch.from_numpy(mono), sr=sr))
            requests = cases()
            orders = [requests, list(reversed(requests)), requests[3:]+requests[:3]]
            for repeat, order in enumerate(orders):
                for case in order:
                    name = lang+'/'+case['name']
                    for variant in (list(models) if repeat % 2 == 0 else list(models)[::-1]):
                        model = models[variant]
                        model.ctc_draft = case['draft']
                        kw = dict(lang=lang, task=case.get('task', 'asr'), timestamps=case['timestamps'],
                                  prompt=case.get('prompt'), beam_size=case['beam'], batch_size=case['batch'])
                        primer = model._primer(lang, kw['task'], kw['timestamps'], kw['prompt'])
                        max_draft_window = max(model._draft_windows(case['batch']), default=0)
                        if case['name'] == 'prompt_greedy_b1':
                            assert len(primer) >= max_draft_window, 'prompt did not exercise long-primer path'
                        inputs = [waves[i] for i in case['indices']]
                        result = model.transcribe(inputs[0] if len(inputs) == 1 else inputs, **kw)
                        values = [result] if len(inputs) == 1 else result
                        records = {rows[i]['key']: result_record(model, value, dict(id=rows[i]['key']),
                                                                 rows[i]['seconds'])
                                   for i, value in zip(case['indices'], values)}
                        assert len(records) == len(inputs)
                        if name in expected and records != expected[name]:
                            differences.append(dict(case=name, variant=variant, repeat=repeat,
                                                    expected=expected[name], actual=records))
                        expected.setdefault(name, records)
                        assert not model.backend.failed and not getattr(model._beam_backend, 'failed', set())
                        assert len(model._sessions) <= model.session_cache_size
                        calls.append(dict(case=name, variant=variant, repeat=repeat,
                                          primer_length=len(primer), max_draft_window=max_draft_window,
                                          records=records,
                                          captures=captures(model),
                                          dispatch={str(key): dispatch(session)
                                                    for key, session in model._sessions.items()}))
                save(args.out/'calls.json', calls)
                save(args.out/'differences.json', differences)
                if progress := os.environ.get('OWSM_PROFILE_PROGRESS'):
                    with Path(progress).open('a') as stream:
                        stream.write(f'{lang}: mixed API repeat{repeat} complete; differences{len(differences)}\n')
        save(args.out/'complete.json', dict(complete=True, samples=12, cases=16, calls=len(calls),
                                            output_visits=sum(len(c['records']) for c in calls),
                                            differences=len(differences), repeat=3))
        if differences:
            raise SystemExit(3)


if __name__ == '__main__':
    main()
