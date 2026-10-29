"""Measure session eviction under alternating B1 and B3 requests with a final B2 group.

The reference intentionally includes repeated session setup/capture. This is a mixed-shape
request workload, not a kernel or fully resident steady-state speedup. The private3-entry
candidate changes no public defaults. Run inference/audio work only in a compute allocation.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import tarfile
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

from profile_expanded_greedy import select_inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--languages', nargs='+', default=['eng', 'jpn'])
    ap.add_argument('--repeat', type=int, default=7)
    ap.add_argument('--isolated-limit', type=int, choices=(2, 3),
                    help='Run one policy in a fresh process for standalone memory assessment')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('model/audio operations require an allocated compute job')
    if args.repeat < 1 or len(set(args.languages)) != len(args.languages):
        ap.error('positive repeat and distinct languages required')

    import soundfile as sf
    import torch
    from profile_beam import command, dispatch, quality, result_record, save, sha256, wall
    from profile_runtime import captures

    from owsm_trt import OWSM, hub
    from owsm_trt.audio import load_audio
    from owsm_trt.build import engine_tag

    limits = dict(reference=2, candidate=3)
    if args.isolated_limit is not None:
        limits = {name: limit for name, limit in limits.items() if limit == args.isolated_limit}
    memory_scope = ('standalone policy in this process' if args.isolated_limit is not None
                    else 'combined two-policy experiment, not standalone deployment memory')

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    device_used_bytes=total-free, device_total_bytes=total)

    with torch.inference_mode(), ExitStack() as stack:
        model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
        selected = select_inputs(json.loads((args.data/'index.json').read_text()), 32, model.cfg.speech_seconds)
        if not set(args.languages) <= selected.keys():
            ap.error('requested language absent from the corpus')
        selected = {lang: selected[lang][:12] for lang in args.languages}
        save(args.out/'selection.json', selected)
        files = hub.resolve(args.model)
        engines = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(model.opts)
        save(args.out/'environment.json', dict(
            commit=os.environ.get('OWSM_PROFILE_COMMIT'), job=os.environ['PBS_JOBID'], load_ms=load_ms,
            checkpoint_sha256=sha256(files.checkpoint), index_sha256=sha256(args.data/'index.json'),
            selection_sha256=sha256(args.out/'selection.json'),
            engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')},
            gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
            versions={name: importlib.metadata.version(name) for name in ('torch', 'triton', 'tensorrt')},
            nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
            repeat=args.repeat, session_limits=limits, memory_scope=memory_scope,
            scope='Private session retention ablation, public default2 unchanged. '
                  'Reference includes cache churn/setup/capture. See memory_scope for pool residency. Loaded audio '
                  'outside timing. Diagnostic factory/decoder counts are separate from timing.'))
        archives = {}
        for lang, rows in selected.items():
            def read_audio():
                waves = []
                for row in rows:
                    if row['shard'] not in archives:
                        archives[row['shard']] = stack.enter_context(tarfile.open(args.data/row['shard']))
                    data = archives[row['shard']].extractfile(row['audio_member']).read()
                    assert hashlib.sha256(data).hexdigest() == row['audio_sha256']
                    values, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
                    mono = values[:, 0] if values.shape[1] == 1 else values.mean(axis=1)
                    waves.append(load_audio(torch.from_numpy(mono), sr=sr))
                return waves

            waves, audio_ms = wall(read_audio)
            model._sessions.clear()
            gc.collect()
            torch.cuda.empty_cache()
            pools = {variant: OrderedDict() for variant in limits}
            expected, records, cold, diagnostics = {}, [], [], []
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = memory()

            def cycle(variant, offset):
                model._sessions = pools[variant]
                model._session_cache_limit = limits[variant]
                single = model.transcribe(waves[offset], lang=lang, beam_size=5, timestamps=True)
                batch = model.transcribe(waves[offset+1:offset+6], lang=lang, beam_size=5,
                                         batch_size=3, timestamps=True)
                assert not model.backend.failed, 'decoder fallback'
                return [single, *batch]

            def check(values, offset):
                actual = {rows[i]['key']: result_record(model, value, dict(id=rows[i]['key']), rows[i]['seconds'])
                          for i, value in zip(range(offset, offset+6), values)}
                assert len(actual) == 6
                for key, value in actual.items():
                    if key in expected:
                        assert value == expected[key], ('changed output', lang, key)
                    expected[key] = value

            # Exercise both input groups before timing. Reference2 cannot retain all three sessions;
            # its repeated captures are the intended measured cost, not an accidental warmup omission.
            for variant in limits:
                for offset in (0, 6):
                    values, ms = wall(lambda: cycle(variant, offset))
                    check(values, offset)
                    cold.append(dict(variant=variant, offset=offset, wall_ms=ms,
                                     captures=captures(model), memory=memory()))
            candidate_captures = captures(model)
            for repeat in range(args.repeat):
                for offset in (0, 6):
                    for variant in (limits if repeat % 2 == 0 else reversed(limits)):
                        torch.cuda.reset_peak_memory_stats()
                        values, ms = wall(lambda: cycle(variant, offset))
                        check(values, offset)
                        if variant == 'candidate':
                            assert captures(model) == candidate_captures, 'new candidate warm capture'
                        records.append(dict(variant=variant, repeat=repeat, offset=offset, wall_ms=ms,
                            ids=[r['key'] for r in rows[offset:offset+6]],
                            audio_seconds=sum(r['seconds'] for r in rows[offset:offset+6]),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(), memory=memory()))
                save(args.out/lang/'wall.json', records)
                save(args.out/lang/'tokens.json', expected)
                if progress := os.environ.get('OWSM_PROFILE_PROGRESS'):
                    with Path(progress).open('a') as stream:
                        stream.write(f'{lang}: session retention repeat{repeat} complete\n')
            # Observe creations in a separate pass; never retain evicted session objects in the observer.
            original = model._session
            for variant in limits:
                created = []

                def observe(batch, beam=1):
                    missing = (batch, beam) not in model._sessions
                    session = original(batch, beam)
                    if missing:
                        created.append(dict(batch=batch, beam=beam, dispatch=dispatch(session)))
                    return session

                model._session = observe
                try:
                    for offset in (0, 6):
                        check(cycle(variant, offset), offset)
                finally:
                    del model._session
                assert len(created) == (6 if variant == 'reference' else 0)
                diagnostics.append(dict(variant=variant, created=created, retained_captures=captures(model)))
            save(args.out/lang/'summary.json', dict(
                complete=True, input_count=12, audio_load_ms=audio_ms, cold=cold, diagnostics=diagnostics,
                baseline_memory=baseline_memory, final_memory=memory(), memory_scope=memory_scope,
                quality=quality([dict(id=r['key'], text=r['text']) for r in rows], expected, lang)))
            # Destroy both pools before the next language; a previous candidate pool must not linger.
            for pool in pools.values():
                pool.clear()
        save(args.out/'complete.json', dict(complete=True, languages=args.languages,
                                           samples=sum(map(len, selected.values())), repeat=args.repeat))


if __name__ == '__main__':
    main()
