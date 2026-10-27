"""Resumable, sharded output regression over the expanded corpus; not a latency benchmark."""

import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import signal
import tarfile
import time
from collections import Counter, OrderedDict
from contextlib import ExitStack
from pathlib import Path

from prepare_large_regression import publish_file, sha256

FIELDS = ('tokens','timestamp_tokens','segments','text')


def record_differences(reference: dict, actual: dict) -> list[str]:
    return [field for field in FIELDS if reference[field]!=actual[field]]


def repetition(tokens: list[int]) -> dict:
    longest = current = 0
    previous = None
    for token in tokens:
        current = current+1 if token==previous else 1
        previous = token
        longest = max(longest,current)
    grams = Counter(tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3)))
    return dict(max_identical_token_run=longest,max_fourgram_count=max(grams.values(),default=0),
                scope='Diagnostic repetition indicators, not annotation-based error labels.')


def select_rows(rows: list[dict], case: str, datasets: list[str] | None) -> list[dict]:
    selected = [r for r in rows if (not datasets or r['dataset'] in datasets)
                and (r['long_form'] if case=='long' else not r['long_form'])]
    # Process Japanese and English early; deterministic hash order mixes lengths within a corpus.
    rank = {'jpn':0,'eng':1}
    selected.sort(key=lambda r:(rank.get(r['lang'],2),r['lang'],r['dataset'],
                               hashlib.sha256(r['key'].encode()).hexdigest()))
    return selected


def capture_failure(call, context: dict, out: Path, publish: Path):
    """Keep exact source/configuration evidence even if decoding invalidates the CUDA context."""
    try:
        return call()
    except Exception as exc:
        failure = context | dict(error_type=type(exc).__name__,error=str(exc),job=os.environ.get('PBS_JOBID'))
        name = 'failure-'+os.environ.get('PBS_JOBID','local')+'.json'
        path = out/name
        path.write_text(json.dumps(failure,indent=2,ensure_ascii=False)+'\n')
        publish_file(path,publish/name)
        raise


def load_completed(publish: Path, rows: list[dict], chunk_size: int, config_digest: str) -> dict:
    prior_chunks = {}
    if (publish/'progress.json').exists():
        prior_chunks = json.loads((publish/'progress.json').read_text())['chunks']
    completed = {}
    for path in sorted(publish.glob('chunk-*.jsonl.gz')):
        if path.name in prior_chunks:
            assert sha256(path)==prior_chunks[path.name]['sha256'],path
        with gzip.open(path,'rt') as f:
            records = [json.loads(line) for line in f]
        expected = rows[int(path.name.split('-')[1].split('.')[0])*chunk_size:][:chunk_size]
        assert expected and [r['key'] for r in records]==[r['key'] for r in expected],path
        assert all(r['config_sha256']==config_digest for r in records),path
        for record in records:
            assert all(set(FIELDS).issubset(record[v]) for v in ('reference','candidate')),path
            assert record['differences']==record_differences(record['reference'],record['candidate']),path
        completed[path.name] = dict(samples=len(records),sha256=sha256(path),differences=sum(bool(r['differences']) for r in records))
    return completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--publish',type=Path,required=True)
    ap.add_argument('--mode',choices=['beam_bound','ring_split','serial64'],default='beam_bound')
    ap.add_argument('--case',choices=['beam_b1','beam_b3','beam_prompt','plain_ts','draft_ts','translation','long'],default='beam_b1')
    ap.add_argument('--datasets',nargs='+')
    ap.add_argument('--budget-seconds',type=int,default=2700)
    ap.add_argument('--chunk-size',type=int,default=128)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('model/audio operations require an allocated compute node')
    if args.budget_seconds<1 or args.chunk_size<1:
        ap.error('positive budget and chunk size required')
    if (args.mode=='beam_bound')!=args.case.startswith('beam_'):
        ap.error('beam_bound mode requires a beam case; other modes require single-row cases')
    args.out.mkdir(parents=True,exist_ok=True)
    args.publish.mkdir(parents=True,exist_ok=True)
    import soundfile as sf
    import torch
    from eval_wer import CER_LANGS, characters, edit_distance, normalize
    from profile_beam import command, result_record

    from owsm_trt import OWSM, hub, megakernel, megakernel_ring
    from owsm_trt.audio import load_audio
    from owsm_trt.build import engine_tag

    rows = select_rows(json.loads((args.data/'index.json').read_text()),args.case,args.datasets)
    if not rows:
        raise RuntimeError('empty selected corpus')
    files = hub.resolve(args.model)
    config = dict(schema=1,commit=os.environ.get('OWSM_PROFILE_COMMIT'),mode=args.mode,case=args.case,
        index_sha256=sha256(args.data/'index.json'),checkpoint_sha256=sha256(files.checkpoint),
        selected_keys_sha256=hashlib.sha256('\n'.join(r['key'] for r in rows).encode()).hexdigest(),
        chunk_size=args.chunk_size,selected_samples=len(rows),datasets=args.datasets,
        scope='Full output/quality regression, one pass per variant. Includes warmup/capture/I/O; no steady-state latency or speedup claim. Long-form has joined text/segments only. No human timestamp or reference translation annotation.')
    config_path = args.publish/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==config,'resume config/source/input changed'
    else:
        (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        publish_file(args.out/'config.json',config_path)
    config_digest = sha256(config_path)
    # Each published chunk is a complete compressed JSONL with all variants; incomplete chunks remain local.
    completed = load_completed(args.publish,rows,args.chunk_size,config_digest)
    stop = [False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    started = time.monotonic()
    with torch.inference_mode():
        model = OWSM.from_pretrained(args.model,decoder_kernel='ring:split' if args.mode=='ring_split' else 'auto')
        assert isinstance(model.backend,megakernel.MegaBackend),'required backend unavailable'
        current = model.backend
        if args.mode=='ring_split':
            assert current.kernel.single_cross_splits in (4,8),'public split did not activate'
            baseline = megakernel.MegaBackend(current.inner,megakernel_ring.RingDecoder(current.kernel.m),
                current.max_tokens,current.max_rows,rows_kernel=current.rows_kernel,trial=True)
        elif args.mode=='serial64':
            assert (current.kernel.m.D,current.kernel.P_solo)==(384,64)
            baseline = megakernel.MegaBackend(current.inner,megakernel_ring.RingDecoder(current.kernel.m,solo_programs=24),
                current.max_tokens,current.max_rows,rows_kernel=current.rows_kernel,trial=True)
        else:
            baseline = current
        backends = {'reference':baseline,'candidate':current}
        pools = {v:OrderedDict() for v in backends}
        engines = hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
        env = dict(engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,cuda=torch.version.cuda,job=os.environ.get('PBS_JOBID'),
            nvidia_smi=command(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv']))
        if (args.publish/'environment.json').exists():
            previous = json.loads((args.publish/'environment.json').read_text())
            assert previous['engine_sha256']==env['engine_sha256']
            assert previous['torch']==env['torch'] and previous['cuda']==env['cuda']
        (args.out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
        publish_file(args.out/'environment.json',args.publish/'environment.json')
        archives = {}
        with ExitStack() as stack:
            def wave(row):
                if row['shard'] not in archives:
                    archives[row['shard']] = stack.enter_context(tarfile.open(args.data/row['shard']))
                raw = archives[row['shard']].extractfile(row['audio_member']).read()
                assert hashlib.sha256(raw).hexdigest()==row['audio_sha256']
                data,sr = sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
                mono = data[:,0] if data.shape[1]==1 else data.mean(axis=1)
                return load_audio(torch.from_numpy(mono),sr=sr)
            for offset in range(0,len(rows),args.chunk_size):
                chunk_id = offset//args.chunk_size
                name = f'chunk-{chunk_id:05d}.jsonl.gz'
                if name in completed:
                    continue
                if stop[0] or time.monotonic()-started>args.budget_seconds:
                    break
                subset = rows[offset:offset+args.chunk_size]
                records = []
                # Keep a chunk one language at a time for the same primer; B3 intentionally mixes lengths.
                for begin in range(0,len(subset),3 if args.case=='beam_b3' else 1):
                    if stop[0]:
                        break
                    group = subset[begin:begin+(3 if args.case=='beam_b3' else 1)]
                    # Corpus boundaries in a chunk cannot share a language primer.
                    groups = [group] if len({r['lang'] for r in group})==1 else [[r] for r in group]
                    for batch in groups:
                        waves = [wave(r) for r in batch]
                        outputs = {}
                        for mode in (list(backends) if chunk_id%2==0 else list(backends)[::-1]):
                            model.backend,model._sessions = backends[mode],pools[mode]
                            model.beam_upper_bound = mode=='candidate' if args.mode=='beam_bound' else True
                            model.ctc_draft = args.case in ('draft_ts','long')
                            kw = dict(lang=batch[0]['lang'],beam_size=5 if args.mode=='beam_bound' else 1,
                                      batch_size=3 if args.case=='beam_b3' else 1,
                                      task='st_eng' if args.case=='translation' else 'asr')
                            if args.case=='beam_prompt':
                                kw['prompt'] = 'Previous speech context. '*20
                            joined = args.case=='long' or any(r['seconds']>model.cfg.speech_seconds for r in batch)
                            def decode():
                                values = (model.transcribe_long(waves,**kw) if joined
                                          else model.transcribe(waves,timestamps=True,**kw))
                                assert not model.backend.failed,f'backend fallback: {mode}'
                                assert len(values)==len(batch),'output count mismatch'
                                return values
                            values = capture_failure(decode,dict(config=config,variant=mode,options=kw,
                                joined=joined,inputs=batch,completed_variant_outputs=outputs),args.out,args.publish)
                            results = []
                            for row,value in zip(batch,values):
                                record = result_record(model,value,row,row['seconds'])
                                record['token_ids_available'] = not joined
                                record['repetition'] = repetition(record['tokens'])
                                if args.case!='translation':
                                    units = characters if row['lang'] in CER_LANGS else normalize
                                    ref,hyp = units(row['text']),units(record['text'])
                                    record['quality'] = dict(metric='CER' if row['lang'] in CER_LANGS else 'WER',
                                        errors=edit_distance(ref,hyp),reference_units=len(ref))
                                else:
                                    record['quality'] = None
                                results.append(record)
                            outputs[mode] = results
                        for index,row in enumerate(batch):
                            ref,actual = outputs['reference'][index],outputs['candidate'][index]
                            records.append(dict(key=row['key'],dataset=row['dataset'],lang=row['lang'],speaker=row['speaker'],
                                duration_bucket=row['duration_bucket'],source=dict(shard=row['shard'],member=row['audio_member'],
                                audio_sha256=row['audio_sha256'],reference_text=row['text']),
                                config_sha256=config_digest,reference=ref,candidate=actual,
                                differences=record_differences(ref,actual)))
                if len(records)!=len(subset):
                    assert stop[0], 'partial chunk without interruption'
                    break  # Complete published chunks remain valid; retry this chunk after termination.
                path = args.out/name
                with gzip.open(path,'wt',encoding='utf-8') as f:
                    for record in records:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n')
                publish_file(path,args.publish/name)
                completed[name] = dict(samples=len(records),sha256=sha256(path),differences=sum(bool(r['differences']) for r in records))
                status = dict(expected=len(rows),completed=sum(x['samples'] for x in completed.values()),
                    differences=sum(x['differences'] for x in completed.values()),chunks=completed,
                    elapsed_seconds=time.monotonic()-started,job=os.environ.get('PBS_JOBID'),complete=False)
                (args.out/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
                publish_file(args.out/'progress.json',args.publish/'progress.json')
                print(f"{args.mode}/{args.case}: {status['completed']}/{len(rows)} outputs, {status['differences']} differing",flush=True)
                gc.collect()
    status = dict(expected=len(rows),completed=sum(x['samples'] for x in completed.values()),
        differences=sum(x['differences'] for x in completed.values()),chunks=completed,
        complete=sum(x['samples'] for x in completed.values())==len(rows),
        classification='N' if args.mode=='ring_split' else 'E',interrupted=stop[0],elapsed_seconds=time.monotonic()-started)
    (args.out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    publish_file(args.out/'status.json',args.publish/'status.json')
    print(json.dumps({k:v for k,v in status.items() if k!='chunks'}),flush=True)
    if status['classification']=='E' and status['differences']:
        raise SystemExit(3)


if __name__=='__main__':
    main()
