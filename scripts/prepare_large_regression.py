"""Pack the expanded local regression corpus into verified WebDataset shards on a compute node."""

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
from collections import Counter
from pathlib import Path

SOURCES = [
    ('librispeech_test_clean','.cache/testdata/librispeech_test_clean/manifest.json','eng'),
    ('librispeech_dev_other','.cache/multilingual/librispeech_dev_other.json','eng'),
    ('fleurs_en_validation','.cache/multilingual/fleurs_en_us_validation.json','eng'),
    ('jvs_parallel10','.cache/multilingual/jvs_test.json','jpn'),
    ('fleurs_ja_validation','.cache/multilingual/fleurs_ja_jp_validation.json','jpn'),
    ('fleurs_zh_test','.cache/multilingual/fleurs_cmn_hans_cn_test.json','zho'),
    ('fleurs_de_test','.cache/multilingual/fleurs_de_de_test.json','deu'),
    ('fleurs_es_test','.cache/multilingual/fleurs_es_419_test.json','spa'),
    ('fleurs_fr_validation','.cache/multilingual/fleurs_fr_fr_validation.json','fra'),
    ('fleurs_ru_validation','.cache/multilingual/fleurs_ru_ru_validation.json','rus'),
    ('librispeech_long','.cache/longform/test_clean/manifest.json','eng'),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def safe_key(value: str) -> str:
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError(f'unsafe sample key: {value!r}')
    return value


def write_shard(path: Path, rows: list[dict]) -> dict:
    """One writer, original encoded audio bytes plus one metadata member per sample."""
    keys = [safe_key(row['key']) for row in rows]
    if not rows or len(set(keys))!=len(keys):
        raise ValueError('nonempty shard with unique sample keys required')
    temp = path.with_suffix('.partial')
    with tarfile.open(temp,'w') as archive:
        for row in rows:
            audio = Path(row['source_path']).read_bytes()
            row['audio_sha256'] = hashlib.sha256(audio).hexdigest()
            row['audio_member'] = row['key']+Path(row['source_path']).suffix.lower()
            for name,data in ((row['audio_member'],audio),(row['key']+'.json',json.dumps(row,ensure_ascii=False).encode())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info,io.BytesIO(data))
    # Verify every member before publication, not just the archive header/count.
    with tarfile.open(temp) as archive:
        assert len(archive.getmembers())==2*len(rows)
        for row in rows:
            assert hashlib.sha256(archive.extractfile(row['audio_member']).read()).hexdigest()==row['audio_sha256']
            assert json.load(archive.extractfile(row['key']+'.json'))==row
    temp.replace(path)
    return dict(name=path.name,samples=len(rows),sha256=sha256(path),bytes=path.stat().st_size)


def publish_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = target.with_name(target.name+'.partial')
    shutil.copyfile(source,temp)
    assert sha256(source)==sha256(temp)
    os.replace(temp,target)


def inventory(repo: Path, jvs: Path) -> tuple[list[dict],dict]:
    rows,provenance = [],{}
    screening = {
        'librispeech_test_clean':set(),
        'fleurs_ja_validation':{x['id'] for x in json.loads((repo/'.cache/multilingual/fleurs_ja_jp_validation.json').read_text())[:12]},
    }
    for dataset,relative,lang in SOURCES:
        path = repo/relative
        items = json.loads(path.read_text())
        provenance[dataset] = dict(manifest=str(path),sha256=sha256(path),samples=len(items))
        for item in items:
            speaker = item['id'].split('-')[0] if dataset.startswith(('librispeech','jvs')) else None
            rows.append(dict(key=safe_key(dataset+'--'+item['id']),id=item['id'],dataset=dataset,lang=lang,
                source_path=item['path'],text=item['text'],speaker=speaker,
                screening_overlap=item['id'] in screening.get(dataset,set()),
                long_form=dataset=='librispeech_long'))
    # JVS parallel100 contributes many repeated sentences; add different sentences from90 speakers.
    # Speakers91..100/nonpara30 were used by the existing calibration study, so exclude that partition.
    hashes = {}
    for speaker in range(1,91):
        root = jvs/f'jvs{speaker:03d}'/'nonpara30'
        transcript = root/'transcripts_utf8.txt'
        hashes[str(transcript)] = sha256(transcript)
        for line in transcript.read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            name,text = line.split(':',1)
            key = f'jvs_nonpara90--jvs{speaker:03d}-{name}'
            rows.append(dict(key=safe_key(key),id=f'jvs{speaker:03d}-{name}',dataset='jvs_nonpara90',lang='jpn',
                source_path=str(root/'wav24kHz16bit'/f'{name}.wav'),text=text,speaker=f'jvs{speaker:03d}',
                screening_overlap=False,long_form=False))
    provenance['jvs_nonpara90'] = dict(transcript_sha256=hashes,samples=sum(r['dataset']=='jvs_nonpara90' for r in rows))
    assert len({r['key'] for r in rows})==len(rows)
    return rows,provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--jvs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--publish',type=Path,required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/header/I/O work requires an allocated compute node')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.publish/'complete.json').exists():
        manifest = json.loads((args.publish/'complete.json').read_text())
        for record in manifest['files']:
            source = args.publish/record['name']
            assert sha256(source)==record['sha256'],source
            shutil.copyfile(source,args.out/source.name)
        shutil.copyfile(args.publish/'complete.json',args.out/'complete.json')
        print('reused verified corpus',manifest['samples'],flush=True)
        return
    import soundfile as sf

    rows,provenance = inventory(args.repo,args.jvs)
    import numpy as np

    # Known nonspeech probes complement real recordings; materialize only under node-local scratch.
    synthetic = args.out/'synthetic'
    synthetic.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    probes = [(f'silence-{seconds:g}s',np.zeros(round(seconds*16000),dtype=np.float32))
              for seconds in (.1,.5,2,10,30)]
    probes += [(f'noise-{seconds}s-{amplitude}',(rng.standard_normal(seconds*16000)*amplitude).astype(np.float32))
               for seconds in (2,10,30) for amplitude in (.01,.1)]
    probes += [(f'tone-{seconds}s',(.05*np.sin(2*np.pi*440*np.arange(seconds*16000)/16000)).astype(np.float32))
               for seconds in (1,10,30)]
    for name,audio in probes:
        name = name.replace('.','p')
        path = synthetic/(name+'.wav')
        sf.write(path,audio,16000,subtype='PCM_16')
        for lang in ('eng','jpn'):
            key = 'stress--'+lang+'-'+name
            rows.append(dict(key=safe_key(key),id=key,dataset='stress',lang=lang,source_path=str(path),
                text='',speaker=None,screening_overlap=False,long_form=False))
    provenance['stress'] = dict(seed=42,samples=len(probes)*2,
        note='Generated silence, Gaussian noise and440Hz tones, no speech expected; not music or natural conversation coverage.')
    summaries = {}
    for row in rows:
        info = sf.info(row['source_path'])
        row.update(seconds=info.frames/info.samplerate,sample_rate=info.samplerate,samples=info.frames,channels=info.channels)
        row['duration_bucket'] = next((f'<={n}s' for n in (2,5,10,20,30) if row['seconds']<=n),'>30s')
    # Stable dataset/id ordering and corpus-qualified keys; no dependency on filesystem enumeration.
    rows.sort(key=lambda r:(r['dataset'],r['id']))
    files = []
    for start in range(0,len(rows),512):
        subset = rows[start:start+512]
        path = args.out/f'audio-{start//512:04d}.tar'
        for row in subset:
            row['shard'] = path.name
        record = write_shard(path,subset)
        publish_file(path,args.publish/path.name)
        files.append(record)
        print('published',record['name'],record['samples'],flush=True)
    for dataset in sorted({r['dataset'] for r in rows}):
        subset = [r for r in rows if r['dataset']==dataset]
        summaries[dataset] = dict(samples=len(subset),hours=sum(r['seconds'] for r in subset)/3600,
            unique_texts=len({r['text'] for r in subset}),known_speakers=len({r['speaker'] for r in subset if r['speaker']}),
            duration_buckets=dict(Counter(r['duration_bucket'] for r in subset)),
            sample_rates=dict(Counter(r['sample_rate'] for r in subset)),screening_overlap=sum(r['screening_overlap'] for r in subset))
    index = args.out/'index.json'
    index.write_text(json.dumps(rows,ensure_ascii=False)+'\n')
    publish_file(index,args.publish/index.name)
    files.append(dict(name=index.name,sha256=sha256(index),bytes=index.stat().st_size))
    manifest = dict(schema=1,samples=len(rows),files=files,datasets=summaries,provenance=provenance,
        scope='Original encoded bytes, no resampling/segmentation in corpus preparation. Local datasets reused for expanded regression, not claimed to be a never-seen holdout. Long LibriSpeech recordings concatenate test-clean utterances and are not additional independent speech. No human timestamp boundaries or target translation references.')
    path = args.out/'complete.json'
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    publish_file(path,args.publish/path.name)
    shutil.rmtree(synthetic)
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
