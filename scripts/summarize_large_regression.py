"""Summarize immutable regression chunks, with language/dataset/length strata and no speed claims."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from audit_large_regression import FIELDS, record_differences
from prepare_large_regression import sha256


def summarize(result_dir: Path) -> dict:
    config_path = result_dir/'config.json'
    config = json.loads(config_path.read_text())
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    progress_path = result_dir/'progress.json'
    if progress_path.exists():
        for name,record in json.loads(progress_path.read_text())['chunks'].items():
            assert Path(name).name==name and name.startswith('chunk-'),name
            assert sha256(result_dir/name)==record['sha256'],name
    totals = defaultdict(lambda:dict(samples=0,agreement={f:dict(equal=0,total=0) for f in FIELDS},
        quality={v:dict(errors=0,reference_units=0,metrics=defaultdict(int),
                        excluded_nonspeech_samples=0,excluded_nonspeech_errors=0)
                 for v in ('reference','candidate')},
        flags={v:defaultdict(int) for v in ('reference','candidate')},different_examples=[]))
    seen = set()
    ordered = []
    for path in sorted(result_dir.glob('chunk-*.jsonl.gz')):
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row = json.loads(line)
                assert row['config_sha256']==digest and row['key'] not in seen,path
                assert row['differences']==record_differences(row['reference'],row['candidate']),path
                seen.add(row['key'])
                ordered.append(row['key'])
                strata = ('all','language:'+row['lang'],'dataset:'+row['dataset'],
                          'duration:'+row['duration_bucket'],row['dataset']+'/'+row['duration_bucket'])
                for stratum in strata:
                    total = totals[stratum]
                    total['samples'] += 1
                    for field in FIELDS:
                        if field in ('tokens','timestamp_tokens') and not all(
                            row[v]['token_ids_available'] for v in ('reference','candidate')
                        ):
                            continue
                        total['agreement'][field]['total'] += 1
                        total['agreement'][field]['equal'] += row['reference'][field]==row['candidate'][field]
                    for mode in ('reference','candidate'):
                        record = row[mode]
                        for field in ('empty_segments','invalid_segments','unpaired_timestamp','nonmonotonic_timestamps'):
                            total['flags'][mode][field] += record[field]
                        total['flags'][mode]['empty_text'] += not record['text'].strip()
                        total['flags'][mode]['identical_token_run_ge8'] += record['repetition']['max_identical_token_run']>=8
                        total['flags'][mode]['fourgram_count_ge4'] += record['repetition']['max_fourgram_count']>=4
                        if record['quality']:
                            quality = total['quality'][mode]
                            if row['dataset']=='stress':
                                quality['excluded_nonspeech_samples'] += 1
                                quality['excluded_nonspeech_errors'] += record['quality']['errors']
                            else:
                                quality['metrics'][record['quality']['metric']] += 1
                                for field in ('errors','reference_units'):
                                    quality[field] += record['quality'][field]
                    if row['differences'] and len(total['different_examples'])<20:
                        total['different_examples'].append(dict(key=row['key'],fields=row['differences'],chunk=path.name))
    assert len(seen)<=config['selected_samples']
    complete = len(seen)==config['selected_samples']
    if complete:
        assert hashlib.sha256('\n'.join(ordered).encode()).hexdigest()==config['selected_keys_sha256']
    for key,value in totals.items():
        for mode,quality in value['quality'].items():
            quality['rate'] = (quality['errors']/quality['reference_units']
                if quality['reference_units'] and len(quality['metrics'])==1
                and (key.startswith(('language:','dataset:')) or '/' in key) else None)
            quality['note'] = ('Speech-only WER or CER; generated nonspeech probes are excluded and counted separately. '
                               'Combined unlike units are not a single accuracy metric.')
    return dict(config=config,completed=len(seen),complete=complete,strata=dict(totals),
        timestamp_reference='unavailable; agreement/invalidity only, not boundary accuracy',
        repetition='heuristic diagnostics only; repeated text can be legitimate',
        timing='regression pass includes I/O, cold graphs and warm calls; no speedup claimed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('result',type=Path)
    args = ap.parse_args()
    result = summarize(args.result)
    temp=args.result/'summary.tmp'
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    temp.replace(args.result/'summary.json')
    print(json.dumps(dict(completed=result['completed'],expected=result['config']['selected_samples'],
                         complete=result['complete'],agreement=result['strata']['all']['agreement']),ensure_ascii=False))


if __name__=='__main__':
    main()
