"""Summarize immutable regression chunks, with language/dataset/length strata and no speed claims."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from audit_large_regression import FIELDS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('result',type=Path)
    args = ap.parse_args()
    config_path = args.result/'config.json'
    config = json.loads(config_path.read_text())
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    totals = defaultdict(lambda:dict(samples=0,agreement={f:dict(equal=0,total=0) for f in FIELDS},
        quality={v:dict(errors=0,reference_units=0) for v in ('reference','candidate')},
        flags={v:defaultdict(int) for v in ('reference','candidate')},different_examples=[]))
    seen = set()
    ordered = []
    for path in sorted(args.result.glob('chunk-*.jsonl.gz')):
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row = json.loads(line)
                assert row['config_sha256']==digest and row['key'] not in seen,path
                seen.add(row['key'])
                ordered.append(row['key'])
                strata = ('all','language:'+row['lang'],'dataset:'+row['dataset'],
                          'duration:'+row['duration_bucket'],row['dataset']+'/'+row['duration_bucket'])
                for stratum in strata:
                    total = totals[stratum]
                    total['samples'] += 1
                    for field in FIELDS:
                        if field in ('tokens','timestamp_tokens') and not row['reference']['token_ids_available']:
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
                            for field in ('errors','reference_units'):
                                total['quality'][mode][field] += record['quality'][field]
                    if row['differences'] and len(total['different_examples'])<20:
                        total['different_examples'].append(dict(key=row['key'],fields=row['differences'],chunk=path.name))
    assert len(seen)<=config['selected_samples']
    complete = len(seen)==config['selected_samples']
    if complete:
        assert hashlib.sha256('\n'.join(ordered).encode()).hexdigest()==config['selected_keys_sha256']
    for key,value in totals.items():
        for mode,quality in value['quality'].items():
            quality['rate'] = (quality['errors']/quality['reference_units']
                if quality['reference_units'] and (key.startswith(('language:','dataset:')) or '/' in key) else None)
            quality['note'] = 'Per-language WER or CER; combined unlike units are not a single accuracy metric.'
    result = dict(config=config,completed=len(seen),complete=complete,strata=dict(totals),
        timestamp_reference='unavailable; agreement/invalidity only, not boundary accuracy',
        repetition='heuristic diagnostics only; repeated text can be legitimate',
        timing='regression pass includes I/O, cold graphs and warm calls; no speedup claimed')
    temp=args.result/'summary.tmp'
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    temp.replace(args.result/'summary.json')
    print(json.dumps(dict(completed=len(seen),expected=config['selected_samples'],complete=complete,
                         agreement=totals['all']['agreement']),ensure_ascii=False))


if __name__=='__main__':
    main()
