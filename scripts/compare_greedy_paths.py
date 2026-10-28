"""Compare saved plain and CTC-drafted greedy outputs; no model/audio operations."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from audit_large_regression import FIELDS, record_differences
from compare_regression_batches import boundary_displacements, distribution, runtime_tree


def compare_paths(plain: Path,drafted: Path) -> dict:
    roots=[plain,drafted]
    configs=[json.loads((r/'config.json').read_text()) for r in roots]
    assert [c['case'] for c in configs]==['plain_ts','draft_ts']
    assert [c['mode'] for c in configs]==['serial64','greedy_kernels']
    for field in ('checkpoint_sha256','index_sha256','selected_keys_sha256','selected_samples'):
        assert configs[0][field]==configs[1][field],field
    environments=[json.loads((r/'environment.json').read_text()) for r in roots]
    for field in ('engine_sha256','torch','cuda','gpu'):
        assert environments[0][field]==environments[1][field],field
    assert environments[0]['decoder_settings']['candidate']==environments[1]['decoder_settings']['candidate']
    trees=[runtime_tree(c['commit']) for c in configs]
    assert trees[0]==trees[1]
    statuses=[json.loads((r/'status.json').read_text()) for r in roots]
    assert all(s['complete'] and s['completed']==configs[0]['selected_samples'] and not s['differences']
               for s in statuses)
    assert statuses[0]['chunks'].keys()==statuses[1]['chunks'].keys()
    digests=[hashlib.sha256((r/'config.json').read_bytes()).hexdigest() for r in roots]
    counts={f:dict(equal=0,total=0) for f in FIELDS}
    quality=defaultdict(lambda:dict(samples=0,reference_units=0,plain_errors=0,drafted_errors=0))
    changed=[]
    keys=[]
    boundaries=[]
    comparable=0
    for name in sorted(statuses[0]['chunks']):
        rows=[]
        for root,status in zip(roots,statuses):
            path=root/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==status['chunks'][name]['sha256']
            with gzip.open(path,'rt') as stream:
                batch=[json.loads(line) for line in stream]
            assert len(batch)==status['chunks'][name]['samples']
            rows.append(batch)
        assert [r['key'] for r in rows[0]]==[r['key'] for r in rows[1]]
        for a,b in zip(*rows):
            assert a['source']==b['source'] and a['input_group']==b['input_group'] and a['api']==b['api']
            assert a['dataset']==b['dataset'] and a['lang']==b['lang']
            for row,digest in zip((a,b),digests):
                assert row['config_sha256']==digest
                assert row['differences']==record_differences(row['reference'],row['candidate'])==[]
            keys.append(a['key'])
            x,y=a['candidate'],b['candidate']
            assert x['token_ids_available']==y['token_ids_available']
            fields=[]
            for field in FIELDS:
                if field in ('tokens','timestamp_tokens') and not x['token_ids_available']:
                    continue
                counts[field]['total']+=1
                counts[field]['equal']+=x[field]==y[field]
                if x[field]!=y[field]:
                    fields.append(field)
            if fields:
                changed.append(dict(key=a['key'],language=a['lang'],dataset=a['dataset'],chunk=name,
                    fields=fields,input_seconds=x['input_seconds'],source=a['source'],
                    plain=x,drafted=y))
            if a['dataset']!='stress':
                qa,qb=x['quality'],y['quality']
                assert qa['metric']==qb['metric'] and qa['reference_units']==qb['reference_units']
                q=quality[a['lang']]
                q['metric']=qa['metric']
                q['samples']+=1
                q['reference_units']+=qa['reference_units']
                q['plain_errors']+=qa['errors']
                q['drafted_errors']+=qb['errors']
            delta=boundary_displacements(x,y)
            if delta is not None:
                comparable+=1
                boundaries.extend(delta)
    assert len(keys)==len(set(keys))==configs[0]['selected_samples']
    assert hashlib.sha256('\n'.join(keys).encode()).hexdigest()==configs[0]['selected_keys_sha256']
    for q in quality.values():
        q['error_delta']=q['drafted_errors']-q['plain_errors']
        for label in ('plain','drafted'):
            q[label+'_rate']=q[label+'_errors']/q['reference_units'] if q['reference_units'] else None
    return dict(complete=True,samples=len(keys),configs=configs,environments=environments,runtime_tree=trees[0],
        compared_variant='candidate in both audits; each within-audit E control must match first',
        classification='N: plain versus CTC-drafted execution, including verification window shapes. Not an isolated search-kernel ablation.',
        agreement=counts,different_inputs=len(changed),changed_inputs=changed,quality=dict(quality),
        boundary_displacements=dict(comparable_inputs=comparable,excluded_inputs=len(keys)-comparable,
            all_boundaries=distribution(boundaries),changed_boundaries=distribution([x for x in boundaries if x>0])),
        scope='Saved-output comparison, no timing claim. Boundary displacement is not human-reference timestamp error. Nonspeech excluded from WER/CER; differing nonspeech outputs remain in changed_inputs.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plain',type=Path)
    ap.add_argument('drafted',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=compare_paths(args.plain,args.drafted)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('configs','environments','changed_inputs')},indent=2))


if __name__=='__main__':
    main()
