"""Validate isolated retention policies and report memory without claiming a timing gain."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def check_memory(value):
    assert all(type(n) is int and n >= 0 for n in value.values())
    assert value['allocated_bytes'] <= value['reserved_bytes']
    assert value['allocated_bytes'] <= value['peak_allocated_bytes']
    assert value['reserved_bytes'] <= value['peak_reserved_bytes']
    assert 0 < value['device_used_bytes'] <= value['device_total_bytes']


def summarize(raw: Path, control: Path):
    scheduler = read(raw/'pbs-final.json')
    assert scheduler['state'] == 'F' and scheduler['exit_status'] == 0
    assert (raw/'exit_status.txt').read_text().strip() == '0'
    source = read(raw/'source.json')
    baseline_env = read(control/'environment.json')
    expected_complete = dict(complete=True, languages=['eng', 'jpn'], samples=24, repeat=3)
    environments, selections, policies, outputs = {}, {}, {}, {}
    for limit, variant in [(2, 'reference'), (3, 'candidate')]:
        profile = raw/f'isolated-{limit}'
        env = read(profile/'environment.json')
        environments[variant] = env
        assert env['commit'] == source['commit'] and env['job'] == scheduler['job']
        assert env['session_limits'] == {variant: limit} and env['repeat'] == 3
        assert env['memory_scope'] == 'standalone policy in this process'
        assert read(profile/'complete.json') == expected_complete
        for field in ('checkpoint_sha256', 'index_sha256', 'engine_sha256', 'gpu', 'cuda', 'versions'):
            assert env[field] == baseline_env[field], field
        selected_path = profile/'selection.json'
        assert hashlib.sha256(selected_path.read_bytes()).hexdigest() == env['selection_sha256']
        selections[variant] = read(selected_path)
        assert selections[variant] == read(control/'selection.json')
        policies[variant], outputs[variant] = {}, {}
        for lang, rows in selections[variant].items():
            folder = profile/lang
            summary, samples, tokens = [read(folder/name) for name in ('summary.json', 'wall.json', 'tokens.json')]
            assert summary['complete'] and summary['input_count'] == len(rows) == 12
            assert set(tokens) == {row['key'] for row in rows}
            assert tokens == read(control/lang/'tokens.json'), (variant, lang, 'paired output control')
            outputs[variant][lang] = tokens
            for name in ('baseline_memory', 'final_memory'):
                check_memory(summary[name])
            assert len(summary['cold']) == 2
            for entry in summary['cold']:
                assert entry['variant'] == variant
                check_memory(entry['memory'])
            assert len(summary['diagnostics']) == 1
            diagnostic = summary['diagnostics'][0]
            assert diagnostic['variant'] == variant
            assert [(v['batch'], v['beam']) for v in diagnostic['created']] == (
                [(1, 5), (3, 5), (2, 5)] * 2 if limit == 2 else [])
            assert len(diagnostic['retained_captures']) == limit
            expected = {(repeat, offset) for repeat in range(3) for offset in (0, 6)}
            observed = set()
            for sample in samples:
                key = sample['repeat'], sample['offset']
                assert sample['variant'] == variant and key in expected and key not in observed
                observed.add(key)
                assert sample['ids'] == [r['key'] for r in rows[key[1]:key[1]+6]]
                check_memory(sample['memory'])
                assert sample['peak_allocated_bytes'] == sample['memory']['peak_allocated_bytes']
            assert observed == expected
            policies[variant][lang] = dict(
                baseline=summary['baseline_memory'], final=summary['final_memory'],
                cold=summary['cold'], cycle_memory=[s['memory'] for s in samples],
                peak_allocated_bytes=max(s['memory']['peak_allocated_bytes'] for s in samples),
                peak_reserved_bytes=max(s['memory']['peak_reserved_bytes'] for s in samples),
                allocated_range_bytes=[min(s['memory']['allocated_bytes'] for s in samples),
                                       max(s['memory']['allocated_bytes'] for s in samples)],
                diagnostics=summary['diagnostics'], quality=summary['quality'])
    assert outputs['reference'] == outputs['candidate']
    delta = {}
    for lang in selections['reference']:
        a, b = policies['reference'][lang], policies['candidate'][lang]
        delta[lang] = dict(
            final_bytes={k: b['final'][k]-a['final'][k] for k in a['final']},
            peak_allocated_bytes=b['peak_allocated_bytes']-a['peak_allocated_bytes'],
            peak_reserved_bytes=b['peak_reserved_bytes']-a['peak_reserved_bytes'])
    return dict(source=source, scheduler=scheduler, environments=environments, policies=policies,
                candidate_minus_reference=delta, output_control=str(control), exact_output_inputs=24,
                scope='Separate fresh processes per policy, same allocation and engine/checkpoint. '
                      'Resident/peak PyTorch allocator bytes and sampled whole-device used bytes are distinct. '
                      'Device samples include context and library allocations; they are not process peaks. '
                      'Each six-input cycle is two requests; sequential policy runs provide no paired timing claim. '
                      'These samples do not bound memory for other models, beams, prompts or workloads.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('raw', type=Path)
    ap.add_argument('--control', type=Path, required=True, help='Validated paired retention profile directory')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.raw, args.control)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result['candidate_minus_reference'], indent=2))


if __name__ == '__main__':
    main()
