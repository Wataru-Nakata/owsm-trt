"""Profile actual warmed CUDA-graph execution before considering multi-step replay changes."""

import argparse
import bisect
import gzip
import importlib.metadata
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import command, save, sha256, stats, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import captures, dispatch

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


class LabelledGraph:
    def __init__(self, graph, label):
        self.graph, self.label = graph, label

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def replay(self):
        with torch.profiler.record_function(self.label):
            self.graph.replay()


@contextmanager
def graph_labels(model, wrapper=LabelledGraph):
    """Only label replay calls in the diagnostic pass; retain every captured graph and address."""
    restore = []

    def table_graph(table, key, label, tuple_entry=False):
        original = table[key]
        graph = original[2] if tuple_entry else original
        if graph is not None:
            proxy = wrapper(graph, label)
            table[key] = (original[0], original[1], proxy) if tuple_entry else proxy
            restore.append(lambda table=table, key=key, value=original: table.__setitem__(key, value))

    try:
        for key, sess in model._sessions.items():
            for name in ('graphs', '_tail_graphs', '_chunk_graphs'):
                table = getattr(sess, name, {})
                for shape in list(table):
                    table_graph(table, shape, f'graph/decoder/{key}/{name}/{shape}')
            for shape in list(sess._enc_graphs):
                table_graph(sess._enc_graphs, shape, f'graph/encoder/{key}/{shape[1:]}', True)
            for name in ('_verify', '_verify_solo'):
                for shape, bound in getattr(sess, name, {}).items():
                    for attr in ('graph', 'notime_graph'):
                        original = getattr(bound, attr, None)
                        if original is not None:
                            setattr(bound, attr, wrapper(original, f'graph/draft/{key}/{name}/{shape}/{attr}'))
                            restore.append(lambda bound=bound, attr=attr, value=original: setattr(bound, attr, value))
        if model._frontend_replay:
            for key in list(model._frontend_replay.entries):
                table_graph(model._frontend_replay.entries, key, f'graph/frontend/{key}', True)
        yield
    finally:
        for reset in reversed(restore):
            reset()


def trace_summary(path: Path) -> dict:
    """All times are trace microseconds; correlated GPU sums are not concurrent wall time."""
    events = json.loads(path.read_text())['traceEvents']
    ranges = defaultdict(list)
    for event in events:
        if event.get('ph') == 'X' and event.get('cat') == 'user_annotation' and event.get('name', '').startswith('graph/'):
            ranges[event['tid']].append(event)
    for values in ranges.values():
        values.sort(key=lambda e: e['ts'])
    starts = {tid: [e['ts'] for e in values] for tid, values in ranges.items()}
    launches = defaultdict(set)
    for host in events:
        if host.get('cat') not in ('cuda_runtime', 'cuda_driver') or 'correlation' not in host.get('args', {}):
            continue
        label = 'unattributed'
        if host['tid'] in starts:
            i = bisect.bisect_right(starts[host['tid']], host['ts']) - 1
            if i >= 0:
                outer = ranges[host['tid']][i]
                if host['ts'] + host.get('dur', 0) <= outer['ts'] + outer['dur']:
                    label = outer['name']
        launches[host['args']['correlation']].add(label)
    api_times, gpu_times = defaultdict(list), defaultdict(list)
    graph_kernels = defaultdict(float)
    for event in events:
        if event.get('cat') in ('cuda_runtime', 'cuda_driver') and 'dur' in event:
            api_times[event['name']].append(event['dur'])
        if event.get('cat') != 'kernel' or 'dur' not in event:
            continue
        gpu_times[event['name']].append(event['dur'])
        labels = launches.get(event.get('args', {}).get('correlation'), set())
        label = next(iter(labels)) if len(labels) == 1 else 'unattributed'
        graph_kernels[label] += event['dur']
    total_gpu = sum(sum(v) for v in gpu_times.values())
    return dict(
        cuda_api={name: dict(total_us=sum(values), duration_us=stats(values)) for name, values in api_times.items()},
        gpu_kernels={name: dict(calls=len(values), total_us=sum(values)) for name, values in gpu_times.items()},
        gpu_kernel_sum_us=total_gpu or None,
        ambiguous_launch_correlations=sum(len(labels) > 1 for labels in launches.values()),
        graph_correlated_kernel_sums_us=dict(graph_kernels) if total_gpu else None,
        graph_annotation_calls={name: sum(e['name'] == name for values in ranges.values() for e in values)
                                for name in {e['name'] for values in ranges.values() for e in values}},
        scope='Instrumented real CUDA-graph execution, not eager body timing. API durations include GPU waits '
              'and overlap device work; neither API nor GPU-kernel sums are wall-time savings. '
              'Graph correlation is reported only where trace IDs and enclosing replay ranges agree; '
              'unattributed includes prefill/eager work and missing graph correlation. No inferred graph-node phase split.')


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=21)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    items = json.loads(args.manifest.read_text())
    waves = [load_audio(x['path']) for x in items]
    if len(waves) < 8:
        ap.error('at least eight inputs required')
    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    mixed = [order[round(i * (len(order)-1) / 7)] for i in range(8)]
    mid = order[len(order)//2]
    cases = {
        'beam_ts_b1': ([mid], 1, 5, True, False, 'asr'),
        'beam_ts_b3': ([order[0], mid, order[-1]], 3, 5, True, False, 'asr'),
        'translation_b1': ([mid], 1, 1, False, False, 'st_eng'),
        'translation_b8': (mixed, 8, 1, False, False, 'st_eng'),
        'drafted_ts_b8': (mixed, 8, 1, True, True, 'asr'),
    }
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
    save(args.out/'environment.json', dict(
        commit=os.environ.get('OWSM_PROFILE_COMMIT'), gpu=torch.cuda.get_device_name(),
        versions={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt')},
        cuda=torch.version.cuda, load_ms=load_ms, checkpoint_sha256=sha256(files.checkpoint),
        engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')},
        manifest_sha256=sha256(args.manifest),
        nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap,memory.total', '--format=csv']),
        scope='Representative fixed loaded-input requests. Default kernels and actual captured execution; '
              'warm wall excludes the separate labelled profiler pass. No optimization/speedup comparison.'))
    report = {}
    for name, (indices, batch, beam, timestamps, drafted, task) in cases.items():
        model._sessions.clear()
        model.ctc_draft = drafted
        inputs = [waves[i] for i in indices]

        def call():
            return model.transcribe(inputs, lang=args.lang, batch_size=batch, beam_size=beam,
                                    timestamps=timestamps, task=task)

        expected, cold_ms = wall(call)
        for sess in model._sessions.values():
            for attr in ('_verify', '_verify_solo'):
                for bound in getattr(sess, attr, {}).values():
                    bound._ensure_graph()
        expected, _ = wall(call)
        before = captures(model)
        times = []
        for _ in range(args.repeat):
            result, ms = wall(call)
            assert result == expected
            times.append(ms)
        assert captures(model) == before
        observed = count_decoder_dispatch(model, call)
        path = args.out/name
        path.mkdir(parents=True, exist_ok=True)
        trace = path/'trace.json'
        with graph_labels(model), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            with torch.profiler.record_function('actual_cuda_graph_request'):
                result = call()
                torch.cuda.synchronize()
        assert result == expected
        assert captures(model) == before
        assert not getattr(model.backend, 'failed', set())
        prof.export_chrome_trace(str(trace))
        traced = trace_summary(trace)
        traced['trace_sha256'] = sha256(trace)
        with trace.open('rb') as source, gzip.open(path/'trace.json.gz', 'wb') as target:
            while data := source.read(1024*1024):
                target.write(data)
        trace.unlink()
        save(path/'trace_summary.json', traced)
        report[name] = dict(ids=[items[i]['id'] for i in indices], cold_warmup_ms=cold_ms,
            wall_ms=stats(times), samples_ms=times, actual_decoder_dispatch=observed,
            captures=before, dispatch=dispatch(model), outputs=[asdict(x) for x in expected],
            traced_output_equal=True, trace_summary=traced)
        save(args.out/'summary.json', report)
        progress = os.environ.get('OWSM_PROFILE_PROGRESS')
        if progress:
            with open(progress, 'a') as stream:
                stream.write(f'{args.lang} {name} actual graph trace complete\n')


if __name__ == '__main__':
    main()
