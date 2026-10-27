"""Same-engine four-step replay ablation; separate lightweight CPU enqueue diagnostics from wall."""

import argparse
import importlib.metadata
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from graph_chunk_experiment import prepare
from profile_beam import command, save, sha256, stats, wall
from profile_graph_launches import LabelledGraph, graph_labels
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag

CASES.update({
    'chunk_beam_ts_b3': (3, 5, True, False, True, 'asr', None, None),
    'chunk_translation_b8': (8, 1, False, False, True, 'st_eng', None, None),
})


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace, args.long_audio, args.variants = False, None, ['reference', 'chunk4']
    items = json.loads(args.manifest.read_text())
    if len(items) < 9:
        ap.error('at least nine inputs required for partial batch coverage')
    waves = [load_audio(x['path']) for x in items]
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    original_session = model._session
    model._chunk_mode = False

    def session(batch, beam=1):
        result = original_session(batch, beam)
        if model._chunk_mode:
            prepare(result)
        return result

    model._session = session
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace('/', '--') / engine_tag(model.opts)
    save(args.out/'environment.json', dict(
        commit=os.environ.get('OWSM_PROFILE_COMMIT'), gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
        cuda=torch.version.cuda, versions={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt')},
        checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name: sha256(p) for p in engines.glob('*.engine')},
        manifest_sha256=sha256(args.manifest), load_ms=load_ms,
        nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap,memory.total', '--format=csv']),
        scope='E-intent profiler-only four-step graph; existing host checks and bucket boundaries retained. '
              'Original single-step graphs retained; default B1 beam is an unchanged chunk1 control. '
              'No CUPTI profiler during paired wall timing or the separate lightweight CPU enqueue diagnostic. '
              'Loaded inputs, unchanged engines/weights/kernels, both session pools resident.'))

    def configure(m, mode):
        m._chunk_mode = mode == 'chunk4'

    def observe(m, call):
        durations = defaultdict(list)

        class TimedGraph(LabelledGraph):
            def replay(self):
                start = time.perf_counter_ns()
                self.graph.replay()
                durations[self.label].append((time.perf_counter_ns() - start)/1000)

        with graph_labels(m, wrapper=TimedGraph):
            result = count_decoder_dispatch(m, call)
        result['cpu_enqueue'] = {name: dict(total_us=sum(values), duration_us=stats(values), samples_us=values)
                                 for name, values in durations.items()}
        result['chunk_graphs'] = {str(k): getattr(s, '_chunk_graph_info', None) for k, s in m._sessions.items()}
        result['cpu_enqueue_scope'] = ('Separate perf_counter diagnostic around actual replay; includes Python timer overhead '
                                       'and may include driver waits. Not additive with device/wall time. No CUPTI.')
        return result

    summary = {}
    for case in ('beam_ts', 'chunk_beam_ts_b3', 'translation', 'chunk_translation_b8'):
        summary[case] = run_case(model, args, case, items, waves, configure=configure, observe=observe)
        save(args.out/'summary.json', summary)
        assert all(v['equal'] == v['total'] for mode in summary[case].values() for v in mode['agreement'].values())
        assert not getattr(model.backend, 'failed', set())
        if case != 'beam_ts':
            assert any(getattr(s, '_chunk_graphs', {}) for s in model._sessions.values())


if __name__ == '__main__':
    main()
