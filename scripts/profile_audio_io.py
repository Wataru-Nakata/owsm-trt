"""Disk-inclusive runtime and isolated warmed audio-load ablations on node-local inputs."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch
from profile_beam import save, sha256, stats, wall
from profile_ring_programs import count_decoder_dispatch
from profile_runtime import run_case

from owsm_trt import OWSM, hub, pipeline
from owsm_trt.audio import duration, load_audio, resample
from owsm_trt.build import engine_tag


def legacy_load_audio(audio, sample_rate=16000, sr=None):
    """Preserved pre-optimization file behavior, independent of the current public loader."""
    if not isinstance(audio, (str, os.PathLike)):
        return load_audio(audio, sample_rate, sr)
    import soundfile as sf

    data, rate = sf.read(os.fspath(audio), dtype="float32", always_2d=True)
    return resample(torch.from_numpy(data.mean(axis=1)), rate, sample_rate)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--lang',default='eng')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repeat',type=int,default=11)
    args=ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    if args.repeat<1:
        ap.error('repeat must be positive')
    args.trace,args.long_audio,args.variants=False,None,['reference','mono','loaded']
    items=json.loads(args.manifest.read_text())
    if len(items)<8:
        ap.error('at least8 inputs required')
    paths=[x['path'] for x in items]
    waves,first_load_ms=wall(lambda:[load_audio(path) for path in paths])
    phases={name:[] for name in ('header','reference','mono')}
    for repeat in range(args.repeat):
        start=time.perf_counter()
        lengths=[duration(path,16000) for path in paths]
        phases['header'].append((time.perf_counter()-start)*1000)
        assert all(abs(sec-len(w)/16000)<1e-6 for sec,w in zip(lengths,waves))
        for mode,loader in ([('reference',legacy_load_audio),('mono',load_audio)] if repeat%2==0
                            else [('mono',load_audio),('reference',legacy_load_audio)]):
            start=time.perf_counter()
            actual=[loader(path) for path in paths]
            phases[mode].append((time.perf_counter()-start)*1000)
            for a,b in zip(actual,waves):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
    del actual
    save(args.out/'audio_phases.json',dict(first_load_ms=first_load_ms,warm_ms={k:stats(v) for k,v in phases.items()},
        samples_ms=phases,waveforms_equal=True,
        scope='CPU file header/decode/mono/resample phases with GPU idle, node-local warm filesystem; '
              'not GPU improvement and not additive with overlapping whole-request wall. Equality checks excluded.'))
    model,load_ms=wall(lambda:OWSM.from_pretrained(args.model))
    files=hub.resolve(args.model)
    engines=hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
    save(args.out/'environment.json',dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,cuda=torch.version.cuda,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt','soundfile')},load_ms=load_ms,
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256={p.name:sha256(p) for p in engines.glob('*.engine')},
        manifest_sha256=sha256(args.manifest),input_modes={'reference':'file paths, preserved legacy mean loader','mono':'file paths, public mono view loader','loaded':'resident tensor control'},
        scope='Same engines/checkpoint/default B1 frontend graph and decoder. E intent; independent public mono integration check. '
              'File paths include headers/read/decode in wall; loaded control excludes them. '
              'Warm OS/filesystem cache after staging; not cold disk/network/Lustre or preprocessing speed on other sample rates.'))
    inputs={'reference':paths,'mono':paths,'loaded':waves}
    summary={}
    original=pipeline.load_audio
    def configure(m,mode):
        pipeline.load_audio=legacy_load_audio if mode=='reference' else load_audio
    try:
        for case in ('single_ts','stream_ts_b8'):
            summary[case]=run_case(model,args,case,items,waves,configure=configure,
                observe=count_decoder_dispatch,input_sources=inputs)
            save(args.out/'summary.json',summary)
            assert not getattr(model.backend,'failed',set())
            assert all(v['equal']==v['total'] for mode in summary[case].values() for v in mode['agreement'].values())
    finally:
        pipeline.load_audio=original


if __name__=='__main__':
    main()
