"""Compute-node public frontend stream reuse / bounded batch option regression."""

import argparse
import faulthandler
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import save, sha256, wall

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--long-audio',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('audio/model work requires a compute allocation')
    faulthandler.enable()
    faulthandler.dump_traceback_later(120,repeat=True)
    items=json.loads(args.manifest.read_text())[:8]
    waves=[load_audio(x['path']) for x in items]
    torch.manual_seed(7)
    waves += [torch.zeros(16000),torch.randn(16000)*.01]
    records,expected,engine_hashes={},None,None
    for mode,limit in [('legacy',1),('default',1),('batch8',8)]:
        model,ms=wall(lambda:OWSM.from_pretrained(args.model,frontend_graph_max_batch=limit))
        assert model._frontend_graph_batch_limit==limit and model._frontend_reuse_warmup_stream
        if mode=='legacy':
            model._frontend_reuse_warmup_stream=False
        files=hub.resolve(args.model)
        engines=hub.cache_dir()/files.name.replace('/','--')/engine_tag(model.opts)
        hashes={p.name:sha256(p) for p in engines.glob('*.engine')}
        if engine_hashes is not None:
            assert hashes==engine_hashes
        engine_hashes=hashes
        outputs,caches={},{}
        cases=[
            ('single_ts',waves[0],True,False,dict(timestamps=True)),
            ('draft8',waves,True,False,dict(batch_size=8,timestamps=True)),
            ('plain3',waves,False,False,dict(batch_size=3,timestamps=True)),
            ('beam3',waves,True,False,dict(batch_size=3,beam_size=5,timestamps=True)),
            ('translation8',waves,True,False,dict(batch_size=8,task='st_eng')),
            ('prompt3',waves[:3],True,False,dict(batch_size=3,timestamps=True,prompt='Previous speech context. '*20)),
            ('lookahead3',waves,False,True,dict(batch_size=3,beam_size=5,timestamps=True)),
            ('single_again',waves[1],True,False,dict(timestamps=True)),
        ]
        for name,audio,draft,lookahead,kwargs in cases:
            if model.ctc_draft!=draft or model._encoder_lookahead!=lookahead:
                model._sessions.clear()
            model.ctc_draft,model._encoder_lookahead=draft,lookahead
            value=model.transcribe(audio,**kwargs)
            outputs[name]=[asdict(x) for x in value] if isinstance(value,list) else asdict(value)
            replay=model._frontend_replay
            assert replay is not None and replay.max_batch==limit
            assert replay.reuse_warmup_stream==(mode!='legacy')
            caches[name]=replay.info()
            assert all(key[0][0]<=limit for key in replay.entries)
        value=model.transcribe_long(args.long_audio)
        outputs['long']=asdict(value)
        assert not getattr(model.backend,'failed',set())
        if expected is None:
            expected=outputs
        else:
            assert outputs==expected,mode
        records[mode]=dict(load_ms=ms,outputs=outputs,caches=caches)
        save(args.out/'partial.json',records)
        torch.cuda.synchronize()
        replay=None
        model._sessions.clear()
        model._frontend_replay=None
        gc.collect()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(mode,'public frontend checks and cleanup complete',flush=True)
    save(args.out/'complete.json',dict(all_equal=True,cases=len(expected),modes=records,
        checkpoint_sha256=sha256(files.checkpoint),engine_sha256=engine_hashes,
        commit=os.environ.get('OWSM_PROFILE_COMMIT'),gpu=torch.cuda.get_device_name(),
        long_token_scope='long-form compares text/segments; joined token IDs unavailable'))


if __name__=='__main__':
    main()
