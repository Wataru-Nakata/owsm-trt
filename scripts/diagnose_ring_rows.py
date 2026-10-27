"""Compute-only numerical isolation of existing 102M Ring row routes; no speed claims."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.megakernel_ring import RingDecoder
from owsm_trt.pipeline import _load_decoder


def compare(actual, reference):
    actual, reference = actual.float().cpu(), reference.float().cpu()
    difference = (actual-reference).abs()
    scale = reference.abs().max().item()
    relative = difference.max().item()/max(scale, 1e-30)
    agreement = (actual.argmax(-1) == reference.argmax(-1)).float().mean().item()
    return dict(max_abs=difference.max().item(), reference_abs_max=scale,
                max_relative_range=relative, argmax_agreement=agreement,
                existing_check_pass=relative < .02 and agreement >= .5,
                finite=bool(torch.isfinite(actual).all()),
                per_position_relative=(difference.amax(-1)/max(scale, 1e-30)).tolist(),
                sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wide-one-token', action='store_true',
                    help='isolated102M/H200 R10/15/16 S1 numerical diagnosis; no production route change')
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('model/audio work requires a compute allocation')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = OWSM.from_pretrained(args.model, use_cuda_graph=False, ctc_draft=False)
    ring = model.backend.kernel
    m, dev = ring.m, model.device
    row_shapes = ((1, 1), (3, 1), (5, 1), (8, 1), (8, 4))
    real_rows = (3, 5, 8)
    wide = {}
    if args.wide_one_token:
        if ((m.D, m.F, m.L, m.H) != (384, 1536, 6, 6)
                or 'H200' not in torch.cuda.get_device_name(dev)
                or torch.cuda.get_device_properties(dev).multi_processor_count != 132):
            raise RuntimeError('wide numerical diagnosis is limited to102M/H200132SM')
        real_rows = (10, 15, 16)
        row_shapes = tuple((r, 1) for r in real_rows)
        wide = {r: RingDecoder(m, row_programs=r*m.H) for r in real_rows}
    files = hub.resolve(args.model)
    decoder = _load_decoder(model.cfg, files.checkpoint, skip_init=True)
    gpu32 = copy.deepcopy(decoder).to(dev)
    gpu16 = copy.deepcopy(decoder).to(device=dev, dtype=torch.float16)
    engine_dir = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(model.opts)
    report = dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'), checkpoint_sha256=sha256(files.checkpoint),
                  engine_sha256={p.name:sha256(p) for p in engine_dir.glob('*.engine')},
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  sm_count=torch.cuda.get_device_properties(dev).multi_processor_count,
                  versions={n:importlib.metadata.version(n) for n in ('torch', 'triton', 'tensorrt')},
                  nvidia_smi=command(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap,memory.total', '--format=csv']),
                  diagnostic_sha256=sha256(Path(__file__)),
                  ring_programs=dict(single=ring.P, solo=ring.P_solo, rows=ring.P_rows),
                  wide_one_token=args.wide_one_token, wide_programs={r:k.P_rows for r,k in wide.items()},
                  cases={}, note='Numerical diagnosis only. Existing 2%/50% gate is unchanged; failures are retained, '
                  'not accepted as speed measurements. Random KV is out of distribution. TF32 disabled for GPU fp32.')

    def check(name, tokens, offsets, k, v, ck, cv, real=False):
        rows, window = tokens.shape
        selected_ring = wide.get(rows, ring)
        record = dict(rows=rows, window=window, cross_length=ck.shape[-2], real_prefill=real, status='running')
        report['cases'][name] = record
        save(args.out/'diagnostics.json', report)
        cpu = [x.cpu() for x in (tokens, offsets, k, v, ck, cv)]
        with torch.autocast('cuda', enabled=False):
            ref, ref_kv = decoder.step(cpu[0], cpu[1], *[list(x.float()) for x in cpu[2:]])
            gpu = [x.to(dev) for x in cpu]
            results = {}
            for label, dec, dtype in (('torch_gpu_fp32', gpu32, torch.float32),
                                      ('torch_gpu_fp16', gpu16, torch.float16)):
                logits, kv = dec.step(gpu[0], gpu[1], *[list(x.to(dtype)) for x in gpu[2:]])
                results[label] = logits.cpu(), kv.cpu()
            for label, backend in [('ring', selected_ring)] + ([('tensorrt', model.backend.inner)]
                                                      if ck.shape[-2] == model.cfg.enc_frames else []):
                logits = torch.zeros(rows, window, m.V, device=dev)
                kv = torch.zeros(m.L, 2, rows, m.H, window, m.D//m.H, device=dev, dtype=torch.float16)
                inputs = [x.clone() for x in gpu]
                kind = 'decode' if window == 1 else 'verify'
                run = backend.step_fn(*inputs, logits, kv, kind=kind)
                run()
                torch.cuda.synchronize()
                if label == 'ring':
                    selected_ring.check()
                    record['actual_ring_programs'] = run.programs
                    record['actual_ring_cross_splits'] = run.const['NS']
                    record['ring_prefix_preserved'] = all(
                        torch.equal(dest[:, row, :, :int(offset), :], source[:, row, :, :int(offset), :])
                        for dest,source in zip(inputs[2:4],gpu[2:4]) for row,offset in enumerate(offsets))
                results[label] = logits.cpu(), kv.cpu()
                if label == 'ring':
                    first = logits.clone()
                    for dest, source in zip(inputs[2:4], gpu[2:4]):
                        dest.copy_(source)
                    run()
                    record['ring_repeat_exact'] = bool(torch.equal(logits, first))
        record['versus_cpu_fp32'] = {}
        for label, (logits, kv) in results.items():
            values = compare(logits, ref)
            values['kv_layer_max_abs'] = (kv.float()-ref_kv).abs().flatten(1).amax(1).tolist()
            record['versus_cpu_fp32'][label] = values
        record['ring_versus_gpu_fp16'] = compare(results['ring'][0], results['torch_gpu_fp16'][0])
        if 'tensorrt' in results:
            record['ring_versus_tensorrt'] = compare(results['ring'][0], results['tensorrt'][0])
        record['status'] = 'complete'
        save(args.out/'diagnostics.json', report)
        print(name, {key:(round(value['max_relative_range'],5),value['argmax_agreement'])
                     for key,value in record['versus_cpu_fp32'].items()}, flush=True)
        if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as f:
                f.write(f'{name} numerical comparison complete\n')

    for time in (64, model.cfg.enc_frames):
        for rows, window in row_shapes:
            g = torch.Generator().manual_seed(0)
            tokens = torch.randint(0, m.V, (rows, window), generator=g)
            offsets = torch.full((rows,), 5, dtype=torch.long)
            kv = [torch.randn(m.L, rows, m.H, n, m.D//m.H, generator=g).half()
                  for n in (64, 64, time, time)]
            check(f'random_t{time}_r{rows}_s{window}', tokens, offsets, *kv)
    items = json.loads(args.manifest.read_text())[:max(real_rows)]
    if len(items) < max(real_rows):
        raise ValueError('manifest has too few real inputs for the numerical row checks')
    waves = [load_audio(it['path']) for it in items]
    for rows in real_rows:
        session = model._session(rows)
        session.encode(model._features(waves[:rows]))
        primer = torch.tensor([model._primer('eng', 'asr', False, None)]*rows, device=dev)
        logits, kv, capacity = session._prefill(primer, session.cross_k, session.cross_v)
        p = primer.shape[1]
        k, v = session.caches[capacity]
        k[:, :, :, :p].copy_(kv[:, 0])
        v[:, :, :, :p].copy_(kv[:, 1])
        session._rules_start(primer)
        tokens = session._rules_mask(logits).argmax(-1).unsqueeze(1)
        check(f'real_prefill_r{rows}_s1', tokens, torch.full((rows,),p,device=dev),
              k, v, session.cross_k, session.cross_v, real=True)
    save(args.out/'complete.json', dict(cases=len(report['cases']), all_complete=True))


if __name__ == '__main__':
    main()
