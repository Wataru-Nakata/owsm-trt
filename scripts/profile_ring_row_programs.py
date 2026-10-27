"""Same-Ring row-program ablation after102M stage attribution; strict output/cache equality."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256, wall
from profile_ring_programs import ROW_CASES, count_decoder_dispatch
from profile_runtime import run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


def row_budget(rows, policy):
    if not 2 <= rows <= 8:
        raise ValueError("row-program experiment requires2..8 rows")
    if policy == "heads":
        return max(24, rows * 6)
    if policy == "tiles":
        return 32 if rows <= 5 else 48
    raise ValueError(f"unknown row policy: {policy}")


class RowPrograms:
    """Only construction-time routing; each captured call owns one unchanged Ring kernel."""

    def __init__(self, original, policy):
        self.original, self.policy = original, policy
        self.kernels = {p: megakernel_ring.RingDecoder(original.m, row_programs=p)
                        for p in {row_budget(r, policy) for r in range(2, 9)}}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def step_fn(self, tokens, *args, **kwargs):
        return self.kernels[row_budget(tokens.shape[0], self.policy)].step_fn(tokens, *args, **kwargs)


def backend_for(base, policy):
    return megakernel.MegaBackend(base.inner, base.kernel, base.max_tokens, base.max_rows,
        rows_kernel=(RowPrograms(base.kernel, policy), megakernel_ring.ROUTE_TOKENS, megakernel_ring.ROUTE_ROWS),
        trial=True)


@torch.inference_mode()
def check_rows(kernels, out):
    """Compare the proposed scheduling to Ring itself; this is not its known fp32 accuracy gate."""
    m = kernels['reference'].m
    torch.manual_seed(19)
    checked = []
    for rows in range(2, 9):
        for window in (1, 4):
            shape = (m.L, rows, m.H, 64, m.D // m.H)
            k, v = (torch.randn(shape, dtype=torch.float16, device=m.device) * .1 for _ in range(2))
            cross_shape = (m.L, rows, m.H, 374, m.D // m.H)
            cross = [torch.randn(cross_shape, dtype=torch.float16, device=m.device) * .1 for _ in range(2)]
            tok = torch.randint(m.V, (rows, window), device=m.device)
            offset = torch.full((rows,), 31, device=m.device, dtype=torch.long)
            expected, ns = None, None
            for mode, kernel in kernels.items():
                kc, vc = k.clone(), v.clone()
                logits = torch.zeros(rows, window, m.V, device=m.device)
                kv = torch.zeros(m.L, 2, rows, m.H, window, m.D//m.H, device=m.device, dtype=torch.float16)
                pred = torch.zeros(rows, window, device=m.device, dtype=torch.long)
                run = kernel.step_fn(tok, offset, kc, vc, *cross, logits, kv, pred=pred)
                run()
                torch.cuda.synchronize()
                kernel.check()
                values = (logits, kc, vc, kv, pred)
                assert all(bool(torch.isfinite(t).all()) for t in values)
                if expected is None:
                    expected, ns = values, run.const['NS']
                else:
                    assert run.const['NS'] == ns, 'attention reduction order changed'
                    for a, b in zip(values, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                checked.append(dict(rows=rows, window=window, mode=mode, programs=run.programs,
                                    cross_splits=run.const['NS'], finite=True, exact_to_ring=True))
                save(out / 'shape_checks.json', checked)
    return checked


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--lang', default='eng')
    ap.add_argument('--repeat', type=int, default=11)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', choices=ROW_CASES, default=list(ROW_CASES))
    args = ap.parse_args()
    if not os.environ.get('PBS_JOBID'):
        ap.error('model/audio work requires a compute allocation')
    if args.repeat < 1:
        ap.error('repeat must be positive')
    args.trace, args.long_audio = False, None
    args.variants = ['reference', 'heads', 'tiles']
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, decoder_kernel='ring:rr'))
    base = model.backend
    if not isinstance(base, megakernel.MegaBackend) or not isinstance(base.kernel, megakernel_ring.RingDecoder):
        raise RuntimeError('existing Ring:rr route required')
    m = base.kernel.m
    if (m.D, m.F, m.L, m.H, base.kernel.P_rows) != (384, 1536, 6, 6, 24) or base.rows_kernel is not None or model.cfg.enc_frames != 374:
        raise RuntimeError('experiment targets102M default Ring row budget24')
    if torch.cuda.get_device_properties(model.device).multi_processor_count < 48:
        raise RuntimeError('cannot guarantee48 resident persistent CTAs')
    backends = {'reference': base} | {p: backend_for(base, p) for p in ('heads', 'tiles')}
    files = hub.resolve(args.model)
    engine_dir = hub.cache_dir()/files.name.replace('/', '--')/engine_tag(model.opts)
    save(args.out/'environment.json', dict(commit=os.environ.get('OWSM_PROFILE_COMMIT'),
        checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name:sha256(p) for p in engine_dir.glob('*.engine')},
        manifest_sha256=sha256(args.manifest), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','tensorrt')}, load_ms=load_ms,
        nvidia_smi=command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv']),
        budgets={p:{r:row_budget(r,p) for r in range(2,9)} for p in ('heads','tiles')},
        classification='E intent versus same opt-in Ring route; existing Ring-versus-TRT N differences remain. '
                       'All attention split counts, full outputs and cache checks must match. '
                       'Cold/compiler ordering and shared resident variants prevent isolated cold/VRAM gains.'))
    check_rows({'reference':base.kernel} | {p:backends[p].rows_kernel[0] for p in ('heads','tiles')}, args.out)
    items = json.loads(args.manifest.read_text())
    waves, disk_ms = wall(lambda: [load_audio(it['path']) for it in items])
    save(args.out/'disk_load.json', dict(aggregate_ms=disk_ms, included_in_wall=False))
    summary = {}
    def configure(m, mode):
        m.backend = backends[mode]
    for name in args.cases:
        summary[name] = run_case(model, args, name, items, waves, configure=configure, observe=count_decoder_dispatch)
        save(args.out/'summary.json', summary)
        assert not any(b.failed for b in backends.values()), 'runtime fallback invalidates comparison'
        assert all(v['equal']==v['total'] for mode in summary[name].values() for v in mode['agreement'].values()), name
        if path := os.environ.get('OWSM_PROFILE_PROGRESS'):
            with Path(path).open('a') as stream:
                stream.write(f'{args.out.name}: {name} Ring row program comparison complete\n')


if __name__ == '__main__':
    main()
