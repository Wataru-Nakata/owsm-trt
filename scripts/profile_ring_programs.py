"""Same-engine experiments for existing Ring scheduling and memory knobs without changing runtime defaults."""

import argparse
import importlib.metadata
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from profile_beam import command, save, sha256, wall
from profile_runtime import CASES, count_draft_dispatch, run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.pipeline import _check_kernel, _load_decoder
from owsm_trt.runtime import BeamSession, GreedySession

ROW_CASES = {
    "plain_b3": (3, 1, False, False, True, "asr", None, None),
    "plain_ts_b3": (3, 1, True, False, True, "asr", None, None),
    "translation_b8": (8, 1, False, False, True, "st_eng", None, None),
    "beam_ts_b1": (1, 5, True, False, True, "asr", None, None),
    "drafted_ts_b8": (8, 1, True, True, True, "asr", None, None),
}
CASES.update(ROW_CASES)


def ring_rows_backend(base):
    """Use the existing opt-in ring:rr row route, sharing weights but isolating failure state."""
    return megakernel.MegaBackend(base.inner, base.kernel, base.max_tokens, base.max_rows,
        rows_kernel=(base.kernel, megakernel_ring.ROUTE_TOKENS, megakernel_ring.ROUTE_ROWS), trial=True)



def count_decoder_dispatch(model, call):
    """Separate diagnostic pass; read positions only after decode's existing output synchronization."""
    plain = []

    def observed(original, beam):
        def decode(session, primer, *args, **kwargs):
            output = original(session, primer, *args, **kwargs)
            calls = int(session.step) if beam else int(session.pos) - primer.shape[1]
            plain.append(dict(beam=beam, rows=session.rows, primer_length=primer.shape[1],
                              decoder_calls=calls, output_lengths=list(map(len, output))))
            return output
        return decode

    with patch.object(GreedySession, 'decode', observed(GreedySession.decode, False)), \
         patch.object(BeamSession, 'decode', observed(BeamSession.decode, True)):
        drafted = count_draft_dispatch(model, call)
    return dict(plain_batches=plain, drafted_calls=drafted,
                scope='Separate diagnostic pass, not added to warm wall samples; no per-step host reads.')


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--sweep", choices=("programs", "prefetch", "attention", "chunk", "rows"), default="programs")
    ap.add_argument("--long-audio")
    ap.add_argument("--split-choose", action="store_true")
    ap.add_argument("--fuse-accept", action="store_true")
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=None,
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.trace = False
    if args.cases is None:
        args.cases = list(ROW_CASES) if args.sweep == "rows" else [
            "single_greedy", "single_ts", "stream_greedy_b1", "long_greedy"]
    if args.fuse_accept and not args.split_choose:
        ap.error("fused acceptance requires split selection")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("audio/model work requires a compute allocation")
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    base = model.backend
    if not isinstance(base, megakernel.MegaBackend) or not isinstance(
        base.kernel, megakernel_ring.RingDecoder
    ):
        raise RuntimeError("experiment requires existing Hopper ring backend")
    original = base.kernel
    if args.sweep == "rows" and (original.m.D != 384 or original.m.F != 1536 or original.m.L != 6
                                 or base.rows_kernel is None or base.rows_kernel[0] is not None):
        raise RuntimeError("row experiment requires102M and the actual unsupported Gluon row route")
    if args.sweep != "rows" and (original.P != 32 or original.P_solo != 64 or original.nw != 4 or original.ck != 2):
        raise RuntimeError("experiment targets the 1B default ring32/solo64, four-warps, chunk2 shape")
    if torch.cuda.get_device_properties(model.device).multi_processor_count < 96:
        raise RuntimeError("cannot keep96 persistent CTAs resident on this GPU")
    files = hub.resolve(args.model)
    decoder = _load_decoder(model.cfg, files.checkpoint, skip_init=args.sweep == "rows")
    settings = {
        "programs": {
            "reference": {},
            "p16": {"programs": 16},
            "p48": {"programs": 48},
            "s96": {"solo_programs": 96},
        },
        "prefetch": {"reference": {}, "ahead1": {"ahead": 1}, "ahead2": {"ahead": 2}},
        "attention": {"reference": {}, "no_attention_prefetch": {"prefetch_attention": False}},
        "chunk": {"reference": {}, "chunk1": {"chunk": 1}, "chunk4": {"chunk": 4}},
        "rows": {"reference": {}, "ring_rows": {}},
    }[args.sweep]
    exact = args.sweep in ("prefetch", "attention")
    args.variants = list(settings)
    backends, checks = {"reference": base}, {}
    for name, options in settings.items():
        kernel = (
            original
            if name in ("reference", "ring_rows")
            else megakernel_ring.RingDecoder(original.m, **options)
        )
        checked = (
            kernel
            if name != "s96"
            else SimpleNamespace(
                m=kernel.m, name=kernel.name, check=kernel.check, step_fn=partial(kernel.step_fn, solo=True)
            )
        )
        _, ms = wall(lambda dec=decoder: _check_kernel(checked, dec, model.device, 1, 8))
        if args.sweep != "programs":
            solo_check = SimpleNamespace(
                m=kernel.m, name=kernel.name, check=kernel.check, step_fn=partial(kernel.step_fn, solo=True)
            )
            _, solo_ms = wall(lambda dec=decoder: _check_kernel(solo_check, dec, model.device, 1, 32))
        row_checks = []
        if args.sweep == "rows" and name == "ring_rows":
            for rows, window in ((3, 1), (5, 1), (8, 1), (8, 4)):
                _, elapsed = wall(lambda dec=decoder: _check_kernel(kernel, dec, model.device, rows, window))
                row_checks.append(dict(rows=rows, window=window, passed=True, check_ms=elapsed))
        checks[name] = dict(
            additional_row_checks=row_checks,
            passed=True,
            solo=name == "s96",
            check_ms=ms,
            solo_window32_check_ms=solo_ms if args.sweep != "programs" else None,
            rows=1,
            window=8,
            criterion="existing fp32 check: max relative error<0.02, argmax>=0.5",
        )
        if name == "ring_rows":
            backends[name] = ring_rows_backend(base)
        elif name != "reference":
            backends[name] = megakernel.MegaBackend(
                base.inner, kernel, base.max_tokens, base.max_rows, rows_kernel=base.rows_kernel, trial=True
            )
    del decoder
    save(args.out / "shape_checks.json", checks)
    engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    save(
        args.out / "environment.json",
        dict(
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            gpu=torch.cuda.get_device_name(),
            sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
            cuda=torch.version.cuda,
            config=asdict(model.cfg),
            model_load_ms=load_ms,
            versions={n: importlib.metadata.version(n) for n in ("torch", "triton", "tensorrt")},
            checkpoint_sha256=sha256(files.checkpoint),
            manifest_sha256=sha256(args.manifest),
            engine_sha256={p.name: sha256(p) for p in engines.glob("*.engine")},
            nvidia_smi=command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"]),
            settings=settings,
            classification=(
                "E intent; same arithmetic/programs/splits, only prefetch scheduling changes"
                if exact
                else "N; existing ring:rr replaces multi-row TRT calls; arithmetic and near-tie outputs may differ"
                if args.sweep == "rows"
                else "N; program budgets can change attention splits; chunk changes FP32 accumulation grouping"
            ),
            reference_time_annotations=False,
        ),
    )
    items = json.loads(Path(args.manifest).read_text())
    waves, disk_ms = wall(lambda: [load_audio(it["path"]) for it in items])
    save(args.out / "disk_load.json", dict(aggregate_ms=disk_ms, included_in_request_samples=False))

    def configure(m, name):
        m.backend = backends[name]
        m.frontend_graph = True
        if args.sweep != "rows":
            m.greedy_split_choose = args.split_choose
            m.greedy_fuse_accept = args.fuse_accept

    summary = {}
    for name in args.cases:
        summary[name] = run_case(
            model, args, name, items, waves, configure=configure,
            observe=count_decoder_dispatch if args.sweep == "rows" else count_draft_dispatch
        )
        if any(backend.failed for backend in backends.values()):
            raise RuntimeError(f"kernel failed during complete warmup/measurement: {name}; refusing fallback timing")
        save(args.out / "summary.json", summary)
        if path := os.environ.get("OWSM_PROFILE_PROGRESS"):
            with Path(path).open('a') as stream:
                stream.write(f'{args.out.name}: {name} Ring {args.sweep} comparison complete\n')
    differences = [
        (c, v)
        for c, modes in summary.items()
        if "not_evaluated" not in modes
        for v, s in modes.items()
        if any(x["equal"] != x["total"] for x in s["agreement"].values())
    ]
    save(args.out / "output_differences.json", differences)
    if exact and differences:
        raise RuntimeError(f"prefetch scheduling changed outputs: {differences}")
    # N tuning differences are retained for quality analysis; dispatch failures are fatal.


if __name__ == "__main__":
    main()
