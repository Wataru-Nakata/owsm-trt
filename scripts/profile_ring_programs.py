"""Same-engine experiments for ring SM budgets, weight-prefetch distance or barrier polling."""

import argparse
import importlib.metadata
import json
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from profile_beam import command, save, sha256, wall
from profile_runtime import CASES, count_draft_dispatch, run_case

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.pipeline import _check_kernel, _load_decoder


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--sweep", choices=("programs", "prefetch", "poll"), default="programs")
    ap.add_argument("--long-audio")
    ap.add_argument("--split-choose", action="store_true")
    ap.add_argument("--fuse-accept", action="store_true")
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=["single_greedy", "single_ts", "stream_greedy_b1", "long_greedy"],
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.trace = False
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
    if original.P != 32 or original.P_solo != 64 or original.nw != 4:
        raise RuntimeError("experiment targets the 1B default ring32/solo64 shape")
    if torch.cuda.get_device_properties(model.device).multi_processor_count < 96:
        raise RuntimeError("cannot keep96 persistent CTAs resident on this GPU")
    files = hub.resolve(args.model)
    decoder = _load_decoder(model.cfg, files.checkpoint)
    settings = (
        {"reference": (32, 64, 0), "p16": (16, 64, 0), "p48": (48, 64, 0), "s96": (32, 96, 0)}
        if args.sweep == "programs"
        else {"reference": (32, 64, 0), "ahead1": (32, 64, 1), "ahead2": (32, 64, 2)}
        if args.sweep == "prefetch"
        else {"reference": (32, 64, 0), "relaxed_poll": (32, 64, 0)}
    )
    args.variants = list(settings)
    backends, checks = {"reference": base}, {}
    for name, (parallel, solo, ahead) in settings.items():
        kernel = (
            original
            if name == "reference"
            else megakernel_ring.RingDecoder(
                original.m,
                programs=parallel,
                solo_programs=solo,
                ahead=ahead,
                relaxed_poll=name == "relaxed_poll",
            )
        )
        checked = (
            kernel
            if name != "s96"
            else SimpleNamespace(
                m=kernel.m, name=kernel.name, check=kernel.check, step_fn=partial(kernel.step_fn, solo=True)
            )
        )
        _, ms = wall(lambda dec=decoder: _check_kernel(checked, dec, model.device, 1, 8))
        if args.sweep in ("prefetch", "poll"):
            solo_check = SimpleNamespace(
                m=kernel.m, name=kernel.name, check=kernel.check, step_fn=partial(kernel.step_fn, solo=True)
            )
            _, solo_ms = wall(lambda dec=decoder: _check_kernel(solo_check, dec, model.device, 1, 32))
        checks[name] = dict(
            passed=True,
            solo=name == "s96",
            check_ms=ms,
            solo_window32_check_ms=solo_ms if args.sweep in ("prefetch", "poll") else None,
            rows=1,
            window=8,
            criterion="existing fp32 check: max relative error<0.02, argmax>=0.5",
        )
        if name != "reference":
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
                "N; P changes cross-attention split count; solo leaves it unchanged"
                if args.sweep == "programs"
                else "E intent; same program/split count/arithmetic, only existing L2 prefetch distance changes"
                if args.sweep == "prefetch"
                else "E intent; relaxed barrier polls then final acquire; same arithmetic"
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
        m.greedy_split_choose = args.split_choose
        m.greedy_fuse_accept = args.fuse_accept

    summary = {}
    for name in args.cases:
        summary[name] = run_case(
            model, args, name, items, waves, configure=configure, observe=count_draft_dispatch
        )
        save(args.out / "summary.json", summary)
    differences = [
        (c, v)
        for c, modes in summary.items()
        if "not_evaluated" not in modes
        for v, s in modes.items()
        if any(x["equal"] != x["total"] for x in s["agreement"].values())
    ]
    save(args.out / "output_differences.json", differences)
    if args.sweep in ("prefetch", "poll") and differences:
        raise RuntimeError(f"{args.sweep} scheduling changed outputs: {differences}")
    # N program-budget differences are retained for quality analysis; dispatch failures are fatal.


if __name__ == "__main__":
    main()
