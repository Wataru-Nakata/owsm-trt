"""Measure plain greedy host-check intervals with unchanged kernels/engines.

Counts are collected in a separate diagnostic pass after warm wall timing. This script changes
session.chunk only inside the experiment; the library default remains4 until validated.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256, wall
from profile_runtime import CASES, run_case

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.runtime import GreedySession


def count_plain_steps(model, call):
    records, restore = [], []
    for sess in model._sessions.values():
        if not isinstance(sess, GreedySession):
            continue
        original = sess._run

        def run(c, pos, steps, done, session=sess, previous=original):
            checks = 0
            old_check = session._all_done

            def check(tensor):
                nonlocal checks
                checks += 1
                return old_check(tensor)

            session._all_done = check
            try:
                previous(c, pos, steps, done)
            finally:
                session._all_done = old_check
            end = int(session.pos.item())
            rows = session.seq[:, pos : end + 1].tolist()
            first_eos = [row.index(session.eos) if session.eos in row else None for row in rows]
            calls = end - pos
            records.append(
                dict(
                    chunk=session.chunk,
                    rows=session.rows,
                    primer_length=pos,
                    decoder_calls=calls,
                    prefill_selection_calls=1,
                    host_checks=checks,
                    first_eos_selection_index=first_eos,
                    excess_decoder_calls=(
                        calls - max(first_eos) if all(x is not None for x in first_eos) else 0
                    ),
                    note="selection index0 is prefill; decoder_calls exclude prefill; no EOS means length limit",
                )
            )

        sess._run = run
        restore.append((sess, original))
    try:
        call()
    finally:
        for sess, original in restore:
            sess._run = original
    return {"separate_diagnostic_pass": True, "calls": records}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--repeat", type=int, default=11)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=("plain_greedy", "plain_ts", "translation"),
        default=["plain_greedy", "plain_ts", "translation"],
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.repeat < 1 or args.limit is not None and args.limit < 1:
        ap.error("repeat and limit must be positive")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("audio/model work requires a compute allocation")
    args.trace, args.long_audio = False, None
    settings = {"reference": 4, "chunk2": 2, "chunk1": 1}
    args.variants = list(settings)
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model))
    original_session, selected = model._session, [4]

    def session(*a, **kw):
        result = original_session(*a, **kw)
        if isinstance(result, GreedySession):
            result.chunk = selected[0]
        return result

    model._session = session

    def configure(m, name):
        selected[0] = settings[name]

    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    save(
        args.out / "environment.json",
        dict(
            commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            checkpoint_sha256=sha256(files.checkpoint),
            engine_sha256={p.name: sha256(p) for p in engines.glob("*.engine")},
            manifest_sha256=sha256(args.manifest),
            model_load_ms=load_ms,
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            versions={n: importlib.metadata.version(n) for n in ("torch", "triton", "tensorrt")},
            nvidia_smi=command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"]),
            settings=settings,
            classification="E intent; host completion checks only, exact outputs required",
            timing_note="warm synchronized wall samples; diagnostic counts collected afterward; resident audio",
            reference_time_annotations=False,
        ),
    )
    items = json.loads(Path(args.manifest).read_text())[: args.limit]
    waves, disk_ms = wall(lambda: [load_audio(x["path"]) for x in items])
    save(args.out / "disk_load.json", dict(aggregate_ms=disk_ms, included_in_request_samples=False))
    summary = {}
    for name in args.cases:
        CASES[name] = (*CASES[name][:-1], args.limit)  # use all73 English inputs unless limited explicitly
        summary[name] = run_case(
            model, args, name, items, waves, configure=configure, observe=count_plain_steps
        )
        save(args.out / "summary.json", summary)
    differences = [
        (name, v)
        for name, variants in summary.items()
        for v, row in variants.items()
        if any(x["equal"] != x["total"] for x in row["agreement"].values())
    ]
    save(args.out / "output_differences.json", differences)
    if differences:
        raise RuntimeError(f"completion frequency changed output: {differences}")


if __name__ == "__main__":
    main()
