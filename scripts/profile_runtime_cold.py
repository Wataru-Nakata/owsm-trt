"""Session-cold costs with warm engines/JIT; defer draft, verification or plain decoder setup."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import save, sha256, stats, wall
from profile_runtime import captures

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--mode", choices=("drafts", "verify", "plain"), default="drafts")
    ap.add_argument(
        "--reuse-first-only", action="store_true",
        help="Check cross-task reuse on the first cycle; timed fresh requests still require exact outputs",
    )
    ap.add_argument(
        "--cases", nargs="+", choices=("translation", "lid", "asr", "prompt", "stream", "stream8")
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.repeat < 1 or args.limit < 1:
        ap.error("repeat and limit must be positive")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("model/audio work requires a compute allocation")
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, frontend_graph=False))
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    items = json.loads(Path(args.manifest).read_text())[: args.limit]
    if args.cases and "stream8" in args.cases and len(items) <= 8:
        ap.error("stream8 needs at least9 recordings (--limit)")
    waves = [load_audio(it["path"]) for it in items]

    def call(task):
        if task == "lid":
            return model.detect_language(waves[0])
        if task in ("stream", "stream8"):
            result = model.transcribe(waves, batch_size=8 if task == "stream8" else 1)
        else:
            result = [
                model.transcribe(
                    waves[0],
                    task="st_eng" if task == "translation" else "asr",
                    prompt="Previous speech context. " * 20 if task == "prompt" else None,
                    timestamps=task == "prompt",
                )
            ]
        return [asdict(r) for r in result]

    cases = args.cases or ("translation", "lid", "asr", "prompt", "stream")
    option = {
        "drafts": "lazy_drafts",
        "verify": "greedy_lazy_verify_graphs",
        "plain": "greedy_lazy_plain",
    }[args.mode]
    records, expected, first = [], {}, []
    for repeat in range(-1, args.repeat):
        for task in cases:
            for lazy in (False, True) if repeat % 2 == 0 else (True, False):
                model._sessions.clear()
                gc.collect()
                torch.cuda.empty_cache()
                setattr(model, option, lazy)
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                result, ms = wall(lambda: call(task))
                if getattr(model.backend, "failed", set()):
                    raise RuntimeError("decoder trial failed; refusing fallback cold timing")
                row = dict(
                    repeat=repeat,
                    task=task,
                    **{option: lazy},
                    session_cold_ms=ms,
                    allocated_delta_bytes=torch.cuda.memory_allocated() - before,
                    peak_delta_bytes=torch.cuda.max_memory_allocated() - before,
                    captures=captures(model),
                )
                key = task
                if key in expected and result != expected[key]:
                    raise RuntimeError(f"output changed: {task} lazy={lazy}")
                expected[key] = result
                (first if repeat < 0 else records).append(row)
                if args.reuse_first_only and repeat >= 0:
                    continue
                # Reuse the same session across plain, long-prompt and streaming draft requests.
                for next_task in cases:
                    result = call(next_task)
                    if next_task in expected and result != expected[next_task]:
                        raise RuntimeError(f"reuse changed output: {task}->{next_task} lazy={lazy}")
                    expected[next_task] = result
        if progress := os.environ.get("OWSM_PROFILE_PROGRESS"):
            with Path(progress).open("a") as f:
                f.write(f"{args.out.name}: cold {args.mode} cycle {repeat} complete\n")
    summary = {}
    for task in cases:
        summary[task] = {
            str(lazy): {
                name: stats([r[name] for r in records if r["task"] == task and r[option] == lazy])
                for name in ("session_cold_ms", "allocated_delta_bytes", "peak_delta_bytes")
            }
            for lazy in (False, True)
        }
    save(
        args.out / "cold.json",
        dict(
            model_load_ms=load_ms,
            mode=args.mode,
            checkpoint_sha256=sha256(files.checkpoint),
            engine_sha256={p.name: sha256(p) for p in engines.glob("*.engine")},
            gpu=torch.cuda.get_device_name(),
            commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            option=option,
            reuse_first_only=args.reuse_first_only,
            frontend_graph=False,
            first_cold=first,
            samples=records,
            summary=summary,
            outputs=expected,
            all_equal=True,
            note="Session reconstruction with the same resident backend; engine/JIT warm. "
            "First observations and model load separate; not steady-state speedup.",
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
