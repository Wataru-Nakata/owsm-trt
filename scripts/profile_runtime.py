"""Same-engine frontend ablations across direct, streaming, beam and long-form APIs.

Audio/model work belongs in an allocated compute job. Wall measurements are uninstrumented;
frontend events/traces and cold warmup are collected separately.
"""

import argparse
import gc
import importlib.metadata
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from profile_beam import command, comparison, quality, result_record, save, sha256, stats, wall

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag

VARIANTS = ("reference", "power", "post", "both", "graph", "graph_both")
CASES = {
    "single_greedy": (1, 1, False, True, False, "asr", None, None),
    "single_ts": (1, 1, True, True, False, "asr", None, None),
    "stream_greedy_b1": (1, 1, False, True, True, "asr", None, None),
    "stream_greedy_b8": (8, 1, False, True, True, "asr", None, None),
    "stream_greedy_b16": (16, 1, False, True, True, "asr", None, None),
    "stream_ts_b8": (8, 1, True, True, True, "asr", None, None),
    "plain_greedy": (1, 1, False, False, False, "asr", None, 12),
    "beam_no_ts": (1, 5, False, True, False, "asr", None, None),
    "beam_ts": (1, 5, True, True, False, "asr", None, 12),
    "translation": (1, 1, False, True, False, "st_eng", None, 12),
    "prompt": (1, 1, True, True, False, "asr", "Previous speech context. " * 20, 6),
    "long_greedy": (1, 1, True, True, True, "asr", None, None),
}


def variant(model, name):
    model.frontend.fused_power = name in ("power", "both", "graph_both")
    model.frontend.fused_logmel = name in ("post", "both", "graph_both")
    model.frontend_graph = name in ("graph", "graph_both")


def captures(model):
    return {
        str(key): {
            "encode": len(getattr(s, "_enc_graphs", {})),
            "ctc": len(getattr(s, "_runs", {})),
            "decode": len(getattr(s, "graphs", {})),
            "verify": len(getattr(s, "_verify", {})) + len(getattr(s, "_verify_solo", {})),
        }
        for key, s in model._sessions.items()
    }


def dispatch(model):
    result = {}
    for key, s in model._sessions.items():
        calls = list(getattr(s, "_steps", {}).values())
        calls += [v[1] for v in getattr(s, "_runs", {}).values()]
        calls += [v.step for v in getattr(s, "_verify", {}).values()]
        counts = {}
        for fn in calls:
            name = getattr(fn, "dispatch", fn.__qualname__)
            reason = getattr(fn, "fallback_reason", None)
            tag = f"{name}: {reason}" if reason else name
            counts[tag] = counts.get(tag, 0) + 1
        result[str(key)] = counts
    return result


def run_case(model, args, name, items, waves):
    batch, beam, timestamps, draft, stream, task, prompt, limit = CASES[name]
    model.ctc_draft = draft
    model._sessions.clear()
    gc.collect()
    torch.cuda.empty_cache()
    items, waves = items[:limit], waves[:limit]
    if name == "long_greedy":
        if args.long_audio is None:
            return {"not_evaluated": "no long recording supplied"}
        waves = [load_audio(args.long_audio)]
        items = [{"id": "long_110s", "path": args.long_audio}]
    chunks = [list(range(len(items)))] if stream else [[i] for i in range(len(items))]
    kw = dict(lang=args.lang, task=task, beam_size=beam, batch_size=batch, prompt=prompt)

    def call(ws):
        if name == "long_greedy":
            return model.transcribe_long(ws, lang=args.lang, task=task, beam_size=beam, batch_size=batch)
        return model.transcribe(ws, timestamps=timestamps, **kw)

    path = args.out / name
    cold, records, samples = {}, {v: {} for v in args.variants}, []
    for v in args.variants:
        variant(model, v)
        before_allocated, before_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        _, ms = wall(lambda: [call([waves[i] for i in chunk]) for chunk in chunks])
        cold[v] = {
            "warmup_ms": ms,
            "captures": captures(model),
            "dispatch": dispatch(model),
            "allocated_delta_bytes": torch.cuda.memory_allocated() - before_allocated,
            "reserved_delta_bytes": torch.cuda.memory_reserved() - before_reserved,
            "frontend_graph": model._frontend_replay.info() if model._frontend_replay else None,
        }
    after_warmup = captures(model)
    frontend_captures = len(model._frontend_replay.entries) if model._frontend_replay else 0
    for repeat in range(args.repeat):
        for v in args.variants if repeat % 2 == 0 else args.variants[::-1]:
            variant(model, v)
            for chunk in chunks:
                group = [waves[i] for i in chunk]
                start_calls = model.draft_stats[1]
                torch.cuda.reset_peak_memory_stats()
                results, ms = wall(lambda: call(group))
                current_frontend_captures = (
                    len(model._frontend_replay.entries) if model._frontend_replay else 0
                )
                if captures(model) != after_warmup or current_frontend_captures != frontend_captures:
                    raise RuntimeError(f"new session/capture in steady-state {name}/{v}")
                samples.append(
                    {
                        "variant": v,
                        "repeat": repeat,
                        "ids": [items[i]["id"] for i in chunk],
                        "wall_ms": ms,
                        "audio_seconds": sum(len(w) for w in group) / model.cfg.sample_rate,
                        "draft_decoder_calls": model.draft_stats[1] - start_calls,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
                for i, result in zip(chunk, results):
                    r = result_record(model, result, items[i], len(waves[i]) / model.cfg.sample_rate)
                    r["token_ids_available"] = name != "long_greedy"
                    previous = records[v].get(items[i]["id"])
                    if previous is not None and any(
                        previous[k] != r[k] for k in ("tokens", "text", "segments")
                    ):
                        raise RuntimeError(f"non-deterministic output {name}/{v}/{items[i]['id']}")
                    records[v][items[i]["id"]] = r
    summary = {}
    for v in args.variants:
        rows = [r for r in samples if r["variant"] == v]
        summary[v] = {
            "wall_ms": stats([r["wall_ms"] for r in rows]),
            "rtfx": sum(r["audio_seconds"] for r in rows) * 1000 / sum(r["wall_ms"] for r in rows),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            "draft_decoder_calls": sum(r["draft_decoder_calls"] for r in rows),
            "agreement": comparison(records["reference"], records[v]),
            "quality": quality(items, records[v], args.lang) if task == "asr" else None,
            "token_comparison_scope": (
                "segment/text only; long-form returns no joined token IDs"
                if name == "long_greedy"
                else "returned token IDs"
            ),
        }
    if args.trace and name in ("single_greedy", "stream_greedy_b8", "long_greedy"):
        for v in args.variants:
            variant(model, v)
            group = [waves[i] for i in chunks[0]]
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                call(group)
                torch.cuda.synchronize()
            path.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(path / f"trace_{v}.json"))
            save(
                path / f"operators_{v}.json",
                [
                    dict(
                        name=e.key,
                        calls=e.count,
                        cpu_total_us=e.cpu_time_total,
                        self_cpu_us=e.self_cpu_time_total,
                        device_total_us=e.device_time_total,
                        self_device_us=e.self_device_time_total,
                    )
                    for e in prof.key_averages()
                ],
            )
    save(path / "summary.json", summary)
    save(path / "wall.json", samples)
    save(path / "tokens.json", records)
    save(path / "cold.json", cold)
    print(name, {v: round(s["rtfx"], 2) for v, s in summary.items()}, flush=True)
    if progress := os.environ.get("OWSM_PROFILE_PROGRESS"):
        with Path(progress).open("a") as f:
            f.write(f"{args.lang} {name} complete\n")
    return summary


def frontend_profile(model, args, waves):
    model._sessions.clear()
    result = {}
    for b in (1, min(8, len(waves))):
        ws = waves[:b]
        variant(model, "reference")
        expected = model._features(ws)
        # Dedicated GPU input isolates frontend from padding/pinning/H2D, at the same length.
        n = min(
            model.cfg.num_samples, (expected.shape[1] - 1) * model.cfg.hop_length + model.cfg.n_fft // 2 + 1
        )
        speech = torch.zeros(b, n, device=model.device)
        for row, w in zip(speech, ws):
            row[: min(n, len(w))].copy_(w[:n])
        result[b] = {}
        for v in args.variants:
            variant(model, v)
            actual = model._features(ws)
            error = (actual - expected).abs()
            frontend_call = model._frontend_replay if model.frontend_graph and b == 1 else model.frontend
            frontend_call(speech)
            times, events = [], []
            for _ in range(30):
                _, ms = wall(lambda: model._features(ws))
                times.append(ms)
                a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                frontend_call(speech)
                z.record()
                events.append((a, z))
            torch.cuda.synchronize()
            result[b][v] = {
                "features_wall_ms": stats(times),
                "frontend_event_ms": stats([a.elapsed_time(z) for a, z in events]),
                "features_exact": torch.equal(actual, expected),
                "max_abs_error": float(error.max()),
                "frontend_graph": model._frontend_replay.info() if model._frontend_replay else None,
            }
            if args.trace:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    model._features(ws)
                prof.export_chrome_trace(str(args.out / f"frontend_b{b}_{v}.json"))
    save(args.out / "frontend.json", result)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=["reference", "power", "post", "both"])
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--long-audio")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--lazy-drafts", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if "reference" not in args.variants or args.repeat < 1:
        ap.error("reference and positive repeat count required")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("audio/model work requires a compute allocation")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "runtime.log")]
    )
    items = json.loads(Path(args.manifest).read_text())[: args.limit]
    if not items or len({it["id"] for it in items}) != len(items):
        ap.error("nonempty manifest with unique ids required")
    start = time.perf_counter()
    waves = [load_audio(it["path"]) for it in items]
    disk_load_ms = (time.perf_counter() - start) * 1000
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, lazy_drafts=args.lazy_drafts))
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    prop = torch.cuda.get_device_properties(model.device)
    save(
        args.out / "environment.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "commit": os.environ.get("OWSM_PROFILE_COMMIT"),
            "gpu": prop.name,
            "sm_count": prop.multi_processor_count,
            "cuda": torch.version.cuda,
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "triton", "tensorrt")},
            "checkpoint_sha256": sha256(files.checkpoint),
            "engine_sha256": {p.name: sha256(p) for p in engines.glob("*.engine")},
            "manifest_sha256": sha256(args.manifest),
            "config": asdict(model.cfg),
            "nvidia_smi": command(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]
            ),
            "cold": {"model_load_ms": load_ms, "disk_load_ms": disk_load_ms},
            "reference_time_annotations": False,
            "timing_note": "single cases are per-request latency; stream cases are whole-manifest throughput; cold/events/traces separate",
        },
    )
    frontend_profile(model, args, waves)
    summary = {}
    for name in args.cases:
        summary[name] = run_case(model, args, name, items, waves)
        save(args.out / "summary.json", summary)
    bad = [
        (case, v)
        for case, modes in summary.items()
        if "not_evaluated" not in modes
        for v, s in modes.items()
        if any(x["equal"] != x["total"] for x in s["agreement"].values())
    ]
    save(args.out / "output_differences.json", bad)
    if bad:
        raise RuntimeError(f"frontend variants changed outputs: {bad}")


if __name__ == "__main__":
    main()
