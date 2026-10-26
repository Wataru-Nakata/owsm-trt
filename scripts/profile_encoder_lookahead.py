"""Fixed-batch encoder lookahead ablation; compute allocation only, unchanged decoder shapes."""

import argparse
import gc
import importlib.metadata
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from profile_beam import command, comparison, quality, result_record, save, sha256, stats, wall

from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag

CASES = {
    "beam_b1": dict(beam_size=5, batch_size=1, timestamps=True),
    "beam_b3": dict(beam_size=5, batch_size=3, timestamps=True),
    "translation_b1": dict(beam_size=1, batch_size=1, task="st_eng"),
    "translation_b8": dict(beam_size=1, batch_size=8, task="st_eng"),
}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--limit", type=int, default=73)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        ap.error("model/audio/profiling work requires an allocated PBS compute job")
    if args.repeat < 1 or args.limit < 1:
        ap.error("repeat and limit must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.manifest.read_text())[:args.limit]
    if len({x["id"] for x in items}) != len(items):
        ap.error("manifest IDs must be unique")
    waves, disk_ms = wall(lambda: [load_audio(x["path"]) for x in items])
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, ctc_draft=False))
    files = hub.resolve(args.model)
    engines = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    save(args.out / "environment.json", dict(
        commit=os.environ.get("OWSM_PROFILE_COMMIT"), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda,
        versions={n: importlib.metadata.version(n) for n in ("triton", "tensorrt")},
        sm_count=torch.cuda.get_device_properties(model.device).multi_processor_count,
        nvidia_smi=command(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]),
        args=vars(args) | {"out": str(args.out), "manifest": str(args.manifest)},
        checkpoint_sha256=sha256(files.checkpoint), engine_sha256={p.name: sha256(p) for p in engines.glob("*.engine")},
        disk_ms=disk_ms, load_ms=load_ms,
        note="Whole-list throughput, identical sorted fixed batches. CUDA synchronization around wall samples; "
             "cold warmup/capture separate. Shared process allocator peak is not isolated variant VRAM.",
    ))
    all_summaries = {}
    for name in args.cases:
        kw = CASES[name] | {"lang": args.lang}
        if len(waves) <= kw["batch_size"]:
            raise ValueError(f"{name} requires more than one batch")
        out = args.out / name
        out.mkdir(exist_ok=True)
        caches = {mode: OrderedDict() for mode in ("reference", "lookahead")}
        records, cold, samples = {}, {}, []
        torch.cuda.reset_peak_memory_stats()

        def call(mode):
            model._sessions = caches[mode]
            model._encoder_lookahead = mode == "lookahead"
            return model.transcribe(waves, **kw)

        def record(results):
            return {it["id"]: result_record(model, value, it, len(wave) / model.cfg.sample_rate)
                    for it, wave, value in zip(items, waves, results)}

        for mode in caches:
            first, cold[mode] = wall(lambda: call(mode))
            records[mode] = record(first)
            warm, _ = wall(lambda: call(mode))
            assert record(warm) == records[mode], "warmup changed output"
        save(out / "tokens.json", records)
        agreement = comparison(records["reference"], records["lookahead"])
        save(out / "agreement.json", agreement)
        if not all(v["equal"] == v["total"] for v in agreement.values()):
            raise RuntimeError("lookahead changed output before timing; inspect saved full tokens")
        for repeat in range(args.repeat):
            for mode in (tuple(caches) if repeat % 2 == 0 else tuple(caches)[::-1]):
                result, elapsed = wall(lambda: call(mode))
                assert record(result) == records[mode], "timed output changed"
                samples.append(dict(mode=mode, repeat=repeat, wall_ms=elapsed))
        save(out / "wall.json", samples)
        summary = dict(cold_ms=cold, agreement=agreement, quality={})
        audio_seconds = sum(len(w) for w in waves) / model.cfg.sample_rate
        for mode in caches:
            values = [r["wall_ms"] for r in samples if r["mode"] == mode]
            summary[mode] = stats(values) | {"rtfx": audio_seconds / (stats(values)["p50"] / 1000)}
            summary["quality"][mode] = (quality(items, records[mode], args.lang) if kw.get("task", "asr") == "asr"
                                        else {"unmeasured": "No reference English translations in this manifest"})
            summary[f"{mode}_resources"] = {
                str(key): dict(encoder_captures=len(s._enc_graphs),
                               staging_bytes=sum(t.numel() * t.element_size()
                                                 for buf in getattr(s, "_lookahead_stage", {}).values() for t in buf))
                for key, s in caches[mode].items()
            }
        summary["shared_process_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        save(out / "summary.json", summary)
        all_summaries[name] = summary
        save(args.out / "summary.json", all_summaries)
        if args.trace and name == "beam_b3":
            # A bounded timeline is separate from whole-list wall measurements.
            model._sessions = caches["lookahead"]
            model._encoder_lookahead = True
            model.transcribe(waves[:12], **kw)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                model.transcribe(waves[:12], **kw)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out / "lookahead_trace.json"))
        if getattr(model.backend, "failed", set()):
            raise RuntimeError("decoder trial failure invalidates the requested dispatch")
        if progress := os.environ.get("OWSM_PROFILE_PROGRESS"):
            with Path(progress).open("a") as f:
                f.write(f"{args.out.name}: {name} lookahead comparison complete\n")
        model._sessions = OrderedDict()
        caches.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
