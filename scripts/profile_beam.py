"""Same-engine beam ablations. Run audio/model work only in a compute allocation.

Wall measurements use sessions without diagnostics/events/profiler. Diagnostic graph runs and
eager CUDA-event attribution are separate; eager phase times are not graph phase estimates.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from eval_wer import CER_LANGS, characters, edit_distance, normalize

import owsm_trt.pipeline as pipeline
from owsm_trt import OWSM, hub
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.runtime import BeamSession


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {e}"


def stats(values):
    if not values:
        return None
    x = sorted(values)

    def percentile(q):
        i = (len(x) - 1) * q
        a = int(i)
        return x[a] + (x[min(a + 1, len(x) - 1)] - x[a]) * (i - a)

    return {"n": len(x), "p50": percentile(.5), "p95": percentile(.95),
            "min": x[0], "max": x[-1], "stdev": statistics.pstdev(x)}


def wall(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - t) * 1000


def dispatch(session):
    return {c: {"implementation": getattr(fn, "dispatch", fn.__qualname__),
                "module": fn.__module__, "fallback_reason": getattr(fn, "fallback_reason", None),
                "writes_cache": getattr(fn, "writes_cache", False)}
            for c, fn in session._steps.items()}


def capture_shapes(session):
    return {"decoder": {f"rows={session.rows},cache={c}": 1 for c in session.graphs},
            "encoder": {str(key[1:]): 1 for key in session._enc_graphs}}


class Events:
    """Instrumentation for a separate eager run. No per-step synchronization."""

    def __init__(self):
        self.samples = []

    def wrap(self, name, fn):
        def run(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            t = time.perf_counter()
            with torch.profiler.record_function(name):
                result = fn(*args, **kwargs)
            cpu_ms = (time.perf_counter() - t) * 1000
            end.record()
            self.samples.append((name, start, end, cpu_ms))
            return result
        run.__dict__.update(getattr(fn, "__dict__", {}))
        return run

    def collect(self):
        torch.cuda.synchronize()
        gpu, cpu = defaultdict(list), defaultdict(list)
        for name, a, b, host in self.samples:
            gpu[name].append(a.elapsed_time(b))
            cpu[name].append(host)
        return {name: {"cuda_event_ms": stats(values), "cuda_event_total_ms": sum(values),
                       "host_ms": stats(cpu[name]), "host_total_ms": sum(cpu[name])}
                for name, values in gpu.items()}


def result_record(model, result, item, duration):
    times = [t for t in result.tokens if model.tokenizer.is_time(t)]
    segments = [asdict(s) for s in result.segments]
    invalid = sum(s["end"] < s["start"] or s["start"] < 0 or s["end"] > duration + .02
                  for s in segments)
    return {"id": item["id"], "input_seconds": duration, "tokens": result.tokens,
            "token_count": len(result.tokens), "timestamp_tokens": times, "timestamp_count": len(times),
            "text": result.text, "segments": segments, "segment_count": len(segments),
            "empty_segments": sum(not s["text"].strip() for s in segments),
            "invalid_segments": invalid, "unpaired_timestamp": bool(len(times) % 2),
            "nonmonotonic_timestamps": any(b < a for a, b in zip(times, times[1:]))}


def comparison(reference, actual):
    fields = ("tokens", "timestamp_tokens", "segments", "text")
    return {f: {"equal": sum(reference[k][f] == actual[k][f] for k in reference),
                "total": len(reference),
                "different_ids": [k for k in reference if reference[k][f] != actual[k][f]]}
            for f in fields}


def quality(items, records, lang):
    units = characters if lang in CER_LANGS else normalize
    pairs = [(units(it["text"]), units(records[it["id"]]["text"])) for it in items if "text" in it]
    n = sum(len(a) for a, _ in pairs)
    errors = sum(edit_distance(a, b) for a, b in pairs)
    return {"metric": "CER" if lang in CER_LANGS else "WER", "reference_units": n,
            "errors": errors, "rate": errors / n if n else None, "annotated_utterances": len(pairs),
            "reference_timestamp_error": None,
            "reference_timestamp_status": "not evaluated; token/segment comparisons are baseline agreement"}


def make_session(model, cls, batch, beam, chunk, diagnostics=False, graph=True):
    extra = {"diagnostics": diagnostics} if cls is BeamSession else {}
    return cls(model.backend, model.cfg, batch, beam, model.opts.capacity, model.tokenizer.eos,
               model.dtype, model.device, use_cuda_graph=graph, chunk=chunk, rules=model.rules, **extra)


def profile_eager(model, args, waves, primer, out):
    s = make_session(model, BeamSession, len(waves), args.beam, args.chunk, diagnostics=True, graph=False)
    s.set_upper_bound(args.upper_bound != "off")
    s.encode(model._features(waves))
    expected = s.decode(primer, model._max_new())  # warm up this shape before attaching events
    events = Events()
    for name, attr in (("prefill", "_prefill"), ("selection", "_rank"), ("reorder", "_reorder"),
                       ("finish_and_history", "_finish"), ("host_sync", "_all_done"),
                       ("cross_kv_expand", "_expand_cross")):
        setattr(s, attr, events.wrap(name, getattr(s, attr)))
    s._steps = {c: events.wrap("decoder", fn) for c, fn in s._steps.items()}
    # Beam's eager encoder calls backend.encode; isolate its time from cross-KV expansion.
    original_encode = model.backend.encode
    model.backend.encode = events.wrap("encoder", original_encode)
    try:
        feats = events.wrap("features_h2d", model._features)(waves)
        s.encode(feats)
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                tokens = s.decode(primer, model._max_new())
            prof.export_chrome_trace(str(out / "eager_trace.json"))
        else:
            tokens = s.decode(primer, model._max_new())
    finally:
        model.backend.encode = original_encode
    return {"mode": "eager, diagnostics enabled; not CUDA graph timing", "phases": events.collect(),
            "warmup_tokens_equal": expected == tokens,
            "graph_internal_phases": None,
            "graph_internal_missing_reason": "CUDA events in eager execution do not attribute graph nodes",
            "kernel_only_ms": None,
            "kernel_only_missing_reason": "event spans include launch gaps; inspect profiler/Nsight trace",
            "diagnostics": s.get_diagnostics()}


def run_batch(model, args, items, waves, batch, legacy):
    out = args.out / f"b{batch}"
    out.mkdir(parents=True, exist_ok=True)
    chunks = [list(range(i, min(i + batch, len(items)))) for i in range(0, len(items), batch)]
    sizes = sorted({len(c) for c in chunks})
    modes = (["baseline"] if legacy else []) + (["off", "on"] if args.upper_bound == "both"
                                               else [args.upper_bound])
    sessions, cold = {}, {}
    for kind, cls in (("fixed", BeamSession), ("baseline", legacy)):
        if cls is None:
            continue
        for b in sizes:
            s, ms = wall(lambda: make_session(model, cls, b, args.beam, args.chunk))
            sessions[kind, b] = s
            cold[f"{kind}/b{b}"] = {"session_create_jit_capture_ms": ms,
                                    "dispatch": dispatch(s)}
    kw = dict(lang=args.lang, task=args.task, timestamps=args.timestamps, prompt=args.prompt,
              batch_size=batch, beam_size=args.beam)

    def select(mode):
        kind = "baseline" if mode == "baseline" else "fixed"
        model._sessions.clear()
        for b in sizes:
            s = sessions[kind, b]
            if kind == "fixed":
                s.set_upper_bound(mode == "on")
            model._sessions[b, args.beam] = s

    # All feature lengths and batch shapes, including the partial final batch, are warmed up.
    for mode in modes:
        select(mode)
        _, ms = wall(lambda: model.transcribe(waves, **kw))
        cold[mode] = {"warmup_ms": ms}
    for (kind, b), s in sessions.items():
        cold[f"{kind}/b{b}"]["capture_shapes"] = capture_shapes(s)
    save(out / "cold.json", cold)
    records, samples = {m: {} for m in modes}, []
    for repeat in range(args.repeat):
        for mode in (modes if repeat % 2 == 0 else modes[::-1]):
            select(mode)
            for indices in chunks:
                group = [waves[i] for i in indices]
                torch.cuda.reset_peak_memory_stats()
                results, full_ms = wall(lambda: model.transcribe(group, **kw))
                s = model._session(len(group), args.beam)
                # A separate decode-only call, using encoder outputs computed outside its timing.
                s.encode(model._features(group))
                primer = torch.tensor([model._primer(args.lang, args.task, args.timestamps, args.prompt)]
                                      * len(group), device=model.device)
                tokens, decode_ms = wall(lambda s=s: s.decode(primer, model._max_new()))
                calls = int(s.step)
                assert tokens == [r.tokens for r in results], "decode-only vs transcribe token difference"
                duration = sum(len(w) for w in group) / model.cfg.sample_rate
                sample = {"mode": mode, "repeat": repeat, "ids": [items[i]["id"] for i in indices],
                          "transcribe_ms": full_ms, "decode_ms": decode_ms, "decoder_calls": calls,
                          "decode_ms_per_selection": decode_ms / (calls + 1), "audio_seconds": duration,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                samples.append(sample)
                for i, result in zip(indices, results):
                    rec = result_record(model, result, items[i], len(waves[i]) / model.cfg.sample_rate)
                    old = records[mode].get(items[i]["id"])
                    if old is not None and old["tokens"] != rec["tokens"]:
                        raise RuntimeError(f"non-deterministic tokens: {mode}/{items[i]['id']}")
                    records[mode][items[i]["id"]] = rec
            print(f"b={batch} repeat={repeat} mode={mode} complete", flush=True)
            save(out / "wall.json", samples)
            save(out / "tokens.json", records)
    summary = {}
    for mode in modes:
        rows = [r for r in samples if r["mode"] == mode]
        summary[mode] = {k: stats([r[k] for r in rows]) for k in
                         ("transcribe_ms", "decode_ms", "decoder_calls", "decode_ms_per_selection")}
        summary[mode]["rtfx"] = sum(r["audio_seconds"] for r in rows) * 1000 / sum(
            r["transcribe_ms"] for r in rows)
        summary[mode]["peak_allocated_bytes"] = max(r["peak_allocated_bytes"] for r in rows)
        summary[mode]["quality"] = quality(items, records[mode], args.lang)
        summary[mode]["agreement"] = comparison(records[modes[0]], records[mode])
    save(out / "summary.json", summary)

    # Disk-inclusive timings and diagnostic histories are not part of steady-state wall samples.
    disk = {}
    if args.disk:
        for mode in modes:
            select(mode)
            results, ms = wall(lambda: model.transcribe([it["path"] for it in items], **kw))
            disk[mode] = {"wall_ms": ms, "file_cache": "OS cache not flushed",
                          "tokens_equal": all(r.tokens == records[mode][it["id"]]["tokens"]
                                              for r, it in zip(results, items))}
    save(out / "disk.json", disk)
    model._sessions.clear()
    sessions.clear()
    del s
    gc.collect()
    torch.cuda.empty_cache()

    diag = {}
    for b in sizes:
        s = make_session(model, BeamSession, b, args.beam, args.chunk, diagnostics=True)
        for mode in [m for m in modes if m != "baseline"]:
            s.set_upper_bound(mode == "on")
            for indices in [c for c in chunks if len(c) == b]:
                group = [waves[i] for i in indices]
                s.encode(model._features(group))
                primer = torch.tensor([model._primer(args.lang, args.task, args.timestamps, args.prompt)] * b,
                                      device=model.device)
                tokens = s.decode(primer, model._max_new())
                assert tokens == [records[mode][items[i]["id"]]["tokens"] for i in indices]
                diag[f"{mode}/{'/'.join(str(i) for i in indices)}"] = s.get_diagnostics()
        del s
    save(out / "diagnostics.json", diag)
    if args.phases:
        group = waves[:batch]
        primer = torch.tensor([model._primer(args.lang, args.task, args.timestamps, args.prompt)] * len(group),
                              device=model.device)
        save(out / "phases.json", profile_eager(model, args, group, primer, out))
    return summary


def extra_checks(model, args, waves, legacy):
    """Quality/compatibility cases, outside the wall benchmark; no reference times invented."""
    g = torch.Generator().manual_seed(0)
    silence = torch.zeros(model.cfg.sample_rate * 2)
    noise = torch.randn(model.cfg.sample_rate * 2, generator=g) * .01
    window = waves[0].repeat(math.ceil(model.cfg.num_samples / len(waves[0])))[:model.cfg.num_samples]
    cases = [("no_timestamps", [waves[0]], {"timestamps": False}),
             ("translation", [waves[0]], {"task": "st_eng"}),
             ("prompt", [waves[0]], {"prompt": "Previous speech context. " * 20}),
             ("silence_noise", [silence, noise], {}), ("30s_repeated_speech", [window], {})]
    if args.long_audio:
        cases.append(("long_audio", [load_audio(args.long_audio)], {"long": True}))
    original = model._session
    results = {}
    for mode in (["baseline"] if legacy else []) + ["off", "on"]:
        cache = {}

        def get(batch, beam=1):
            if beam == 1:
                return original(batch, beam)
            key = (batch, beam)
            if key not in cache:
                cls = legacy if mode == "baseline" else BeamSession
                cache[key] = make_session(model, cls, batch, beam, args.chunk)
                if mode != "baseline":
                    cache[key].set_upper_bound(mode == "on")
            return cache[key]

        model._session = get
        try:
            results[mode] = {}
            for name, audio, extra in cases:
                kw = dict(lang=args.lang, beam_size=args.beam, batch_size=1)
                if extra.get("long"):
                    value, ms = wall(lambda: model.transcribe_long(audio, **kw))
                else:
                    kw.update(timestamps=True, **{k: v for k, v in extra.items() if k != "timestamps"})
                    kw["timestamps"] = extra.get("timestamps", True)
                    value, ms = wall(lambda: model.transcribe(audio, **kw))
                results[mode][name] = {"results": [asdict(r) for r in value], "cold_inclusive_ms": ms}
                save(args.out / "extra_checks.json", results)
                print(f"extra {mode}/{name} complete", flush=True)
        finally:
            model._session = original
        cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
    reference = results.get("baseline", results["off"])
    agreement = {mode: {name: val["results"] == reference[name]["results"] for name, val in rows.items()}
                 for mode, rows in results.items()}
    save(args.out / "extra_agreement.json", agreement)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--prompt")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--task", default="asr")
    ap.add_argument("--backend", choices=["tensorrt", "torch"], default="tensorrt")
    ap.add_argument("--decoder-kernel", default="auto")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--upper-bound", choices=["off", "on", "both"], default="both")
    ap.add_argument("--baseline-runtime", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--disk", action="store_true")
    ap.add_argument("--extra-checks", action="store_true")
    ap.add_argument("--long-audio")
    args = ap.parse_args()
    if args.beam < 2 or min(args.batch) < 1 or max(args.batch) * args.beam > 16:
        ap.error("require beam >= 2 and 1 <= batch * beam <= 16 for the default engines")
    if args.repeat < 1 or args.chunk < 1 or (args.limit is not None and args.limit < 1):
        ap.error("repeat, chunk and limit must be positive")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("audio and inference must run in an allocated compute job")
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.out / "runtime.log")])
    items = json.loads(Path(args.manifest).read_text())[:args.limit]
    if not items or len({it["id"] for it in items}) != len(items):
        ap.error("manifest must be nonempty with unique ids")
    t = time.perf_counter()
    waves = [load_audio(it["path"]) for it in items]
    disk_load_ms = (time.perf_counter() - t) * 1000
    cold = defaultdict(list)
    originals = {}
    for name in ("build_engines", "load_model", "_load_frontend", "_load_decoder"):
        fn = getattr(pipeline, name)
        originals[name] = fn

        def measured(*a, _fn=fn, _name=name, **kw):
            result, ms = wall(lambda: _fn(*a, **kw))
            cold[_name].append(ms)
            return result
        setattr(pipeline, name, measured)
    model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, backend=args.backend,
                          decoder_kernel=args.decoder_kernel, ctc_draft=False))
    for name, fn in originals.items():
        setattr(pipeline, name, fn)
    if any(len(w) > model.cfg.num_samples for w in waves):
        ap.error("manifest contains audio longer than a model window; use a separate long-form evaluation")
    files = hub.resolve(args.model)
    engine_dir = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
    versions = {}
    for name in ("torch", "triton", "tensorrt", "numpy", "sentencepiece"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    prop = torch.cuda.get_device_properties(model.device)
    env = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "git_commit": command(["git", "rev-parse", "HEAD"]),
           "git_diff": command(["git", "diff", "--stat"]), "python": platform.python_version(),
           "versions": versions, "cuda": torch.version.cuda, "gpu": prop.name,
           "sm_count": prop.multi_processor_count, "gpu_memory": prop.total_memory,
           "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]),
           "nsys": shutil.which("nsys"), "ncu": shutil.which("ncu"),
           "environment": {k: v for k, v in os.environ.items() if k.startswith(("OWSM_", "TORCH", "CUDA"))
                           or k in ("PBS_JOBID", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "checkpoint_sha256": sha256(files.checkpoint), "manifest_sha256": sha256(args.manifest),
           "engine_dir": str(engine_dir),
           "engine_sha256": {p.name: sha256(p) for p in engine_dir.glob("*.engine")},
           "build_metadata": json.loads((engine_dir / "build.json").read_text())
                             if (engine_dir / "build.json").exists() else None,
           "cold": {"disk_load_ms": disk_load_ms, "from_pretrained_ms": load_ms,
                    "components_ms": dict(cold), "jit_separate": None,
                    "jit_note": "included in backend/session setup; not independently attributed"},
           "reference_time_annotations": "not evaluated", "items": items}
    save(args.out / "environment.json", env)
    legacy = None
    if args.baseline_runtime:
        spec = importlib.util.spec_from_file_location("owsm_trt._baseline_runtime", args.baseline_runtime)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        legacy = mod.BeamSession
        env["baseline_runtime_sha256"] = sha256(args.baseline_runtime)
        save(args.out / "environment.json", env)
    summaries = {}
    for batch in args.batch:
        summaries[batch] = run_batch(model, args, items, waves, batch, legacy)
        save(args.out / "summary.json", summaries)
        gc.collect()
        torch.cuda.empty_cache()
    if args.extra_checks:
        extra_checks(model, args, waves, legacy)


if __name__ == "__main__":
    main()
