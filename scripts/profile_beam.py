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
from bisect import bisect_right
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
    steps = dict(session._steps)
    steps.update({f"tail/group={b}/cache={c}": fn for (c, b), fn in getattr(session, "_tail_steps", {}).items()})
    return {c: {"implementation": getattr(fn, "dispatch", fn.__qualname__),
                "module": fn.__module__, "fallback_reason": getattr(fn, "fallback_reason", None),
                "writes_cache": getattr(fn, "writes_cache", False)}
            for c, fn in steps.items()}


def capture_shapes(session):
    return {"decoder": {f"rows={session.rows},cache={c}": 1 for c in session.graphs},
            "tail_decoder": {f"rows={session.beam},group={b},cache={c}": 1
                             for c, b in getattr(session, "_tail_graphs", {})},
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


def kernel_totals(path):
    """Sum eager trace kernel durations by the enclosing host phase of their launch correlation."""
    trace = json.loads(Path(path).read_text())["traceEvents"]
    ranges = sorted((e for e in trace if e.get("cat") == "user_annotation" and e.get("ph") == "X"),
                    key=lambda e: e["ts"])
    starts = [e["ts"] for e in ranges]
    launches = {e["args"]["correlation"]: e for e in trace
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    totals = defaultdict(float)
    for event in trace:
        if event.get("cat") != "kernel":
            continue
        host = launches.get(event.get("args", {}).get("correlation"))
        name = "unattributed"
        if host:
            i = bisect_right(starts, host["ts"]) - 1
            if i >= 0 and host["ts"] <= ranges[i]["ts"] + ranges[i]["dur"] and host["tid"] == ranges[i]["tid"]:
                name = ranges[i]["name"]
        totals[name] += event["dur"] / 1000  # trace microseconds -> milliseconds
    return dict(totals)


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


def make_session(model, cls, batch, beam, chunk, diagnostics=False, graph=True,
                 fused_rank=False, fused_finish=False, kv_reorder="torch", share_cross_kv=False, fused_state=False, compact_tail=False, local_topk=False):
    extra = {"diagnostics": diagnostics, "fused_rank": fused_rank,
             "fused_finish": fused_finish, "kv_reorder": kv_reorder, "share_cross_kv": share_cross_kv, "fused_state": fused_state, "compact_tail": compact_tail, "local_topk": local_topk} if cls is BeamSession else {}
    backend = getattr(model, "_profile_backend", model.backend) if cls is BeamSession else model.backend
    return cls(backend, model.cfg, batch, beam, model.opts.capacity, model.tokenizer.eos,
               model.dtype, model.device, use_cuda_graph=graph, chunk=chunk, rules=model.rules, **extra)


def configure_gluon(model, programs=None, prefetch=None, group_queries=False, cross_splits=None, max_rows=None):
    """Independent dispatch state, shared immutable weights/engines; baseline closures stay unchanged."""
    from owsm_trt.megakernel import MegaBackend
    from owsm_trt.megakernel_gluon import GluonDecoder

    base = model.backend
    if not isinstance(base, MegaBackend):
        raise ValueError("Gluon options require a successfully initialized megakernel backend")
    source = base.rows_kernel[0] if base.rows_kernel is not None else base.kernel
    if not isinstance(source, GluonDecoder) or id(source) in base.failed:
        raise ValueError("Gluon options require an available multi-row Gluon decoder")
    if programs is not None and not 1 <= programs <= torch.cuda.get_device_properties(model.device).multi_processor_count:
        raise ValueError("program count must fit the GPU's SMs for the persistent grid barrier")
    if max_rows is not None and max_rows not in (8, 16):
        raise ValueError("experimental row limit must be8 or16; token capacity stays unchanged")
    kernel = GluonDecoder(source.m, programs=programs or source.P, num_warps=source.nw,
                          prefetch_attention=source.pfa if prefetch is None else prefetch,
                          group_cross_queries=group_queries, cross_splits=cross_splits)
    backend = MegaBackend(base.inner, base.kernel, base.max_tokens, base.max_rows,
                          rows_kernel=base.rows_kernel, trial=base.trial)
    backend.failed = set(base.failed)
    if base.rows_kernel is None:
        backend.kernel = kernel
        if max_rows is not None:
            backend.max_rows = max_rows
    else:
        backend.rows_kernel = (kernel, base.rows_kernel[1], max_rows or base.rows_kernel[2])
    model._profile_backend = backend
    return {"programs": kernel.P, "warps": kernel.nw, "prefetch_attention": kernel.pfa,
            "reference_programs": source.P, "reference_prefetch_attention": source.pfa,
            "routing_limits_unchanged": max_rows is None, "max_rows_override": max_rows,
            "group_cross_queries": group_queries,
            "cross_splits": cross_splits}


def profile_eager(model, args, waves, primer, out):
    s = make_session(model, BeamSession, len(waves), args.beam, args.chunk, diagnostics=True, graph=False,
                     fused_rank=args.fused_rank, fused_finish=args.fused_finish, kv_reorder=args.kv_reorder, share_cross_kv=args.share_cross_kv, fused_state=args.fused_state, compact_tail=args.compact_tail, local_topk=args.local_topk)
    s.set_upper_bound(args.upper_bound != "off")
    s.encode(model._features(waves))
    expected = s.decode(primer, model._max_new())  # warm up this shape before attaching events
    events = Events()
    phases = [("prefill", "_prefill"), ("finish_and_history", "_finish"), ("host_sync", "_all_done"),
              ("cross_kv_expand", "_expand_cross")]
    phases += ([("selection", "_topk"), ("state_update", "_prepare_state"), ("reorder", "_reorder_cache")]
               if s.fused_state else [("selection", "_rank"), ("reorder", "_reorder")])
    for name, attr in phases:
        setattr(s, attr, events.wrap(name, getattr(s, attr)))
    s._steps = {c: events.wrap("decoder", fn) for c, fn in s._steps.items()}
    s._tail_steps = {key: events.wrap("decoder", fn) for key, fn in s._tail_steps.items()}
    # Beam's eager encoder calls backend.encode; isolate its time from cross-KV expansion.
    original_encode = s.backend.encode
    s.backend.encode = events.wrap("encoder", original_encode)
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
        s.backend.encode = original_encode
    kernels = kernel_totals(out / "eager_trace.json") if args.trace else None
    return {"mode": "eager, diagnostics enabled; not CUDA graph timing", "phases": events.collect(),
            "warmup_tokens_equal": expected == tokens,
            "graph_internal_phases": None,
            "graph_internal_missing_reason": "CUDA events in eager execution do not attribute graph nodes",
            "kernel_only_ms": sum(kernels.values()) if kernels is not None else None,
            "kernel_only_ms_by_phase": kernels,
            "kernel_only_scope": "eager decode trace only; excludes encoder, copies and host launch gaps",
            "kernel_only_missing_reason": None if kernels is not None else "enable --trace for attribution",
            "diagnostics": s.get_diagnostics()}


def run_batch(model, args, items, waves, batch, legacy):
    out = args.out / f"b{batch}"
    out.mkdir(parents=True, exist_ok=True)
    chunks = [list(range(i, min(i + batch, len(items)))) for i in range(0, len(items), batch)]
    sizes = sorted({len(c) for c in chunks})
    modes = (["baseline"] if legacy else []) + (["off", "on"] if args.upper_bound == "both"
                                               else [args.upper_bound])
    modes += [f"on_chunk{n}" for n in args.compare_chunks if n != args.chunk]
    sessions, cold = {}, {}
    for kind, cls in (("fixed", BeamSession), ("baseline", legacy)):
        if cls is None:
            continue
        for b in sizes:
            s, ms = wall(lambda: make_session(model, cls, b, args.beam, args.chunk,
                        fused_rank=args.fused_rank, fused_finish=args.fused_finish, kv_reorder=args.kv_reorder, share_cross_kv=args.share_cross_kv, fused_state=args.fused_state, compact_tail=args.compact_tail, local_topk=args.local_topk))
            sessions[kind, b] = s
            cross = {id(t): t for t in (s.cross_k, s.cross_v, s.cross_kb, s.cross_vb)}
            cross_bytes = sum(t.numel() * t.element_size() for t in cross.values())
            del cross
            cold[f"{kind}/b{b}"] = {"session_create_jit_capture_ms": ms,
                                    "dispatch": dispatch(s),
                                    "search_kernels": {"rank": getattr(s, "fused_rank", False),
                                                       "finish": getattr(s, "fused_finish", False),
                                                       "kv_reorder": getattr(s, "kv_reorder", "torch"),
                                                       "share_cross_kv": getattr(s, "share_cross_kv", False),
                                                       "state": getattr(s, "fused_state", False),
                                                       "local_topk": getattr(s, "local_topk", False),
                                                       "compact_tail": getattr(s, "compact_tail", False),
                                                       "compact_tail_fallback": getattr(s, "compact_tail_fallback", None),
                                                       "cross_kv_fallback": getattr(s, "cross_kv_fallback", None),
                                                       "cross_kv_bytes": cross_bytes}}
    if getattr(getattr(model, "_profile_backend", None), "failed", set()):
        raise RuntimeError("candidate trial failed before timing; requested dispatch unavailable")
    kw = dict(lang=args.lang, task=args.task, timestamps=args.timestamps, prompt=args.prompt,
              batch_size=batch, beam_size=args.beam)

    def select(mode):
        kind = "baseline" if mode == "baseline" else "fixed"
        model._sessions.clear()
        for b in sizes:
            s = sessions[kind, b]
            if kind == "fixed":
                s.set_upper_bound(mode != "off")
                s.chunk = int(mode.removeprefix("on_chunk")) if mode.startswith("on_chunk") else args.chunk
            model._sessions[b, args.beam] = s

    # All feature lengths and batch shapes, including the partial final batch, are warmed up.
    for mode in modes:
        select(mode)
        def warm():
            # transcribe sorts only within each call. Warming the whole manifest would create
            # different B>1 feature shapes from the actual per-chunk latency measurements.
            for indices in chunks:
                model.transcribe([waves[i] for i in indices], **kw)
        _, ms = wall(warm)
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
                s = model._session(len(group), args.beam)
                captures_before = len(s._enc_graphs)
                torch.cuda.reset_peak_memory_stats()
                results, full_ms = wall(lambda: model.transcribe(group, **kw))
                s = model._session(len(group), args.beam)
                if len(s._enc_graphs) != captures_before:
                    raise RuntimeError("new encoder capture in steady-state timing; warm-up incomplete")
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
        s = make_session(model, BeamSession, b, args.beam, args.chunk, diagnostics=True,
                         fused_rank=args.fused_rank, fused_finish=args.fused_finish, kv_reorder=args.kv_reorder, share_cross_kv=args.share_cross_kv, fused_state=args.fused_state, compact_tail=args.compact_tail, local_topk=args.local_topk)
        for mode in [m for m in modes if m != "baseline"]:
            s.set_upper_bound(mode != "off")
            s.chunk = int(mode.removeprefix("on_chunk")) if mode.startswith("on_chunk") else args.chunk
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
    if args.verify_api:
        # Exercise OWSM's production session factory, independently of profiler-supplied sessions.
        original_options = model.beam_compact_tail, model.beam_grouped_cross, model._beam_backend
        model.beam_compact_tail = args.compact_tail
        model.beam_grouped_cross = True
        model._beam_backend = None
        model._sessions.clear()
        actual, api_dispatch = {}, {}
        try:
            for indices in chunks:
                results = model.transcribe([waves[i] for i in indices], **kw)
                api_session = model._sessions[len(indices), args.beam]
                api_dispatch[len(indices)] = dispatch(api_session)
                assert api_dispatch[len(indices)] == cold[f"fixed/b{len(indices)}"]["dispatch"], "API dispatch differs"
                assert not getattr(api_session.backend, "failed", set()), "API decoder trial failed"
                for i, result in zip(indices, results):
                    actual[items[i]["id"]] = result_record(model, result, items[i], len(waves[i]) / model.cfg.sample_rate)
            agreement = comparison(records["on"], actual)
            save(out / "api_agreement.json", agreement)
            save(out / "api_dispatch.json", api_dispatch)
            assert all(v["equal"] == v["total"] for v in agreement.values()), "public API and measured mode differ"
        finally:
            model._sessions.clear()
            model.beam_compact_tail, model.beam_grouped_cross, model._beam_backend = original_options
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
    if args.extra_batch > 1:
        cases = [(name, audio + waves[:max(0, args.extra_batch - len(audio))], extra)
                 if not extra.get("long") else (name, audio, extra) for name, audio, extra in cases]
    original = model._session
    original_options = model.beam_compact_tail, model.beam_grouped_cross, model._beam_backend
    results = {}
    for mode in (["baseline"] if legacy else []) + ["off", "on"] + (["api"] if args.verify_api else []):
        cache = {}

        def get(batch, beam=1):
            if beam == 1:
                return original(batch, beam)
            key = (batch, beam)
            if key not in cache:
                cls = legacy if mode == "baseline" else BeamSession
                cache[key] = make_session(model, cls, batch, beam, args.chunk,
                                         fused_rank=args.fused_rank, fused_finish=args.fused_finish,
                                         kv_reorder=args.kv_reorder, share_cross_kv=args.share_cross_kv, fused_state=args.fused_state, compact_tail=args.compact_tail, local_topk=args.local_topk)
                if mode != "baseline":
                    cache[key].set_upper_bound(mode == "on")
            return cache[key]

        model._session = original if mode == "api" else get
        if mode == "api":
            model._sessions.clear()
            model.beam_compact_tail, model.beam_grouped_cross, model._beam_backend = args.compact_tail, True, None
        try:
            results[mode] = {}
            for name, audio, extra in cases:
                kw = dict(lang=args.lang, beam_size=args.beam, batch_size=args.extra_batch)
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
            if mode == "api":
                model._sessions.clear()
                model.beam_compact_tail, model.beam_grouped_cross, model._beam_backend = original_options
        cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
    reference = results.get("baseline", results["off"])
    agreement = {mode: {name: val["results"] == reference[name]["results"] for name, val in rows.items()}
                 for mode, rows in results.items()}
    save(args.out / "extra_agreement.json", agreement)
    if args.verify_api:
        api_matches = {name: val["results"] == results["on"][name]["results"]
                       for name, val in results["api"].items()}
        save(args.out / "extra_api_agreement.json", api_matches)
        assert all(api_matches.values()), "extra-case public API and candidate differ"


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
    ap.add_argument("--decoder-kernel", default="auto", help="auto, 0 to disable, or a pipeline kernel name")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--compare-chunks", type=int, nargs="*", default=[],
                    help="additional upper-bound ON host-check frequencies, on the same graphs")
    ap.add_argument("--upper-bound", choices=["off", "on", "both"], default="both")
    ap.add_argument("--compact-tail", action="store_true", help="opt-in N: decode only the final active utterance group")
    ap.add_argument("--fused-state", action="store_true", help="candidate only: fuse beam parent/token/history updates")
    ap.add_argument("--fused-rank", action="store_true", help="opt-in exact candidate-mask kernel")
    ap.add_argument("--fused-finish", action="store_true", help="opt-in exact beam-state kernel")
    ap.add_argument("--kv-reorder", choices=["torch", "full", "prefix"], default="torch")
    ap.add_argument("--share-cross-kv", action="store_true")
    ap.add_argument("--gluon-group-queries", action="store_true",
                    help="candidate only: group beam queries sharing cross KV in one attention tile")
    ap.add_argument("--gluon-cross-splits", type=int, choices=[1, 2, 4, 8],
                    help="candidate only: key slices per cross-attention head (changes FP reduction order)")
    ap.add_argument("--gluon-beam-splits", action="store_true",
                    help="candidate only: measured K5 row preset R5/10/15 uses8/4/2 key slices")
    ap.add_argument("--gluon-max-rows", type=int, choices=[8, 16],
                    help="experimental N-class multi-row routing, existing token/tile capacity unchanged")
    ap.add_argument("--gluon-programs", type=int, help="candidate only: persistent CTA count")
    ap.add_argument("--gluon-prefetch", action=argparse.BooleanOptionalAction, default=None,
                    help="candidate only: attention prefetch; keep arithmetic and routing unchanged")
    ap.add_argument("--local-topk", action="store_true",
                    help="candidate only: parent-local then global top-K; equal-score ordering may differ")
    ap.add_argument("--baseline-runtime", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--disk", action="store_true")
    ap.add_argument("--verify-api", action="store_true", help="check production API with both measured opt-in options")
    ap.add_argument("--extra-checks", action="store_true")
    ap.add_argument("--extra-batch", type=int, default=1, help="batch size for extra mixed-window cases")
    ap.add_argument("--long-audio")
    args = ap.parse_args()
    if args.beam < 2 or min(args.batch) < 1 or max(args.batch) * args.beam > 16:
        ap.error("require beam >= 2 and 1 <= batch * beam <= 16 for the default engines")
    if not 1 <= args.extra_batch <= 16 // args.beam:
        ap.error("extra-batch must fit batch * beam <=16")
    if args.repeat < 1 or args.chunk < 1 or (args.limit is not None and args.limit < 1):
        ap.error("repeat, chunk and limit must be positive")
    if any(n < 1 for n in args.compare_chunks) or len(set(args.compare_chunks)) != len(args.compare_chunks):
        ap.error("compare-chunks must be distinct positive integers")
    tuning = (args.gluon_programs is not None or args.gluon_prefetch is not None or args.gluon_group_queries
              or args.gluon_cross_splits is not None or args.gluon_max_rows is not None or args.gluon_beam_splits)
    if (tuning or args.local_topk) and not args.baseline_runtime:
        ap.error("selection/Gluon ablations require --baseline-runtime for the unchanged reference")
    if args.verify_api and args.local_topk:
        ap.error("local-topk is not a public API option")
    if args.gluon_beam_splits and not (args.beam == 5 and args.gluon_max_rows == 16
                                      and args.gluon_group_queries and args.gluon_cross_splits is None):
        ap.error("gluon-beam-splits requires K5, grouped queries, max-rows16 and no fixed cross-splits")
    if args.verify_api and not (args.beam == 5 and args.upper_bound == "on" and args.fused_state
                                and args.fused_rank and args.fused_finish and args.kv_reorder == "prefix"
                                and args.share_cross_kv and args.gluon_group_queries
                                and args.gluon_beam_splits):
        ap.error("verify-api requires the current K5 grouped row preset with fused/prefix/shared upper-bound ON")
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
    decoder_options = None
    if tuning:
        from owsm_trt.megakernel_gluon import BEAM_CROSS_SPLITS

        splits = BEAM_CROSS_SPLITS if args.gluon_beam_splits else args.gluon_cross_splits
        decoder_options = configure_gluon(model, args.gluon_programs, args.gluon_prefetch,
                                          args.gluon_group_queries, splits, args.gluon_max_rows)
        if args.gluon_max_rows is not None:
            # Shape-specific numeric checks precede real search; synthetic shared-KV/cache tests
            # are run separately by the PBS driver. No public dispatch limits are changed.
            from owsm_trt.pipeline import _check_kernel, _load_decoder

            kernel = (model._profile_backend.rows_kernel or (model._profile_backend.kernel,))[0]
            decoder = _load_decoder(model.cfg, hub.resolve(args.model).checkpoint)
            checks = {}
            for rows in sorted({b * args.beam for b in args.batch}):
                _, ms = wall(lambda: _check_kernel(kernel, decoder, model.device, rows, 1))
                checks[rows] = {"passed": True, "check_ms": ms, "window": 1,
                                "criterion": "max relative logits error<0.02, argmax>=0.5 vs fp32 CPU"}
            del decoder
            save(args.out / "shape_checks.json", checks)
    if any(len(w) > model.cfg.num_samples for w in waves):
        ap.error("manifest contains audio longer than a model window; use a separate long-form evaluation")
    files = hub.resolve(args.model)
    engine_dir = (hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
                  if args.backend == "tensorrt" else None)
    versions = {}
    for name in ("torch", "triton", "tensorrt", "numpy", "sentencepiece"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    prop = torch.cuda.get_device_properties(model.device)
    env = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "git_commit": os.environ.get("OWSM_PROFILE_COMMIT") or command(["git", "rev-parse", "HEAD"]),
           "git_diff": None if os.environ.get("OWSM_PROFILE_COMMIT") else command(["git", "diff", "--stat"]),
           "python": platform.python_version(),
           "versions": versions, "cuda": torch.version.cuda, "gpu": prop.name,
           "sm_count": prop.multi_processor_count, "gpu_memory": prop.total_memory,
           "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]),
           "nsys": shutil.which("nsys"), "ncu": shutil.which("ncu"),
           "environment": {k: v for k, v in os.environ.items() if k.startswith(("OWSM_", "TORCH", "CUDA"))
                           or k in ("PBS_JOBID", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "checkpoint_sha256": sha256(files.checkpoint), "manifest_sha256": sha256(args.manifest),
           "engine_dir": str(engine_dir) if engine_dir is not None else None,
           "engine_sha256": {p.name: sha256(p) for p in engine_dir.glob("*.engine")}
                            if engine_dir is not None else {},
           "build_metadata": json.loads((engine_dir / "build.json").read_text())
                             if engine_dir is not None and (engine_dir / "build.json").exists() else None,
           "cold": {"disk_load_ms": disk_load_ms, "from_pretrained_ms": load_ms,
                    "components_ms": dict(cold), "jit_separate": None,
                    "jit_note": "included in backend/session setup; not independently attributed"},
           "decoder_options": decoder_options,
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
    if tuning and model._profile_backend.failed:
        raise RuntimeError("candidate decoder trial failed; comparison is not the requested dispatch")


if __name__ == "__main__":
    main()
