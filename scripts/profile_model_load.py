"""Fresh-process cached-engine loading; checkpoint mapping is an isolated opt-in experiment.

Run on a compute node. OS cache and JIT are warm; Python/CUDA initialization and instrumented
helper timings are separate from load samples. This does not measure cold disk or engine builds.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def worker(args):
    import importlib.metadata
    import resource
    from dataclasses import asdict
    from functools import partial, wraps

    import torch

    from owsm_trt import OWSM, hub, modeling, pipeline
    from owsm_trt.audio import load_audio
    from owsm_trt.build import engine_tag

    selected = partial(modeling.load_checkpoint, memory_map=args.worker_mode == "mapped")
    phases = []
    load_calls = []

    def instrument(name, fn):
        @wraps(fn)
        def measured(*a, **kw):
            start = time.perf_counter()
            result = fn(*a, **kw)
            phases.append(dict(helper=name, host_elapsed_ms=(time.perf_counter() - start) * 1000))
            return result

        return measured

    if args.instrument:
        original_load = torch.load

        def traced_load(*a, **kw):
            load_calls.append({key: kw.get(key) for key in ("mmap", "map_location", "weights_only")})
            return original_load(*a, **kw)

        torch.load = traced_load
        selected = instrument("load_checkpoint", selected)
        for name in ("_load_frontend", "_load_decoder"):
            setattr(pipeline, name, instrument(name, getattr(pipeline, name)))
    modeling.load_checkpoint = pipeline.load_checkpoint = selected
    files = hub.resolve(args.model)
    cache = hub.cache_dir() / files.name.replace("/", "--")

    def engine_files():
        return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in cache.rglob("*.engine")}

    before = engine_files()
    if not before:
        raise RuntimeError("cached engines are required; refusing a build benchmark")
    start = time.perf_counter()
    torch.cuda.synchronize()
    cuda_init_ms = (time.perf_counter() - start) * 1000
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    model = OWSM.from_pretrained(args.model)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_cuda = torch.cuda.max_memory_allocated()
    if engine_files() != before:
        raise RuntimeError("engine files changed during load; result is not cached-engine startup")
    outputs = {}
    with torch.inference_mode():
        for lang, manifest in (("eng", args.manifest), ("jpn", args.extra_manifest)):
            if not manifest:
                continue
            items = json.loads(Path(manifest).read_text())[:3]
            waves = [load_audio(x["path"]) for x in items]
            for name, kwargs in (
                ("asr", {}),
                ("ts", {"timestamps": True}),
                ("translation", {"task": "st_eng"}),
                ("beam", {"beam_size": 5, "timestamps": True}),
            ):
                outputs[f"{lang}/{name}"] = [asdict(model.transcribe(w, lang=lang, **kwargs)) for w in waves]
            outputs[f"{lang}/lid"] = model.detect_language(waves[0])
    if getattr(model.backend, "failed", set()):
        raise RuntimeError("decoder failed; refusing fallback load measurement")

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    engines = cache / engine_tag(model.opts)
    print(
        json.dumps(
            dict(
                mode=args.worker_mode,
                instrumented=args.instrument,
                model_load_ms=load_ms,
                cuda_init_ms=cuda_init_ms,
                pre_load_peak_rss_kib=rss_before,
                load_peak_rss_kib=peak_rss,
                load_peak_cuda_bytes=peak_cuda,
                helper_host_timings=phases,
                checkpoint_load_calls=load_calls,
                checkpoint_zip=zipfile.is_zipfile(files.checkpoint),
                mmap_flags=torch.serialization.get_default_mmap_options(),
                outputs=outputs,
                checkpoint_sha256=sha(files.checkpoint),
                engine_sha256={p.name: sha(p) for p in engines.glob("*.engine")},
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda,
                versions={n: importlib.metadata.version(n) for n in ("torch", "triton", "tensorrt")},
                commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            )
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--extra-manifest")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker-mode", choices=("eager", "mapped"), help=argparse.SUPPRESS)
    ap.add_argument("--instrument", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("model/audio work requires a compute allocation")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    if args.worker_mode:
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    records, first, detail, expected, hashes = [], [], [], None, None
    for repeat in range(-1, args.repeat + 1):
        instrumented = repeat == args.repeat
        for mode in ("eager", "mapped") if repeat % 2 == 0 else ("mapped", "eager"):
            cmd = [
                sys.executable,
                __file__,
                "--model",
                args.model,
                "--manifest",
                args.manifest,
                "--out",
                str(args.out),
                "--worker-mode",
                mode,
            ]
            if args.extra_manifest:
                cmd += ["--extra-manifest", args.extra_manifest]
            if instrumented:
                cmd += ["--instrument"]
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = (time.perf_counter() - start) * 1000
            with (args.out / "workers.log").open("a") as log:
                log.write(f"cycle={repeat} mode={mode} exit={proc.returncode}\n{proc.stderr}\n")
            if proc.returncode:
                raise RuntimeError(f"{mode} worker failed: {proc.stderr[-4000:]}")
            record = json.loads(proc.stdout.strip().splitlines()[-1])
            current = (record["checkpoint_sha256"], record["engine_sha256"])
            if hashes is not None and current != hashes:
                raise RuntimeError("checkpoint or engines changed between processes")
            hashes = current
            if expected is not None and record["outputs"] != expected:
                raise RuntimeError(f"{mode} outputs changed in cycle{repeat}")
            expected = record["outputs"]
            record.update(repeat=repeat, worker_wall_ms=elapsed)
            (detail if instrumented else first if repeat < 0 else records).append(record)
            if progress := os.environ.get("OWSM_PROFILE_PROGRESS"):
                with Path(progress).open("a") as f:
                    f.write(f"{args.out.name}: model load {mode} cycle {repeat} complete\n")
    summary = {}
    for mode in ("eager", "mapped"):
        rows = [r for r in records if r["mode"] == mode]
        summary[mode] = {}
        for field in ("model_load_ms", "load_peak_rss_kib", "load_peak_cuda_bytes"):
            values = [r[field] for r in rows]
            summary[mode][field] = dict(
                n=len(values),
                median=statistics.median(values),
                min=min(values),
                max=max(values),
                stdev=statistics.pstdev(values),
            )
    result = dict(
        summary=summary,
        samples=records,
        first=first,
        instrumented=detail,
        all_equal=True,
        notes="Fresh processes, cached engines, warmed OS cache/JIT. Load wall excludes Python import/CUDA init. "
        "Worker wall includes validation/hash IO, not startup. CPU peak RSS includes imports/driver; CUDA peak "
        "excludes validation. Helper host timings are inclusive/overlapping and separately instrumented. "
        "No OS caches are dropped. Changed engine files fail the experiment.",
    )
    (args.out / "model_load.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
