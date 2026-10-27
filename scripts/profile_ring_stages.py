"""Profile unchanged Ring stages with real weights and fixed synthetic or actual-prefill states.

Run in a compute allocation. Uninstrumented CUDA-event samples, instrumented globaltimer
stages, and optional Nsight counters are separate measurements; none is request latency.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256, stats, wall

from owsm_trt import OWSM, hub, megakernel, megakernel_ring
from owsm_trt.audio import load_audio
from owsm_trt.build import engine_tag
from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.pipeline import _load_decoder

STAGES = ("qkv", "self", "out", "crossq", "cross", "crossout", "ffn1", "ffn2")
CASES = {
    "plain": (1, 1, False, False),
    "plain_solo": (1, 1, True, True),
    "verify8": (1, 8, False, True),
    "verify8_solo": (1, 8, True, True),
    "verify32_solo": (1, 32, True, True),
    "rows3": (3, 1, False, False),
    "rows5": (5, 1, False, False),
    "rows8": (8, 1, False, False),
}


def stage_record(timestamps, layers, programs, argmax):
    n = 8 * layers + 2 + int(argmax)
    # [stage, CTA, record]; ns globaltimer, copied once after this diagnostic launch.
    t = timestamps[: n * programs * 8].cpu().reshape(n, programs, 8).double()
    start, exit_, arrive = (t[:, :, i] for i in (0, 1, 5))
    if not bool((start > 0).all() and (arrive >= start).all()):
        raise RuntimeError("missing or unordered Ring stage stamps")
    release = exit_.min(1).values
    release[-1] = arrive[-1].max()  # final stage has no barrier/release stamp
    previous = torch.cat((start[0].min().view(1), release[:-1]))
    elapsed = release - previous
    groups = {"embed": [0], "output": [8 * layers + 1]}
    groups.update({name: [1 + 8 * layer + i for layer in range(layers)] for i, name in enumerate(STAGES)})
    if argmax:
        groups["argmax"] = [n - 1]
    result = {}
    for name, idx in groups.items():
        work = arrive[idx] - start[idx]
        barriers = [i for i in idx if i < n - 1]
        result[name] = dict(
            elapsed_us=elapsed[idx].sum().item() / 1e3,
            mean_work_median_us=work.median(1).values.mean().item() / 1e3,
            mean_work_max_us=work.max(1).values.mean().item() / 1e3,
            mean_arrival_skew_us=(arrive[idx].max(1).values - arrive[idx].min(1).values).mean().item() / 1e3,
            mean_barrier_us=(
                (exit_[barriers].min(1).values - arrive[barriers].max(1).values).mean().item() / 1e3
                if barriers
                else None
            ),
        )
    return dict(total_us=elapsed.sum().item() / 1e3, stages=result)


def capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    return graph


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", nargs="+", choices=CASES, default=["plain", "verify8", "verify8_solo", "verify32_solo"])
    ap.add_argument("--repeat", type=int, default=11)
    ap.add_argument("--manifest", type=Path, help="use real audio/prefill states; S1 cases only")
    ap.add_argument("--ncu", action="store_true", help="bracket one warmed launch per case for Nsight")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("repeat must be positive")
    if args.manifest and any(CASES[name][1] != 1 for name in args.cases):
        ap.error("real-prefill profiling requires single-token cases")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("model work requires a compute allocation")
    dev = torch.device("cuda")
    if not megakernel_ring.available(dev):
        raise RuntimeError("Ring profiling requires the supported Hopper/Triton environment")
    files = hub.resolve(args.model)
    cfg = parse_config(load_espnet_config(files.config))
    model, waves, engines = None, None, {}
    if args.manifest:
        model, load_ms = wall(lambda: OWSM.from_pretrained(args.model, ctc_draft=False, use_cuda_graph=False))
        if not isinstance(model.backend.kernel, megakernel_ring.RingDecoder):
            raise RuntimeError("real-prefill profile requires the actual Ring main kernel")
        mega, prepare_ms = model.backend.kernel.m, None
        items = json.loads(args.manifest.read_text())[:max(CASES[name][0] for name in args.cases)]
        if len(items) < max(CASES[name][0] for name in args.cases):
            ap.error("manifest has too few inputs for requested rows")
        waves = [load_audio(it["path"]) for it in items]
        engine_dir = hub.cache_dir() / files.name.replace("/", "--") / engine_tag(model.opts)
        engines = {p.name: sha256(p) for p in engine_dir.glob("*.engine")}
    else:
        decoder, load_ms = wall(lambda: _load_decoder(cfg, files.checkpoint))
        mega, prepare_ms = wall(lambda dec=decoder: megakernel.MegaDecoder(dec, dev))
        del decoder
    kernels = {"reference": megakernel_ring.RingDecoder(mega)}
    if not args.ncu:
        kernels["instrumented"] = megakernel_ring.RingDecoder(mega, ts=True)
    save(
        args.out / "environment.json",
        dict(
            commit=os.environ.get("OWSM_PROFILE_COMMIT"),
            checkpoint_sha256=sha256(files.checkpoint),
            gpu=torch.cuda.get_device_name(),
            sm_count=torch.cuda.get_device_properties(dev).multi_processor_count,
            cuda=torch.version.cuda,
            versions={n: importlib.metadata.version(n) for n in ("torch", "triton")},
            nvidia_smi=command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"]),
            load_ms=load_ms,
            weight_prepare_ms=prepare_ms,
            ncu=args.ncu,
            scope=("real audio/prefill fixed states" if args.manifest else "real weights, synthetic fixed states")
                  + "; decoder-only timings exclude audio/encoder/prefill; not request latency",
            manifest_sha256=sha256(args.manifest) if args.manifest else None,
            engine_sha256=engines,
            default_ring=dict(
                programs=kernels["reference"].P,
                solo=kernels["reference"].P_solo,
                rows=kernels["reference"].P_rows,
                warps=kernels["reference"].nw,
                chunk=kernels["reference"].ck,
            ),
        ),
    )
    torch.manual_seed(18)
    layers, h, d, c = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads, 128
    synthetic = {}
    if model is None:
        for rows in sorted({CASES[name][0] for name in args.cases}):
            cross = [torch.randn(layers, rows, h, cfg.enc_frames, d, device=dev, dtype=torch.float16) * .1
                     for _ in range(2)]
            prefix = [torch.randn(layers, rows, h, c, d, device=dev, dtype=torch.float16) * .1
                      for _ in range(2)]
            synthetic[rows] = (prefix, cross)
    summary = {}
    for name in args.cases:
        rows, w, solo, argmax = CASES[name]
        if model is not None:
            session = model._session(rows)
            session.encode(model._features(waves[:rows]))
            primer = torch.tensor([model._primer("eng", "asr", False, None)] * rows, device=dev)
            logits, kv, c = session._prefill(primer, session.cross_k, session.cross_v)
            length = primer.shape[1]
            k, v = session.caches[c]
            k[:, :, :, :length].copy_(kv[:, 0])
            v[:, :, :, :length].copy_(kv[:, 1])
            session._rules_start(primer)
            tokens = session._rules_mask(logits).argmax(-1).unsqueeze(1)
            offset = torch.full((rows,), length, device=dev, dtype=torch.long)
            prefix, cross = (k, v), (session.cross_k, session.cross_v)
        else:
            length = 32
            tokens = torch.randint(cfg.vocab_size, (rows, w), device=dev)
            offset = torch.full((rows,), length, device=dev, dtype=torch.long)
            prefix, cross = synthetic[rows]
        runs, state = {}, {}
        for mode, kernel in kernels.items():
            k, v = (x.clone() for x in prefix)
            logits = torch.zeros(rows, w, cfg.vocab_size, device=dev)
            new = torch.zeros(layers, 2, rows, h, w, d, device=dev, dtype=torch.float16)
            pred = torch.zeros(rows, w, device=dev, dtype=torch.long) if argmax else None
            runs[mode] = kernel.step_fn(tokens, offset, k, v, *cross, logits, new, solo=solo, pred=pred)
            state[mode] = (logits, k, v, pred)
            for _ in range(3):
                runs[mode]()
            torch.cuda.synchronize()
            kernel.check()
        if args.ncu:
            torch.cuda.profiler.start()
            runs["reference"]()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            save(
                args.out / f"{name}.json", dict(completed=True, rows=rows, window=w, programs=runs["reference"].programs)
            )
            continue
        for a, b in zip(state["reference"], state["instrumented"]):
            if a is not None and not torch.equal(a, b):
                raise RuntimeError(f"instrumentation changed {name} outputs/cache")
        graphs = {mode: capture(run) for mode, run in runs.items()}
        samples = {mode: [] for mode in runs}
        stages = []
        for rep in range(args.repeat):
            for mode in list(runs)[:: 1 if rep % 2 == 0 else -1]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):
                    graphs[mode].replay()
                end.record()
                end.synchronize()
                samples[mode].append(start.elapsed_time(end) * 1000 / 30)
            kernel = kernels["instrumented"]
            kernel.ts.zero_()
            runs["instrumented"]()
            torch.cuda.synchronize()
            kernel.check()
            stages.append(stage_record(kernel.ts, layers, runs["instrumented"].programs, argmax))
        summary[name] = dict(
            rows=rows,
            window=w,
            programs=runs["reference"].programs,
            cross_splits=runs["reference"].const["NS"],
            argmax=argmax,
            cache=c,
            offset=length,
            real_prefill=args.manifest is not None,
            outputs_and_cache_exact=True,
            kernel_us={mode: stats(x) for mode, x in samples.items()},
            samples_us=samples,
            stage_samples=stages,
            instrumented_stages_are_not_uninstrumented_kernel_time=True,
        )
        save(args.out / "summary.json", summary)
        print(f"{name}: {summary[name]['kernel_us']}", flush=True)


if __name__ == "__main__":
    main()
