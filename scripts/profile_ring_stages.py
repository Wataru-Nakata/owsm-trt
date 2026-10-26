"""Profile unchanged Ring stages with real weights and fixed synthetic decoder state.

Run in a compute allocation. Uninstrumented CUDA-event samples, instrumented globaltimer
stages, and optional Nsight counters are separate measurements; none is request latency.
"""

import argparse
import importlib.metadata
import os
from pathlib import Path

import torch
from profile_beam import command, save, sha256, stats, wall

from owsm_trt import hub, megakernel, megakernel_ring
from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.pipeline import _load_decoder

STAGES = ("qkv", "self", "out", "crossq", "cross", "crossout", "ffn1", "ffn2")
CASES = {
    "plain": (1, False, False),
    "verify8": (8, False, True),
    "verify8_solo": (8, True, True),
    "verify32_solo": (32, True, True),
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
    ap.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    ap.add_argument("--repeat", type=int, default=11)
    ap.add_argument("--ncu", action="store_true", help="bracket one warmed launch per case for Nsight")
    ap.add_argument("--ncu-relaxed-poll", action="store_true", help="profile the experimental polling variant")
    args = ap.parse_args()
    if args.ncu_relaxed_poll and not args.ncu:
        ap.error("ncu-relaxed-poll is only for the separate counter pass")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("model work requires a compute allocation")
    dev = torch.device("cuda")
    if not megakernel_ring.available(dev):
        raise RuntimeError("Ring profiling requires the supported Hopper/Triton environment")
    files = hub.resolve(args.model)
    cfg = parse_config(load_espnet_config(files.config))
    decoder, load_ms = wall(lambda: _load_decoder(cfg, files.checkpoint))
    mega, prepare_ms = wall(lambda dec=decoder: megakernel.MegaDecoder(dec, dev))
    del decoder
    kernels = {"reference": megakernel_ring.RingDecoder(mega, relaxed_poll=args.ncu_relaxed_poll)}
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
            relaxed_poll=args.ncu_relaxed_poll,
            scope="real model weights, synthetic fixed self/cross state; no encoder, audio, or request timing",
            default_ring=dict(
                programs=kernels["reference"].P,
                solo=kernels["reference"].P_solo,
                warps=kernels["reference"].nw,
                chunk=kernels["reference"].ck,
            ),
        ),
    )
    torch.manual_seed(18)
    layers, h, d, c = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads, 128
    cross = [torch.randn(layers, 1, h, cfg.enc_frames, d, device=dev, dtype=torch.float16) * 0.1 for _ in range(2)]
    prefix = [torch.randn(layers, 1, h, c, d, device=dev, dtype=torch.float16) * 0.1 for _ in range(2)]
    summary = {}
    for name in args.cases:
        w, solo, argmax = CASES[name]
        tokens = torch.randint(cfg.vocab_size, (1, w), device=dev)
        offset = torch.full((1,), 32, device=dev, dtype=torch.long)
        runs, state = {}, {}
        for mode, kernel in kernels.items():
            k, v = (x.clone() for x in prefix)
            logits = torch.zeros(1, w, cfg.vocab_size, device=dev)
            new = torch.zeros(layers, 2, 1, h, w, d, device=dev, dtype=torch.float16)
            pred = torch.zeros(1, w, device=dev, dtype=torch.long) if argmax else None
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
                args.out / f"{name}.json", dict(completed=True, window=w, programs=runs["reference"].programs)
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
            window=w,
            programs=runs["reference"].programs,
            cross_splits=runs["reference"].const["NS"],
            argmax=argmax,
            cache=c,
            offset=32,
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
