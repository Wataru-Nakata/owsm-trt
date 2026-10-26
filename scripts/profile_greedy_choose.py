"""Same-logit CUDA-graph timing of serial and block-parallel greedy timestamp selection."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from profile_beam import save, stats

from owsm_trt import kernels
from owsm_trt.runtime import TimestampRules, capture_graph


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fused", action="store_true", help="Include accept in all variants; add fusion")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("GPU profiling requires a compute allocation")
    # The config supplies the measured model's vocabulary size; synthetic rules exercise all states.
    import yaml

    cfg = yaml.safe_load(args.config.read_text())
    vocab = len(cfg["token_list"])
    first, last = vocab - 1501, vocab - 1
    rules = TimestampRules(4, first, last, 2, vocab, torch.device("cuda"))
    result = {"vocab": vocab, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cases": {}}
    for rows, window in [(1, 1), (1, 4), (1, 16), (1, 64), (3, 4), (8, 1), (8, 4), (8, 16), (16, 16)]:
        torch.manual_seed(rows * window)
        logits = torch.randn(rows, window, vocab, device="cuda")
        inp = torch.zeros(rows, window + 4, device="cuda", dtype=torch.long)
        inp[:, :window] = torch.randint(5, vocab, (rows, window), device="cuda")
        inp[:, window + 1] = 1
        inp[:, window + 2] = first
        inp[:, window + 3] = -1
        notime = torch.zeros(rows, device="cuda", dtype=torch.bool)
        pred = torch.zeros(rows, window, device="cuda", dtype=torch.long)
        scratch = kernels.choose_workspace(rows, window, vocab, "cuda")
        graphs, cold, output = {}, {}, {}
        modes = ("serial", "split", "fused") if args.fused else ("serial", "split")
        res = torch.zeros(rows, 2, device="cuda", dtype=torch.long)
        for mode in modes:
            workspace = scratch if mode != "serial" else None

            def body():
                if mode == "fused":
                    kernels.choose_accept(logits, inp, notime, pred, rules, res, 2, workspace)
                else:
                    kernels.choose(logits, inp, notime, pred, rules, workspace=workspace)
                    if args.fused:
                        kernels.accept(pred, inp, res, 2)

            torch.cuda.synchronize()
            start = time.perf_counter()
            graphs[mode] = capture_graph(body)
            torch.cuda.synchronize()
            cold[mode] = (time.perf_counter() - start) * 1000
            graphs[mode].replay()
            output[mode] = (pred.clone(), res.clone())
        for mode in modes:
            assert all(torch.equal(a, b) for a, b in zip(output["serial"], output[mode]))
        samples = {mode: [] for mode in modes}
        for rep in range(15):
            for mode in modes if rep % 2 == 0 else modes[::-1]:
                a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(100):
                    graphs[mode].replay()
                z.record()
                z.synchronize()
                samples[mode].append(a.elapsed_time(z) / 100)
        result["cases"][f"r{rows}_w{window}"] = {
            "event_ms": {mode: stats(times) for mode, times in samples.items()},
            "samples_ms": samples,
            "cold_capture_ms": cold,
            "exact_tokens": True,
            "workspace_bytes": sum(x.numel() * x.element_size() for x in scratch),
            "scope": ("choose+accept" if args.fused else "choose only")
            + "; CUDA event per graph replay; excludes host decode loop",
        }
    save(args.out, result)
    print(json.dumps({k: v["event_ms"] for k, v in result["cases"].items()}))


if __name__ == "__main__":
    main()
