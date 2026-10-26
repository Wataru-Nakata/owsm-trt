"""Synthetic plain-greedy selection graph timing; separate from model/end-to-end latency."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from profile_beam import save, stats, wall

from owsm_trt.runtime import GreedySession, TimestampRules, capture_graph


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and os.uname().nodename.startswith("login"):
        ap.error("GPU profiling requires a compute allocation")
    vocab = len(yaml.safe_load(args.config.read_text())["token_list"])
    cfg = SimpleNamespace(vocab_size=vocab, dec_layers=1, dec_heads=1, d_model=32, enc_frames=1)
    dev = torch.device("cuda")

    class Backend:
        def step_fn(self, *args, **kwargs):
            return lambda: None  # only selection is executed in this synthetic benchmark

    result = dict(vocab=vocab, gpu=torch.cuda.get_device_name(), torch=torch.__version__, cases={})
    for rows in (1, 8, 16):
        for mode in ("free", "notime", "timed"):
            rules = None if mode == "free" else TimestampRules(4, vocab - 1501, vocab - 1, 2, vocab, dev)
            sessions, graphs, cold = {}, {}, {}
            for name, fused in (("torch", False), ("fused", True)):
                sess = GreedySession(
                    Backend(),
                    cfg,
                    rows,
                    512,
                    2,
                    torch.float32,
                    dev,
                    use_cuda_graph=False,
                    rules=rules,
                    fused_select=fused,
                )
                sess.logits.zero_()
                sess.logits[:, :, 5000] = 100
                sess.t_notime.fill_(mode == "notime")
                sess.t_count.fill_(1)
                sess.t_last.fill_(vocab - 1501)
                sess.tok.fill_(5000)
                sess.pos.fill_(10)
                sessions[name] = sess
                graphs[name], cold[name] = wall(lambda s=sess: capture_graph(lambda: s._select(512)))
            for key in ("tok", "done", "seq", "t_first", "t_count", "t_last"):
                assert torch.equal(getattr(sessions["torch"], key), getattr(sessions["fused"], key))
            samples = {name: [] for name in graphs}
            for repeat in range(15):
                for name in ("torch", "fused") if repeat % 2 == 0 else ("fused", "torch"):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / 100)
            result["cases"][f"r{rows}_{mode}"] = dict(
                event_ms={name: stats(values) for name, values in samples.items()},
                samples_ms=samples,
                cold_capture_ms=cold,
                exact_state=True,
                workspace_bytes=sum(x.numel() * x.element_size() for x in sessions["fused"].select_workspace),
                scope="selection only; CUDA event per graph replay, no decoder or host decode loop",
            )
    save(args.out, result)
    print(json.dumps({k: v["event_ms"] for k, v in result["cases"].items()}))


if __name__ == "__main__":
    main()
