"""Kernel timeline of the encoder (encoder-decoder models) or the CTC engine (OWSM-CTC) at batch 1: every
kernel of a CUDA-graph replay with its duration and the gap before it (torch.profiler / CUPTI), and the
single-utterance latency of ``transcribe``.

python scripts/b1_timeline.py --model .cache/models/owsm_v4_medium_1B [--seconds 8]
"""

import argparse
import json
import re
import statistics
import time
from collections import defaultdict

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio
from owsm_trt.runtime import CTCSession


def kind(name: str) -> str:
    """A coarse class of a kernel name."""
    for pat, k in (
        (r"dwconv1d", "plugin dwconv"),
        (r"softmax_rows|argmax_rows", "plugin ctc rows"),
        (r"conv1_relu|relu_flatten", "plugin subsampling"),
        (r"mha|fmha|flash", "attention (mha)"),
        (r"fprop|implicit_gemm|conv", "convolution"),
        (r"gemm|gemv|matmul", "gemm"),
        (r"^__myl|myl", "myelin elementwise/norm"),
        (r"[Rr]eformat|copy|Copy|transpose|Transpose", "copy/reformat"),
    ):
        if re.search(pat, name):
            return k
    return "other"


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pos", nargs="?", help="(gpu_eval.pbs pyfile stage: the model as argument)")
    ap.add_argument("--model")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--seconds", type=float, default=8.0, help="pick an utterance about this long")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--latency-iters", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.model = args.model or args.model_pos

    model = OWSM.from_pretrained(args.model, backend="tensorrt")
    items = json.load(open(args.data))
    waves = [load_audio(it["path"]) for it in items]
    i = min(range(len(waves)), key=lambda j: abs(len(waves[j]) / 16000 - args.seconds))
    wav = waves[i]
    print(f"utterance {items[i]['id']}: {len(wav) / 16000:.2f} s")
    feats = model._features([wav])
    t = model.tokenizer
    if model.cfg.ctc_only:
        sess = CTCSession(model.backend, model.cfg, 1, 1, "cuda", use_cuda_graph=True)
        prompt = torch.tensor([[t.na]], device="cuda")
        prefix = torch.tensor([[t.lang_id("eng"), t.task_id("asr")]], device="cuda")

        def call():
            sess(feats, prompt, prefix)

        what = "CTC engine"
    else:
        sess = model._session(1)

        def call():
            sess.encode(feats)

        what = "encoder"
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(args.iters):
        st.record()
        call()
        en.record()
        torch.cuda.synchronize()
        ms.append(st.elapsed_time(en))
    wall = statistics.median(ms)

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            call()
            torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    # split into calls at gaps > 50 us (calls are separated by host synchronisations)
    calls, cur = [], []
    for e in evs:
        if cur and e.time_range.start - cur[-1].time_range.end > 50:
            calls.append(cur)
            cur = []
        cur.append(e)
    if cur:
        calls.append(cur)
    calls = calls[1:-1] or calls  # (first and last may be partial)
    n_k = statistics.median(len(c) for c in calls)
    busy = statistics.median(sum(e.time_range.elapsed_us() for e in c) for c in calls) / 1000
    span = statistics.median((c[-1].time_range.end - c[0].time_range.start) for c in calls) / 1000
    print(f"== {what} at batch 1, feats {tuple(feats.shape)}: {wall:.3f} ms per call (events), "
          f"{n_k} kernels, {busy:.3f} ms in kernels, {span:.3f} ms first to last kernel, "
          f"{span - busy:.3f} ms of gaps ({1000 * (span - busy) / max(n_k - 1, 1):.2f} us per gap)")  # fmt: skip
    c = max(calls, key=len)
    by_kind = defaultdict(lambda: [0, 0.0])
    by_name = defaultdict(lambda: [0, 0.0])
    durs = []
    for e in c:
        d = e.time_range.elapsed_us()
        durs.append(d)
        by_kind[kind(e.name)][0] += 1
        by_kind[kind(e.name)][1] += d
        by_name[e.name[:100]][0] += 1
        by_name[e.name[:100]][1] += d
    print("  by kind (kernels, us):")
    for k, (n, d) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        print(f"    {n:5d} {d:9.1f}  {k}")
    print("  kernel durations (us): " + ", ".join(
        f"<{lim}: {sum(d < lim for d in durs)}" for lim in (3, 5, 10, 20, 50, 100, 1e9)))  # fmt: skip
    print("  top kernels by total time (count, us, us each):")
    for name, (n, d) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:40]:
        print(f"    {n:4d} {d:8.1f} {d / n:7.1f}  {name}")
    if args.out:
        rows = [(e.name, e.time_range.start - c[0].time_range.start, e.time_range.elapsed_us()) for e in c]
        json.dump(rows, open(args.out, "w"))

    # single-utterance latency: transcribe() of the file's waveform, end to end
    for _ in range(5):
        model.transcribe(wav)
    lat = []
    for _ in range(args.latency_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.transcribe(wav)
        torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - t0))
    print(f"transcribe latency ({len(wav) / 16000:.2f} s): median {statistics.median(lat):.2f} ms, "
          f"min {min(lat):.2f} ms over {len(lat)}")  # fmt: skip

    # where the latency goes: host time before the first GPU kernel, GPU kernels outside the engine
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.cuda.synchronize()
            model.transcribe(wav)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=60))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=60))


if __name__ == "__main__":
    main()
