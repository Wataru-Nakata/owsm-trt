"""The decoder megakernel (owsm_trt.megakernel.MegaDecoder) against Decoder.step on the GPU: accuracy (fp32
step as truth, fp16 step for comparison) and time per call with CUDA graphs, for decode (S=1) and verify
(S=2..32) calls of 1..16 rows.

python scripts/mega_decoder_bench.py MODEL   (env: MEGA_OPTS="PF=1;num_warps=8" option sets to time,
                                              MEGA_CASES=1x1:16x16 (rows x window), MEGA_PROFILE=1x8 (breakdown),
                                              MEGA_RANDOM=1 random weights with the model's shapes)
"""

import os
import sys
import time
import traceback

import torch

from owsm_trt import hub
from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.megakernel import MegaDecoder
from owsm_trt.modeling import Decoder, load_model


def bench(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1000)
    return best


STAGE_NAMES = ("qkv", "self", "out", "crossq", "cross", "crossout", "ffn1", "ffn2")


def breakdown(ts, L, P):
    """Wall time per stage type (from the previous barrier's release to this one's), the median / max
    work time of the programs (GEMMs: until the loads are consumed / the epilogue is done), barrier
    latency (first exit - last arrival) and arrival skew. ts [stages, P, 6]: start, exit, -, loads done,
    epilogue done, arrival."""
    n = 8 * L + 2
    t = ts[: n * P * 6].view(n, P, 6).double()
    t0 = t[0, :, 0].min()
    start, exit_, loads, epi, arrive = (t[:, :, i] - t0 for i in (0, 1, 3, 4, 5))
    rel = exit_.min(1).values
    rel[-1] = arrive[-1].max()  # the last stage has no barrier: its last program finishing
    wall = torch.cat([rel[:1], rel[1:] - rel[:-1]])
    work = arrive - start
    gl = (loads - start).clamp(min=0)
    ge = (epi - loads).clamp(min=0)
    lat = exit_.min(1).values - arrive.max(1).values
    skew = arrive.max(1).values - arrive.min(1).values
    rows = {"embed": [0], "output": [8 * L + 1]}
    for i, name in enumerate(STAGE_NAMES):
        rows[name] = [1 + 8 * layer + i for layer in range(L)]
    out = [f"total {rel[-1].item() / 1e3:.1f} us"]
    for name, idx in rows.items():
        idx = torch.tensor(idx)
        line = (
            f"{name:9s} wall {wall[idx].sum().item() / 1e3:7.1f} us ({wall[idx].mean().item() / 1e3:5.2f}/stage)"
            f"  work med {work[idx].median(1).values.mean().item() / 1e3:5.2f} max "
            f"{work[idx].max(1).values.mean().item() / 1e3:5.2f}"
        )
        if name in ("qkv", "out", "crossq", "crossout", "ffn1", "ffn2", "output") and loads[idx].max() > 0:
            line += (
                f" [loads {gl[idx].median(1).values.mean().item() / 1e3:5.2f}"
                f" epilogue {ge[idx].median(1).values.mean().item() / 1e3:5.2f}]"
            )
        line += f"  barrier lat {lat[idx].mean().item() / 1e3:5.2f} skew {skew[idx].mean().item() / 1e3:5.2f}"
        out.append(line)
    return "\n      ".join(out)


def one_stage(run, ts, L, P, reps=40):
    """Each stage type alone, repeated (its code stays hot): wall time per repetition, work, GEMM loads."""
    out = []
    names = [("embed", 0)] + [(n, 1 + i) for i, n in enumerate(STAGE_NAMES)] + [("output", 8 * L + 1)]
    for name, g in names:
        ts.zero_()
        run.one(g, reps)
        torch.cuda.synchronize()
        t = ts[(g + 5) * P * 6 : (g + reps) * P * 6].view(reps - 5, P, 6).double()  # skip 5 warm-up reps
        exit_ = t[:, :, 1].min(1).values
        wall = (exit_[-1] - exit_[0]).item() / (reps - 6) / 1e3
        work = (t[:, :, 5] - t[:, :, 0]).median(1).values.mean().item() / 1e3
        line = f"{name:9s} alone: {wall:5.2f} us/stage, work med {work:5.2f}"
        if t[:, :, 3].max() > 0:
            gl = (t[:, :, 3] - t[:, :, 0]).median(1).values.mean().item() / 1e3
            ge = (t[:, :, 4] - t[:, :, 3]).median(1).values.mean().item() / 1e3
            line += f" [loads {gl:5.2f} epilogue {ge:5.2f}]"
        out.append(line)
    return "\n      ".join(out)


def parse_opts(text):
    out = []
    for part in text.split(";"):
        d = {}
        for kv in filter(None, part.split(",")):
            k, v = kv.split("=")
            d[k] = int(v)
        d.setdefault("TS", 1)
        out.append(d)
    return out


@torch.inference_mode()
def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else ".cache/models/owsm_v4_medium_1B"
    files = hub.resolve(model_dir)
    cfg = parse_config(load_espnet_config(files.config))
    dev = torch.device("cuda")
    if os.environ.get("MEGA_RANDOM") == "1":
        torch.manual_seed(0)
        dec32 = Decoder(cfg).to(dev).eval()
        for layer in dec32.decoders:
            layer.self_attn.fuse_qkv()
    else:
        dec32 = load_model(cfg, str(files.checkpoint), "cpu", torch.float32).decoder.to(dev)
    import copy

    dec16 = copy.deepcopy(dec32).half()
    L, H, dk, T = cfg.dec_layers, cfg.dec_heads, cfg.d_model // cfg.dec_heads, cfg.enc_frames
    V = cfg.vocab_size
    torch.manual_seed(1)
    rows = 16
    # plausible cross-attention keys/values: the layers' projections of random "encoder output"
    enc = torch.randn(rows, T, cfg.d_model, device=dev) * 0.5
    ckv = dec32.cross_kv(enc)  # [L, 2, B, H, T, dk]
    ck32, cv32 = ckv[:, 0].contiguous(), ckv[:, 1].contiguous()
    ck16, cv16 = ck32.half(), cv32.half()
    C = 128
    # a plausible self-attention cache: keys/values of a random prefix run through the decoder
    prefix = torch.randint(0, V, (rows, C - 32), device=dev)
    sk32 = torch.zeros(L, rows, H, C, dk, device=dev)
    sv32 = torch.zeros(L, rows, H, C, dk, device=dev)
    _, kv = dec32.step(prefix, torch.zeros(rows, dtype=torch.long, device=dev), list(sk32), list(sv32),
                       list(ck32), list(cv32))  # fmt: skip
    sk32[:, :, :, : C - 32] = kv[:, 0]
    sv32[:, :, :, : C - 32] = kv[:, 1]
    sk16, sv16 = sk32.half(), sv32.half()

    opts_list = parse_opts(os.environ.get("MEGA_OPTS", "BK=128,STAGES=3"))
    mks = [MegaDecoder(dec16, dev, **o) for o in opts_list]
    print(f"model {model_dir}: L={L} d={cfg.d_model} H={H} V={V} T={T}; option sets {opts_list}", flush=True)

    cases = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (4, 8), (8, 1), (8, 8), (16, 1), (16, 4), (16, 8),
             (16, 16)]  # fmt: skip
    if os.environ.get("MEGA_CASES"):
        cases = [tuple(int(v) for v in c.split("x")) for c in os.environ["MEGA_CASES"].split(":")]
    profile = os.environ.get("MEGA_PROFILE", "1x1:1x8:16x16").split(":")
    for R, S in cases:
        tokens = torch.randint(0, V, (R, S), device=dev)
        offset = torch.randint(1, C - 32 - S, (R,), device=dev)
        ref32, kv32 = dec32.step(tokens, offset, list(sk32[:, :R]), list(sv32[:, :R]), list(ck32[:, :R]),
                                 list(cv32[:, :R]))  # fmt: skip
        ref16, _ = dec16.step(tokens, offset, list(sk16[:, :R]), list(sv16[:, :R]), list(ck16[:, :R]),
                              list(cv16[:, :R]))  # fmt: skip
        line = f"R={R:2d} S={S:2d}:"
        for mk, o in zip(mks, opts_list):
            try:
                k = sk16[:, :R].clone()
                v = sv16[:, :R].clone()
                ck, cv = ck16[:, :R].contiguous(), cv16[:, :R].contiguous()
                logits = torch.zeros(R, S, V, device=dev)
                new_kv = torch.zeros(L, 2, R, H, S, dk, device=dev, dtype=torch.float16)
                mk.err.zero_()
                run = mk.step_fn(tokens, offset, k, v, ck, cv, logits, new_kv)
                run()
                torch.cuda.synchronize()
                e_mk = (logits - ref32).abs().max().item()
                e_16 = (ref16 - ref32).abs().max().item()
                a_mk = (logits.argmax(-1) == ref32.argmax(-1)).sum().item()
                a_16 = (ref16.argmax(-1) == ref32.argmax(-1)).sum().item()
                e_kv = (new_kv.float() - kv32).abs().max().item()
                slots = (offset[:, None] + torch.arange(S, device=dev))[None, :, None, :, None].expand(
                    L, R, H, S, dk
                )
                rk = sk16[:, :R].clone()
                rk.scatter_(3, slots, new_kv[:, 0])
                cache_ok = torch.equal(k, rk)
                # time on the bucket the TensorRT numbers use (64 slots, offset 20)
                k64 = torch.zeros(L, R, H, 64, dk, device=dev, dtype=torch.float16)
                v64 = torch.zeros_like(k64)
                off20 = torch.full((R,), 20, dtype=torch.long, device=dev)
                run64 = mk.step_fn(tokens, off20, k64, v64, ck, cv, logits, new_kv)
                ms = bench(run64)
                if mk.opts.get("TS") and f"{R}x{S}" in profile:
                    run64()
                    torch.cuda.synchronize()
                    prof = "\n      " + breakdown(mk.ts, L, mk.programs)
                    if os.environ.get("MEGA_ONE") == "1":
                        prof += "\n      " + one_stage(run64, mk.ts, L, mk.programs)
                else:
                    prof = ""
                line += (
                    f"\n   {o}: {ms:.3f} ms | logits err {e_mk:.3f} (fp16 step {e_16:.3f}), argmax = fp32 "
                    f"{a_mk}/{R * S} (fp16 step {a_16}), new_kv err {e_kv:.3f}, cache {'ok' if cache_ok else 'WRONG'}"
                    f", err flag {mk.err.item()}{prof}"
                )
            except Exception:
                traceback.print_exc()
                line += f"\n   {o}: FAILED"
        print(line, flush=True)


if __name__ == "__main__":
    main()
