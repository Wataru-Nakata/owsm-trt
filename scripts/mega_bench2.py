"""Where does the time of the persistent linear chain go? (stage 2 of the megakernel, round 2)

Instrumented chain of the 1B decoder's 72 linears at M <= 16: per program and linear, %globaltimer stamps at
the start of the GEMM, at its end (arrival at the barrier) and at the barrier exit. Variants: register
prefetch of the next linear's first weight chunk before the barrier (RP), more pipeline stages, 4 warps,
2 programs per SM.

python scripts/mega_bench2.py [MODEL]
"""

import os
import time
import traceback

import torch
import triton
import triton.language as tl
from mega_bench import _grid_sync, _prefetch_l2

from owsm_trt.megakernel import _grid_exit

MAXS = 1 << 22


@triton.jit
def _now():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _rows(N, pid, NU: tl.constexpr):
    nu = pid % NU
    return nu * N // NU, (nu + 1) * N // NU


@triton.jit
def _load_first(w, N, K: tl.constexpr, pid, NU: tl.constexpr, BN: tl.constexpr, KC: tl.constexpr):
    r0, r1 = _rows(N, pid, NU)
    rn = r0 + tl.arange(0, BN)
    return tl.load(w + rn[:, None] * K + tl.arange(0, KC)[None, :], mask=(rn < r1)[:, None], other=0.0)


@triton.jit
def _gemm1(pid, M, x, xs, w, b, N, K: tl.constexpr, y, ys, wpre, RELU: tl.constexpr, RP: tl.constexpr,
           BMT: tl.constexpr, NU: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KC: tl.constexpr,
           STAGES: tl.constexpr):  # fmt: skip
    """One unit per program (TB = 1): rows r0..r1 of y = x w^T + b; with RP the first KC columns of the
    weights are already in ``wpre``."""
    if pid < NU:
        r0, r1 = _rows(N, pid, NU)
        rn = r0 + tl.arange(0, BN)
        nmask = rn < r1
        tok = tl.arange(0, BMT)
        tmask = tok < M
        acc = tl.zeros([BN, BMT], dtype=tl.float32)
        k_start = 0
        if RP:
            kc = tl.arange(0, KC)
            x0 = tl.load(
                x + tok[None, :] * xs + kc[:, None], mask=tmask[None, :], other=0.0, cache_modifier=".cg"
            )
            acc = tl.dot(wpre, x0, acc)
            k_start = KC
        for k0 in tl.range(k_start, K, BK, num_stages=STAGES):
            rk = k0 + tl.arange(0, BK)
            wt = tl.load(w + rn[:, None] * K + rk[None, :], mask=nmask[:, None], other=0.0)
            xt = tl.load(
                x + tok[None, :] * xs + rk[:, None], mask=tmask[None, :], other=0.0, cache_modifier=".cg"
            )
            acc = tl.dot(wt, xt, acc)
        acc += tl.load(b + rn, mask=nmask, other=0.0).to(tl.float32)[:, None]
        if RELU:
            acc = tl.maximum(acc, 0.0)
        tl.store(
            y + tok[None, :] * ys + rn[:, None], acc.to(tl.float16), mask=nmask[:, None] & tmask[None, :]
        )


@triton.jit
def _stamp(ts, idx, TS: tl.constexpr):
    if TS:
        tl.store(ts + idx, _now())


@triton.jit
def _chain2(
    xa, xb, wq, bq, wo, bo, w1, b1, w2, b2, bar, err, done, M, L, ts,
    BMT: tl.constexpr, NU: tl.constexpr, BNQ: tl.constexpr, BNO: tl.constexpr, BN1: tl.constexpr,
    BN2: tl.constexpr, BK: tl.constexpr, BK2: tl.constexpr, STAGES: tl.constexpr, RP: tl.constexpr,
    KC: tl.constexpr, KC2: tl.constexpr, PF: tl.constexpr, TS: tl.constexpr, MAXS: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    D: tl.constexpr = 1024
    F: tl.constexpr = 4096
    S: tl.constexpr = 4096
    pq = _load_first(wq, 3 * D, D, pid, NU, BNQ, KC)
    for layer in range(L):
        lq = wq + layer.to(tl.int64) * 3 * D * D
        lo = wo + layer.to(tl.int64) * D * D
        l1 = w1 + layer.to(tl.int64) * F * D
        l2 = w2 + layer.to(tl.int64) * D * F
        nq = wq + tl.minimum(layer + 1, L - 1).to(tl.int64) * 3 * D * D
        base = (layer * 4) * P * 3 + pid * 3
        _stamp(ts, base, TS)
        _gemm1(
            pid,
            M,
            xa,
            S,
            lq,
            bq + layer * 3 * D,
            3 * D,
            D,
            xb,
            S,
            pq,
            False,
            RP,
            BMT,
            NU,
            BNQ,
            BK,
            KC,
            STAGES,
        )
        po = _load_first(lo, D, D, pid, NU, BNO, KC)
        if PF:
            r0, r1 = _rows(D, pid, NU)
            _prefetch_l2(lo + r0 * D, (r1 - r0) * D * 2, 1, 256)
        _stamp(ts, base + 1, TS)
        _grid_sync(bar, err, (layer * 4 + 1) * P, MAXS, 0, 0)
        _stamp(ts, base + 2, TS)
        base += P * 3
        _stamp(ts, base, TS)
        _gemm1(pid, M, xb, S, lo, bo + layer * D, D, D, xa, S, po, False, RP, BMT, NU, BNO, BK, KC, STAGES)
        p1 = _load_first(l1, F, D, pid, NU, BN1, KC)
        if PF:
            r0, r1 = _rows(F, pid, NU)
            _prefetch_l2(l1 + r0 * D, (r1 - r0) * D * 2, 1, 256)
        _stamp(ts, base + 1, TS)
        _grid_sync(bar, err, (layer * 4 + 2) * P, MAXS, 0, 0)
        _stamp(ts, base + 2, TS)
        base += P * 3
        _stamp(ts, base, TS)
        _gemm1(pid, M, xa, S, l1, b1 + layer * F, F, D, xb, S, p1, True, RP, BMT, NU, BN1, BK, KC, STAGES)
        p2 = _load_first(l2, D, F, pid, NU, BN2, KC2)
        if PF:
            r0, r1 = _rows(D, pid, NU)
            _prefetch_l2(l2 + r0 * F, (r1 - r0) * F * 2, 1, 256)
        _stamp(ts, base + 1, TS)
        _grid_sync(bar, err, (layer * 4 + 3) * P, MAXS, 0, 0)
        _stamp(ts, base + 2, TS)
        base += P * 3
        _stamp(ts, base, TS)
        _gemm1(pid, M, xb, S, l2, b2 + layer * D, D, F, xa, S, p2, False, RP, BMT, NU, BN2, BK2, KC2, STAGES)
        pq = _load_first(nq, 3 * D, D, pid, NU, BNQ, KC)
        if PF:
            r0, r1 = _rows(3 * D, pid, NU)
            _prefetch_l2(nq + r0 * D, (r1 - r0) * D * 2, 1, 256)
        _stamp(ts, base + 1, TS)
        _grid_sync(bar, err, (layer * 4 + 4) * P, MAXS, 0, 0)
        _stamp(ts, base + 2, TS)
    _grid_exit(bar, done, P)


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


def pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def analyse(ts, P, L):
    """ts [4L, P, 3] ns: (start, arrive, exit). Per linear kind: GEMM time (median / max over programs),
    barrier latency (first exit - last arrival), arrival skew (last - first arrival)."""
    t = ts.view(4 * L, P, 3).double()
    t = t - t[0, :, 0].min()
    gemm = t[:, :, 1] - t[:, :, 0]
    lat = t[:, :, 2].min(1).values - t[:, :, 1].max(1).values
    skew = t[:, :, 1].max(1).values - t[:, :, 1].min(1).values
    start_skew = t[:, :, 0].max(1).values - t[:, :, 0].min(1).values
    out = []
    for j, name in enumerate(("qkv", "out", "ffn1", "ffn2")):
        g = gemm[j::4]
        out.append(
            f"{name}: gemm med {g.median().item() / 1e3:.2f} max {g.max(1).values.mean().item() / 1e3:.2f} us, "
            f"start skew {start_skew[j::4].mean().item() / 1e3:.2f}, arrive skew {skew[j::4].mean().item() / 1e3:.2f}, "
            f"barrier latency {lat[j::4].mean().item() / 1e3:.2f} us"
        )
    total = (t[-1, :, 2].max() - t[0, :, 0].min()).item() / 1e3
    return f"total {total:.1f} us\n    " + "\n    ".join(out)


def main():
    P0 = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"
    torch.manual_seed(0)
    L, D, F = 18, 1024, 4096
    h = {"device": dev, "dtype": torch.float16}
    ws = [
        torch.randn(L, 3 * D, D, **h) / D**0.5, torch.randn(L, 3 * D, **h) * 0.1,
        torch.randn(L, D, D, **h) / D**0.5, torch.randn(L, D, **h) * 0.1,
        torch.randn(L, F, D, **h) / D**0.5, torch.randn(L, F, **h) * 0.1,
        torch.randn(L, D, F, **h) * (2 / F) ** 0.5, torch.randn(L, D, **h) * 0.1,
    ]  # fmt: skip
    bar = torch.zeros(1, dtype=torch.int32, device=dev)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    done = torch.zeros(1, dtype=torch.int32, device=dev)

    variants = {
        "base bk256 s4": {},
        "warps4": {"warps": 4},
        "bk256 s5": {"stages": 5},
        "bk512 s3 (ffn2 bk512)": {"bk": 512, "bk2": 512, "stages": 3},
        "RP kc256": {"rp": 1, "kc": 256, "kc2": 256},
        "RP kc512": {"rp": 1, "kc": 512, "kc2": 512},
        "RP kc1024 (whole K=1024 slab)": {"rp": 1, "kc": 1024, "kc2": 512},
        "PF line": {"pf": 1},
        "2 per SM, warps4": {"progs": 2 * P0, "warps": 4},
        "2 per SM, warps4, RP kc512": {"progs": 2 * P0, "warps": 4, "rp": 1, "kc": 512, "kc2": 512},
    }
    only = os.environ.get("MEGA_VARIANTS")
    for name, v in variants.items():
        if only and name not in only.split(";"):
            continue
        P = v.get("progs", P0)
        nu = P
        per = {n: -(-n // nu) for n in (D, 3 * D, F)}
        bn = {n: max(16, pow2(per[n])) for n in per}
        ts = torch.zeros(4 * L * P * 3, dtype=torch.int64, device=dev)
        for M in (1, 8, 16):
            xa = torch.randn(M, 4096, **h)
            xb = torch.zeros(M, 4096, **h)

            def launch(TS=False, M=M, xa=xa, xb=xb, ts=ts, v=v, P=P, nu=nu, bn=bn):
                return _chain2[(P,)](
                    xa, xb, *ws, bar, err, done, M, L, ts, BMT=16, NU=nu, BNQ=bn[3 * D], BNO=bn[D], BN1=bn[F],
                    BN2=bn[D], BK=v.get("bk", 256), BK2=v.get("bk2", 256), STAGES=v.get("stages", 4),
                    RP=v.get("rp", 0), KC=v.get("kc", 256), KC2=v.get("kc2", 256), PF=v.get("pf", 0), TS=TS,
                    MAXS=MAXS, num_warps=v.get("warps", 8),
                )  # fmt: skip

            try:
                err.zero_()
                k = launch()
                ms = bench(launch)
                launch(TS=True)
                torch.cuda.synchronize()
                line = f"[{name}] M={M}: {ms:.3f} ms  (regs {k.n_regs}, spills {k.n_spills}, smem {k.metadata.shared // 1024} KB, err {err.item()})"
                if M == 8:
                    line += "\n    " + analyse(ts, P, L)
                print(line, flush=True)
            except Exception:
                traceback.print_exc()
                print(f"[{name}] M={M}: FAILED", flush=True)
                break


if __name__ == "__main__":
    main()
