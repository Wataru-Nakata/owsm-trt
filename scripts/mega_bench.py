"""Megakernel building blocks on the GPU (stages 1-2 of the persistent decoder kernel):

1. grid barrier: cost per barrier with 1 program per SM, empty and with 32 KB of loads per program between
   barriers; the arrival counter is checked (one arrival per program and barrier).
2. a persistent chain of the 1B decoder's 72 linears (18 x [QKV 3072x1024, out 1024x1024, FFN1 4096x1024,
   FFN2 1024x4096], random weights) at M = 1, 8, 16, 256 rows, one grid barrier between linears; tile
   configurations and L2 prefetch of the next linear's weights are swept.

python scripts/mega_bench.py [MODEL]  (env: MEGA_PARTS=barrier,chain  MEGA_QUICK=1)
"""

import os
import sys
import time
import traceback

import torch
import triton
import triton.language as tl

from owsm_trt.megakernel import _grid_exit, _ld_acquire

# ------------------------------------------------------------------------------- building blocks (variants)


@triton.jit
def _arrive(bar, ARRIVE: tl.constexpr):
    if ARRIVE == 0:
        tl.atomic_add(bar, 1, sem="release", scope="gpu")  # scalar atomics run on one thread
    else:
        tl.inline_asm_elementwise(
            """{
            .reg .pred p;
            .reg .b32 t;
            mov.u32 t, %tid.x;
            setp.eq.u32 p, t, 0;
            @p fence.acq_rel.gpu;
            @p red.relaxed.gpu.global.add.s32 [$1], 1;
            mov.u32 $0, 0;
            }""",
            "=r,l",
            [bar],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _grid_sync(bar, err, target, MAX_SPIN: tl.constexpr, ARRIVE: tl.constexpr, SPIN: tl.constexpr):
    """Grid barrier variants: arrival by atom.release (0) or fence + red (1); spin with ld.acquire (0) or
    volatile loads + a final acquire (1)."""
    tl.debug_barrier()
    _arrive(bar, ARRIVE)
    if SPIN == 0:
        v = _ld_acquire(bar)
    else:
        v = tl.load(bar, volatile=True)
    e = tl.load(err, volatile=True)
    n = tl.full([], 0, tl.int32)
    while (v < target) & (e == 0) & (n < MAX_SPIN):
        if SPIN == 0:
            v = _ld_acquire(bar)
        else:
            v = tl.load(bar, volatile=True)
        e = tl.load(err, volatile=True)
        n += 1
    if SPIN != 0:
        tl.atomic_add(bar, 0, sem="acquire", scope="gpu")
    if n >= MAX_SPIN:
        tl.atomic_xchg(err, 1, sem="relaxed", scope="gpu")
    tl.debug_barrier()


@triton.jit
def _prefetch_l2(base, nbytes, PF: tl.constexpr, NT: tl.constexpr):
    """Ask L2 for ``nbytes`` bytes at ``base`` (an fp16 pointer). PF=1: one prefetch per 128-byte line,
    PF=2: one bulk prefetch (sm90) per thread of an equal share."""
    if PF == 1:
        lines = (nbytes + 127) // 128
        i = tl.arange(0, NT)
        for i0 in range(0, lines, NT):
            li = tl.minimum(i0 + i, lines - 1)
            tl.inline_asm_elementwise(
                "prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [base + li * 64], dtype=tl.int32,
                is_pure=False, pack=1,
            )  # fmt: skip
    elif PF == 2:
        share = ((nbytes + NT * 16 - 1) // (NT * 16)) * 16  # bytes per thread, a multiple of 16
        i = tl.arange(0, NT)
        start = tl.minimum(i * share, nbytes)
        size = tl.minimum(share, nbytes - start)
        tl.inline_asm_elementwise(
            """{
            .reg .pred p;
            setp.gt.u32 p, $2, 0;
            @p cp.async.bulk.prefetch.L2.global [$1], $2;
            mov.u32 $0, 0;
            }""",
            "=r,l,r",
            [base + start // 2, size.to(tl.uint32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _gemm(pid, P, M, x, x_stride, w, bias, N, K: tl.constexpr, y, y_stride, RELU: tl.constexpr,
          BMT: tl.constexpr, TB: tl.constexpr, NU: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          STAGES: tl.constexpr):  # fmt: skip
    """y[m, n] = act(sum_k x[m, k] w[n, k] + bias[n]) (fp16), units of BMT tokens x a range of features."""
    for u in range(pid, TB * NU, P):
        tb = u // NU
        nu = u % NU
        r0 = nu * N // NU
        r1 = (nu + 1) * N // NU
        tok = tb * BMT + tl.arange(0, BMT)
        tmask = tok < M
        for n0 in range(r0, r1, BN):
            rn = n0 + tl.arange(0, BN)
            nmask = rn < r1
            acc = tl.zeros([BN, BMT], dtype=tl.float32)
            for k0 in tl.range(0, K, BK, num_stages=STAGES):
                rk = k0 + tl.arange(0, BK)
                wt = tl.load(w + rn[:, None] * K + rk[None, :], mask=nmask[:, None], other=0.0)
                xt = tl.load(x + tok[None, :] * x_stride + rk[:, None], mask=tmask[None, :], other=0.0,
                             cache_modifier=".cg")  # fmt: skip
                acc = tl.dot(wt, xt, acc)
            acc += tl.load(bias + rn, mask=nmask, other=0.0).to(tl.float32)[:, None]
            if RELU:
                acc = tl.maximum(acc, 0.0)
            tl.store(y + tok[None, :] * y_stride + rn[:, None], acc.to(tl.float16),
                     mask=nmask[:, None] & tmask[None, :])  # fmt: skip


MAX_SPIN = 1 << 22


def bench(fn, iters=50, warmup=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
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


# ----------------------------------------------------------------------------------------------- barrier


@triton.jit
def _bar_kernel(bar, err, done, buf, nbuf, n_iter, out, LOAD: tl.constexpr, SYNC: tl.constexpr,
                ARRIVE: tl.constexpr, SPIN: tl.constexpr, RESET: tl.constexpr, MAXS: tl.constexpr):  # fmt: skip
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    acc = tl.zeros([LOAD], dtype=tl.float32)
    for i in range(n_iter):
        if LOAD > 1:
            offs = ((i * P + pid).to(tl.int64) * LOAD + tl.arange(0, LOAD)) % nbuf
            acc += tl.load(buf + offs).to(tl.float32)
        if SYNC:
            _grid_sync(bar, err, (i + 1) * P, MAXS, ARRIVE, SPIN)
    tl.store(out + pid, tl.sum(acc, 0))
    if RESET:
        _grid_exit(bar, done, P)


def run_barrier(P):
    dev = "cuda"
    bar = torch.zeros(1, dtype=torch.int32, device=dev)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    done = torch.zeros(1, dtype=torch.int32, device=dev)
    buf = torch.randn(256 << 20, device=dev, dtype=torch.float16)  # 512 MB: loads come from DRAM
    out = torch.zeros(P, device=dev)
    n = 2000
    print(f"== grid barrier, {P} programs, {n} barriers per launch", flush=True)
    for arrive, spin in ((0, 0), (1, 0), (0, 1), (1, 1)):
        name = f"arrive={'atom' if arrive == 0 else 'red'} spin={'ld.acquire' if spin == 0 else 'volatile'}"
        try:
            # arrival count check (no reset)
            bar.zero_()
            err.zero_()
            _bar_kernel[(P,)](bar, err, done, buf, buf.numel(), 10, out, LOAD=1, SYNC=True, ARRIVE=arrive,
                              SPIN=spin, RESET=False, MAXS=MAX_SPIN, num_warps=8)  # fmt: skip
            torch.cuda.synchronize()
            count_ok = bar.item() == 10 * P and err.item() == 0
            bar.zero_()
            res = []
            for load in (1, 16384):
                for sync in (True, False):
                    if load == 1 and not sync:
                        continue

                    def fn(load=load, sync=sync, nn=n):
                        _bar_kernel[(P,)](bar, err, done, buf, buf.numel(), nn, out, LOAD=load, SYNC=sync,
                                          ARRIVE=arrive, SPIN=spin, RESET=True, MAXS=MAX_SPIN, num_warps=8)  # fmt: skip

                    res.append(bench(fn, iters=5) / n * 1000)  # us per iteration
            print(
                f"{name}: arrivals {'ok' if count_ok else 'WRONG ' + str(bar.item())}, err={err.item()}; "
                f"empty barrier {res[0]:.2f} us; 32 KB loads + barrier {res[1]:.2f} us vs loads only "
                f"{res[2]:.2f} us -> barrier under load {res[1] - res[2]:.2f} us",
                flush=True,
            )
        except Exception:
            traceback.print_exc()
            print(f"{name}: FAILED", flush=True)


# ------------------------------------------------------------------------------------------------- chain


@triton.jit
def _slab(w, N, K: tl.constexpr, pid, TB: tl.constexpr, NU: tl.constexpr, PF: tl.constexpr):
    if PF > 0:
        if pid < TB * NU:
            nu = pid % NU
            r0 = nu * N // NU
            r1 = (nu + 1) * N // NU
            _prefetch_l2(w + r0.to(tl.int64) * K, (r1 - r0) * K * 2, PF, 256)


@triton.jit
def _chain_kernel(
    xa, xb, wq, bq, wo, bo, w1, b1, w2, b2, bar, err, done, M, L, st,
    BMT: tl.constexpr, TB: tl.constexpr, NU: tl.constexpr,
    BNQ: tl.constexpr, BNO: tl.constexpr, BN1: tl.constexpr, BN2: tl.constexpr,
    BK: tl.constexpr, STAGES: tl.constexpr, PF: tl.constexpr, PFD: tl.constexpr, MAXS: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    D: tl.constexpr = 1024
    F: tl.constexpr = 4096
    S: tl.constexpr = 4096  # row stride of xa / xb
    if PF > 0:
        _slab(wq, 3 * D, D, pid, TB, NU, PF)
        if PFD > 1:
            _slab(wo, D, D, pid, TB, NU, PF)
    for layer in range(L):
        lq = wq + layer.to(tl.int64) * 3 * D * D
        lo = wo + layer.to(tl.int64) * D * D
        l1 = w1 + layer.to(tl.int64) * F * D
        l2 = w2 + layer.to(tl.int64) * D * F
        nq = wq + (layer + 1).to(tl.int64) * 3 * D * D
        no = wo + (layer + 1).to(tl.int64) * D * D
        more = layer + 1 < L
        # QKV: xa -> xb
        if PF > 0:
            _slab(lo if PFD == 1 else l1, D if PFD == 1 else F, D, pid, TB, NU, PF)
        _gemm(pid, P, M, xa, S, lq, bq + layer * 3 * D, 3 * D, D, xb, S, False, BMT, TB, NU, BNQ, BK, STAGES)  # fmt: skip
        _grid_sync(bar, err, (layer * 4 + 1) * P, MAXS, 0, 0)
        # out: xb[:, :D] -> xa
        if PF > 0:
            if PFD == 1:
                _slab(l1, F, D, pid, TB, NU, PF)
            else:
                _slab(l2, D, F, pid, TB, NU, PF)
        _gemm(pid, P, M, xb, S, lo, bo + layer * D, D, D, xa, S, False, BMT, TB, NU, BNO, BK, STAGES)  # fmt: skip
        _grid_sync(bar, err, (layer * 4 + 2) * P, MAXS, 0, 0)
        # FFN1: xa -> xb (ReLU)
        if PF > 0:
            if PFD == 1:
                _slab(l2, D, F, pid, TB, NU, PF)
            elif more:
                _slab(nq, 3 * D, D, pid, TB, NU, PF)
        _gemm(pid, P, M, xa, S, l1, b1 + layer * F, F, D, xb, S, True, BMT, TB, NU, BN1, BK, STAGES)  # fmt: skip
        _grid_sync(bar, err, (layer * 4 + 3) * P, MAXS, 0, 0)
        # FFN2: xb -> xa
        if PF > 0:
            if more:
                if PFD == 1:
                    _slab(nq, 3 * D, D, pid, TB, NU, PF)
                else:
                    _slab(no, D, D, pid, TB, NU, PF)
        _gemm(pid, P, M, xb, S, l2, b2 + layer * D, D, F, xa, S, False, BMT, TB, NU, BN2, BK, STAGES)  # fmt: skip
        _grid_sync(bar, err, (layer * 4 + 4) * P, MAXS, 0, 0)
    _grid_exit(bar, done, P)


def chain_ref(x, ws, M):
    import torch.nn.functional as F

    wq, bq, wo, bo, w1, b1, w2, b2 = ws
    h = x[:M, :1024]
    for i in range(wq.shape[0]):
        q = F.linear(h, wq[i], bq[i])
        h = F.linear(q[:, :1024], wo[i], bo[i])
        h = F.relu(F.linear(h, w1[i], b1[i]))
        h = F.linear(h, w2[i], b2[i])
    return h


def pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def tiles(M, P, bmt=None):
    """(BMT, TB, NU) for M rows."""
    mp = max(16, pow2(M))
    if bmt is None:
        bmt = min(mp, 64)
    tb = -(-M // bmt)
    return bmt, tb, P // tb


def run_chain(P, quick):
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
    mb = sum(t.numel() * 2 for t in ws) / 1e6
    print(f"== chain of {4 * L} linears, {mb:.0f} MB weights ({mb / 4e3 * 1e3:.3f} ms at 4 TB/s)", flush=True)
    bar = torch.zeros(1, dtype=torch.int32, device=dev)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    done = torch.zeros(1, dtype=torch.int32, device=dev)
    st = torch.zeros(4096, device=dev)

    def launch(M, cfg, xa, xb):
        bmt, tb, nu = cfg["tiles"]
        per = {n: -(-n // nu) for n in (D, 3 * D, F)}
        cap = cfg.get("bn_cap", 128)
        bn = {n: min(cap, max(16, pow2(per[n]))) for n in per}
        return _chain_kernel[(P,)](
            xa, xb, *ws, bar, err, done, M, L, st, BMT=bmt, TB=tb, NU=nu, BNQ=bn[3 * D], BNO=bn[D],
            BN1=bn[F], BN2=bn[D], BK=cfg["bk"], STAGES=cfg["stages"], PF=cfg.get("pf", 0),
            PFD=cfg.get("pfd", 1), MAXS=MAX_SPIN, num_warps=cfg.get("warps", 8),
        )  # fmt: skip

    def measure(M, cfg):
        xa = torch.zeros(M, 4096, **h)
        xb = torch.zeros(M, 4096, **h)
        x0 = torch.randn(M, 4096, **h)
        xa.copy_(x0)
        err.zero_()
        k = launch(M, cfg, xa, xb)
        torch.cuda.synchronize()
        ref = chain_ref(x0, ws, M).float()
        got = xa[:, :D].float()
        diff = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        ms = bench(lambda: launch(M, cfg, xa, xb))
        meta = f"regs {k.n_regs} spills {k.n_spills} smem {k.metadata.shared // 1024} KB"
        return ms, diff, err.item(), meta

    small = [
        {"bk": bk, "stages": s} for bk, s in ((128, 3), (256, 2), (256, 3), (256, 4), (512, 2), (512, 3))
    ]
    if quick:
        small = small[2:3]
    results = {}
    for cfg in small:
        cfg["tiles"] = tiles(16, P)
        line = f"M<=16 bk={cfg['bk']} stages={cfg['stages']}:"
        try:
            for M in (1, 8, 16):
                ms, diff, e, meta = measure(M, cfg)
                results.setdefault(M, []).append((ms, dict(cfg)))
                line += f"  M={M} {ms:.3f} ms (rel err {diff:.1e}{', ERR FLAG' if e else ''})"
            print(line + "  " + meta, flush=True)
        except Exception:
            traceback.print_exc()
            print(line + " FAILED", flush=True)
    if results:
        best = min(results[8], key=lambda r: r[0])[1]
        for pf, pfd in ((1, 1), (2, 1), (1, 2), (2, 2)):
            cfg = dict(best, pf=pf, pfd=pfd)
            line = f"M<=16 bk={cfg['bk']} stages={cfg['stages']} prefetch={'line' if pf == 1 else 'bulk'} dist={pfd}:"
            try:
                for M in (1, 8, 16):
                    ms, diff, e, meta = measure(M, cfg)
                    line += f"  M={M} {ms:.3f} ms (rel err {diff:.1e}{', ERR FLAG' if e else ''})"
                print(line + "  " + meta, flush=True)
            except Exception:
                traceback.print_exc()
                print(line + " FAILED", flush=True)
    big = [
        {"bk": 64, "stages": 3, "tiles": tiles(256, P, 64)},
        {"bk": 64, "stages": 3, "tiles": tiles(256, P, 128), "bn_cap": 64},
        {"bk": 64, "stages": 4, "tiles": tiles(256, P, 64), "pf": 1},
        {"bk": 128, "stages": 3, "tiles": tiles(256, P, 64), "bn_cap": 64},
    ]
    if quick:
        big = big[:1]
    for cfg in big:
        line = f"M=256 tiles={cfg['tiles']} bk={cfg['bk']} stages={cfg['stages']} pf={cfg.get('pf', 0)}:"
        try:
            ms, diff, e, meta = measure(256, cfg)
            print(f"{line} {ms:.3f} ms (rel err {diff:.1e}{', ERR FLAG' if e else ''})  {meta}", flush=True)
        except Exception:
            traceback.print_exc()
            print(line + " FAILED", flush=True)


def main():
    P = torch.cuda.get_device_properties(0).multi_processor_count
    print(torch.cuda.get_device_name(), P, "SMs", "triton", triton.__version__, flush=True)
    parts = os.environ.get("MEGA_PARTS", "barrier,chain").replace(":", ",").split(",")
    quick = os.environ.get("MEGA_QUICK", "0") == "1"
    if "barrier" in parts:
        run_barrier(P)
    if "chain" in parts:
        run_chain(P, quick)


if __name__ == "__main__":
    _ = sys.argv  # MODEL (unused)
    main()
