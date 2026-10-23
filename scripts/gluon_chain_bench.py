"""Gate (a), part 2: the persistent chain of the 1B decoder's 72 linears in Gluon, weights prefetched across
the grid barriers.

18 x [QKV 3072x1024, out 1024x1024, FFN1 4096x1024 + ReLU, FFN2 1024x4096], random weights, one program per
SM, a grid barrier between linears. Every program owns BN weight rows of a linear (QKV / FFN1: 32, out / FFN2:
8) and streams its slabs through a two-slot shared-memory ring with TMA: the slab of linear j + 2 is
requested as soon as linear j's MMAs are done, so it arrives while the program waits at the barriers. After a
barrier only the activations (from L2) remain to be loaded. GEMM unit as in scripts/gluon_unit_bench.py.

python scripts/gluon_chain_bench.py [MODEL]
"""

import time

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

D, F, L = 1024, 4096, 18


@gluon.jit
def _now():
    return gl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=gl.int64, is_pure=False, pack=1
    )


@gluon.jit
def _ld_acquire(p):
    return gl.inline_asm_elementwise(
        "ld.global.acquire.gpu.b32 $0, [$1];", "=r,l", [p], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _garrive(bar):
    gl.barrier()
    gl.atomic_add(bar, 1, sem="release", scope="gpu")


@gluon.jit
def _gwait(bar, err, target, MAXS: gl.constexpr):
    e = gl.load(err, volatile=True)
    v = _ld_acquire(bar)
    n = 0
    while (v < target) & (n < MAXS) & (e == 0):
        v = _ld_acquire(bar)
        n += 1
    if (v < target) & (e == 0):
        gl.atomic_xchg(err, 1, sem="relaxed", scope="gpu")
    gl.barrier()


@gluon.jit
def _gexit(bar, done, P):
    gl.barrier()
    t = gl.atomic_add(done, 1, sem="acq_rel", scope="gpu")
    if t == P - 1:
        gl.atomic_xchg(done, 0, sem="relaxed", scope="gpu")
        gl.atomic_xchg(bar, 0, sem="relaxed", scope="gpu")


@gluon.jit
def _issue_w(desc, row0, bar, wslot, BN: gl.constexpr, K: gl.constexpr, pred):
    """One TMA: the program's slab of BN weight rows from row0, as A tiles (K chunk, K block, row); desc is the
    weights viewed as [K / S / 64, S, rows, 64] with S = 64 / BN."""
    S: gl.constexpr = 64 // BN
    NKC: gl.constexpr = K // S // 64
    l4: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=4)
    dst = wslot._reinterpret(gl.float16, [8, S, BN, 64], l4).slice(0, NKC, dim=0)  # the slot holds 8 chunks
    mbarrier.expect(bar, BN * K * 2, pred=pred)
    tma.async_load(desc, [0, 0, row0, 0], bar, dst, pred=pred)


@gluon.jit
def _unit_mma(x, x_stride, M, wslot, xbuf, bar, phase, bias, BN: gl.constexpr, K: gl.constexpr, MT: gl.constexpr,
              LBN: gl.constexpr, LMT: gl.constexpr, NW: gl.constexpr, tsp, TS: gl.constexpr,
              NACC: gl.constexpr):  # fmt: skip
    """X (B side) via cp.async, then the wgmma chain over the weight slab in wslot (A side); [64, S * MT]. The
    bias [BN] is loaded first, so that its latency hides behind X's."""
    ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
    b = gl.load(bias + gl.arange(0, BN, layout=gl.SliceLayout(1, ol))).to(gl.float32)
    S: gl.constexpr = 64 // BN
    KP: gl.constexpr = K // S
    NKC: gl.constexpr = KP // 64
    NB: gl.constexpr = S * MT
    xl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    rho = gl.arange(0, NKC * NB, layout=gl.SliceLayout(1, xl))  # (kc, s, t)
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, xl))
    t = rho & (MT - 1)
    s_ = (rho >> LMT) & (S - 1)
    kc_ = rho >> (LMT + 6 - LBN)
    xp = x + t[:, None] * x_stride + (s_ * KP + kc_ * 64)[:, None] + kk[None, :]
    xm = (t < M)[:, None] & (kk >= 0)[None, :]
    xs = xbuf.slice(0, NKC * NB)
    async_copy.async_load(xs, xp, mask=xm, cache_modifier=".cg")
    async_copy.commit_group()
    mbarrier.wait(bar, phase)
    if TS:
        gl.store(tsp, _now())
    async_copy.wait_group(0)
    fence_async_shared()
    gl.barrier()
    if TS:
        gl.store(tsp + 1, _now())
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, NW // 4], instr_shape=[16, NB // (NW // 4), 16]
    )
    NA: gl.constexpr = NACC if NKC >= NACC else NKC  # independent accumulators over alternating K chunks
    acc0 = gl.zeros([64, NB], gl.float32, layout=mma)
    acc1 = gl.zeros([64, NB], gl.float32, layout=mma)
    acc2 = gl.zeros([64, NB], gl.float32, layout=mma)
    acc3 = gl.zeros([64, NB], gl.float32, layout=mma)
    for kc in gl.static_range(NKC):
        a_t = wslot.slice(kc * 64, 64)
        b_t = xs.slice(kc * NB, NB).permute((1, 0))
        if kc % NA == 0:
            acc0 = warpgroup_mma(a_t, b_t, acc0, is_async=True)
        elif kc % NA == 1:
            acc1 = warpgroup_mma(a_t, b_t, acc1, is_async=True)
        elif kc % NA == 2:
            acc2 = warpgroup_mma(a_t, b_t, acc2, is_async=True)
        else:
            acc3 = warpgroup_mma(a_t, b_t, acc3, is_async=True)
    if NA == 1:
        acc = warpgroup_mma_wait(0, deps=[acc0])
    elif NA == 2:
        acc0, acc1 = warpgroup_mma_wait(0, deps=[acc0, acc1])
        acc = acc0 + acc1
    else:
        acc0, acc1, acc2, acc3 = warpgroup_mma_wait(0, deps=[acc0, acc1, acc2, acc3])
        acc = (acc0 + acc1) + (acc2 + acc3)
    if TS:
        gl.store(tsp + 2, _now())
    return acc, b


@gluon.jit
def _unit_out(acc, b, scratch, y, y_stride, n0, M, BN: gl.constexpr, MT: gl.constexpr, RELU: gl.constexpr,
              NW: gl.constexpr):  # fmt: skip
    """y[t, n0 + r] = sum of the diagonal blocks of acc + bias (fp16)."""
    S: gl.constexpr = 64 // BN
    NB: gl.constexpr = S * MT
    sc = scratch.slice(0, NB, dim=1)
    sc.store(acc)
    gl.barrier()
    ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
    out = gl.zeros([BN, MT], gl.float32, layout=ol)
    for s in gl.static_range(S):
        out += sc.slice(s * BN, BN, dim=0).slice(s * MT, MT, dim=1).load(ol)
    rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
    tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
    out += b[:, None]
    if RELU:
        out = gl.maximum(out, 0.0)
    gl.store(
        y + tt[None, :] * y_stride + n0 + rr[:, None],
        out.to(gl.float16),
        mask=(tt < M)[None, :] & (rr >= 0)[:, None],
    )
    gl.barrier()


@gluon.jit
def _chain(dq, do, d1, d2, bq, bo, b1, b2, xa, xb, bar, err, done, ts, M, NL,
           MT: gl.constexpr, LMT: gl.constexpr, NW: gl.constexpr, TS: gl.constexpr, MAXS: gl.constexpr,
           NACC: gl.constexpr):  # fmt: skip
    pid = gl.program_id(0)
    P = gl.num_programs(0)
    D: gl.constexpr = 1024
    F: gl.constexpr = 4096
    SX: gl.constexpr = 4096  # row stride of xa / xb
    sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    w0 = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
    w1 = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
    xbuf = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
    scratch = gl.allocate_shared_memory(gl.float32, [64, 64], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    mb = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(mb.index(0), count=1)
    mbarrier.init(mb.index(1), count=1)
    fence_async_shared()
    has_q = pid < 3 * D // 32
    has_o = pid < D // 8
    has_1 = pid < F // 32
    has_2 = pid < D // 8
    nq = pid * 32
    no = pid * 8
    n1 = pid * 32
    n2 = pid * 8
    _issue_w(dq, nq, mb.index(0), w0, 32, D, has_q)
    _issue_w(do, no, mb.index(1), w1, 8, D, has_o)
    g = 0
    p0 = 0  # mbarrier phases of the two slots (a program only waits for the slabs it requested)
    p1 = 0
    for layer in range(NL):
        nxt = layer + 1 < NL
        # QKV: xa -> xb (slot 0); then the slot takes FFN1's slab
        if TS:
            gl.store(ts + (g * P + pid) * 6, _now())
        if has_q:
            acc, b = _unit_mma(xa, SX, M, w0, xbuf, mb.index(0), p0, bq + layer * 3 * D + nq, 32, D, MT, 5, LMT, NW,
                               ts + (g * P + pid) * 6 + 2, TS, NACC)  # fmt: skip
            p0 = p0 ^ 1
            _unit_out(acc, b, scratch, xb, SX, nq, M, 32, MT, False, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 6 + 1, _now())
        g += 1
        _garrive(bar)
        _issue_w(d1, layer * F + n1, mb.index(0), w0, 32, D, has_1)
        _gwait(bar, err, g * P, MAXS)
        # out: xb[:, :D] -> xa (slot 1); then FFN2's slab
        if TS:
            gl.store(ts + (g * P + pid) * 6, _now())
        if has_o:
            acc, b = _unit_mma(xb, SX, M, w1, xbuf, mb.index(1), p1, bo + layer * D + no, 8, D, MT, 3, LMT, NW,
                               ts + (g * P + pid) * 6 + 2, TS, NACC)  # fmt: skip
            p1 = p1 ^ 1
            _unit_out(acc, b, scratch, xa, SX, no, M, 8, MT, False, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 6 + 1, _now())
        g += 1
        _garrive(bar)
        _issue_w(d2, layer * D + n2, mb.index(1), w1, 8, F, has_2)
        _gwait(bar, err, g * P, MAXS)
        # FFN1: xa -> xb, ReLU (slot 0); then the next layer's QKV slab
        if TS:
            gl.store(ts + (g * P + pid) * 6, _now())
        if has_1:
            acc, b = _unit_mma(xa, SX, M, w0, xbuf, mb.index(0), p0, b1 + layer * F + n1, 32, D, MT, 5, LMT, NW,
                               ts + (g * P + pid) * 6 + 2, TS, NACC)  # fmt: skip
            p0 = p0 ^ 1
            _unit_out(acc, b, scratch, xb, SX, n1, M, 32, MT, True, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 6 + 1, _now())
        g += 1
        _garrive(bar)
        _issue_w(dq, (layer + 1) * 3 * D + nq, mb.index(0), w0, 32, D, has_q & nxt)
        _gwait(bar, err, g * P, MAXS)
        # FFN2: xb -> xa (slot 1); then the next layer's out-projection slab
        if TS:
            gl.store(ts + (g * P + pid) * 6, _now())
        if has_2:
            acc, b = _unit_mma(xb, SX, M, w1, xbuf, mb.index(1), p1, b2 + layer * D + n2, 8, F, MT, 3, LMT, NW,
                               ts + (g * P + pid) * 6 + 2, TS, NACC)  # fmt: skip
            p1 = p1 ^ 1
            _unit_out(acc, b, scratch, xa, SX, n2, M, 8, MT, False, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 6 + 1, _now())
        g += 1
        _garrive(bar)
        _issue_w(do, (layer + 1) * D + no, mb.index(1), w1, 8, D, has_o & nxt)
        _gwait(bar, err, g * P, MAXS)
    mbarrier.invalidate(mb.index(0))
    mbarrier.invalidate(mb.index(1))
    _gexit(bar, done, P)


def slab_desc(w, bn):
    """TMA descriptor of weights [..., rows, K] viewed as [K / S / 64, S, rows, 64] (S = 64 / bn), box = one
    program's slab of bn rows."""
    k = w.shape[-1]
    s = 64 // bn
    v = w.reshape(-1, s, k // s // 64, 64).permute(2, 1, 0, 3)
    lay = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=4)
    return TensorDescriptor.from_tensor(v, [k // s // 64, s, bn, 64], lay)


def chain_ref(x, ws, bs, M):
    import torch.nn.functional as F

    wq, wo, w1, w2 = ws
    bq, bo, b1, b2 = bs
    h = x[:M, :D].float()
    for i in range(L):
        q = F.linear(h.half(), wq[i], bq[i]).float()
        h = F.linear(q[:, :D].half(), wo[i], bo[i]).float()
        h = F.relu(F.linear(h.half(), w1[i], b1[i])).float()
        h = F.linear(h.half(), w2[i], b2[i]).float()
    return h


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


def main():
    P = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"
    torch.manual_seed(0)
    h = {"device": dev, "dtype": torch.float16}
    wq = torch.randn(L, 3 * D, D, **h) / D**0.5
    wo = torch.randn(L, D, D, **h) / D**0.5
    w1 = torch.randn(L, F, D, **h) / D**0.5
    w2 = torch.randn(L, D, F, **h) * (2 / F) ** 0.5
    bq, bo, b1, b2 = (torch.randn(L, n, **h) * 0.1 for n in (3 * D, D, F, D))
    dq, do, d1, d2 = (slab_desc(w, bn) for w, bn in ((wq, 32), (wo, 8), (w1, 32), (w2, 8)))
    mb = sum(t.numel() * 2 for t in (wq, wo, w1, w2)) / 1e6
    print(f"{P} programs; 72 linears, {mb:.0f} MB of weights ({mb / 4e3 * 1e3:.3f} ms at 4 TB/s)", flush=True)
    bar = torch.zeros(1, dtype=torch.int32, device=dev)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    done = torch.zeros(1, dtype=torch.int32, device=dev)
    ts = torch.zeros(4 * L * P * 6, dtype=torch.int64, device=dev)
    for nw, M, MT, nacc in ((4, 8, 8, 1), (4, 1, 8, 2), (4, 8, 8, 2), (4, 1, 8, 4), (4, 8, 8, 4)):
        xa = torch.zeros(16, 4096, **h)
        xb = torch.zeros(16, 4096, **h)
        x0 = torch.randn(16, 4096, **h)

        def launch(TS=False, M=M, MT=MT, xa=xa, xb=xb, nw=nw, nacc=nacc):
            _chain[(P,)](dq, do, d1, d2, bq, bo, b1, b2, xa, xb, bar, err, done, ts, M, L, MT=MT, LMT=3, NW=nw, TS=TS,
                         MAXS=1 << 22, NACC=nacc, num_warps=nw)  # fmt: skip

        xa.copy_(x0)
        err.zero_()
        launch()
        torch.cuda.synchronize()
        ref = chain_ref(x0, (wq, wo, w1, w2), (bq, bo, b1, b2), M)
        got = xa[:M, :D].float()
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        ms = bench(launch)
        launch(TS=True)
        torch.cuda.synchronize()
        t = ts.view(4 * L, P, 6).double()
        t = t - t[0, :, 0].min()
        st, ar, wr, xr, mm, iss = (t[:, :, i] for i in range(6))
        lat = st[1:].min(1).values - ar[:-1].max(1).values  # barrier: last arrival -> first exit
        skew = ar.max(1).values - ar.min(1).values
        names = ("qkv", "out", "ffn1", "ffn2")
        parts = []
        for j, n in enumerate(names):
            used = wr[j::4] > 0  # programs with a unit of this linear

            def med(a, used=used):
                return a[used].median().item() / 1e3

            sj = st[j::4]
            parts.append(
                f"\n   {n}: work {med(ar[j::4] - sj):.2f} us = weights {med(wr[j::4] - sj):.2f}, X {med(xr[j::4] - sj):.2f}"
                f", MMA +{med(mm[j::4] - xr[j::4]):.2f}, epilogue +{med(ar[j::4] - mm[j::4]):.2f}; arrival skew {skew[j::4].mean().item() / 1e3:.2f}, barrier "
                f"{lat[j::4].mean().item() / 1e3 if j < 3 else lat[3::4].mean().item() / 1e3:.2f} us"
            )
        print(
            f"{nw} warps, {nacc} accumulators, M={M}: {ms:.3f} ms (rel err {rel:.1e}, err flag {err.item()})"
            + "".join(parts),
            flush=True,
        )


if __name__ == "__main__":
    main()
