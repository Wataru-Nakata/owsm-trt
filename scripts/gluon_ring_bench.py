"""Prototype for the decoder megakernel with any number of programs: the 1B decoder's 6 linears per layer
(QKV, out, cross-Q, cross-out, FFN1, FFN2 split-K) x 18 layers as a persistent chain with a grid barrier
between linears, every program owning units of 64 weight rows (x all of K, FFN2: K / KS) at a time.

The weights stream through a ring of NSLOT [64, 64] chunks in shared memory, fed by a producer warp
(warp specialization) that runs ahead over the program's whole chunk sequence, across the grid barriers; the
four consumer warps wait for each chunk, run its wgmma (weights on the A side, the tokens on the B side) and
release it. B side: [HL x tokens, K] from global memory (cp.async), hi/lo rows summed through a padded scratch.

python scripts/gluon_ring_bench.py   (env: RING_P=64:132 programs, RING_MT=1:8:16:32 tokens, RING_SLOTS=16)
"""

import os
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

from owsm_trt.megakernel_gluon import _garrive, _gexit, _gwait, _now

D, F, L = 1024, 4096, 18


@gluon.jit
def _prod_units(desc, row0, c, U, pid, P, ring, full, empty, LKS: gl.constexpr, KW: gl.constexpr,
                NKC: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr):  # fmt: skip
    """Producer: the chunks of this program's units u = pid, pid + P, ... < U of one linear (unit u = rows
    row0 + (u >> LKS) * 64, K slice u & (2^LKS - 1) of KW columns), in order."""
    for u in range(pid, U, P):
        row = row0 + (u >> LKS) * 64
        col0 = (u & ((1 << LKS) - 1)) * KW
        for kc in range(NKC):
            slot = c & (NSLOT - 1)
            mbarrier.wait(empty.index(slot), ((c >> LNS) & 1) ^ 1)
            mbarrier.expect(full.index(slot), 8192)
            tma.async_load(desc, [row, col0 + kc * 64], full.index(slot), ring.index(slot))
            c += 1
    return c


@gluon.jit
def _producer(dqkv, do_, dcq, dco, d1, d2, ring, full, empty, pid, P, L, D: gl.constexpr, F: gl.constexpr,
              NSLOT: gl.constexpr, LNS: gl.constexpr):  # fmt: skip
    NKC: gl.constexpr = D // 64
    LKS: gl.constexpr = 2  # FFN2: F / D = 4 slices of K
    c = 0
    for layer in range(L):
        c = _prod_units(dqkv, layer * 3 * D, c, 3 * D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
        c = _prod_units(do_, layer * D, c, D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
        c = _prod_units(dcq, layer * D, c, D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
        c = _prod_units(dco, layer * D, c, D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
        c = _prod_units(d1, layer * F, c, F // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
        c = _prod_units(d2, layer * D, c, F // 64, pid, P, ring, full, empty, LKS, D, NKC, NSLOT, LNS)


@gluon.jit
def _x_load(x, XS, HO, M, xbuf, NKC: gl.constexpr, N: gl.constexpr, LN_: gl.constexpr, MTP: gl.constexpr,
            LMTP: gl.constexpr, HL: gl.constexpr, NW: gl.constexpr):  # fmt: skip
    """cp.async of the B side: xbuf rows (K chunk, h, t) = x[t, h * HO + chunk * 64 + 0..63] (token stride XS);
    rows past HL x MTP and tokens >= M are zero."""
    ROWS: gl.constexpr = NKC * N
    xl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, xl))
    rho = gl.arange(0, ROWS, layout=gl.SliceLayout(1, xl))
    r = rho & (N - 1)
    kc = rho >> LN_
    t = r & (MTP - 1)
    h = r >> LMTP
    ok = (h < HL) & (t < M)
    xp = x + t[:, None] * XS + (h * HO + kc * 64)[:, None] + kk[None, :]
    async_copy.async_load(
        xbuf.slice(0, ROWS), xp, mask=ok[:, None] & (kk >= 0)[None, :], cache_modifier=".cg"
    )
    async_copy.commit_group()


@gluon.jit
def _consume(ring, full, empty, c, xbuf, N: gl.constexpr, NKC: gl.constexpr, NSLOT: gl.constexpr,
             LNS: gl.constexpr, NW: gl.constexpr, LAG: gl.constexpr, SPLIT: gl.constexpr = False):  # fmt: skip
    """The MMAs of one unit: chunk after chunk from the ring against the B side in xbuf -> [64, N] fp32 (SPLIT:
    the two halves of each B tile into one accumulator -> [64, N / 2]); up to LAG chunks' wgmmas in flight, a
    chunk released once its wgmma is done."""
    NA: gl.constexpr = N // 2 if SPLIT else N
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, NA, 16]
    )
    acc = gl.zeros([64, NA], gl.float32, layout=mma)
    for kc in gl.static_range(NKC):
        slot = c & (NSLOT - 1)
        mbarrier.wait(full.index(slot), (c >> LNS) & 1)
        if SPLIT:
            acc = warpgroup_mma(ring.index(slot), xbuf.slice(kc * N, NA).permute((1, 0)), acc, is_async=True)
            acc = warpgroup_mma(
                ring.index(slot), xbuf.slice(kc * N + NA, NA).permute((1, 0)), acc, is_async=True
            )
        else:
            acc = warpgroup_mma(ring.index(slot), xbuf.slice(kc * N, N).permute((1, 0)), acc, is_async=True)
        if kc >= LAG:
            acc = warpgroup_mma_wait(LAG, deps=[acc])
            mbarrier.arrive(empty.index((c - LAG) & (NSLOT - 1)))
        c += 1
    acc = warpgroup_mma_wait(0, deps=[acc])
    for j in gl.static_range(LAG):
        mbarrier.arrive(empty.index((c - LAG + j) & (NSLOT - 1)))
    return acc, c


@gluon.jit
def _sum_hl(acc, scratch, N: gl.constexpr, MTP: gl.constexpr, HL: gl.constexpr, NW: gl.constexpr):
    """[64, MTP] = the sum of acc's HL column groups of MTP (hi/lo rows of the B side), via shared memory."""
    el: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, NW], [0, 1])
    sc = scratch.slice(0, N, dim=1)
    sc.store(acc)
    gl.barrier()
    out = sc.slice(0, MTP, dim=1).load(el)
    if HL == 2:
        out += sc.slice(MTP, MTP, dim=1).load(el)
    gl.barrier()
    return out


@gluon.jit
def _linear(x, XS, HO, M, ring, full, empty, c, xbuf, scratch, bias, y, YS, pid, P, U, ts, TSI,
            NKC: gl.constexpr, N: gl.constexpr, LN_: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr,
            HL: gl.constexpr, NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr,
            TS: gl.constexpr, LAG: gl.constexpr, HLK: gl.constexpr = False):  # fmt: skip
    """The units pid, pid + P, ... < U of y = x @ W^T + b (fp16 out, 64 features per unit)."""
    el: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, NW], [0, 1])
    rr = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
    tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
    if pid < U:
        _x_load(x, XS, HO, M, xbuf, NKC, N, LN_, MTP, LMTP, HL, NW)
        async_copy.wait_group(0)
        fence_async_shared()
        gl.barrier()
        if TS:
            gl.store(ts + TSI + 2, _now())
    for u in range(pid, U, P):
        b = gl.load(bias + u * 64 + rr)
        if HLK and HL == 2 and N == 2 * MTP and MTP >= 8:
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, LAG, True)
            if TS:
                gl.store(ts + TSI + 3, _now())
            out = _sum_hl(acc, scratch, MTP, MTP, 1, NW) + b[:, None]
        else:
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, LAG)
            if TS:
                gl.store(ts + TSI + 3, _now())
            out = _sum_hl(acc, scratch, N, MTP, HL, NW) + b[:, None]
        gl.store(y + tt[None, :] * YS + u * 64 + rr[:, None], out.to(gl.float16),
                 mask=(tt < M)[None, :] & (rr >= 0)[:, None])  # fmt: skip
    return c


@gluon.jit
def _ffn2(hb, M, ring, full, empty, c, xbuf, scratch, bias, y, part, cnt, pid, P, ts, TSI, D: gl.constexpr,
          F: gl.constexpr, N: gl.constexpr, LN_: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr,
          NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, TS: gl.constexpr, LAG: gl.constexpr):  # fmt: skip
    """FFN2 split-K: unit u = (row block u >> 2, K slice u & 3 of D columns); partials [64, MTP] fp32 in part,
    the last slice of a block sums them (+ bias) into y (fp32)."""
    KS: gl.constexpr = F // D
    NKC: gl.constexpr = D // 64
    el: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, NW], [0, 1])
    rr = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
    tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
    po = tt[None, :] * 64 + rr[:, None]  # token-major: coalesced
    for u in range(pid, F // 64, P):
        rb = u >> 2
        ks = u & 3
        _x_load(hb + ks * D, F, 0, M, xbuf, NKC, N, LN_, MTP, LMTP, 1, NW)
        async_copy.wait_group(0)
        fence_async_shared()
        gl.barrier()
        if TS:
            gl.store(ts + TSI + 2, _now())
        acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, LAG)
        if TS:
            gl.store(ts + TSI + 3, _now())
        out = _sum_hl(acc, scratch, N, MTP, 1, NW)
        gl.store(part + (u * 64) * MTP + po, out)
        gl.barrier()
        arrived = gl.atomic_add(cnt + rb, 1, sem="acq_rel", scope="gpu")
        gl.barrier()
        if arrived == KS - 1:
            tot = out + gl.load(bias + rb * 64 + rr)[:, None]
            for j in gl.static_range(KS):
                if j != ks:
                    tot += gl.load(part + ((rb * KS + j) * 64) * MTP + po, cache_modifier=".cg")
            gl.atomic_xchg(cnt + rb, 0, sem="relaxed", scope="gpu")
            gl.store(
                y + tt[None, :] * D + rb * 64 + rr[:, None], tot, mask=(tt < M)[None, :] & (rr >= 0)[:, None]
            )
    return c


@gluon.jit
def _consumer(x2, ob, hb, M, ring, full, empty, xbuf, scratch, bq, bo, bcq, bco, b1, b2, yq, yo, ycq, yco, y1, y2,
              part, cnt, bar, err, done, ts, pid, P, L, D: gl.constexpr, F: gl.constexpr, NL: gl.constexpr,
              LNL: gl.constexpr, N1: gl.constexpr, LN1: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr,
              HL: gl.constexpr, NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, MAXS: gl.constexpr,
              TS: gl.constexpr, LAG: gl.constexpr, HLK: gl.constexpr):  # fmt: skip
    NKC: gl.constexpr = D // 64
    c = 0
    g = 0
    for layer in range(L):
        for s in gl.static_range(6):
            TSI = (g * P + pid) * 8
            if TS:
                gl.store(ts + TSI, _now())
            if s == 0:
                c = _linear(x2, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, bq + layer * 3 * D, yq, 3 * D, pid,
                            P, 3 * D // 64, ts, TSI, NKC, NL, LNL, MTP, LMTP, HL, NW, NSLOT, LNS, TS, LAG, HLK)  # fmt: skip
            elif s == 1:
                c = _linear(ob, D, 0, M, ring, full, empty, c, xbuf, scratch, bo + layer * D, yo, D, pid, P, D // 64,
                            ts, TSI, NKC, N1, LN1, MTP, LMTP, 1, NW, NSLOT, LNS, TS, LAG)  # fmt: skip
            elif s == 2:
                c = _linear(x2, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, bcq + layer * D, ycq, D, pid, P,
                            D // 64, ts, TSI, NKC, NL, LNL, MTP, LMTP, HL, NW, NSLOT, LNS, TS, LAG, HLK)  # fmt: skip
            elif s == 3:
                c = _linear(ob, D, 0, M, ring, full, empty, c, xbuf, scratch, bco + layer * D, yco, D, pid, P,
                            D // 64, ts, TSI, NKC, N1, LN1, MTP, LMTP, 1, NW, NSLOT, LNS, TS, LAG)  # fmt: skip
            elif s == 4:
                c = _linear(x2, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, b1 + layer * F, y1, F, pid, P,
                            F // 64, ts, TSI, NKC, NL, LNL, MTP, LMTP, HL, NW, NSLOT, LNS, TS, LAG, HLK)  # fmt: skip
            else:
                c = _ffn2(hb, M, ring, full, empty, c, xbuf, scratch, b2 + layer * D, y2, part, cnt, pid, P, ts, TSI,
                          D, F, N1, LN1, MTP, LMTP, NW, NSLOT, LNS, TS, LAG)  # fmt: skip
            if TS:
                gl.store(ts + TSI + 5, _now())
            g += 1
            _garrive(bar)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + TSI + 1, _now())
    _gexit(bar, done, P)


@gluon.jit
def _chain(dqkv, do_, dcq, dco, d1, d2, x2, ob, hb, M, bq, bo, bcq, bco, b1, b2, yq, yo, ycq, yco, y1, y2, part, cnt,
           bar, err, done, ts, L, D: gl.constexpr, F: gl.constexpr, NL: gl.constexpr, LNL: gl.constexpr,
           N1: gl.constexpr, LN1: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr, HL: gl.constexpr,
           NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, MAXS: gl.constexpr, TS: gl.constexpr,
           LAG: gl.constexpr, HLK: gl.constexpr):  # fmt: skip
    pid = gl.program_id(0)
    P = gl.num_programs(0)
    sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    ring = gl.allocate_shared_memory(gl.float16, [NSLOT, 64, 64], sl)
    xbuf = gl.allocate_shared_memory(gl.float16, [(D // 64) * NL, 64], sl)
    pl: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NL, 2]], [64, NL], [1, 0])
    scratch = gl.allocate_shared_memory(gl.float32, [64, NL], pl)
    full = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NSLOT):
        mbarrier.init(full.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    fence_async_shared()
    gl.warp_specialize(
        [
            (_consumer, (x2, ob, hb, M, ring, full, empty, xbuf, scratch, bq, bo, bcq, bco, b1, b2, yq, yo, ycq, yco,
                         y1, y2, part, cnt, bar, err, done, ts, pid, P, L, D, F, NL, LNL, N1, LN1, MTP, LMTP, HL, NW,
                         NSLOT, LNS, MAXS, TS, LAG, HLK)),
            (_producer, (dqkv, do_, dcq, dco, d1, d2, ring, full, empty, pid, P, L, D, F, NSLOT, LNS)),
        ],
        [1],
        [48],
    )  # fmt: skip


def _log2(n):
    return int(n).bit_length() - 1


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    f16 = {"device": dev, "dtype": torch.float16}
    w = {
        "qkv": torch.randn(L * 3 * D, D, **f16) * 0.03,
        "o": torch.randn(L * D, D, **f16) * 0.03,
        "cq": torch.randn(L * D, D, **f16) * 0.03,
        "co": torch.randn(L * D, D, **f16) * 0.03,
        "1": torch.randn(L * F, D, **f16) * 0.03,
        "2": torch.randn(L * D, F, **f16) * 0.03,
    }
    b = {k: torch.randn(v.shape[0], device=dev) * 0.1 for k, v in w.items()}
    lay = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    descs = [TensorDescriptor.from_tensor(w[k], [64, 64], lay) for k in ("qkv", "o", "cq", "co", "1", "2")]
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    ps = [int(p) for p in os.environ.get("RING_P", f"64:{nsm}").split(":")]
    mts = [int(m) for m in os.environ.get("RING_MT", "1:8:16:32").split(":")]
    nslot = int(os.environ.get("RING_SLOTS", "16"))
    lag = int(os.environ.get("RING_LAG", "1"))
    zero = os.environ.get("RING_ZERO") == "1"
    hlk = os.environ.get("RING_HLK") == "1"  # LayerNorm GEMMs: hi and lo B tiles as two wgmmas
    force_hl = int(os.environ.get("RING_HL", "0"))  # LayerNorm GEMMs' hi/lo rows (0: 2 up to 16 tokens)
    for mt in mts:
        M = mt
        mtp = max(4, mt)
        hl = force_hl or (2 if mt <= 16 else 1)
        nl, n1 = max(8, hl * mtp), max(8, mtp)
        x2 = torch.randn(M, 2, D, **f16)
        x2[:, 1] *= 1e-3  # a lo part
        ob = torch.randn(M, D, **f16)
        hb = torch.randn(M, F, **f16).relu()
        if zero:
            x2.zero_()
            ob.zero_()
            hb.zero_()
        ys = [torch.zeros(M, 3 * D, **f16), torch.zeros(M, D, **f16), torch.zeros(M, D, **f16),
              torch.zeros(M, D, **f16), torch.zeros(M, F, **f16), torch.zeros(M, D, device=dev)]  # fmt: skip
        part = torch.zeros(F // 64, 64, mtp, device=dev)
        cnt = torch.zeros(D // 64, dtype=torch.int32, device=dev)
        for P in ps:
            bar = torch.zeros(1, dtype=torch.int32, device=dev)
            done = torch.zeros(1, dtype=torch.int32, device=dev)
            err = torch.zeros(1, dtype=torch.int32, device=dev)
            ts = torch.zeros(6 * L * P * 8, dtype=torch.int64, device=dev)

            def run(tsflag=False):
                _chain[(P,)](*descs, x2, ob, hb, M, b["qkv"], b["o"], b["cq"], b["co"], b["1"], b["2"], *ys, part,
                             cnt, bar, err, done, ts, L, D=D, F=F, NL=nl, LNL=_log2(nl), N1=n1, LN1=_log2(n1),
                             MTP=mtp, LMTP=_log2(mtp), HL=hl, NW=4, NSLOT=nslot, LNS=_log2(nslot), MAXS=1 << 24,
                             TS=tsflag, LAG=lag, HLK=hlk, num_warps=4)  # fmt: skip

            try:
                run()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"MT={mt} P={P}: FAILED {type(e).__name__}: {str(e)[:2000]}", flush=True)
                continue
            # check the last layer's outputs
            lq = slice((L - 1) * 3 * D, L * 3 * D)
            xs = (x2[:, 0].float() + x2[:, 1].float()) if hl == 2 else x2[:, 0].float()
            refs = [
                xs @ w["qkv"][lq].float().T + b["qkv"][lq],
                ob.float() @ w["o"][(L - 1) * D :].float().T + b["o"][(L - 1) * D :],
                xs @ w["cq"][(L - 1) * D :].float().T + b["cq"][(L - 1) * D :],
                ob.float() @ w["co"][(L - 1) * D :].float().T + b["co"][(L - 1) * D :],
                xs @ w["1"][(L - 1) * F :].float().T + b["1"][(L - 1) * F :],
                hb.float() @ w["2"][(L - 1) * D :].float().T + b["2"][(L - 1) * D :],
            ]
            errs = [((y.float() - r).abs().max() / r.abs().max()).item() for y, r in zip(ys, refs)]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            g.replay()
            torch.cuda.synchronize()
            best = 1e9
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(20):
                    g.replay()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - t0) / 20 * 1e3)
            ts.zero_()
            run(True)
            torch.cuda.synchronize()
            t = ts.view(6 * L, P, 8).double()
            t0 = t[0, :, 0].min()
            rel = t[:, :, 1].min(1).values
            wall = torch.cat([rel[:1] - t0, rel[1:] - rel[:-1]]).view(L, 6)
            live = t[:, :, 3] > 0
            xw = (t[:, :, 2] - t[:, :, 0]).where(live, torch.nan).nanmedian(1).values.view(L, 6)
            mm = (t[:, :, 3] - t[:, :, 2]).where(live, torch.nan).nanmedian(1).values.view(L, 6)
            ep = (t[:, :, 5] - t[:, :, 3]).where(live, torch.nan).nanmedian(1).values.view(L, 6)
            names = ("qkv", "out", "crossq", "crossout", "ffn1", "ffn2")
            prof = "  ".join(
                f"{n} {wall[1:, i].mean() / 1e3:.2f} [x {xw[1:, i].mean() / 1e3:.2f} mma {mm[1:, i].mean() / 1e3:.2f}"
                f" ep {ep[1:, i].mean() / 1e3:.2f}]"
                for i, n in enumerate(names)
            )
            print(f"MT={mt:2d} P={P:3d} slots={nslot} lag={lag} zero={int(zero)} hl={hl} hlk={int(hlk)}: {best:.3f} ms per call ({best / L * 1e3:.1f} us/layer), "
                  f"rel err {max(errs):.1e}, err flag {err.item()}\n   {prof}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
