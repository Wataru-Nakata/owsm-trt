"""The decoder megakernel in Gluon (Hopper, sm90) for any number of programs: weights stream through a ring.

Same stages, grid barriers, attention and folded LayerNorms as :mod:`owsm_trt.megakernel_gluon`, but the
GEMM work is cut into units of BM = 64 weight rows (128 with 8 warps) over all of K (FFN2: split-K, K / KS),
and program pid takes units pid, pid + P, ... of a stage. The units' [BM, 64] weight tiles stream through a
ring in shared memory (slots of CK tiles), fed by a producer warp (warp specialization) that runs ahead over
the program's whole sequence of tiles, across the grid barriers; the consumer warps wait for each slot, run its
wgmmas (weights on the A side, the tokens on the B side: [hi/lo x tokens, K] from L2) and release it. The
output layer can also leave the argmax of the logits (then the runtime skips its own argmax kernel).

So a call needs no more programs than its largest stage has units (64 for the 1B model: FFN1 / FFN2) and
leaves the other SMs free, e.g. to the encoder that runs concurrently in decode_stream: an all-SM persistent
kernel stalls it. Fewer programs stream fewer weight bytes per microsecond, though (about 100 GB/s per SM):
32 programs (the single-row default) take about 0.70-0.95 ms per call of 1-32 tokens on an H200, 64 about
0.59-0.78 ms.

Loops over K tiles are runtime loops and the layer loop adds an opaque zero to its base pointers (see
_opaque0): unrolled or hoisted, their address and descriptor tensors stay live and spill.
"""

import torch

from .megakernel_gluon import MAX_TOKENS, _log2, _pow2, available, gluon

if gluon is not None:
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

    from .megakernel_gluon import (
        _cross_attn_g,
        _el,
        _els,
        _embed_g,
        _garrive,
        _gexit,
        _gwait,
        _now,
        _prefetch_rows,
        _self_attn_g,
        _store_res,
    )

__all__ = ["RingDecoder", "available", "MAX_TOKENS", "ROUTE_TOKENS", "ROUTE_ROWS"]


# calls of up to ROUTE_TOKENS tokens and ROUTE_ROWS rows go to the kernel (MegaBackend)
ROUTE_TOKENS = MAX_TOKENS
ROUTE_ROWS = 8

if gluon is not None:
    from triton.experimental.gluon import constexpr_function

    @constexpr_function
    def _np2(n):
        """The smallest power of 2 >= n (block shapes must be powers of 2)."""
        return 1 << max(0, (int(n) - 1).bit_length())

    @gluon.jit
    def _prod_units(desc, row0, c, U, pid, P, ring, full, empty, LKS: gl.constexpr, KW: gl.constexpr,
                    NKC: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr,
                    BM: gl.constexpr):  # fmt: skip
        """Producer: the chunks of units u = pid, pid + P, ... < U of a GEMM (unit u: rows row0 + (u >> LKS) *
        BM, K slice u & (2^LKS - 1) of KW columns), in order, each into the next ring slot once it is free."""
        for u in range(pid, U, P):
            row = row0 + (u >> LKS) * BM
            col0 = (u & ((1 << LKS) - 1)) * KW
            for kc in range(NKC // CK):  # chunks of CK [BM, 64] tiles
                slot = c & (NSLOT - 1)
                mbarrier.wait(empty.index(slot), ((c >> LNS) & 1) ^ 1)
                mbarrier.expect(full.index(slot), BM * 128 * CK)
                for j in gl.static_range(CK):
                    tma.async_load(
                        desc, [row, col0 + (kc * CK + j) * 64], full.index(slot), ring.index(slot * CK + j)
                    )
                c += 1
        return c

    @gluon.jit
    def _l2_units(
        w, row0, U, pid, P, RW: gl.constexpr, LKS: gl.constexpr, KW: gl.constexpr, BM: gl.constexpr
    ):
        """Producer warp: bulk-prefetch this program's units u = pid, pid + P, ... < U of a GEMM into L2 (unit u:
        rows row0 + (u >> LKS) * BM + 0..BM-1 of the weights w (row width RW), columns (u & (2^LKS - 1)) * KW +
        0..KW-1), BM / 32 rows per lane."""
        ll: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        lane = gl.arange(0, 32, layout=ll)
        nb = gl.full([32], KW * 2, gl.int32, layout=ll)
        for u in range(pid, U, P):
            row = row0 + (u >> LKS) * BM + (BM // 32) * lane
            p = w + row.to(gl.int64) * RW + (u & ((1 << LKS) - 1)) * KW
            for j in gl.static_range(BM // 32):
                gl.inline_asm_elementwise(
                    "cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;", "=r,l,r", [p + j * RW, nb],
                    dtype=gl.int32, is_pure=False, pack=1,
                )  # fmt: skip

    @gluon.jit
    def _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, s: gl.constexpr, pid, P, L, D: gl.constexpr, F: gl.constexpr,
                  LKS: gl.constexpr, AHEAD: gl.constexpr, BM: gl.constexpr):  # fmt: skip
        """Prefetch into L2 this program's units of the GEMM stage AHEAD GEMM stages after stage s of layer."""
        t: gl.constexpr = (s + AHEAD) % 6
        lt = layer + (s + AHEAD) // 6
        if AHEAD > 0:
            if lt < L:
                if t == 0:
                    _l2_units(wqkv, lt * 3 * D, 3 * D // BM, pid, P, D, 0, D, BM)
                elif t == 1:
                    _l2_units(wo, lt * D, D // BM, pid, P, D, 0, D, BM)
                elif t == 2:
                    _l2_units(wcq, lt * D, D // BM, pid, P, D, 0, D, BM)
                elif t == 3:
                    _l2_units(wco, lt * D, D // BM, pid, P, D, 0, D, BM)
                elif t == 4:
                    _l2_units(w1, lt * F, F // BM, pid, P, D, 0, D, BM)
                else:
                    _l2_units(w2, lt * D, F // BM, pid, P, F, LKS, D, BM)

    @gluon.jit
    def _producer(dqkv, do_, dcq, dco, d1, d2, dout, wqkv, wo, wcq, wco, w1, w2, ring, full, empty, pid, P, L, nrb,
                  D: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr,
                  AHEAD: gl.constexpr, BM: gl.constexpr):  # fmt: skip
        """The program's whole chunk sequence: per layer QKV, out, cross-Q, cross-out, FFN1, FFN2 (split-K),
        then the output layer's blocks. With AHEAD, each GEMM stage first prefetches into L2 the program's units
        of the stage AHEAD stages later (so that their chunks come from L2)."""
        NKC: gl.constexpr = D // 64
        c = 0
        for i in gl.static_range(AHEAD):  # the first stages' prefetches
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, 0, i - AHEAD, pid, P, L, D, F, LKS, AHEAD, BM)
        for layer in range(L):
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 0, pid, P, L, D, F, LKS, AHEAD, BM)
            c = _prod_units(
                dqkv, layer * 3 * D, c, 3 * D // BM, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS, CK, BM
            )
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 1, pid, P, L, D, F, LKS, AHEAD, BM)
            c = _prod_units(
                do_, layer * D, c, D // BM, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS, CK, BM
            )
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 2, pid, P, L, D, F, LKS, AHEAD, BM)
            c = _prod_units(
                dcq, layer * D, c, D // BM, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS, CK, BM
            )
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 3, pid, P, L, D, F, LKS, AHEAD, BM)
            c = _prod_units(
                dco, layer * D, c, D // BM, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS, CK, BM
            )
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 4, pid, P, L, D, F, LKS, AHEAD, BM)
            c = _prod_units(
                d1, layer * F, c, F // BM, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS, CK, BM
            )
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 5, pid, P, L, D, F, LKS, AHEAD, BM)
            c = _prod_units(
                d2, layer * D, c, F // BM, pid, P, ring, full, empty, LKS, D, NKC, NSLOT, LNS, CK, BM
            )
        _prod_units(dout, 0, c, nrb, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS, CK, BM)

    @gluon.jit
    def _opaque0():
        """A runtime 0 the compiler cannot see through. Added to the program id and the base pointers at the top
        of the layer loop, it keeps the stages' index and address tensors inside the loop: hoisted out of it
        (loop-invariant), they stay live across the whole loop and spill."""
        return gl.inline_asm_elementwise("mov.u32 $0, 0;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)

    @gluon.jit
    def _stats_r(st, M, eps, NPART: gl.constexpr, PART: gl.constexpr, D: gl.constexpr, MTP: gl.constexpr,
                 NW: gl.constexpr):  # fmt: skip
        """Mean and 1/std per token of the residual stream from its NPART parts' statistics in st [2, NPART, M]
        (mean, sum of squared deviations over PART features each; exact parallel combination; NPART need not be a
        power of 2)."""
        NPP: gl.constexpr = _np2(NPART)
        # 32 tokens: 4 parts per thread, over 4 lanes (short reductions; as below, slower for fewer tokens)
        if MTP >= 32:
            ol: gl.constexpr = gl.BlockedLayout([_np2(NPP // 4), 1], [_np2(min(4, NPP)), 32 // _np2(min(4, NPP))],
                                                [1, NW], [0, 1])  # fmt: skip
        else:
            ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MTP, MTP], [NW, 1], [1, 0])
        p = gl.arange(0, NPP, layout=gl.SliceLayout(1, ol))
        t = gl.arange(0, MTP, layout=gl.SliceLayout(0, ol))
        m = (t < M)[None, :] & (p < NPART)[:, None]
        mu = gl.load(st + p[:, None] * M + t[None, :], mask=m, other=0.0, cache_modifier=".cg")
        m2 = gl.load(st + (NPART + p[:, None]) * M + t[None, :], mask=m, other=0.0, cache_modifier=".cg")
        mean = gl.sum(mu, axis=0) / NPART
        dev = gl.where(m, mu - mean[None, :], 0.0)
        var = (gl.sum(m2, axis=0) + PART * gl.sum(dev * dev, axis=0)) / D
        return mean, 1.0 / gl.sqrt(var + eps)

    @gluon.jit
    def _x_load(x, XS, HO, M, xbuf, NKC: gl.constexpr, N: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr,
                HL: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """cp.async of the B side, one K chunk after another (small address tensors): xbuf[chunk] rows (h, t) =
        x[t, h * HO + chunk * 64 + 0..63] (token stride XS; h < HL: hi/lo halves HO apart); rows of h >= HL or
        tokens >= M are zero."""
        xl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
        kk = gl.arange(0, 64, layout=gl.SliceLayout(0, xl))
        r = gl.arange(0, N, layout=gl.SliceLayout(1, xl))
        t = r & (MTP - 1)
        h = r >> LMTP
        ok = ((h < HL) & (t < M))[:, None] & (kk >= 0)[None, :]
        off = t[:, None] * XS + (h * HO)[:, None] + kk[None, :]
        for kc in range(NKC):
            async_copy.async_load(
                xbuf.index(kc).slice(0, N), x + kc * 64 + off, mask=ok, cache_modifier=".cg"
            )
        async_copy.commit_group()

    @gluon.jit
    def _x_wait():
        async_copy.wait_group(0)
        fence_async_shared()
        gl.barrier()

    @gluon.jit
    def _consume(ring, full, empty, c, xbuf, N: gl.constexpr, NKC: gl.constexpr, NSLOT: gl.constexpr,
                 LNS: gl.constexpr, NW: gl.constexpr, CK: gl.constexpr, BM: gl.constexpr):  # fmt: skip
        """The MMAs of one unit: chunk after chunk from the ring (each released as soon as its wgmma is done)
        against the B side in xbuf -> [BM, N] fp32. (A runtime loop: unrolled, its operand descriptors stay
        live in registers.)"""
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, N, 16]
        )
        acc = gl.zeros([BM, N], gl.float32, layout=mma)
        # even / odd slots into two accumulators: independent wgmma chains
        acc1 = gl.zeros([BM, N], gl.float32, layout=mma)
        for k2 in range(NKC // CK // 2):
            slot = c & (NSLOT - 1)
            mbarrier.wait(full.index(slot), (c >> LNS) & 1)
            for j in gl.static_range(CK):
                acc = warpgroup_mma(ring.index(slot * CK + j),
                                    xbuf.index(2 * k2 * CK + j).slice(0, N).permute((1, 0)), acc, is_async=True)  # fmt: skip
            acc, acc1 = warpgroup_mma_wait(CK, deps=[acc, acc1])
            if k2 > 0:
                mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
            c += 1
            slot = c & (NSLOT - 1)
            mbarrier.wait(full.index(slot), (c >> LNS) & 1)
            for j in gl.static_range(CK):
                acc1 = warpgroup_mma(ring.index(slot * CK + j),
                                     xbuf.index((2 * k2 + 1) * CK + j).slice(0, N).permute((1, 0)), acc1,
                                     is_async=True)  # fmt: skip
            acc, acc1 = warpgroup_mma_wait(CK, deps=[acc, acc1])
            mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
            c += 1
        acc, acc1 = warpgroup_mma_wait(0, deps=[acc, acc1])
        mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
        acc = acc + acc1
        return acc, c

    @gluon.jit
    def _sum_hl(acc, scratch, N: gl.constexpr, MTP: gl.constexpr, HL: gl.constexpr, NW: gl.constexpr,
                BM: gl.constexpr):  # fmt: skip
        """[BM, MTP] (epilogue layout) = the sum of acc's HL column groups of MTP (the B side's hi/lo rows),
        through a padded shared-memory scratch."""
        el: gl.constexpr = _el(BM, NW)
        sc = scratch.slice(0, N, dim=1)
        sc.store(acc)
        gl.barrier()
        out = sc.slice(0, MTP, dim=1).load(el)
        if HL == 2:
            out += sc.slice(MTP, MTP, dim=1).load(el)
        gl.barrier()
        return out

    @gluon.jit
    def _gemm_r(x, XS, HO, M, ring, full, empty, c, xbuf, scratch, bias, cvec, st, eps, res, x2, y, YS, pid, P, U,
                tsp, NKC: gl.constexpr, N: gl.constexpr, LN_: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr,
                HL: gl.constexpr, LN: gl.constexpr, RELU: gl.constexpr, RES: gl.constexpr, D: gl.constexpr,
                NPART: gl.constexpr, NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr,
                TS: gl.constexpr, BM: gl.constexpr):  # fmt: skip
        """Units pid, pid + P, ... < U of y = x @ W^T + b, BM features each, from the B side x (one load per
        stage). LN: the LayerNorm folded in (statistics in st, column sums cvec). RES: + the residual stream
        res (fp32), stored back as the residual stream (res, hi/lo x2, statistics); else stored as fp16 into
        y (token stride YS), with RELU a ReLU first."""
        el: gl.constexpr = _el(BM, NW)
        rr = gl.arange(0, BM, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        tm = (tt < M)[None, :] & (rr >= 0)[:, None]
        if pid < U:
            _x_load(x, XS, HO, M, xbuf, NKC, N, MTP, LMTP, HL, NW)
        if LN:
            mean, rstd = _stats_r(st, M, eps, NPART, BM, D, MTP, NW)
            mean = gl.convert_layout(mean, gl.SliceLayout(0, el))
            rstd = gl.convert_layout(rstd, gl.SliceLayout(0, el))
        if pid < U:
            _x_wait()
            if TS:
                gl.store(tsp + 2, _now())
                gl.store(tsp + 3, _now())
        for u in range(pid, U, P):
            n0 = u * BM
            b = gl.load(bias + n0 + rr)
            if LN:
                cv = gl.load(cvec + n0 + rr)
            if RES:
                r = gl.load(
                    res + tt[None, :] * D + n0 + rr[:, None], mask=tm, other=0.0, cache_modifier=".cg"
                )
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, CK, BM)
            if TS:
                gl.store(tsp + 4, _now())
            out = _sum_hl(acc, scratch, N, MTP, HL, NW, BM)
            if TS:
                gl.store(tsp + 6, _now())
            if LN:
                out = (out - cv[:, None] * mean[None, :]) * rstd[None, :]
            out += b[:, None]
            if RES:
                out += r
                _store_res(out, res, x2, st, n0, u, M, BM, MTP, D, NPART, NW)
            else:
                if RELU:
                    out = gl.maximum(out, 0.0)
                gl.store(y + tt[None, :] * YS + n0 + rr[:, None], out.to(gl.float16), mask=tm)
            if TS:
                gl.store(tsp + 7, _now())
        return c

    @gluon.jit
    def _ffn2_sum(x, x2, st, bias, part, cnt, rb, M, D: gl.constexpr, KS: gl.constexpr, MTP: gl.constexpr,
                  NPART: gl.constexpr, NW: gl.constexpr, BM: gl.constexpr):  # fmt: skip
        """The residual stream of block rb: x + bias + its KS partials, in slice order."""
        el: gl.constexpr = _el(BM, NW)
        rr = gl.arange(0, BM, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        po = tt[None, :] * BM + rr[:, None]
        tm = (tt < M)[None, :] & (rr >= 0)[:, None]
        n0 = rb * BM
        tot = gl.load(x + tt[None, :] * D + n0 + rr[:, None], mask=tm, other=0.0, cache_modifier=".cg")
        tot += gl.load(bias + n0 + rr)[:, None]
        for j in gl.static_range(KS):
            tot += gl.load(part + ((rb * KS + j) * BM) * MTP + po, cache_modifier=".cg")
        gl.atomic_xchg(cnt + rb, 0, sem="relaxed", scope="gpu")
        _store_res(tot, x, x2, st, n0, rb, M, BM, MTP, D, NPART, NW)

    @gluon.jit
    def _ffn2_r(hb, M, ring, full, empty, c, xbuf, scratch, bias, x, x2, st, part, cnt, pid, P, tsp,
                D: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr, N: gl.constexpr, LN_: gl.constexpr,
                MTP: gl.constexpr, LMTP: gl.constexpr, NPART: gl.constexpr, NW: gl.constexpr,
                NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr, TS: gl.constexpr,
                BM: gl.constexpr):  # fmt: skip
        """FFN2 + residual, split-K: unit u = (block of BM features u >> LKS, K slice u & (KS - 1) of D
        columns). Every slice stores its partial [tokens, BM] (fp32) into part and counts itself; the last one
        of a block adds them (in slice order), bias and residual and stores the residual stream."""
        KS: gl.constexpr = F // D
        NKC: gl.constexpr = D // 64
        el: gl.constexpr = _el(BM, NW)
        rr = gl.arange(0, BM, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        po = tt[None, :] * BM + rr[:, None]  # token-major partials: coalesced
        tm = (tt < M)[None, :] & (rr >= 0)[:, None]
        NU: gl.constexpr = F // BM
        # programs with two units (fewer than 32 tokens: their sums are short): both partials first, then both
        # counts at once (one atomic round trip, not two); else unit after unit (a block's sum overlaps the
        # other programs' second units)
        two = False
        if MTP < 32:
            two = (2 * P >= NU) & (P < NU)
        if two:
            ll: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
            i = gl.arange(0, 32 * NW, layout=ll)
            ones = gl.full([32 * NW], 1, gl.int32, layout=ll)
            u1 = pid + P
            has1 = u1 < NU
            rb0 = pid >> LKS
            rb1 = u1 >> LKS
            _x_load(hb + (pid & (KS - 1)) * D, F, 0, M, xbuf, NKC, N, MTP, LMTP, 1, NW)
            _x_wait()
            if TS:
                gl.store(tsp + 2, _now())
                gl.store(tsp + 3, _now())
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, CK, BM)
            gl.store(part + (pid * BM) * MTP + po, _sum_hl(acc, scratch, N, MTP, 1, NW, BM))
            if has1:
                if (P & (KS - 1)) != 0:
                    _x_load(hb + (u1 & (KS - 1)) * D, F, 0, M, xbuf, NKC, N, MTP, LMTP, 1, NW)
                    _x_wait()
                acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, CK, BM)
                gl.store(part + (u1 * BM) * MTP + po, _sum_hl(acc, scratch, N, MTP, 1, NW, BM))
            if TS:
                gl.store(tsp + 4, _now())
            gl.barrier()
            a0 = gl.atomic_add(cnt + rb0 + i * 0, ones, mask=i == 0, sem="acq_rel", scope="gpu")
            a1 = gl.atomic_add(cnt + rb1 + i * 0, ones, mask=(i == 0) & has1, sem="acq_rel", scope="gpu")
            arrived0 = gl.sum(gl.where(i == 0, a0, 0), axis=0)
            arrived1 = gl.sum(gl.where((i == 0) & has1, a1, 0), axis=0)
            if TS:
                gl.store(tsp + 6, _now())
            if arrived0 == KS - 1:
                _ffn2_sum(x, x2, st, bias, part, cnt, rb0, M, D, KS, MTP, NPART, NW, BM)
            if has1 & (arrived1 == KS - 1):
                _ffn2_sum(x, x2, st, bias, part, cnt, rb1, M, D, KS, MTP, NPART, NW, BM)
            if TS:
                gl.store(tsp + 7, _now())
        for u in range(pid, NU if not two else 0, P):
            rb = u >> LKS
            ks = u & (KS - 1)
            n0 = rb * BM
            if (u == pid) | ((P & (KS - 1)) != 0):  # (with P a multiple of KS all units share the K slice)
                _x_load(hb + ks * D, F, 0, M, xbuf, NKC, N, MTP, LMTP, 1, NW)
                _x_wait()
            if TS:
                gl.store(tsp + 2, _now())
                gl.store(tsp + 3, _now())
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, CK, BM)
            if TS:
                gl.store(tsp + 4, _now())
            out = _sum_hl(acc, scratch, N, MTP, 1, NW, BM)
            gl.store(part + (u * BM) * MTP + po, out)
            gl.barrier()
            arrived = gl.atomic_add(cnt + rb, 1, sem="acq_rel", scope="gpu")
            gl.barrier()
            if TS:
                gl.store(tsp + 6, _now())
            if arrived == KS - 1:  # the last slice of the block: the sum in slice order, whichever it is
                tot = gl.load(
                    x + tt[None, :] * D + n0 + rr[:, None], mask=tm, other=0.0, cache_modifier=".cg"
                )
                tot += gl.load(bias + n0 + rr)[:, None]
                for j in gl.static_range(KS):
                    if j == ks:
                        tot += out
                    else:
                        tot += gl.load(part + ((rb * KS + j) * BM) * MTP + po, cache_modifier=".cg")
                gl.atomic_xchg(cnt + rb, 0, sem="relaxed", scope="gpu")
                _store_res(tot, x, x2, st, n0, rb, M, BM, MTP, D, NPART, NW)
            if TS:
                gl.store(tsp + 7, _now())
        return c

    @gluon.jit
    def _out_r(x2, M, ring, full, empty, c, xbuf, scratch, st, eps, bout, cout, logits, amax, aidx, ntm, FT, LT, S, V,
               nrb, pid, P,
               D: gl.constexpr, N: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr, NPART: gl.constexpr,
               NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr, ARGMAX: gl.constexpr,
               BM: gl.constexpr, NOTIME: gl.constexpr):  # fmt: skip
        """Final LayerNorm (folded) + output layer: blocks of BM rows pid, pid + P, ..., the B side (the
        residual stream's hi part) loaded once. ARGMAX: every block also stores its maximum per token and the
        first row where it occurs (amax, aidx [blocks, MTP]); NOTIME: not over the timestamp tokens FT .. LT for
        the tokens of rows whose flag ntm [R] is set (<notimestamps>)."""
        NKC: gl.constexpr = D // 64
        el: gl.constexpr = _el(BM, NW)
        rr = gl.arange(0, BM, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        _x_load(x2, 2 * D, D, M, xbuf, NKC, N, MTP, LMTP, 1, NW)
        mean, rstd = _stats_r(st, M, eps, NPART, BM, D, MTP, NW)
        mean = gl.convert_layout(mean, gl.SliceLayout(0, el))
        rstd = gl.convert_layout(rstd, gl.SliceLayout(0, el))
        _x_wait()
        if NOTIME:
            nt = gl.load(ntm + tt // S, mask=tt < M, other=0).to(gl.int32)
        for rb in range(pid, nrb, P):
            n0 = rb * BM
            rm = n0 + rr < V
            cv = gl.load(cout + n0 + rr, mask=rm, other=0.0)
            b = gl.load(bout + n0 + rr, mask=rm, other=0.0)
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW, CK, BM)
            out = _sum_hl(acc, scratch, N, MTP, 1, NW, BM)
            out = (out - cv[:, None] * mean[None, :]) * rstd[None, :] + b[:, None]
            gl.store(logits + tt[None, :] * V + n0 + rr[:, None], out, mask=(tt < M)[None, :] & rm[:, None])
            if ARGMAX:
                om = gl.where(rm[:, None] & (tt >= 0)[None, :], out, float("-inf"))
                if NOTIME:
                    vv = n0 + rr
                    om = gl.where(((vv >= FT) & (vv <= LT))[:, None] & (nt != 0)[None, :], float("-inf"), om)
                # the reductions over the block's rows with the rows within threads (see _store_res)
                sl: gl.constexpr = _els(BM, MTP, NW)
                om2 = gl.convert_layout(om, sl)
                r2 = gl.arange(0, BM, layout=gl.SliceLayout(1, sl))
                t2 = gl.arange(0, MTP, layout=gl.SliceLayout(0, sl))
                mx = gl.max(om2, axis=0)
                ix = gl.min(gl.where(om2 == mx[None, :], r2[:, None], BM), axis=0) + n0
                gl.store(amax + rb * MTP + t2, mx)
                gl.store(aidx + rb * MTP + t2, ix)
        return c

    @gluon.jit
    def _argmax_r(amax, aidx, pred, M, nrb, pid, P, MTP: gl.constexpr, NW: gl.constexpr):
        """pred[t] = argmax over the vocabulary for tokens t = pid, pid + P, ... < M, from the output blocks'
        partials (ties: the first index, as torch.argmax)."""
        BR: gl.constexpr = 1024
        bl: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
        j = gl.arange(0, BR, layout=bl)
        for t in range(pid, M, P):
            best = gl.max(gl.full([BR], float("-inf"), gl.float32, layout=bl), axis=0)
            bi = gl.max(gl.zeros([BR], gl.int32, layout=bl), axis=0)
            for j0 in range(0, nrb, BR):
                jm = j0 + j < nrb
                v = gl.load(amax + (j0 + j) * MTP + t, mask=jm, other=float("-inf"), cache_modifier=".cg")
                ix = gl.load(aidx + (j0 + j) * MTP + t, mask=jm, other=0, cache_modifier=".cg")
                m = gl.max(v, axis=0)
                i = gl.min(gl.where(jm & (v == m), ix, 1 << 30), axis=0)
                bi = gl.where(m > best, i, bi)  # an earlier chunk keeps a tie
                best = gl.maximum(best, m)
            gl.store(pred + t, bi.to(gl.int64))

    @gluon.jit
    def _stage_end(bar, err, g, P, pid, ts, MAXS: gl.constexpr, TS: gl.constexpr):
        """Arrive at the grid barrier that ends stage g - 1 and wait for the others (stamps: arrival of stage
        g - 1, its release, the start of stage g)."""
        if TS:
            gl.store(ts + ((g - 1) * P + pid) * 8 + 5, _now())
        _garrive(bar)
        _gwait(bar, err, g * P, MAXS)
        if TS:
            gl.store(ts + ((g - 1) * P + pid) * 8 + 1, _now())
            gl.store(ts + (g * P + pid) * 8, _now())

    @gluon.jit
    def _consumer(tok, offs, TSTR, OSTR, emb, pe, xscale, bqkv, cqkv, bo, bcq, ccq, bco, b1, c1, b2, bout, cout,
                  kc, vc, C, ck, cv, T, logits, new_kv, x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt,
                  bar, err, done, ts, amax, aidx, pred, ntm, FT, LT, R, S, L, V, eps, ring, full, empty, xbuf, scratch, pid, P,
                  D: gl.constexpr, H: gl.constexpr, DK: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr,
                  MTP: gl.constexpr, LMTP: gl.constexpr, HL: gl.constexpr, NL: gl.constexpr, LNL: gl.constexpr,
                  N1: gl.constexpr, LN1: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr,
                  NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr, MAXS: gl.constexpr, PFA: gl.constexpr,
                  TS: gl.constexpr, ARGMAX: gl.constexpr, BM: gl.constexpr, NOTIME: gl.constexpr):  # fmt: skip
        """The consumer warps: embedding, L layers of 8 stages, the output layer; a grid barrier between
        stages."""
        M = R * S
        NKC: gl.constexpr = D // 64
        NPART: gl.constexpr = D // BM
        nrb = (V + BM - 1) // BM
        if TS:
            gl.store(ts + pid * 8, _now())
        for nu in range(pid, NPART, P):  # parts of BM features
            _embed_g(nu, M, S, tok, offs, TSTR, OSTR, emb, pe, xscale, V, x, x2, st, D, BM, NPART, MTP, NW)
        g = 1
        _stage_end(bar, err, g, P, pid, ts, MAXS, TS)
        c = 0
        for layer in range(L):
            z = _opaque0() * 128  # (see _opaque0; a multiple of 128 keeps the alignment the compiler knows)
            x2_z = x2 + z
            x_z = x + z
            qkv_z = qkv + z
            ob_z = ob + z
            q2_z = q2 + z
            hb_z = hb + z
            st_z = st + z
            part_z = part + z
            cnt2_z = cnt2 + z
            xp_z = xp + z
            xml_z = xml + z
            xcnt_z = xcnt + z
            kc_z = kc + z
            vc_z = vc + z
            ck_z = ck + z
            cv_z = cv + z
            new_kv_z = new_kv + z
            offs_z = offs + z
            pid_z = pid + z
            # LN1 + QKV -> qkv; meanwhile the self-attention's cache rows into L2
            c = _gemm_r(x2_z, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, bqkv + layer * 3 * D,
                        cqkv + layer * 3 * D, st_z, eps, x_z, x2_z, qkv_z, 3 * D, pid_z, P, 3 * D // BM, ts + (g * P + pid_z) * 8,
                        NKC, NL, LNL, MTP, LMTP, HL, True, False, False, D, NPART, NW, NSLOT, LNS, CK, TS, BM)  # fmt: skip
            g += 1
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 5, _now())
            _garrive(bar)
            if PFA:
                if pid_z < R * H:
                    cb = ((layer * R + pid_z // H) * H + pid_z % H).to(gl.int64) * C * DK
                    off = gl.load(offs_z + (pid_z // H) * OSTR)
                    _prefetch_rows(kc_z + cb, off, 128, NW)
                    _prefetch_rows(vc_z + cb, off, 128, NW)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # self-attention -> ob
            _self_attn_g(
                pid_z,
                P,
                R,
                S,
                layer,
                qkv_z,
                offs_z,
                OSTR,
                kc_z,
                vc_z,
                C,
                new_kv_z,
                ob_z,
                D,
                H,
                DK,
                SP,
                BT,
                NW,
                ts + (g * P + pid_z) * 8,
                TS,
            )
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # out-projection + residual
            c = _gemm_r(ob_z, D, 0, M, ring, full, empty, c, xbuf, scratch, bo + layer * D, bo, st_z, eps, x_z, x2_z, x_z, D,
                        pid_z, P, D // BM, ts + (g * P + pid_z) * 8, NKC, N1, LN1, MTP, LMTP, 1, False, False, True, D,
                        NPART, NW, NSLOT, LNS, CK, TS, BM)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # LN2 + cross-Q -> q2; meanwhile the cross-attention's keys/values into L2
            c = _gemm_r(x2_z, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, bcq + layer * D, ccq + layer * D, st_z,
                        eps, x_z, x2_z, q2_z, D, pid_z, P, D // BM, ts + (g * P + pid_z) * 8, NKC, NL, LNL, MTP, LMTP, HL, True,
                        False, False, D, NPART, NW, NSLOT, LNS, CK, TS, BM)  # fmt: skip
            g += 1
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 5, _now())
            _garrive(bar)
            if PFA:
                if pid_z < R * H * NS:
                    sp = pid_z % NS
                    bh = pid_z // NS
                    tch = ((T + NS - 1) // NS + 15) // 16 * 16
                    t_lo = sp * tch
                    nk = gl.minimum(T, t_lo + tch) - t_lo
                    cb = ((layer * R + bh // H) * H + bh % H).to(gl.int64) * T * DK + t_lo * DK
                    _prefetch_rows(ck_z + cb, nk, 128, NW)
                    _prefetch_rows(cv_z + cb, nk, 128, NW)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # cross-attention -> ob
            _cross_attn_g(
                pid_z,
                P,
                R,
                S,
                layer,
                q2_z,
                ck_z,
                cv_z,
                T,
                ob_z,
                xp_z,
                xml_z,
                xcnt_z,
                D,
                H,
                DK,
                SP,
                BT,
                NS,
                NW,
                ts + (g * P + pid_z) * 8,
                TS,
            )
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # cross-out + residual
            c = _gemm_r(ob_z, D, 0, M, ring, full, empty, c, xbuf, scratch, bco + layer * D, bco, st_z, eps, x_z, x2_z, x_z, D,
                        pid_z, P, D // BM, ts + (g * P + pid_z) * 8, NKC, N1, LN1, MTP, LMTP, 1, False, False, True, D,
                        NPART, NW, NSLOT, LNS, CK, TS, BM)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # LN3 + FFN1 + ReLU -> hb
            c = _gemm_r(x2_z, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, b1 + layer * F, c1 + layer * F, st_z,
                        eps, x_z, x2_z, hb_z, F, pid_z, P, F // BM, ts + (g * P + pid_z) * 8, NKC, NL, LNL, MTP, LMTP, HL, True,
                        True, False, D, NPART, NW, NSLOT, LNS, CK, TS, BM)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # FFN2 + residual (split-K)
            c = _ffn2_r(hb_z, M, ring, full, empty, c, xbuf, scratch, b2 + layer * D, x_z, x2_z, st_z, part_z, cnt2_z, pid_z, P,
                        ts + (g * P + pid_z) * 8, D, F, LKS, N1, LN1, MTP, LMTP, NPART, NW, NSLOT, LNS, CK, TS, BM)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
        _out_r(x2, M, ring, full, empty, c, xbuf, scratch, st, eps, bout, cout, logits, amax, aidx, ntm, FT, LT, S, V,
               nrb, pid, P, D, N1, MTP, LMTP, NPART, NW, NSLOT, LNS, CK, ARGMAX, BM, NOTIME)  # fmt: skip
        if ARGMAX:  # the argmax over the output blocks' partials
            g += 1
            _stage_end(bar, err, g, P, pid, ts, MAXS, TS)
            _argmax_r(amax, aidx, pred, M, nrb, pid, P, MTP, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 8 + 5, _now())
        _gexit(bar, done, P)

    @gluon.jit
    def _decoder_kernel_r(
        tok, offs, TSTR, OSTR, emb, pe, xscale,
        dqkv, bqkv, cqkv, do_, bo, dcq, bcq, ccq, dco, bco, d1, b1, c1, d2, b2, dout, bout, cout,
        wqkv, wo, wcq, wco, w1, w2,
        kc, vc, C, ck, cv, T, logits, new_kv,
        x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, err, done, ts, amax, aidx, pred, ntm, FT, LT,
        R, S, L, V, eps, stop,
        D: gl.constexpr, H: gl.constexpr, DK: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr,
        MTP: gl.constexpr, LMTP: gl.constexpr, HL: gl.constexpr, NL: gl.constexpr, LNL: gl.constexpr,
        N1: gl.constexpr, LN1: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr,
        NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, CK: gl.constexpr, MAXS: gl.constexpr, PFA: gl.constexpr,
        AHEAD: gl.constexpr = 0, TS: gl.constexpr = False, ARGMAX: gl.constexpr = False, PW: gl.constexpr = 1,
        NOTIME: gl.constexpr = False,
        BM: gl.constexpr = 64, HAS_STOP: gl.constexpr = False,
    ):  # fmt: skip
        """A decoder call (see the module docstring): shared memory, then the consumer warps and the producer
        warp."""
        if HAS_STOP:
            # Every CTA reads the same immutable scalar before any producer/barrier starts.
            # The caller must order flag writes before this launch, never change it in flight.
            if gl.load(stop):
                return
        pid = gl.program_id(0)
        P = gl.num_programs(0)
        sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        ring = gl.allocate_shared_memory(gl.float16, [NSLOT * CK, BM, 64], sl)  # NSLOT chunks of CK tiles
        xbuf = gl.allocate_shared_memory(gl.float16, [_np2(D // 64), NL, 64], sl)  # [K chunk, B rows, 64]
        # padding 2 (not 1): the scratch is written with 8-byte vectors
        pl: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NL, 2]], [BM, NL], [1, 0])
        scratch = gl.allocate_shared_memory(gl.float32, [BM, NL], pl)
        full = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NSLOT):
            mbarrier.init(full.index(i), count=1)
            mbarrier.init(empty.index(i), count=1)
        fence_async_shared()
        nrb = (V + BM - 1) // BM
        gl.warp_specialize(
            [
                (_consumer, (tok, offs, TSTR, OSTR, emb, pe, xscale, bqkv, cqkv, bo, bcq, ccq, bco, b1, c1, b2, bout, cout, kc, vc,
                             C, ck, cv, T, logits, new_kv, x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar,
                             err, done, ts, amax, aidx, pred, ntm, FT, LT, R, S, L, V, eps, ring, full, empty, xbuf, scratch, pid, P,
                             D, H, DK, F, LKS, MTP, LMTP, HL, NL, LNL, N1, LN1, SP, BT, NS, NW, NSLOT, LNS, CK, MAXS, PFA,
                             TS, ARGMAX, BM, NOTIME)),
                (_producer, (dqkv, do_, dcq, dco, d1, d2, dout, wqkv, wo, wcq, wco, w1, w2, ring, full, empty, pid, P,
                             L, nrb, D, F, LKS, NSLOT, LNS, CK, AHEAD, BM)),
            ],
            [PW],
            [48],
        )  # fmt: skip


def weight_desc(w: torch.Tensor, bm: int = 64):
    """TMA descriptor of weights [rows, K] (fp16) in [bm, 64] tiles."""
    lay = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    return TensorDescriptor.from_tensor(w.reshape(-1, w.shape[-1]), [bm, 64], lay)


class RingDecoder:
    """The decoder megakernel for any number of programs (see the module docstring); built on the weights
    :class:`owsm_trt.megakernel.MegaDecoder` prepares (LayerNorms folded in).

    ``programs``: the number of programs (SMs it occupies) for single-row calls while an encoder runs
    concurrently (decode_stream; default: half as many as the largest stage has units, 32 for the 1B model:
    the most SMs left to the encoder, at two units per program); ``solo_programs`` for single-row calls with
    nothing alongside (``step_fn(..., solo=True)``; measured H200/102M default64) and ``row_programs``
    for calls of several rows (default:
    as many as the largest stage has units, 64; H200/102M automatically uses one CTA per head
    for R5..8, while an explicit value retains a fixed budget). ``chunk``: [64, 64] weight tiles per ring slot (and per
    wait / release); the ring holds ``ring_kb`` KB. ``ahead``: GEMM stages ahead whose weights the producer
    also prefetches into L2. ``exclusive``: 3 idle warps more, so that a program takes all of its SM's
    registers (no other kernel's blocks beside it)."""

    name = "megakernel-ring"
    argmax = True  # step_fn takes pred
    supports_finished_stop = True  # One immutable scalar may skip an entire R1/S1 call.

    def __init__(self, mega, programs: int | None = None, row_programs: int | None = None,
                 solo_programs: int | None = None, num_warps: int = 4, chunk: int = 2, ring_kb: int = 128,
                 maxs: int = 1 << 24, ts: bool = False, prefetch_attention: bool = True, ahead: int = 0,
                 exclusive: bool = False):  # fmt: skip
        self.m = mega
        d, f = mega.D, mega.F
        dev = mega.device
        bm = 16 * num_warps
        ks = f // d if f % d == 0 else 0
        if (d % bm or d % mega.H or d // mega.H != 64 or (d // 64) % 2 or not 256 <= d <= 1024 or ks < 1
                or ks & (ks - 1) or chunk not in (1, 2, 4) or num_warps not in (4, 8)):  # fmt: skip
            raise ValueError("the model's shapes do not fit the ring megakernel's tiling")
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        self.bm = bm  # weight rows per unit: one wgmma M tile per warpgroup
        h200_base = ((d, f, mega.H, mega.L) == (384, 1536, 6, 6) and num_warps == 4
                     and nsm == 132 and "H200" in torch.cuda.get_device_name(dev))
        self.P = min(nsm, programs or f // bm // 2)
        # Serial H200/102M benefits from more CTAs; NS still derives from P so the
        # reduction order matches concurrent calls. Explicit solo budgets take precedence.
        self.P_solo = min(nsm, solo_programs or (64 if h200_base else f // bm))
        self.P_rows = min(nsm, row_programs or f // bm)
        # H200/102M R5..8 has more attention heads than the default24 CTAs. The
        # measured schedule gives each head a CTA while retaining the original NS.
        # An explicit row_programs value keeps the requested fixed schedule.
        self.row_head_parallel = row_programs is None and h200_base
        while (d // 64) % (2 * chunk):  # (an even number of ring slots per unit: two accumulators)
            chunk //= 2
        self.ck = chunk
        self.slots = 1 << (ring_kb // (bm // 8 * self.ck)).bit_length() - 1  # a power of 2
        self.pw = 4 if exclusive else 1
        self.ahead = ahead
        self.nw, self.maxs, self.pfa = num_warps, maxs, prefetch_attention
        self.descs = tuple(
            weight_desc(w, bm) for w in (mega.wqkv, mega.wo, mega.wcq, mega.wco, mega.w1, mega.w2, mega.wout)
        )
        self.err = mega.err
        self.opts = {"TS": ts}
        self.programs = self.P
        pmax = max(self.P, self.P_solo, self.P_rows, 48 if self.row_head_parallel else 0)
        self.ts = torch.zeros((8 * mega.L + 3) * pmax * 8 if ts else 1, dtype=torch.int64, device=dev)

    def check(self) -> None:
        self.m.check()

    @property
    def solo_differs(self) -> bool:
        """Whether single-row calls take more SMs when nothing runs alongside (step_fn's ``solo``)."""
        return self.P_solo != self.P

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, kind="decode", pred=None,
                solo=False, notime=None, stop=None):  # fmt: skip
        """The call on these buffers (see MegaDecoder.step_fn); with ``pred`` [R, S] (int64) it also writes the
        argmax of the logits there (``run.writes_pred``), with ``notime`` = (flags [R] bool, first timestamp
        token, last one) not over the timestamp tokens in the flagged rows (see MegaBackend.argmax). ``solo``: no
        encoder runs alongside (single-row calls take ``solo_programs`` then)."""
        m = self.m
        R, S = tokens.shape
        if stop is not None and (R != 1 or S != 1 or stop.numel() != 1
                                 or stop.dtype != torch.bool or stop.device != tokens.device):
            raise ValueError("finished-stop requires R1/S1 and one boolean on the decoder device")
        M = R * S
        if M > MAX_TOKENS:
            raise ValueError(f"the ring decoder kernel takes up to {MAX_TOKENS} tokens per call, not {M}")
        D, F, L, H = m.D, m.F, m.L, m.H
        P = (self.P_solo if solo else self.P) if R == 1 else self.P_rows
        if self.row_head_parallel and 5 <= R <= 8:
            P = R * m.H
        dev = m.device
        mtp = max(4, _pow2(M))
        hl = 2 if mtp <= 16 else 1  # LayerNorm GEMMs: the residual stream as a hi/lo pair while it fits
        nl, n1 = max(8, hl * mtp), max(8, mtp)
        # cross-attention key slices: for single-row calls from the programs alongside an encoder, also when
        # solo (more programs), so that a call gives the same result either way
        ns = max(1, min(8, (self.P if R == 1 else self.P_rows) // (R * H)))
        f16 = {"device": dev, "dtype": torch.float16}
        i32 = {"device": dev, "dtype": torch.int32}
        x = torch.zeros(M, D, device=dev)
        x2 = torch.zeros(M, 2, D, **f16)
        qkv = torch.zeros(M, 3 * D, **f16)
        ob, q2 = torch.zeros(M, D, **f16), torch.zeros(M, D, **f16)
        hb = torch.zeros(M, F, **f16)
        bm = self.bm
        st = torch.zeros(2, D // bm, M, device=dev)
        part = torch.zeros(F // bm, mtp, bm, device=dev)
        cnt2 = torch.zeros(D // bm, **i32)
        xp = torch.zeros(ns, M, D, device=dev)
        xml = torch.zeros(ns, M, H, 2, device=dev)
        xcnt = torch.zeros(R * H, **i32)
        bar, done = torch.zeros(1, **i32), torch.zeros(1, **i32)
        nrb = (m.V + bm - 1) // bm
        amax = torch.zeros(nrb, mtp, device=dev)
        aidx = torch.zeros(nrb, mtp, **i32)
        argmax = pred is not None
        if pred is None:
            pred = torch.zeros(1, device=dev, dtype=torch.long)
        ntm, ft, lt = notime if notime is not None else (torch.zeros(1, dtype=torch.bool, device=dev), 0, -1)
        dq, do_, dcq, dco, d1, d2, dout = self.descs
        C, T = self_k.shape[3], cross_k.shape[3]
        args = (
            tokens, offset, tokens.stride(0), offset.stride(0), m.emb, m.pe, m.xscale,
            dq, m.bqkv, m.cqkv, do_, m.bo, dcq, m.bcq, m.ccq, dco, m.bco, d1, m.b1, m.c1, d2, m.b2, dout, m.bout,
            m.cout, m.wqkv, m.wo, m.wcq, m.wco, m.w1, m.w2,
            self_k, self_v, C, cross_k, cross_v, T, logits, new_kv,
            x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, self.err, done, self.ts, amax, aidx, pred, ntm, ft, lt,
            R, S, L, m.V, m.eps, stop if stop is not None else bar,
        )  # fmt: skip
        const = dict(
            D=D, H=H, DK=D // H, F=F, LKS=_log2(F // D), MTP=mtp, LMTP=_log2(mtp), HL=hl, NL=nl, LNL=_log2(nl), N1=n1,
            LN1=_log2(n1), SP=max(16, _pow2(S)), BT=64, NS=ns, NW=self.nw, NSLOT=self.slots, LNS=_log2(self.slots), CK=self.ck,
            MAXS=self.maxs, PFA=self.pfa, AHEAD=self.ahead, TS=bool(self.opts["TS"]), ARGMAX=argmax, PW=self.pw, BM=bm, NOTIME=argmax and notime is not None, HAS_STOP=stop is not None,
        )  # fmt: skip

        def run() -> None:
            _decoder_kernel_r[(P,)](*args, **const, num_warps=self.nw)

        run.buffers = (x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, done, amax, aidx, pred)
        run.const = const
        run.writes_cache = True
        run.writes_pred = argmax
        run.skips_finished = stop is not None
        run.programs = P
        run.kernel = self  # (profiling: the time stamps in self.ts)
        return run
