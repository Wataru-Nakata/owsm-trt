"""The decoder megakernel in Gluon (Hopper, sm90) for any number of programs: weights stream through a ring.

Same stages, grid barriers, attention and folded LayerNorms as :mod:`owsm_trt.megakernel_gluon`, but the
GEMM work is cut into units of 64 weight rows over all of K (FFN2: K / KS, split-K), and program pid takes
units pid, pid + P, ... of a stage. The units' [64 rows, 64 columns] chunks stream through a ring of NSLOT
chunks in shared memory, fed by a producer warp (warp specialization) that runs ahead over the program's
whole sequence of chunks, across the grid barriers; the four consumer warps wait for each chunk, run its
wgmma (weights on the A side, the tokens on the B side: [hi/lo x tokens, K] from L2) and release it.

So a call needs no more programs than its largest stage has units (64 for the 1B model: FFN1 / FFN2) and
leaves the other SMs free, e.g. to the encoder that runs concurrently in decode_stream: an all-SM
persistent kernel stalls it.
"""

import os

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
        _embed_g,
        _garrive,
        _gexit,
        _gwait,
        _now,
        _prefetch_rows,
        _self_attn_g,
        _stats_finish,
        _stats_load,
        _store_res,
    )

__all__ = ["RingDecoder", "available", "MAX_TOKENS", "ROUTE_TOKENS", "ROUTE_ROWS"]

# two accumulators per unit (experiment: OWSM_RING_ACC2=1)
ACC2 = gl.constexpr(os.environ.get("OWSM_RING_ACC2", "1") == "1") if gluon is not None else False

# calls of up to ROUTE_TOKENS tokens and ROUTE_ROWS rows go to the kernel (MegaBackend)
ROUTE_TOKENS = MAX_TOKENS
ROUTE_ROWS = 8

if gluon is not None:

    @gluon.jit
    def _prod_units(desc, row0, c, U, pid, P, ring, full, empty, LKS: gl.constexpr, KW: gl.constexpr,
                    NKC: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr):  # fmt: skip
        """Producer: the chunks of units u = pid, pid + P, ... < U of a GEMM (unit u: rows row0 + (u >> LKS) *
        64, K slice u & (2^LKS - 1) of KW columns), in order, each into the next ring slot once it is free."""
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
    def _l2_units(w, row0, U, pid, P, RW: gl.constexpr, LKS: gl.constexpr, KW: gl.constexpr):
        """Producer warp: bulk-prefetch this program's units u = pid, pid + P, ... < U of a GEMM into L2 (unit u:
        rows row0 + (u >> LKS) * 64 + 0..63 of the weights w (row width RW), columns (u & (2^LKS - 1)) * KW +
        0..KW-1), two rows per lane."""
        ll: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        lane = gl.arange(0, 32, layout=ll)
        nb = gl.full([32], KW * 2, gl.int32, layout=ll)
        for u in range(pid, U, P):
            row = row0 + (u >> LKS) * 64 + 2 * lane
            p = w + row.to(gl.int64) * RW + (u & ((1 << LKS) - 1)) * KW
            for j in gl.static_range(2):
                gl.inline_asm_elementwise(
                    "cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0, 0;", "=r,l,r", [p + j * RW, nb],
                    dtype=gl.int32, is_pure=False, pack=1,
                )  # fmt: skip

    @gluon.jit
    def _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, s: gl.constexpr, pid, P, L, D: gl.constexpr, F: gl.constexpr,
                  LKS: gl.constexpr, AHEAD: gl.constexpr):  # fmt: skip
        """Prefetch into L2 this program's units of the GEMM stage AHEAD GEMM stages after stage s of layer."""
        t: gl.constexpr = (s + AHEAD) % 6
        lt = layer + (s + AHEAD) // 6
        if AHEAD > 0:
            if lt < L:
                if t == 0:
                    _l2_units(wqkv, lt * 3 * D, 3 * D // 64, pid, P, D, 0, D)
                elif t == 1:
                    _l2_units(wo, lt * D, D // 64, pid, P, D, 0, D)
                elif t == 2:
                    _l2_units(wcq, lt * D, D // 64, pid, P, D, 0, D)
                elif t == 3:
                    _l2_units(wco, lt * D, D // 64, pid, P, D, 0, D)
                elif t == 4:
                    _l2_units(w1, lt * F, F // 64, pid, P, D, 0, D)
                else:
                    _l2_units(w2, lt * D, F // 64, pid, P, F, LKS, D)

    @gluon.jit
    def _producer(dqkv, do_, dcq, dco, d1, d2, dout, wqkv, wo, wcq, wco, w1, w2, ring, full, empty, pid, P, L, nrb,
                  D: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr,
                  AHEAD: gl.constexpr):  # fmt: skip
        """The program's whole chunk sequence: per layer QKV, out, cross-Q, cross-out, FFN1, FFN2 (split-K),
        then the output layer's blocks. With AHEAD, each GEMM stage first prefetches into L2 the program's units
        of the stage AHEAD stages later (so that their chunks come from L2)."""
        NKC: gl.constexpr = D // 64
        c = 0
        for i in gl.static_range(AHEAD):  # the first stages' prefetches
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, 0, i - AHEAD, pid, P, L, D, F, LKS, AHEAD)
        for layer in range(L):
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 0, pid, P, L, D, F, LKS, AHEAD)
            c = _prod_units(
                dqkv, layer * 3 * D, c, 3 * D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS
            )
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 1, pid, P, L, D, F, LKS, AHEAD)
            c = _prod_units(do_, layer * D, c, D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 2, pid, P, L, D, F, LKS, AHEAD)
            c = _prod_units(dcq, layer * D, c, D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 3, pid, P, L, D, F, LKS, AHEAD)
            c = _prod_units(dco, layer * D, c, D // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 4, pid, P, L, D, F, LKS, AHEAD)
            c = _prod_units(d1, layer * F, c, F // 64, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)
            _l2_stage(wqkv, wo, wcq, wco, w1, w2, layer, 5, pid, P, L, D, F, LKS, AHEAD)
            c = _prod_units(d2, layer * D, c, F // 64, pid, P, ring, full, empty, LKS, D, NKC, NSLOT, LNS)
        _prod_units(dout, 0, c, nrb, pid, P, ring, full, empty, 0, D, NKC, NSLOT, LNS)

    @gluon.jit
    def _opaque0():
        """A runtime 0 the compiler cannot see through. Added to the program id and the base pointers at the top
        of the layer loop, it keeps the stages' index and address tensors inside the loop: hoisted out of it
        (loop-invariant), they stay live across the whole loop and spill."""
        return gl.inline_asm_elementwise("mov.u32 $0, 0;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)

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
                 LNS: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """The MMAs of one unit: chunk after chunk from the ring (each released as soon as its wgmma is done)
        against the B side in xbuf -> [64, N] fp32. (A runtime loop: unrolled, its operand descriptors stay
        live in registers.)"""
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, N, 16]
        )
        acc = gl.zeros([64, N], gl.float32, layout=mma)
        if ACC2:  # even / odd chunks into two accumulators: independent wgmma chains
            acc1 = gl.zeros([64, N], gl.float32, layout=mma)
            for k2 in range(NKC // 2):
                slot = c & (NSLOT - 1)
                mbarrier.wait(full.index(slot), (c >> LNS) & 1)
                acc = warpgroup_mma(
                    ring.index(slot), xbuf.index(2 * k2).slice(0, N).permute((1, 0)), acc, is_async=True
                )
                acc, acc1 = warpgroup_mma_wait(1, deps=[acc, acc1])
                if k2 > 0:
                    mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
                c += 1
                slot = c & (NSLOT - 1)
                mbarrier.wait(full.index(slot), (c >> LNS) & 1)
                acc1 = warpgroup_mma(
                    ring.index(slot), xbuf.index(2 * k2 + 1).slice(0, N).permute((1, 0)), acc1, is_async=True
                )
                acc, acc1 = warpgroup_mma_wait(1, deps=[acc, acc1])
                mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
                c += 1
            acc, acc1 = warpgroup_mma_wait(0, deps=[acc, acc1])
            mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
            acc = acc + acc1
        else:
            for kc in range(NKC):
                slot = c & (NSLOT - 1)
                mbarrier.wait(full.index(slot), (c >> LNS) & 1)
                acc = warpgroup_mma(
                    ring.index(slot), xbuf.index(kc).slice(0, N).permute((1, 0)), acc, is_async=True
                )
                acc = warpgroup_mma_wait(1, deps=[acc])
                if kc > 0:
                    mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
                c += 1
            acc = warpgroup_mma_wait(0, deps=[acc])
            mbarrier.arrive(empty.index((c - 1) & (NSLOT - 1)))
        return acc, c

    @gluon.jit
    def _sum_hl(acc, scratch, N: gl.constexpr, MTP: gl.constexpr, HL: gl.constexpr, NW: gl.constexpr):
        """[64, MTP] (epilogue layout) = the sum of acc's HL column groups of MTP (the B side's hi/lo rows),
        through a padded shared-memory scratch."""
        el: gl.constexpr = _el(64, NW)
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
                NPART: gl.constexpr, NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr,
                TS: gl.constexpr):  # fmt: skip
        """Units pid, pid + P, ... < U of y = x @ W^T + b, 64 features each, from the B side x (one load per
        stage). LN: the LayerNorm folded in (statistics in st, column sums cvec). RES: + the residual stream
        res (fp32), stored back as the residual stream (res, hi/lo x2, statistics); else stored as fp16 into
        y (token stride YS), with RELU a ReLU first."""
        el: gl.constexpr = _el(64, NW)
        rr = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        tm = (tt < M)[None, :] & (rr >= 0)[:, None]
        if pid < U:
            _x_load(x, XS, HO, M, xbuf, NKC, N, MTP, LMTP, HL, NW)
        if LN:
            mu, m2 = _stats_load(st, M, NPART, MTP, NW)
            mean, rstd = _stats_finish(mu, m2, eps, NPART, 64, D)
            mean = gl.convert_layout(mean, gl.SliceLayout(0, el))
            rstd = gl.convert_layout(rstd, gl.SliceLayout(0, el))
        if pid < U:
            _x_wait()
            if TS:
                gl.store(tsp + 2, _now())
                gl.store(tsp + 3, _now())
        for u in range(pid, U, P):
            n0 = u * 64
            b = gl.load(bias + n0 + rr)
            if LN:
                cv = gl.load(cvec + n0 + rr)
            if RES:
                r = gl.load(
                    res + tt[None, :] * D + n0 + rr[:, None], mask=tm, other=0.0, cache_modifier=".cg"
                )
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW)
            if TS:
                gl.store(tsp + 4, _now())
            out = _sum_hl(acc, scratch, N, MTP, HL, NW)
            if TS:
                gl.store(tsp + 6, _now())
            if LN:
                out = (out - cv[:, None] * mean[None, :]) * rstd[None, :]
            out += b[:, None]
            if RES:
                out += r
                _store_res(out, res, x2, st, n0, u, M, 64, MTP, D, NPART, NW)
            else:
                if RELU:
                    out = gl.maximum(out, 0.0)
                gl.store(y + tt[None, :] * YS + n0 + rr[:, None], out.to(gl.float16), mask=tm)
            if TS:
                gl.store(tsp + 7, _now())
        return c

    @gluon.jit
    def _ffn2_r(hb, M, ring, full, empty, c, xbuf, scratch, bias, x, x2, st, part, cnt, pid, P, tsp,
                D: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr, N: gl.constexpr, LN_: gl.constexpr,
                MTP: gl.constexpr, LMTP: gl.constexpr, NPART: gl.constexpr, NW: gl.constexpr,
                NSLOT: gl.constexpr, LNS: gl.constexpr, TS: gl.constexpr):  # fmt: skip
        """FFN2 + residual, split-K: unit u = (block of 64 features u >> LKS, K slice u & (KS - 1) of D
        columns). Every slice stores its partial [tokens, 64] (fp32) into part and counts itself; the last one
        of a block adds the others, bias and residual and stores the residual stream."""
        KS: gl.constexpr = F // D
        NKC: gl.constexpr = D // 64
        el: gl.constexpr = _el(64, NW)
        rr = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        po = tt[None, :] * 64 + rr[:, None]  # token-major partials: coalesced
        tm = (tt < M)[None, :] & (rr >= 0)[:, None]
        for u in range(pid, F // 64, P):
            rb = u >> LKS
            ks = u & (KS - 1)
            n0 = rb * 64
            _x_load(hb + ks * D, F, 0, M, xbuf, NKC, N, MTP, LMTP, 1, NW)
            _x_wait()
            if TS:
                gl.store(tsp + 2, _now())
                gl.store(tsp + 3, _now())
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW)
            if TS:
                gl.store(tsp + 4, _now())
            out = _sum_hl(acc, scratch, N, MTP, 1, NW)
            gl.store(part + (u * 64) * MTP + po, out)
            gl.barrier()
            arrived = gl.atomic_add(cnt + rb, 1, sem="acq_rel", scope="gpu")
            gl.barrier()
            if TS:
                gl.store(tsp + 6, _now())
            if arrived == KS - 1:  # the last slice of the block
                tot = out + gl.load(bias + n0 + rr)[:, None]
                tot += gl.load(
                    x + tt[None, :] * D + n0 + rr[:, None], mask=tm, other=0.0, cache_modifier=".cg"
                )
                for j in gl.static_range(KS):
                    if j != ks:
                        tot += gl.load(part + ((rb * KS + j) * 64) * MTP + po, cache_modifier=".cg")
                gl.atomic_xchg(cnt + rb, 0, sem="relaxed", scope="gpu")
                _store_res(tot, x, x2, st, n0, rb, M, 64, MTP, D, NPART, NW)
            if TS:
                gl.store(tsp + 7, _now())
        return c

    @gluon.jit
    def _out_r(x2, M, ring, full, empty, c, xbuf, scratch, st, eps, bout, cout, logits, amax, aidx, V, nrb, pid, P,
               D: gl.constexpr, N: gl.constexpr, MTP: gl.constexpr, LMTP: gl.constexpr, NPART: gl.constexpr,
               NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, ARGMAX: gl.constexpr):  # fmt: skip
        """Final LayerNorm (folded) + output layer: blocks of 64 rows pid, pid + P, ..., the B side (the
        residual stream's hi part) loaded once. ARGMAX: every block also stores its maximum per token and the
        first row where it occurs (amax, aidx [blocks, MTP])."""
        NKC: gl.constexpr = D // 64
        el: gl.constexpr = _el(64, NW)
        rr = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
        tt = gl.arange(0, MTP, layout=gl.SliceLayout(0, el))
        _x_load(x2, 2 * D, D, M, xbuf, NKC, N, MTP, LMTP, 1, NW)
        mu, m2 = _stats_load(st, M, NPART, MTP, NW)
        mean, rstd = _stats_finish(mu, m2, eps, NPART, 64, D)
        mean = gl.convert_layout(mean, gl.SliceLayout(0, el))
        rstd = gl.convert_layout(rstd, gl.SliceLayout(0, el))
        _x_wait()
        for rb in range(pid, nrb, P):
            n0 = rb * 64
            rm = n0 + rr < V
            cv = gl.load(cout + n0 + rr, mask=rm, other=0.0)
            b = gl.load(bout + n0 + rr, mask=rm, other=0.0)
            acc, c = _consume(ring, full, empty, c, xbuf, N, NKC, NSLOT, LNS, NW)
            out = _sum_hl(acc, scratch, N, MTP, 1, NW)
            out = (out - cv[:, None] * mean[None, :]) * rstd[None, :] + b[:, None]
            gl.store(logits + tt[None, :] * V + n0 + rr[:, None], out, mask=(tt < M)[None, :] & rm[:, None])
            if ARGMAX:
                om = gl.where(rm[:, None] & (tt >= 0)[None, :], out, float("-inf"))
                mx = gl.max(om, axis=0)
                ix = gl.min(gl.where(om == mx[None, :], rr[:, None], 64), axis=0) + n0
                gl.store(amax + rb * MTP + tt, mx)
                gl.store(aidx + rb * MTP + tt, ix)
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
    def _consumer(tok, offs, emb, pe, xscale, bqkv, cqkv, bo, bcq, ccq, bco, b1, c1, b2, bout, cout,
                  kc, vc, C, ck, cv, T, logits, new_kv, x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt,
                  bar, err, done, ts, amax, aidx, pred, R, S, L, V, eps, ring, full, empty, xbuf, scratch, pid, P,
                  D: gl.constexpr, H: gl.constexpr, DK: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr,
                  MTP: gl.constexpr, LMTP: gl.constexpr, HL: gl.constexpr, NL: gl.constexpr, LNL: gl.constexpr,
                  N1: gl.constexpr, LN1: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr,
                  NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, MAXS: gl.constexpr, PFA: gl.constexpr,
                  TS: gl.constexpr, ARGMAX: gl.constexpr):  # fmt: skip
        """The four consumer warps: embedding, L layers of 8 stages, the output layer; a grid barrier between
        stages."""
        M = R * S
        NKC: gl.constexpr = D // 64
        NPART: gl.constexpr = D // 64
        nrb = (V + 63) // 64
        if TS:
            gl.store(ts + pid * 8, _now())
        for nu in range(pid, NPART, P):  # parts of 64 features
            _embed_g(nu, M, S, tok, offs, emb, pe, xscale, V, x, x2, st, D, 64, NPART, MTP, NW)
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
                        cqkv + layer * 3 * D, st_z, eps, x_z, x2_z, qkv_z, 3 * D, pid_z, P, 3 * D // 64, ts + (g * P + pid_z) * 8,
                        NKC, NL, LNL, MTP, LMTP, HL, True, False, False, D, NPART, NW, NSLOT, LNS, TS)  # fmt: skip
            g += 1
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 5, _now())
            _garrive(bar)
            if PFA:
                if pid_z < R * H:
                    cb = ((layer * R + pid_z // H) * H + pid_z % H).to(gl.int64) * C * DK
                    off = gl.load(offs_z + pid_z // H)
                    _prefetch_rows(kc_z + cb, off, 128, NW)
                    _prefetch_rows(vc_z + cb, off, 128, NW)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # self-attention -> ob
            _self_attn_g(
                pid_z, P, R, S, layer, qkv_z, offs_z, kc_z, vc_z, C, new_kv_z, ob_z, D, H, DK, SP, BT, NW
            )
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # out-projection + residual
            c = _gemm_r(ob_z, D, 0, M, ring, full, empty, c, xbuf, scratch, bo + layer * D, bo, st_z, eps, x_z, x2_z, x_z, D,
                        pid_z, P, D // 64, ts + (g * P + pid_z) * 8, NKC, N1, LN1, MTP, LMTP, 1, False, False, True, D,
                        NPART, NW, NSLOT, LNS, TS)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # LN2 + cross-Q -> q2; meanwhile the cross-attention's keys/values into L2
            c = _gemm_r(x2_z, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, bcq + layer * D, ccq + layer * D, st_z,
                        eps, x_z, x2_z, q2_z, D, pid_z, P, D // 64, ts + (g * P + pid_z) * 8, NKC, NL, LNL, MTP, LMTP, HL, True,
                        False, False, D, NPART, NW, NSLOT, LNS, TS)  # fmt: skip
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
            )
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # cross-out + residual
            c = _gemm_r(ob_z, D, 0, M, ring, full, empty, c, xbuf, scratch, bco + layer * D, bco, st_z, eps, x_z, x2_z, x_z, D,
                        pid_z, P, D // 64, ts + (g * P + pid_z) * 8, NKC, N1, LN1, MTP, LMTP, 1, False, False, True, D,
                        NPART, NW, NSLOT, LNS, TS)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # LN3 + FFN1 + ReLU -> hb
            c = _gemm_r(x2_z, 2 * D, D, M, ring, full, empty, c, xbuf, scratch, b1 + layer * F, c1 + layer * F, st_z,
                        eps, x_z, x2_z, hb_z, F, pid_z, P, F // 64, ts + (g * P + pid_z) * 8, NKC, NL, LNL, MTP, LMTP, HL, True,
                        True, False, D, NPART, NW, NSLOT, LNS, TS)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
            # FFN2 + residual (split-K)
            c = _ffn2_r(hb_z, M, ring, full, empty, c, xbuf, scratch, b2 + layer * D, x_z, x2_z, st_z, part_z, cnt2_z, pid_z, P,
                        ts + (g * P + pid_z) * 8, D, F, LKS, N1, LN1, MTP, LMTP, NPART, NW, NSLOT, LNS, TS)  # fmt: skip
            g += 1
            _stage_end(bar, err, g, P, pid_z, ts, MAXS, TS)
        _out_r(x2, M, ring, full, empty, c, xbuf, scratch, st, eps, bout, cout, logits, amax, aidx, V, nrb, pid, P, D,
               N1, MTP, LMTP, NPART, NW, NSLOT, LNS, ARGMAX)  # fmt: skip
        if ARGMAX:  # the argmax over the output blocks' partials
            g += 1
            _stage_end(bar, err, g, P, pid, ts, MAXS, TS)
            _argmax_r(amax, aidx, pred, M, nrb, pid, P, MTP, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 8 + 5, _now())
        _gexit(bar, done, P)

    @gluon.jit
    def _decoder_kernel_r(
        tok, offs, emb, pe, xscale,
        dqkv, bqkv, cqkv, do_, bo, dcq, bcq, ccq, dco, bco, d1, b1, c1, d2, b2, dout, bout, cout,
        wqkv, wo, wcq, wco, w1, w2,
        kc, vc, C, ck, cv, T, logits, new_kv,
        x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, err, done, ts, amax, aidx, pred,
        R, S, L, V, eps,
        D: gl.constexpr, H: gl.constexpr, DK: gl.constexpr, F: gl.constexpr, LKS: gl.constexpr,
        MTP: gl.constexpr, LMTP: gl.constexpr, HL: gl.constexpr, NL: gl.constexpr, LNL: gl.constexpr,
        N1: gl.constexpr, LN1: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr,
        NW: gl.constexpr, NSLOT: gl.constexpr, LNS: gl.constexpr, MAXS: gl.constexpr, PFA: gl.constexpr,
        AHEAD: gl.constexpr = 0, TS: gl.constexpr = False, ARGMAX: gl.constexpr = False, PW: gl.constexpr = 1,
    ):  # fmt: skip
        """A decoder call (see the module docstring): shared memory, then the consumer warps and the producer
        warp."""
        pid = gl.program_id(0)
        P = gl.num_programs(0)
        sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        ring = gl.allocate_shared_memory(gl.float16, [NSLOT, 64, 64], sl)
        xbuf = gl.allocate_shared_memory(gl.float16, [D // 64, NL, 64], sl)  # [K chunk, B rows, 64]
        # padding 2 (not 1): the scratch is written with 8-byte vectors
        pl: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NL, 2]], [64, NL], [1, 0])
        scratch = gl.allocate_shared_memory(gl.float32, [64, NL], pl)
        full = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NSLOT, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NSLOT):
            mbarrier.init(full.index(i), count=1)
            mbarrier.init(empty.index(i), count=1)
        fence_async_shared()
        nrb = (V + 63) // 64
        gl.warp_specialize(
            [
                (_consumer, (tok, offs, emb, pe, xscale, bqkv, cqkv, bo, bcq, ccq, bco, b1, c1, b2, bout, cout, kc, vc,
                             C, ck, cv, T, logits, new_kv, x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar,
                             err, done, ts, amax, aidx, pred, R, S, L, V, eps, ring, full, empty, xbuf, scratch, pid, P,
                             D, H, DK, F, LKS, MTP, LMTP, HL, NL, LNL, N1, LN1, SP, BT, NS, NW, NSLOT, LNS, MAXS, PFA,
                             TS, ARGMAX)),
                (_producer, (dqkv, do_, dcq, dco, d1, d2, dout, wqkv, wo, wcq, wco, w1, w2, ring, full, empty, pid, P,
                             L, nrb, D, F, LKS, NSLOT, LNS, AHEAD)),
            ],
            [PW],
            [48],
        )  # fmt: skip


def weight_desc(w: torch.Tensor):
    """TMA descriptor of weights [rows, K] (fp16) in [64, 64] chunks."""
    lay = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    return TensorDescriptor.from_tensor(w.reshape(-1, w.shape[-1]), [64, 64], lay)


class RingDecoder:
    """The decoder megakernel for any number of programs (see the module docstring); built on the weights
    :class:`owsm_trt.megakernel.MegaDecoder` prepares (LayerNorms folded in). ``programs``: the number of
    programs (SMs it occupies), by default as many as the largest stage has units."""

    name = "megakernel-ring"
    argmax = True  # step_fn takes pred

    def __init__(self, mega, programs: int | None = None, num_warps: int = 4, slots: int = 16, maxs: int = 1 << 24,
                 ts: bool = False, prefetch_attention: bool = True, ahead: int = 0,
                 exclusive: bool = False):  # fmt: skip
        self.m = mega
        # exclusive: 3 idle warps more, so that a program takes all of its SM's registers (no other kernel's
        # blocks beside it)
        self.pw = 4 if exclusive else 1
        self.ahead = ahead  # GEMM stages ahead whose weights the producer prefetches into L2 (0: none)
        d, f = mega.D, mega.F
        dev = mega.device
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        self.P = min(nsm, programs or f // 64)
        pow2 = (d & (d - 1)) == 0 and f % d == 0 and ((f // d) & (f // d - 1)) == 0
        if not pow2 or not 256 <= d <= 1024 or d % mega.H or d // mega.H != 64:
            raise ValueError("the model's shapes do not fit the ring megakernel's tiling")
        self.nw, self.maxs, self.pfa, self.slots = num_warps, maxs, prefetch_attention, slots
        self.descs = tuple(
            weight_desc(w) for w in (mega.wqkv, mega.wo, mega.wcq, mega.wco, mega.w1, mega.w2, mega.wout)
        )
        self.err = mega.err
        self.opts = {"TS": ts}
        self.programs = self.P
        self.ts = torch.zeros((8 * mega.L + 3) * self.P * 8 if ts else 1, dtype=torch.int64, device=dev)

    def check(self) -> None:
        self.m.check()

    def step_fn(
        self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, kind="decode", pred=None
    ):
        """The call on these buffers (see MegaDecoder.step_fn); with ``pred`` [R, S] (int64) it also writes the
        argmax of the logits there (``run.writes_pred``)."""
        m = self.m
        R, S = tokens.shape
        M = R * S
        if M > MAX_TOKENS:
            raise ValueError(f"the ring decoder kernel takes up to {MAX_TOKENS} tokens per call, not {M}")
        D, F, L, H, P = m.D, m.F, m.L, m.H, self.P
        dev = m.device
        mtp = max(4, _pow2(M))
        hl = 2 if mtp <= 16 else 1  # LayerNorm GEMMs: the residual stream as a hi/lo pair while it fits
        nl, n1 = max(8, hl * mtp), max(8, mtp)
        ns = max(1, min(8, P // (R * H)))
        f16 = {"device": dev, "dtype": torch.float16}
        i32 = {"device": dev, "dtype": torch.int32}
        x = torch.zeros(M, D, device=dev)
        x2 = torch.zeros(M, 2, D, **f16)
        qkv = torch.zeros(M, 3 * D, **f16)
        ob, q2 = torch.zeros(M, D, **f16), torch.zeros(M, D, **f16)
        hb = torch.zeros(M, F, **f16)
        st = torch.zeros(2, D // 64, M, device=dev)
        part = torch.zeros(F // 64, mtp, 64, device=dev)
        cnt2 = torch.zeros(D // 64, **i32)
        xp = torch.zeros(ns, M, D, device=dev)
        xml = torch.zeros(ns, M, H, 2, device=dev)
        xcnt = torch.zeros(R * H, **i32)
        bar, done = torch.zeros(1, **i32), torch.zeros(1, **i32)
        nrb = (m.V + 63) // 64
        amax = torch.zeros(nrb, mtp, device=dev)
        aidx = torch.zeros(nrb, mtp, **i32)
        argmax = pred is not None
        if pred is None:
            pred = torch.zeros(1, device=dev, dtype=torch.long)
        dq, do_, dcq, dco, d1, d2, dout = self.descs
        C, T = self_k.shape[3], cross_k.shape[3]
        args = (
            tokens, offset, m.emb, m.pe, m.xscale,
            dq, m.bqkv, m.cqkv, do_, m.bo, dcq, m.bcq, m.ccq, dco, m.bco, d1, m.b1, m.c1, d2, m.b2, dout, m.bout,
            m.cout, m.wqkv, m.wo, m.wcq, m.wco, m.w1, m.w2,
            self_k, self_v, C, cross_k, cross_v, T, logits, new_kv,
            x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, self.err, done, self.ts, amax, aidx, pred,
            R, S, L, m.V, m.eps,
        )  # fmt: skip
        const = dict(
            D=D, H=H, DK=D // H, F=F, LKS=_log2(F // D), MTP=mtp, LMTP=_log2(mtp), HL=hl, NL=nl, LNL=_log2(nl), N1=n1,
            LN1=_log2(n1), SP=max(16, _pow2(S)), BT=64, NS=ns, NW=self.nw, NSLOT=self.slots, LNS=_log2(self.slots),
            MAXS=self.maxs, PFA=self.pfa, AHEAD=self.ahead, TS=bool(self.opts["TS"]), ARGMAX=argmax, PW=self.pw,
        )  # fmt: skip

        def run() -> None:
            _decoder_kernel_r[(P,)](*args, **const, num_warps=self.nw)

        run.buffers = (x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, done, amax, aidx, pred)
        run.const = const
        run.writes_cache = True
        run.writes_pred = argmax
        return run
