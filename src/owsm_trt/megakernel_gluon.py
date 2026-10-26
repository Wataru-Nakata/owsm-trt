"""The decoder megakernel in Gluon (Hopper, sm90): weights stream into shared memory ahead of the stages.

Same stages and buffers as :mod:`owsm_trt.megakernel` (Triton), for calls of up to 8 tokens (rows x window):

* Every GEMM stage gives each program one unit of BN weight rows over the whole K. Its slab is copied into a
  two-slot shared-memory ring by one TMA instruction two GEMM stages ahead (right after the program arrives at
  the barrier that ends the stage before), so it is there when the stage starts; after a barrier only the
  activations (L2) are loaded.
* wgmma wants 64 rows on its A side: the slab [BN, K] is stacked as S = 64 / BN column blocks into
  A = [64, K / S] and the activations go to the B side as [S x MT tokens, K / S]; one wgmma chain computes all
  block products and the result is the sum of the S diagonal blocks.
* LayerNorms are folded into the weights (see owsm_trt.megakernel._gemm): the residual stream comes as an fp16
  hi/lo pair, i.e. 2 x MT B-side rows, and the statistics come from residual-producing units of PART features
  each (no integer division by the number of parts: Triton miscompiles that in some builds).
* Attention stages use mma_v2 (m16n8k16) tiles with an online softmax; cross-attention is cut into slices
  of the keys for few rows and combined by the last slice of each (row, head).
"""

import torch

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import constexpr_function
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import async_copy, mma_v2
    from triton.experimental.gluon.language.nvidia.hopper import (
        fence_async_shared,
        mbarrier,
        tma,
        warpgroup_mma,
        warpgroup_mma_wait,
    )
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
except ImportError:  # no Triton, or a Triton without Gluon
    gluon = None

MAX_TOKENS = 32  # rows x window of a call
# calls of up to ROUTE_TOKENS tokens and ROUTE_ROWS rows go to the kernel (MegaBackend): faster than TensorRT
# with attention plugins per call there (H200, 1B model, whole graphed call: 1 row 0.66-1.05 ms vs 0.77-1.10 ms
# for windows 1..16), slower with 32 tokens (1.45 vs 1.13 ms) and with 16 rows (1.32-1.46 vs 1.24-1.35 ms)
ROUTE_TOKENS = 16
ROUTE_ROWS = 8


def available(device: torch.device) -> bool:
    return gluon is not None and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9


TESTED = ("3.8", "2.14")  # the Triton (and torch) versions the kernels are tested with; older ones lack APIs


def triton_version() -> str:
    try:
        import triton

        return triton.__version__
    except ImportError:
        return "not installed"


if gluon is not None:

    @constexpr_function
    def _el(bn, nw):
        """Layout of [bn features, tokens] epilogue tiles: threads along the features (contiguous in memory)."""
        tr = min(bn, 32)
        return gl.BlockedLayout([1, 1], [tr, 32 // tr], [1, nw], [0, 1])

    @constexpr_function
    def _els(bn, mt, nw):
        """Layout of [bn features, mt tokens] tiles for per-token reductions over the features: a thread holds
        consecutive features of one token (as few lanes per token as fit)."""
        lf = 32 // max(1, min(32, mt // nw))  # lanes along the features
        return gl.BlockedLayout([max(1, bn // lf), 1], [lf, 32 // lf], [1, nw], [0, 1])

    @gluon.jit
    def _now():
        return gl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=gl.int64, is_pure=False, pack=1
        )

    @gluon.jit
    def _opaque0():
        """A runtime 0 the compiler cannot see through. Added (times 128: the compiler keeps the alignment it
        knows) to the program id and the base pointers at the top of the layer loop, it keeps the stages' index
        and address tensors inside the loop: hoisted out of it (loop-invariant), they stay live across the
        whole loop and spill to local memory."""
        return gl.inline_asm_elementwise("mov.u32 $0, 0;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)

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
    def _issue_w(desc, row0, kc0, bar, wslot, BN: gl.constexpr, K: gl.constexpr, pred):
        """One TMA: BN weight rows from row0, K columns from chunk kc0 (of 64), as A tiles (K chunk, K block,
        row); desc = the weights viewed as [all K / S / 64, S, rows, 64] with S = 64 / BN (see slab_desc)."""
        S: gl.constexpr = 64 // BN
        NKC: gl.constexpr = K // S // 64
        l4: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=4)
        dst = wslot._reinterpret(gl.float16, [8, S, BN, 64], l4).slice(0, NKC, dim=0)  # a slot holds 8 chunks
        mbarrier.expect(bar, BN * K * 2, pred=pred)
        tma.async_load(desc, [kc0, 0, row0, 0], bar, dst, pred=pred)

    @gluon.jit
    def _x_issue(x, XS, M, xbuf, K: gl.constexpr, BN: gl.constexpr, LBN: gl.constexpr, MT: gl.constexpr,
                 LMT: gl.constexpr, HL: gl.constexpr, D: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """cp.async of the B side: rows (K chunk, K block s, [hi/lo,] token) of K / S / 64 x 64 columns, from x
        (token stride XS); HL = 2: hi/lo pairs (lo at x + D). (Shifts and masks only: Triton miscompiles some
        vector divisions by constants.)"""
        S: gl.constexpr = 64 // BN
        LS: gl.constexpr = 6 - LBN
        KP: gl.constexpr = K // S
        NKC: gl.constexpr = KP // 64
        ROWS: gl.constexpr = NKC * S * HL * MT
        xl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
        kk = gl.arange(0, 64, layout=gl.SliceLayout(0, xl))
        rho = gl.arange(0, ROWS, layout=gl.SliceLayout(1, xl))
        t = rho & (MT - 1)
        if HL == 2:
            h = (rho >> LMT) & 1
            sb = (rho >> (LMT + 1)) & (S - 1)
            kc = rho >> (LMT + 1 + LS)
            off = h * D + sb * KP + kc * 64
        else:
            sb = (rho >> LMT) & (S - 1)
            kc = rho >> (LMT + LS)
            off = sb * KP + kc * 64
        xp = x + t[:, None] * XS + off[:, None] + kk[None, :]
        xm = (t < M)[:, None] & (kk >= 0)[None, :]
        async_copy.async_load(xbuf.slice(0, ROWS), xp, mask=xm, cache_modifier=".cg")
        async_copy.commit_group()

    @gluon.jit
    def _mma(wslot, xb, wbar, phase, K: gl.constexpr, BN: gl.constexpr, NB: gl.constexpr, NW: gl.constexpr,
             tsp=None, TS: gl.constexpr = False):  # fmt: skip
        """Wait for the slab and the B side (xb: B tiles of NB rows per K chunk), then the wgmma chain (two
        accumulators while they are small) -> [64, NB] fp32. TS: stamps slab / B side ready, MMAs done at
        tsp + 2, 3, 4."""
        S: gl.constexpr = 64 // BN
        NKC: gl.constexpr = K // S // 64
        mbarrier.wait(wbar, phase)
        if TS:
            gl.store(tsp + 2, _now())
        async_copy.wait_group(0)
        if TS:
            gl.store(tsp + 3, _now())
        fence_async_shared()
        gl.barrier()
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, NB, 16]
        )
        acc0 = gl.zeros([64, NB], gl.float32, layout=mma)
        if NB <= 64:
            acc1 = gl.zeros([64, NB], gl.float32, layout=mma)
            for kc in gl.static_range(NKC):
                a_t = wslot.slice(kc * 64, 64)
                b_t = xb.slice(kc * NB, NB).permute((1, 0))
                if kc % 2 == 0:
                    acc0 = warpgroup_mma(a_t, b_t, acc0, is_async=True)
                else:
                    acc1 = warpgroup_mma(a_t, b_t, acc1, is_async=True)
            acc0, acc1 = warpgroup_mma_wait(0, deps=[acc0, acc1])
            acc0 = acc0 + acc1
            if TS:
                gl.store(tsp + 4, _now())
        else:
            for kc in gl.static_range(NKC):
                acc0 = warpgroup_mma(wslot.slice(kc * 64, 64), xb.slice(kc * NB, NB).permute((1, 0)), acc0,
                                     is_async=True)  # fmt: skip
            acc0 = warpgroup_mma_wait(0, deps=[acc0])
            if TS:
                gl.store(tsp + 4, _now())
        return acc0

    @gluon.jit
    def _prefetch_rows(base, n_rows, NT: gl.constexpr, NW: gl.constexpr):
        """L2 prefetch of n_rows rows of 128 bytes (a head's keys or values) from base (fp16 pointer)."""
        pl: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
        i = gl.arange(0, NT, layout=pl)
        for i0 in range(0, n_rows, NT):
            r = gl.minimum(i0 + i, n_rows - 1)
            gl.inline_asm_elementwise(
                "prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [base + r * 64], dtype=gl.int32, is_pure=False,
                pack=1,
            )  # fmt: skip

    @gluon.jit
    def _extract(acc, scratch, BN: gl.constexpr, MT: gl.constexpr, HL: gl.constexpr, NB: gl.constexpr,
                 NW: gl.constexpr):  # fmt: skip
        """[BN, MT] = sum of the S diagonal blocks of acc [64, NB] (and of the hi/lo halves with HL = 2), through
        shared memory (scratch [64, >= NB] fp32)."""
        S: gl.constexpr = 64 // BN
        sc = scratch.slice(0, NB, dim=1)
        sc.store(acc)
        gl.barrier()
        ol: gl.constexpr = _el(BN, NW)
        out = gl.zeros([BN, MT], gl.float32, layout=ol)
        for s in gl.static_range(S):
            blk = sc.slice(s * BN, BN, dim=0)
            if HL == 2:
                out += blk.slice(s * 2 * MT, MT, dim=1).load(ol) + blk.slice(s * 2 * MT + MT, MT, dim=1).load(
                    ol
                )
            else:
                out += blk.slice(s * MT, MT, dim=1).load(ol)
        gl.barrier()
        return out

    @gluon.jit
    def _stats_load(st, M, NPART: gl.constexpr, MT: gl.constexpr, NW: gl.constexpr):
        """The LayerNorm statistics partials [2, NPART, M] of the residual-producing units."""
        ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
        p = gl.arange(0, NPART, layout=gl.SliceLayout(1, ol))
        t = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        m = (t < M)[None, :] & (p >= 0)[:, None]
        mu = gl.load(st + p[:, None] * M + t[None, :], mask=m, other=0.0, cache_modifier=".cg")
        m2 = gl.load(st + (NPART + p[:, None]) * M + t[None, :], mask=m, other=0.0, cache_modifier=".cg")
        return mu, m2

    @gluon.jit
    def _stats_finish(mu, m2, eps, NPART: gl.constexpr, PART: gl.constexpr, D: gl.constexpr):
        """Mean and 1/std per token from equal-sized parts (exact parallel combination)."""
        mean = gl.sum(mu, axis=0) / NPART
        dev = mu - mean[None, :]
        var = (gl.sum(m2, axis=0) + PART * gl.sum(dev * dev, axis=0)) / D
        return mean, 1.0 / gl.sqrt(var + eps)

    @gluon.jit
    def _store_res(val, x, x2, st, n0, nu, M, BN: gl.constexpr, MT: gl.constexpr, D: gl.constexpr,
                   NPART: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """Residual-stream values val [BN features from n0, MT tokens]: fp32 into x, hi/lo into x2 [M, 2, D],
        the unit's statistics (mean, sum of squared deviations over its BN features) into st."""
        ol: gl.constexpr = _el(BN, NW)
        rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        m = (tt < M)[None, :] & (rr >= 0)[:, None]
        gl.store(x + tt[None, :] * D + n0 + rr[:, None], val, mask=m)
        hi = val.to(gl.float16)
        o2 = x2 + tt[None, :] * (2 * D) + n0 + rr[:, None]
        gl.store(o2, hi, mask=m)
        gl.store(o2 + D, (val - hi.to(gl.float32)).to(gl.float16), mask=m)
        # the statistics in a layout with the features within threads: over the lanes (the store layout), the
        # reductions take shuffle chains per token (~1 us per unit of 32 tokens)
        sl: gl.constexpr = _els(BN, MT, NW)
        v2 = gl.convert_layout(val, sl)
        t2 = gl.arange(0, MT, layout=gl.SliceLayout(0, sl))
        mu = gl.sum(v2, axis=0) / BN
        dev = v2 - mu[None, :]
        gl.store(st + nu * M + t2, mu, mask=t2 < M)
        gl.store(st + (NPART + nu) * M + t2, gl.sum(dev * dev, axis=0), mask=t2 < M)

    @gluon.jit
    def _embed_g(pid, M, S, tok, offs, TSTR, OSTR, emb, pe, xscale, V, x, x2, st, D: gl.constexpr, PART: gl.constexpr,
                 NPART: gl.constexpr, MT: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """x = embedding * sqrt(d) + positional encoding, for the PART features of unit pid."""
        if pid < NPART:
            ol: gl.constexpr = _el(PART, NW)
            rr = gl.arange(0, PART, layout=gl.SliceLayout(1, ol))
            tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
            tm = tt < M
            ids = gl.load(tok + (tt // S) * TSTR + tt % S, mask=tm, other=0)
            ids = gl.minimum(gl.maximum(ids, 0), V - 1).to(gl.int64)
            pos = gl.load(offs + (tt // S) * OSTR, mask=tm, other=0) + tt % S
            n0 = pid * PART
            m = tm[None, :] & (rr >= 0)[:, None]
            e = gl.load(emb + ids[None, :] * D + n0 + rr[:, None], mask=m, other=0.0).to(gl.float32)
            p = gl.load(pe + pos[None, :] * D + n0 + rr[:, None], mask=m, other=0.0)
            val = gl.where(m, e * xscale + p, 0.0)
            _store_res(val, x, x2, st, n0, pid, M, PART, MT, D, NPART, NW)

    @gluon.jit
    def _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP: gl.constexpr, BT: gl.constexpr, DK: gl.constexpr,
                      NW: gl.constexpr):  # fmt: skip
        """Keys kt [DK, BT] / values vt [BT, DK] (any layout) into the running softmax of the queries qa
        (dot-operand layout); ok [SP, BT] (mma layout): which keys each query sees."""
        mma2: gl.constexpr = gl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8]
        )
        la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2)
        lb: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma2, k_width=2)
        sc = mma_v2(qa, gl.convert_layout(kt, lb), gl.zeros([SP, BT], gl.float32, layout=mma2))
        sc = gl.where(ok, sc, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(sc, axis=1))
        alpha = gl.exp(m_i - m_new)
        p = gl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        pv = mma_v2(gl.convert_layout(p.to(gl.float16), la), gl.convert_layout(vt, lb),
                    gl.zeros([SP, DK], gl.float32, layout=mma2))  # fmt: skip
        return m_new, l_i, acc * alpha[:, None] + pv

    @gluon.jit
    def _kv_block(k, v, t0, t_end, tT, dT, tV, d_, DK: gl.constexpr):
        """Keys [DK, BT] (transposed) and values [BT, DK] t0 .. t0 + BT - 1 (those < t_end; the rest zero)."""
        kt = gl.load(k + (t0 + tT)[None, :] * DK + dT[:, None], mask=((t0 + tT) < t_end)[None, :] & (dT >= 0)[:, None],
                     other=0.0, cache_modifier=".cg")  # fmt: skip
        vt = gl.load(v + (t0 + tV)[:, None] * DK + d_[None, :], mask=((t0 + tV) < t_end)[:, None] & (d_ >= 0)[None, :],
                     other=0.0, cache_modifier=".cg")  # fmt: skip
        return kt, vt

    @gluon.jit
    def _self_attn_g(pid, P, R, S, layer, qkv, offs, OSTR, kc, vc, C, new_kv, o, D: gl.constexpr, H: gl.constexpr,
                     DK: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NW: gl.constexpr, tsp=None,
                     TS: gl.constexpr = False):  # fmt: skip
        """Per (row, head): the queries attend to the row's cache slots < offset and causally to its new keys
        (from registers); then the new keys/values go into the cache at the offset and to new_kv."""
        mma2: gl.constexpr = gl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8]
        )
        la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2)
        bl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [NW, 1], [1, 0])  # [rows, DK], DK contiguous
        bt: gl.constexpr = gl.BlockedLayout([8, 1], [4, 8], [1, NW], [0, 1])  # [DK, cols], DK contiguous
        s_ = gl.arange(0, SP, layout=gl.SliceLayout(1, bl))
        d_ = gl.arange(0, DK, layout=gl.SliceLayout(0, bl))
        dT = gl.arange(0, DK, layout=gl.SliceLayout(1, bt))
        tT = gl.arange(0, BT, layout=gl.SliceLayout(0, bt))
        tV = gl.arange(0, BT, layout=gl.SliceLayout(1, bl))
        tS = gl.arange(0, BT, layout=gl.SliceLayout(0, mma2))
        iS = gl.arange(0, SP, layout=gl.SliceLayout(1, mma2))
        jS = gl.arange(0, SP, layout=gl.SliceLayout(0, mma2))
        sm = (s_ < S)[:, None] & (d_ >= 0)[None, :]
        for u in range(pid, R * H, P):
            b = u // H
            h = u % H
            off = gl.load(offs + b * OSTR)
            rows = b * S + s_
            qp = qkv + rows[:, None] * (3 * D) + h * DK + d_[None, :]
            q = gl.load(qp, mask=sm, other=0.0, cache_modifier=".cg")
            kn = gl.load(qp + D, mask=sm, other=0.0, cache_modifier=".cg")
            vn = gl.load(qp + 2 * D, mask=sm, other=0.0, cache_modifier=".cg")
            qa = gl.convert_layout((q.to(gl.float32) * (DK**-0.5)).to(gl.float16), la)
            if TS:
                gl.store(tsp + 2, _now())
            cb = ((layer * R + b) * H + h).to(gl.int64) * C * DK
            m_i = gl.full([SP], float("-inf"), gl.float32, layout=gl.SliceLayout(1, mma2))
            l_i = gl.zeros([SP], gl.float32, layout=gl.SliceLayout(1, mma2))
            acc = gl.zeros([SP, DK], gl.float32, layout=mma2)
            if SP <= 16:  # the next block's keys / values load during this block's MMAs (more rows: spills)
                kt, vt = _kv_block(kc + cb, vc + cb, 0, off, tT, dT, tV, d_, DK)
                for t0 in range(0, off, BT):
                    kx, vx = _kv_block(kc + cb, vc + cb, t0 + BT, off, tT, dT, tV, d_, DK)
                    ok = ((t0 + tS) < off)[None, :] & (iS >= 0)[:, None]
                    m_i, l_i, acc = _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP, BT, DK, NW)
                    kt, vt = kx, vx
            else:
                for t0 in range(0, off, BT):
                    kt, vt = _kv_block(kc + cb, vc + cb, t0, off, tT, dT, tV, d_, DK)
                    ok = ((t0 + tS) < off)[None, :] & (iS >= 0)[:, None]
                    m_i, l_i, acc = _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP, BT, DK, NW)
            if TS:
                gl.store(tsp + 3, _now())
            ok = (jS[None, :] <= iS[:, None]) & (jS < S)[None, :]
            m_i, l_i, acc = _softmax_step(qa, gl.permute(kn, (1, 0)), vn, ok, m_i, l_i, acc, SP, SP, DK, NW)
            out = gl.convert_layout((acc / l_i[:, None]).to(gl.float16), bl)
            gl.store(o + rows[:, None] * D + h * DK + d_[None, :], out, mask=sm)
            dst = (off + s_)[:, None] * DK + d_[None, :]
            gl.store(kc + cb + dst, kn, mask=sm)
            gl.store(vc + cb + dst, vn, mask=sm)
            nk = (((layer * 2) * R + b) * H + h).to(gl.int64) * S * DK + s_[:, None] * DK + d_[None, :]
            gl.store(new_kv + nk, kn, mask=sm)
            gl.store(new_kv + nk + R * H * S * DK, vn, mask=sm)
            if TS:
                gl.store(tsp + 4, _now())

    @gluon.jit
    def _cross_attn_g(pid, P, R, S, layer, q2, ck, cv, T, o, xp, xml, xcnt, D: gl.constexpr, H: gl.constexpr,
                      DK: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr, NW: gl.constexpr,
                      tsp=None, TS: gl.constexpr = False, CR: gl.constexpr = None, CG: gl.constexpr = 1):  # fmt: skip
        """Per (row, head, slice of the T keys); with NS > 1 slices every slice stores its partial softmax
        (running max, sum, output) in xml / xp and the last one of a (row, head) combines them."""
        mma2: gl.constexpr = gl.NVMMADistributedLayout(
            version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8]
        )
        la: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma2, k_width=2)
        bl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [NW, 1], [1, 0])
        bt: gl.constexpr = gl.BlockedLayout([8, 1], [4, 8], [1, NW], [0, 1])
        s_ = gl.arange(0, SP, layout=gl.SliceLayout(1, bl))
        d_ = gl.arange(0, DK, layout=gl.SliceLayout(0, bl))
        dT = gl.arange(0, DK, layout=gl.SliceLayout(1, bt))
        tT = gl.arange(0, BT, layout=gl.SliceLayout(0, bt))
        tV = gl.arange(0, BT, layout=gl.SliceLayout(1, bl))
        tS = gl.arange(0, BT, layout=gl.SliceLayout(0, mma2))
        iS = gl.arange(0, SP, layout=gl.SliceLayout(1, mma2))
        sm = (s_ < S)[:, None] & (d_ >= 0)[None, :]
        M = R * S
        tch = ((T + NS - 1) // NS + 15) // 16 * 16
        for u in range(pid, R * H * NS, P):
            sp = u % NS
            bh = u // NS
            b = bh // H
            h = bh % H
            rows = b * S + s_
            q = gl.load(
                q2 + rows[:, None] * D + h * DK + d_[None, :], mask=sm, other=0.0, cache_modifier=".cg"
            )
            qa = gl.convert_layout((q.to(gl.float32) * (DK**-0.5)).to(gl.float16), la)
            if TS:
                gl.store(tsp + 2, _now())
            if CG == 1:
                cb = ((layer * R + b) * H + h).to(gl.int64) * T * DK
            else:
                cb = ((layer * CR + b // CG) * H + h).to(gl.int64) * T * DK
            t_lo = sp * tch
            t_hi = gl.minimum(T, t_lo + tch)
            m_i = gl.full([SP], float("-inf"), gl.float32, layout=gl.SliceLayout(1, mma2))
            l_i = gl.zeros([SP], gl.float32, layout=gl.SliceLayout(1, mma2))
            acc = gl.zeros([SP, DK], gl.float32, layout=mma2)
            if SP <= 16:  # the next block's keys / values load during this block's MMAs (more rows: spills)
                kt, vt = _kv_block(ck + cb, cv + cb, t_lo, t_hi, tT, dT, tV, d_, DK)
                for t0 in range(t_lo, t_hi, BT):
                    kx, vx = _kv_block(ck + cb, cv + cb, t0 + BT, t_hi, tT, dT, tV, d_, DK)
                    ok = ((t0 + tS) < t_hi)[None, :] & (iS >= 0)[:, None]
                    m_i, l_i, acc = _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP, BT, DK, NW)
                    kt, vt = kx, vx
            else:
                for t0 in range(t_lo, t_hi, BT):
                    kt, vt = _kv_block(ck + cb, cv + cb, t0, t_hi, tT, dT, tV, d_, DK)
                    ok = ((t0 + tS) < t_hi)[None, :] & (iS >= 0)[:, None]
                    m_i, l_i, acc = _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP, BT, DK, NW)
            if TS:
                gl.store(tsp + 3, _now())
            out = o + rows[:, None] * D + h * DK + d_[None, :]
            if NS == 1:
                gl.store(out, gl.convert_layout((acc / l_i[:, None]).to(gl.float16), bl), mask=sm)
            else:
                pa = xp + (sp * M + rows)[:, None] * D + h * DK + d_[None, :]
                pm = xml + ((sp * M + rows) * H + h) * 2
                gl.store(pa, gl.convert_layout(acc, bl), mask=sm)
                gl.store(pm, gl.convert_layout(m_i, gl.SliceLayout(1, bl)), mask=s_ < S)
                gl.store(pm + 1, gl.convert_layout(l_i, gl.SliceLayout(1, bl)), mask=s_ < S)
                gl.barrier()
                arrived = gl.atomic_add(xcnt + bh, 1, sem="acq_rel", scope="gpu")
                gl.barrier()
                if TS:
                    gl.store(tsp + 4, _now())
                if arrived == NS - 1:  # the last slice of (row, head): combine
                    mx = gl.full([SP], float("-inf"), gl.float32, layout=gl.SliceLayout(1, bl))
                    for j in gl.static_range(NS):
                        pmj = xml + ((j * M + rows) * H + h) * 2
                        mx = gl.maximum(mx, gl.load(pmj, mask=s_ < S, other=0.0, cache_modifier=".cg"))
                    num = gl.zeros([SP, DK], gl.float32, layout=bl)
                    den = gl.zeros([SP], gl.float32, layout=gl.SliceLayout(1, bl))
                    for j in gl.static_range(NS):
                        pmj = xml + ((j * M + rows) * H + h) * 2
                        wj = gl.exp(gl.load(pmj, mask=s_ < S, other=0.0, cache_modifier=".cg") - mx)
                        lj = gl.load(pmj + 1, mask=s_ < S, other=0.0, cache_modifier=".cg")
                        aj = gl.load(xp + (j * M + rows)[:, None] * D + h * DK + d_[None, :], mask=sm, other=0.0,
                                     cache_modifier=".cg")  # fmt: skip
                        num += wj[:, None] * aj
                        den += wj * lj
                    gl.atomic_xchg(xcnt + bh, 0, sem="relaxed", scope="gpu")
                    gl.store(out, (num / den[:, None]).to(gl.float16), mask=sm)
                if TS:
                    gl.store(tsp + 6, _now())

    @gluon.jit
    def _gemm_g(x, XS, M, wslot, xbuf, scratch, wbar, phase, bias, cvec, st, eps, res, n0,
                K: gl.constexpr, BN: gl.constexpr, LBN: gl.constexpr, MT: gl.constexpr, LMT: gl.constexpr,
                LN: gl.constexpr, HL: gl.constexpr, RES: gl.constexpr, D: gl.constexpr, NPART: gl.constexpr,
                PART: gl.constexpr, NW: gl.constexpr, tsp=None, TS: gl.constexpr = False):  # fmt: skip
        """One program's unit of y = X @ W^T + b: BN weight rows from n0 (slab already requested into wslot),
        MT tokens; with LN the LayerNorm is folded in (statistics from NPART parts of PART features). Loads
        that do not wait for the MMAs (bias, column sums, statistics, residual) are issued first. Returns
        [BN, MT] fp32 (before activation / store)."""
        S: gl.constexpr = 64 // BN
        NB: gl.constexpr = S * HL * MT
        ol: gl.constexpr = _el(BN, NW)
        rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        b = gl.load(bias + n0 + rr)
        if LN:
            cv = gl.load(cvec + n0 + rr)
            mu, m2 = _stats_load(st, M, NPART, MT, NW)
        if RES:
            r = gl.load(res + tt[None, :] * D + n0 + rr[:, None], mask=(tt < M)[None, :] & (rr >= 0)[:, None],
                        other=0.0, cache_modifier=".cg")  # fmt: skip
        _x_issue(x, XS, M, xbuf, K, BN, LBN, MT, LMT, HL, D, NW)
        acc = _mma(wslot, xbuf, wbar, phase, K, BN, NB, NW, tsp, TS)
        out = _extract(acc, scratch, BN, MT, HL, NB, NW)
        if TS:
            gl.store(tsp + 6, _now())
        if LN:
            mean, rstd = _stats_finish(mu, m2, eps, NPART, PART, D)
            mean = gl.convert_layout(mean, gl.SliceLayout(0, ol))
            rstd = gl.convert_layout(rstd, gl.SliceLayout(0, ol))
            out = (out - cv[:, None] * mean[None, :]) * rstd[None, :]
        out += b[:, None]
        if RES:
            out += r
        if TS:
            gl.store(tsp + 7, _now())
        return out

    @gluon.jit
    def _store16(out, y, y_stride, n0, M, BN: gl.constexpr, MT: gl.constexpr, RELU: gl.constexpr,
                 NW: gl.constexpr):  # fmt: skip
        ol: gl.constexpr = _el(BN, NW)
        rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        if RELU:
            out = gl.maximum(out, 0.0)
        gl.store(y + tt[None, :] * y_stride + n0 + rr[:, None], out.to(gl.float16),
                 mask=(tt < M)[None, :] & (rr >= 0)[:, None])  # fmt: skip

    @gluon.jit
    def _ffn2_g(pid, hb, M, wslot, xbuf, wbar, phase, bias, x, x2, st, part, cnt, D: gl.constexpr,
                F: gl.constexpr, LKS: gl.constexpr, MT: gl.constexpr, LMT: gl.constexpr, NW: gl.constexpr,
                tsp=None, TS: gl.constexpr = False):  # fmt: skip
        """FFN2 + residual, split-K: unit pid = (block of 64 features rb, slice ks of 512 columns of K; F / 512 =
        2^LKS slices). Every slice stores its partial sum and counts itself; the last one of a block adds the
        partials up in order, adds bias and residual and stores the residual stream (statistics in D / 64 parts
        of 64)."""
        KS: gl.constexpr = F // 512
        ks = pid & (KS - 1)
        rb = pid >> LKS
        n0 = rb * 64
        _x_issue(hb + ks * 512, F, M, xbuf, 512, 64, 6, MT, LMT, 1, D, NW)
        acc = _mma(wslot, xbuf, wbar, phase, 512, 64, MT, NW, tsp, TS)
        el: gl.constexpr = _el(64, NW)
        ra = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
        ta = gl.arange(0, MT, layout=gl.SliceLayout(0, el))
        po = ta[None, :] * D + n0 + ra[:, None]
        gl.store(
            part + ks * M * D + po, gl.convert_layout(acc, el), mask=(ta < M)[None, :] & (ra >= 0)[:, None]
        )
        gl.barrier()
        arrived = gl.atomic_add(cnt + rb, 1, sem="acq_rel", scope="gpu")
        gl.barrier()
        if TS:
            gl.store(tsp + 6, _now())
        if arrived == KS - 1:
            ol: gl.constexpr = _el(64, NW)
            rr = gl.arange(0, 64, layout=gl.SliceLayout(1, ol))
            tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
            m = (tt < M)[None, :] & (rr >= 0)[:, None]
            o = tt[None, :] * D + n0 + rr[:, None]
            tot = gl.load(x + o, mask=m, other=0.0, cache_modifier=".cg") + gl.load(bias + n0 + rr)[:, None]
            for j in gl.static_range(KS):
                tot += gl.load(part + j * M * D + o, mask=m, other=0.0, cache_modifier=".cg")
            gl.atomic_xchg(cnt + rb, 0, sem="relaxed", scope="gpu")
            _store_res(tot, x, x2, st, n0, rb, M, 64, MT, D, D // 64, NW)
        if TS:
            gl.store(tsp + 7, _now())

    @gluon.jit
    def _out_half(wslot, xb, wbar, phase, KH: gl.constexpr, MT: gl.constexpr, NW: gl.constexpr):
        """One half (KH columns of K) of a block of 64 output rows -> [64, MT] fp32."""
        return _mma(wslot, xb, wbar, phase, KH, 64, MT, NW)

    @gluon.jit
    def _out_store(acc, mean, rstd, bout, cout, logits, M, V, n0, MT: gl.constexpr, NW: gl.constexpr):
        """logits of a block of 64 rows from its K sums (final LayerNorm folded in; mean / rstd in the
        epilogue layout's token slice)."""
        el: gl.constexpr = _el(64, NW)
        ra = gl.arange(0, 64, layout=gl.SliceLayout(1, el))
        ta = gl.arange(0, MT, layout=gl.SliceLayout(0, el))
        rm = n0 + ra < V
        cv = gl.load(cout + n0 + ra, mask=rm, other=0.0)
        b = gl.load(bout + n0 + ra, mask=rm, other=0.0)
        out = (gl.convert_layout(acc, el) - cv[:, None] * mean[None, :]) * rstd[None, :] + b[:, None]
        gl.store(logits + ta[None, :] * V + n0 + ra[:, None], out, mask=(ta < M)[None, :] & rm[:, None])

    @gluon.jit
    def _decoder_kernel_g(
        tok, offs, TSTR, OSTR, emb, pe, xscale,
        dqkv, bqkv, cqkv, do_, bo, dcq, bcq, ccq, dco, bco, d1, b1, c1, d2, d2s, b2, dout, bout, cout,
        kc, vc, C, ck, cv, T, logits, new_kv,
        x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, err, done, ts,
        R, S, L, V, eps,
        D: gl.constexpr, H: gl.constexpr, DK: gl.constexpr, F: gl.constexpr, LF: gl.constexpr,
        MT: gl.constexpr, LMT: gl.constexpr, HL: gl.constexpr, MTF: gl.constexpr, LMTF: gl.constexpr,
        BNQ: gl.constexpr, LBNQ: gl.constexpr, BNF: gl.constexpr, LBNF: gl.constexpr, PART: gl.constexpr,
        LPART: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr, NW: gl.constexpr,
        MAXS: gl.constexpr, PFA: gl.constexpr, F2S: gl.constexpr, CR: gl.constexpr, CG: gl.constexpr,
        GQ: gl.constexpr, CSP: gl.constexpr, TS: gl.constexpr = False,
    ):  # fmt: skip
        """A decoder call: embedding, L layers (LN1+QKV, self-attention, out-projection, LN2+cross-Q,
        cross-attention, cross-out, LN3+FFN1, FFN2), final LN + output layer, a grid barrier between stages.
        QKV / FFN1 units of BNQ / BNF rows, out-projections of PART (statistics in D / PART parts), FFN2 split-K
        blocks of 64 rows (and the embedding: statistics in D / 64 parts), the output layer blocks of 64 rows in
        two halves of K. MT tokens (MTF >= 8 for the split-K / output GEMMs); LayerNorm GEMMs read the residual
        stream as HL = 2 (hi/lo) or 1 (hi only) fp16 rows. With PFA, attention inputs are prefetched into L2
        while a program waits at the barrier before the attention stage."""
        pid = gl.program_id(0)
        P = gl.num_programs(0)
        M = R * S
        NPO: gl.constexpr = D // PART
        NP64: gl.constexpr = D // 64
        # FFN2 with F2S: stacked units of PART rows (its statistics, and the embedding's, in D / PART parts),
        # else split-K blocks of 64 rows (statistics in D / 64 parts): the B side of a stacked unit holds all
        # of K for MT tokens, which fits up to 8 tokens
        P2: gl.constexpr = PART if F2S else 64
        NP2: gl.constexpr = D // P2
        sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        w0 = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
        w1 = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
        xbuf = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
        scratch = xbuf._reinterpret(gl.float32, [64, 256], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
        mb = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(mb.index(0), count=1)
        mbarrier.init(mb.index(1), count=1)
        fence_async_shared()
        has_q = pid < 3 * D // BNQ
        has_p = pid < NPO
        has_f = pid < F // BNF
        LKS: gl.constexpr = LF - 9  # FFN2: F / 512 slices of K
        has_2 = pid < (NP64 << LKS)
        nq = pid * BNQ
        npt = pid * PART
        nf = pid * BNF
        n2 = (pid >> LKS) * 64
        k2 = (pid & ((F // 512) - 1)) * 8
        nrb = (V + 63) // 64  # output blocks: pid, pid + P, ...
        NKH: gl.constexpr = D // 128  # K chunks of 64 in half of the output layer's K
        _issue_w(dqkv, nq, 0, mb.index(0), w0, BNQ, D, has_q)
        _issue_w(do_, npt, 0, mb.index(1), w1, PART, D, has_p)
        p0 = 0  # phases of the two slots' barriers
        p1 = 0
        if TS:
            gl.store(ts + pid * 8, _now())
        _embed_g(pid, M, S, tok, offs, TSTR, OSTR, emb, pe, xscale, V, x, x2, st, D, P2, NP2, MTF, NW)
        g = 1
        if TS:
            gl.store(ts + pid * 8 + 5, _now())
        _garrive(bar)
        _gwait(bar, err, g * P, MAXS)
        if TS:
            gl.store(ts + pid * 8 + 1, _now())
            gl.store(ts + (g * P + pid) * 8, _now())
        for layer in range(L):
            z = _opaque0() * 128  # (see _opaque0)
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
            nq_z = nq + z
            npt_z = npt + z
            nf_z = nf + z
            n2_z = n2 + z
            last = layer + 1 == L
            # LN1 + QKV (slot 0) -> qkv; then LN2+cross-Q's slab, and the self-attention's cache rows into L2
            if has_q:
                out = _gemm_g(x2_z, 2 * D, M, w0, xbuf, scratch, mb.index(0), p0, bqkv + layer * 3 * D,
                              cqkv + layer * 3 * D, st_z, eps, x_z, nq_z, D, BNQ, LBNQ, MT, LMT, True, HL, False, D, NP2,
                              P2, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                p0 = p0 ^ 1
                _store16(out, qkv_z, 3 * D, nq_z, M, BNQ, MT, False, NW)
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            _issue_w(dcq, layer * D + npt_z, 0, mb.index(0), w0, PART, D, has_p)
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
            )
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # out-projection + residual (slot 1); then cross-out's slab
            if has_p:
                out = _gemm_g(ob_z, D, M, w1, xbuf, scratch, mb.index(1), p1, bo + layer * D, bo, st_z, eps, x_z, npt_z,
                              D, PART, LPART, MT, LMT, False, 1, True, D, NPO, PART, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                p1 = p1 ^ 1
                _store_res(out, x_z, x2_z, st_z, npt_z, pid_z, M, PART, MT, D, NPO, NW)
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            _issue_w(dco, layer * D + npt_z, 0, mb.index(1), w1, PART, D, has_p)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # LN2 + cross-Q (slot 0) -> q2; then LN3+FFN1's slab, and the cross-attention's keys/values into L2
            if has_p:
                out = _gemm_g(x2_z, 2 * D, M, w0, xbuf, scratch, mb.index(0), p0, bcq + layer * D, ccq + layer * D,
                              st_z, eps, x_z, npt_z, D, PART, LPART, MT, LMT, True, HL, False, D, NPO, PART, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                p0 = p0 ^ 1
                _store16(out, q2_z, D, npt_z, M, PART, MT, False, NW)
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            _issue_w(d1, layer * F + nf_z, 0, mb.index(0), w0, BNF, D, has_f)
            if PFA:
                prefetch_rows = R // CG if GQ else R
                if pid_z < prefetch_rows * H * NS:
                    sp = pid_z % NS
                    bh = pid_z // NS
                    tch = ((T + NS - 1) // NS + 15) // 16 * 16
                    t_lo = sp * tch
                    nk = gl.minimum(T, t_lo + tch) - t_lo
                    if GQ:
                        cb = ((layer * CR + bh // H) * H + bh % H).to(gl.int64) * T * DK + t_lo * DK
                    elif CG == 1:
                        cb = ((layer * R + bh // H) * H + bh % H).to(gl.int64) * T * DK + t_lo * DK
                    else:
                        cb = ((layer * CR + (bh // H) // CG) * H + bh % H).to(gl.int64) * T * DK + t_lo * DK
                    _prefetch_rows(ck_z + cb, nk, 128, NW)
                    _prefetch_rows(cv_z + cb, nk, 128, NW)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # cross-attention -> ob
            # The same SP=16 tile can carry several beam queries. Keep each query's softmax and
            # the original key slicing/reduction order; only share its K/V loads with its siblings.
            cross_rows = R // CG if GQ else R
            cross_window = S * CG if GQ else S
            _cross_attn_g(
                pid_z,
                P,
                cross_rows,
                cross_window,
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
                CSP,
                BT,
                NS,
                NW,
                CR=CR,
                CG=1 if GQ else CG,
            )
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # cross-out + residual (slot 1); then FFN2's slab
            if has_p:
                out = _gemm_g(ob_z, D, M, w1, xbuf, scratch, mb.index(1), p1, bco + layer * D, bco, st_z, eps, x_z, npt_z,
                              D, PART, LPART, MT, LMT, False, 1, True, D, NPO, PART, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                p1 = p1 ^ 1
                _store_res(out, x_z, x2_z, st_z, npt_z, pid_z, M, PART, MT, D, NPO, NW)
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            if F2S:
                _issue_w(d2s, layer * D + npt_z, 0, mb.index(1), w1, PART, F, has_p)
            else:
                _issue_w(d2, layer * D + n2_z, k2, mb.index(1), w1, 64, 512, has_2)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # LN3 + FFN1 + ReLU (slot 0) -> hb; then the next layer's QKV slab (or the first output half)
            if has_f:
                out = _gemm_g(x2_z, 2 * D, M, w0, xbuf, scratch, mb.index(0), p0, b1 + layer * F, c1 + layer * F, st_z,
                              eps, x_z, nf_z, D, BNF, LBNF, MT, LMT, True, HL, False, D, NPO, PART, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                p0 = p0 ^ 1
                _store16(out, hb_z, F, nf_z, M, BNF, MT, True, NW)
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            if last:
                _issue_w(dout, pid_z * 64, 0, mb.index(0), w0, 64, D // 2, pid_z < nrb)
            else:
                _issue_w(dqkv, (layer + 1) * 3 * D + nq_z, 0, mb.index(0), w0, BNQ, D, has_q)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
            # FFN2 + residual (slot 1, split-K); then the next layer's out-projection slab (or the second half)
            if F2S:
                if has_p:
                    out = _gemm_g(hb_z, F, M, w1, xbuf, scratch, mb.index(1), p1, b2 + layer * D, b2, st_z, eps, x_z, npt_z,
                                  F, PART, LPART, MT, LMT, False, 1, True, D, NPO, PART, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                    p1 = p1 ^ 1
                    _store_res(out, x_z, x2_z, st_z, npt_z, pid_z, M, PART, MT, D, NPO, NW)
            elif has_2:
                _ffn2_g(pid_z, hb_z, M, w1, xbuf, mb.index(1), p1, b2 + layer * D, x_z, x2_z, st_z, part_z, cnt2_z, D, F, LKS,
                        MTF, LMTF, NW,
                              ts + (g * P + pid_z) * 8, TS)  # fmt: skip
                p1 = p1 ^ 1
            if TS:
                gl.store(ts + (g * P + pid_z) * 8 + 5, _now())
            g += 1
            _garrive(bar)
            if last:
                _issue_w(dout, pid_z * 64, NKH, mb.index(1), w1, 64, D // 2, pid_z < nrb)
            else:
                _issue_w(do_, (layer + 1) * D + npt_z, 0, mb.index(1), w1, PART, D, has_p)
            _gwait(bar, err, g * P, MAXS)
            if TS:
                gl.store(ts + ((g - 1) * P + pid_z) * 8 + 1, _now())
                gl.store(ts + (g * P + pid_z) * 8, _now())
        # final LayerNorm + output layer: blocks of 64 rows pid, pid + P, ..., each in two halves of K (slots 0, 1)
        _x_issue(
            x2, 2 * D, M, xbuf, D, 64, 6, MTF, LMTF, 1, D, NW
        )  # the B side once (hi only), for all blocks
        mu, m2 = _stats_load(st, M, NP2, MTF, NW)
        mean, rstd = _stats_finish(mu, m2, eps, NP2, P2, D)
        el: gl.constexpr = _el(64, NW)
        mean = gl.convert_layout(mean, gl.SliceLayout(0, el))
        rstd = gl.convert_layout(rstd, gl.SliceLayout(0, el))
        for rb in range(pid, nrb, P):
            nxt = rb + P < nrb
            acc = _out_half(w0, xbuf.slice(0, NKH * MTF), mb.index(0), p0, D // 2, MTF, NW)
            p0 = p0 ^ 1
            _issue_w(dout, (rb + P) * 64, 0, mb.index(0), w0, 64, D // 2, nxt)
            acc += _out_half(w1, xbuf.slice(NKH * MTF, NKH * MTF), mb.index(1), p1, D // 2, MTF, NW)
            p1 = p1 ^ 1
            _issue_w(dout, (rb + P) * 64, NKH, mb.index(1), w1, 64, D // 2, nxt)
            _out_store(acc, mean, rstd, bout, cout, logits, M, V, rb * 64, MTF, NW)
        if TS:
            gl.store(ts + (g * P + pid) * 8 + 5, _now())
        mbarrier.invalidate(mb.index(0))
        mbarrier.invalidate(mb.index(1))
        _gexit(bar, done, P)


def _log2(n: int) -> int:
    return int(n).bit_length() - 1


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def slab_desc(w: torch.Tensor, bn: int, kslice: int | None = None):
    """TMA descriptor of weights [rows, K] viewed as [K / S / 64, S, rows, 64] (S = 64 / bn): one box = one
    program's slab of bn rows and ``kslice`` columns (all of K by default), landing as stacked A tiles (see
    _issue_w)."""
    rows, k = w.reshape(-1, w.shape[-1]).shape
    s = 64 // bn
    v = w.reshape(rows, s, k // s // 64, 64).permute(2, 1, 0, 3)
    lay = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=4)
    return TensorDescriptor.from_tensor(v, [(kslice or k) // s // 64, s, bn, 64], lay)


class GluonDecoder:
    """The decoder megakernel in Gluon for calls of up to MAX_TOKENS tokens (see the module docstring); built on
    the weights :class:`owsm_trt.megakernel.MegaDecoder` prepares (LayerNorms folded in)."""

    name = "megakernel-gluon"
    shared_cross_kv = True

    def __init__(self, mega, programs: int | None = None, num_warps: int = 4, maxs: int = 1 << 24, ts: bool = False,
                 prefetch_attention: bool = True, group_cross_queries: bool = False,
                 cross_splits: int | None = None):  # fmt: skip
        if cross_splits is not None and cross_splits not in (1, 2, 4, 8):
            raise ValueError("cross_splits must be 1, 2, 4 or 8")
        self.m = mega
        d, f = mega.D, mega.F
        dev = mega.device
        self.P = programs or torch.cuda.get_device_properties(dev).multi_processor_count
        self.part = max(8, 4096 // d)  # rows of an out-projection unit: K / (64 / part) >= 64
        self.bnq = self.bnf = 32
        units = (3 * d // self.bnq, d // self.part, f // self.bnf, d // 64 * (f // 512))
        pow2 = (d & (d - 1)) == 0 and (f & (f - 1)) == 0
        if max(units) > self.P or not pow2 or not 256 <= d <= 1024 or f < 512:
            raise ValueError("the model's GEMMs do not fit this GPU's SMs (or the kernel's tiling)")
        self.nw, self.maxs, self.pfa = num_warps, maxs, prefetch_attention
        self.group_cross_queries = group_cross_queries
        self.cross_splits = cross_splits
        self.descs = (
            slab_desc(mega.wqkv, self.bnq), slab_desc(mega.wo, self.part), slab_desc(mega.wcq, self.part),
            slab_desc(mega.wco, self.part), slab_desc(mega.w1, self.bnf), slab_desc(mega.w2, 64, 512),
            slab_desc(mega.w2, self.part),
            slab_desc(mega.wout, 64, d // 2),
        )  # fmt: skip
        self.err = mega.err
        self.opts = {"TS": ts}
        self.programs = self.P
        self.ts = torch.zeros((8 * mega.L + 2) * self.P * 8 if ts else 1, dtype=torch.int64, device=dev)

    def check(self) -> None:
        self.m.check()

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, kind="decode"):
        m = self.m
        R, S = tokens.shape
        cr = cross_k.shape[1]
        if cr < 1 or R % cr or cross_v.shape != cross_k.shape:
            raise ValueError("cross KV rows must divide decoder rows into equal utterance-major groups")
        M = R * S
        if M > MAX_TOKENS:
            raise ValueError(f"the Gluon decoder kernel takes up to {MAX_TOKENS} tokens per call, not {M}")
        D, F, L, H, P = m.D, m.F, m.L, m.H, self.P
        dev = m.device
        mt = max(2, _pow2(M))
        mtf = max(8, mt)
        hl = 2 if mt <= 16 else 1  # LayerNorm GEMMs: the residual stream as a hi/lo pair while it fits
        ns = self.cross_splits or max(1, min(8, P // (R * H)))
        f16 = {"device": dev, "dtype": torch.float16}
        i32 = {"device": dev, "dtype": torch.int32}
        x = torch.zeros(M, D, device=dev)
        x2 = torch.zeros(M, 2, D, **f16)
        qkv = torch.zeros(M, 3 * D, **f16)
        ob, q2 = torch.zeros(M, D, **f16), torch.zeros(M, D, **f16)
        hb = torch.zeros(M, F, **f16)
        st = torch.zeros(2, D // min(self.part, 64), M, device=dev)
        part = torch.zeros(F // 512, M, D, device=dev)
        cnt2 = torch.zeros(D // 64, **i32)
        xp = torch.zeros(ns, M, D, device=dev)
        xml = torch.zeros(ns, M, H, 2, device=dev)
        xcnt = torch.zeros(R * H, **i32)
        bar, done = torch.zeros(1, **i32), torch.zeros(1, **i32)
        dq, do_, dcq, dco, d1, d2, d2s, dout = self.descs
        C, T = self_k.shape[3], cross_k.shape[3]
        args = (
            tokens, offset, tokens.stride(0), offset.stride(0), m.emb, m.pe, m.xscale,
            dq, m.bqkv, m.cqkv, do_, m.bo, dcq, m.bcq, m.ccq, dco, m.bco, d1, m.b1, m.c1, d2, d2s, m.b2, dout, m.bout,
            m.cout,
            self_k, self_v, C, cross_k, cross_v, T, logits, new_kv,
            x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, self.err, done, self.ts,
            R, S, L, m.V, m.eps,
        )  # fmt: skip
        const = dict(
            D=D, H=H, DK=D // H, F=F, LF=_log2(F), MT=mt, LMT=_log2(mt), HL=hl, MTF=mtf, LMTF=_log2(mtf), BNQ=self.bnq,
            LBNQ=_log2(self.bnq), BNF=self.bnf, LBNF=_log2(self.bnf), PART=self.part, LPART=_log2(self.part),
            SP=max(16, _pow2(S)), BT=64, NS=ns, NW=self.nw, MAXS=self.maxs, PFA=self.pfa, F2S=mt <= 8,
            CR=cr, CG=R // cr,
            GQ=self.group_cross_queries,
            CSP=max(16, _pow2(S * (R // cr if self.group_cross_queries else 1))),
            TS=bool(self.opts["TS"]),
        )  # fmt: skip

        def run() -> None:
            _decoder_kernel_g[(P,)](*args, **const, num_warps=self.nw)

        run.buffers = (x, x2, qkv, ob, q2, hb, st, part, cnt2, xp, xml, xcnt, bar, done)
        run.const = const
        run.writes_cache = True
        return run
