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

MAX_TOKENS = 8  # rows x window of a call


def available(device: torch.device) -> bool:
    return gluon is not None and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9


if gluon is not None:

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
        """One TMA: BN weight rows from row0 as A tiles (K chunk, K block, row); desc = the weights viewed as
        [K / S / 64, S, rows, 64] with S = 64 / BN (see slab_desc)."""
        S: gl.constexpr = 64 // BN
        NKC: gl.constexpr = K // S // 64
        l4: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=4)
        dst = wslot._reinterpret(gl.float16, [8, S, BN, 64], l4).slice(0, NKC, dim=0)  # a slot holds 8 chunks
        mbarrier.expect(bar, BN * K * 2, pred=pred)
        tma.async_load(desc, [0, 0, row0, 0], bar, dst, pred=pred)

    @gluon.jit
    def _x_issue(x, M, xbuf, K: gl.constexpr, BN: gl.constexpr, LBN: gl.constexpr, MT: gl.constexpr,
                 LMT: gl.constexpr, LN: gl.constexpr, D: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """cp.async of the B side: rows (K chunk, K block s, [hi/lo,] token) of K / S / 64 x 64 columns.
        Without LN, x = [M, K] fp16; with LN, x = the residual stream's hi/lo pair [M, 2, D]. (Shifts and masks
        only: Triton miscompiles some vector divisions by constants.)"""
        S: gl.constexpr = 64 // BN
        LS: gl.constexpr = 6 - LBN
        KP: gl.constexpr = K // S
        NKC: gl.constexpr = KP // 64
        xl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
        kk = gl.arange(0, 64, layout=gl.SliceLayout(0, xl))
        if LN:
            rho = gl.arange(0, NKC * S * 2 * MT, layout=gl.SliceLayout(1, xl))
            t = rho & (MT - 1)
            h = (rho >> LMT) & 1
            sb = (rho >> (LMT + 1)) & (S - 1)
            kc = rho >> (LMT + 1 + LS)
            xp = x + t[:, None] * (2 * D) + (h * D + sb * KP + kc * 64)[:, None] + kk[None, :]
            xm = (t < M)[:, None] & (kk >= 0)[None, :]
            async_copy.async_load(xbuf.slice(0, NKC * S * 2 * MT), xp, mask=xm, cache_modifier=".cg")
        else:
            rho = gl.arange(0, NKC * S * MT, layout=gl.SliceLayout(1, xl))
            t = rho & (MT - 1)
            sb = (rho >> LMT) & (S - 1)
            kc = rho >> (LMT + LS)
            xp = x + t[:, None] * K + (sb * KP + kc * 64)[:, None] + kk[None, :]
            xm = (t < M)[:, None] & (kk >= 0)[None, :]
            async_copy.async_load(xbuf.slice(0, NKC * S * MT), xp, mask=xm, cache_modifier=".cg")
        async_copy.commit_group()

    @gluon.jit
    def _mma(wslot, xbuf, wbar, phase, K: gl.constexpr, BN: gl.constexpr, NB: gl.constexpr, NW: gl.constexpr):
        """Wait for the slab and the B side, then the wgmma chain (two accumulators) -> [64, NB] fp32."""
        S: gl.constexpr = 64 // BN
        NKC: gl.constexpr = K // S // 64
        mbarrier.wait(wbar, phase)
        async_copy.wait_group(0)
        fence_async_shared()
        gl.barrier()
        mma: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, NB, 16]
        )
        acc0 = gl.zeros([64, NB], gl.float32, layout=mma)
        acc1 = gl.zeros([64, NB], gl.float32, layout=mma)
        for kc in gl.static_range(NKC):
            a_t = wslot.slice(kc * 64, 64)
            b_t = xbuf.slice(kc * NB, NB).permute((1, 0))
            if kc % 2 == 0:
                acc0 = warpgroup_mma(a_t, b_t, acc0, is_async=True)
            else:
                acc1 = warpgroup_mma(a_t, b_t, acc1, is_async=True)
        acc0, acc1 = warpgroup_mma_wait(0, deps=[acc0, acc1])
        return acc0 + acc1

    @gluon.jit
    def _extract(acc, scratch, BN: gl.constexpr, MT: gl.constexpr, LN: gl.constexpr, NB: gl.constexpr,
                 NW: gl.constexpr):  # fmt: skip
        """[BN, MT] = sum of the S diagonal blocks of acc [64, NB] (and of the hi/lo halves with LN), through
        shared memory (scratch [64, 128] fp32)."""
        S: gl.constexpr = 64 // BN
        sc = scratch.slice(0, NB, dim=1)
        sc.store(acc)
        gl.barrier()
        ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
        out = gl.zeros([BN, MT], gl.float32, layout=ol)
        for s in gl.static_range(S):
            blk = sc.slice(s * BN, BN, dim=0)
            if LN:
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
        ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
        rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        m = (tt < M)[None, :] & (rr >= 0)[:, None]
        gl.store(x + tt[None, :] * D + n0 + rr[:, None], val, mask=m)
        hi = val.to(gl.float16)
        o2 = x2 + tt[None, :] * (2 * D) + n0 + rr[:, None]
        gl.store(o2, hi, mask=m)
        gl.store(o2 + D, (val - hi.to(gl.float32)).to(gl.float16), mask=m)
        mu = gl.sum(val, axis=0) / BN
        dev = val - mu[None, :]
        gl.store(st + nu * M + tt, mu, mask=tt < M)
        gl.store(st + (NPART + nu) * M + tt, gl.sum(dev * dev, axis=0), mask=tt < M)

    @gluon.jit
    def _embed_g(pid, M, S, tok, offs, emb, pe, xscale, V, x, x2, st, D: gl.constexpr, PART: gl.constexpr,
                 NPART: gl.constexpr, MT: gl.constexpr, NW: gl.constexpr):  # fmt: skip
        """x = embedding * sqrt(d) + positional encoding, for the PART features of unit pid."""
        if pid < NPART:
            ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
            rr = gl.arange(0, PART, layout=gl.SliceLayout(1, ol))
            tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
            tm = tt < M
            ids = gl.load(tok + tt, mask=tm, other=0)
            ids = gl.minimum(gl.maximum(ids, 0), V - 1).to(gl.int64)
            pos = gl.load(offs + tt // S, mask=tm, other=0) + tt % S
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
    def _self_attn_g(pid, P, R, S, layer, qkv, offs, kc, vc, C, new_kv, o, D: gl.constexpr, H: gl.constexpr,
                     DK: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NW: gl.constexpr):  # fmt: skip
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
            off = gl.load(offs + b)
            rows = b * S + s_
            qp = qkv + rows[:, None] * (3 * D) + h * DK + d_[None, :]
            q = gl.load(qp, mask=sm, other=0.0, cache_modifier=".cg")
            kn = gl.load(qp + D, mask=sm, other=0.0, cache_modifier=".cg")
            vn = gl.load(qp + 2 * D, mask=sm, other=0.0, cache_modifier=".cg")
            qa = gl.convert_layout((q.to(gl.float32) * (DK**-0.5)).to(gl.float16), la)
            cb = ((layer * R + b) * H + h).to(gl.int64) * C * DK
            m_i = gl.full([SP], float("-inf"), gl.float32, layout=gl.SliceLayout(1, mma2))
            l_i = gl.zeros([SP], gl.float32, layout=gl.SliceLayout(1, mma2))
            acc = gl.zeros([SP, DK], gl.float32, layout=mma2)
            for t0 in range(0, off, BT):
                kt = gl.load(kc + cb + (t0 + tT)[None, :] * DK + dT[:, None], mask=((t0 + tT) < off)[None, :] & (dT >= 0)[:, None],
                             other=0.0, cache_modifier=".cg")  # fmt: skip
                vt = gl.load(vc + cb + (t0 + tV)[:, None] * DK + d_[None, :], mask=((t0 + tV) < off)[:, None] & (d_ >= 0)[None, :],
                             other=0.0, cache_modifier=".cg")  # fmt: skip
                ok = ((t0 + tS) < off)[None, :] & (iS >= 0)[:, None]
                m_i, l_i, acc = _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP, BT, DK, NW)
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

    @gluon.jit
    def _cross_attn_g(pid, P, R, S, layer, q2, ck, cv, T, o, xp, xml, xcnt, D: gl.constexpr, H: gl.constexpr,
                      DK: gl.constexpr, SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr, NW: gl.constexpr):  # fmt: skip
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
            cb = ((layer * R + b) * H + h).to(gl.int64) * T * DK
            t_lo = sp * tch
            t_hi = gl.minimum(T, t_lo + tch)
            m_i = gl.full([SP], float("-inf"), gl.float32, layout=gl.SliceLayout(1, mma2))
            l_i = gl.zeros([SP], gl.float32, layout=gl.SliceLayout(1, mma2))
            acc = gl.zeros([SP, DK], gl.float32, layout=mma2)
            for t0 in range(t_lo, t_hi, BT):
                kt = gl.load(ck + cb + (t0 + tT)[None, :] * DK + dT[:, None], mask=((t0 + tT) < t_hi)[None, :] & (dT >= 0)[:, None],
                             other=0.0, cache_modifier=".cg")  # fmt: skip
                vt = gl.load(cv + cb + (t0 + tV)[:, None] * DK + d_[None, :], mask=((t0 + tV) < t_hi)[:, None] & (d_ >= 0)[None, :],
                             other=0.0, cache_modifier=".cg")  # fmt: skip
                ok = ((t0 + tS) < t_hi)[None, :] & (iS >= 0)[:, None]
                m_i, l_i, acc = _softmax_step(qa, kt, vt, ok, m_i, l_i, acc, SP, BT, DK, NW)
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

    @gluon.jit
    def _gemm_g(x, M, wslot, xbuf, scratch, wbar, phase, bias, cvec, st, eps, res, n0, nlim,
                K: gl.constexpr, BN: gl.constexpr, LBN: gl.constexpr, MT: gl.constexpr, LMT: gl.constexpr,
                LN: gl.constexpr, RES: gl.constexpr, D: gl.constexpr, NPART: gl.constexpr, PART: gl.constexpr,
                NW: gl.constexpr, XL: gl.constexpr = True):  # fmt: skip
        """One program's unit of y = act(X @ W^T + b): BN weight rows from n0 (< nlim; slab already requested
        into wslot), MT tokens. Loads that do not wait for the MMAs (bias, LayerNorm column sums and statistics,
        residual) are issued first; with XL=False the B side is already in xbuf. Returns [BN, MT] fp32."""
        S: gl.constexpr = 64 // BN
        NB: gl.constexpr = S * MT * 2 if LN else S * MT
        ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
        rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        b = gl.load(bias + n0 + rr, mask=n0 + rr < nlim, other=0.0)
        if LN:
            cv = gl.load(cvec + n0 + rr, mask=n0 + rr < nlim, other=0.0)
            mu, m2 = _stats_load(st, M, NPART, MT, NW)
        if RES:
            r = gl.load(res + tt[None, :] * D + n0 + rr[:, None], mask=(tt < M)[None, :] & (rr >= 0)[:, None],
                        other=0.0, cache_modifier=".cg")  # fmt: skip
        if XL:
            _x_issue(x, M, xbuf, K, BN, LBN, MT, LMT, LN, D, NW)
        acc = _mma(wslot, xbuf, wbar, phase, K, BN, NB, NW)
        out = _extract(acc, scratch, BN, MT, LN, NB, NW)
        if LN:
            mean, rstd = _stats_finish(mu, m2, eps, NPART, PART, D)
            out = (out - cv[:, None] * mean[None, :]) * rstd[None, :]
        out += b[:, None]
        if RES:
            out += r
        return out

    @gluon.jit
    def _store16(out, y, y_stride, n0, M, BN: gl.constexpr, MT: gl.constexpr, RELU: gl.constexpr,
                 NW: gl.constexpr):  # fmt: skip
        ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
        rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        if RELU:
            out = gl.maximum(out, 0.0)
        gl.store(y + tt[None, :] * y_stride + n0 + rr[:, None], out.to(gl.float16),
                 mask=(tt < M)[None, :] & (rr >= 0)[:, None])  # fmt: skip

    @gluon.jit
    def _decoder_kernel_g(
        tok, offs, emb, pe, xscale,
        dqkv, bqkv, cqkv, do_, bo, dcq, bcq, ccq, dco, bco, d1, b1, c1, d2, b2, dout, bout, cout,
        kc, vc, C, ck, cv, T, logits, new_kv,
        x, x2, qkv, ob, q2, hb, st, xp, xml, xcnt, bar, err, done,
        R, S, L, V, eps,
        D: gl.constexpr, H: gl.constexpr, DK: gl.constexpr, F: gl.constexpr,
        MT: gl.constexpr, LMT: gl.constexpr, BNQ: gl.constexpr, LBNQ: gl.constexpr, BNF: gl.constexpr,
        LBNF: gl.constexpr, PART: gl.constexpr, LPART: gl.constexpr, BNV: gl.constexpr, LBNV: gl.constexpr,
        SP: gl.constexpr, BT: gl.constexpr, NS: gl.constexpr, NW: gl.constexpr, MAXS: gl.constexpr,
    ):  # fmt: skip
        """A decoder call: embedding, L layers (LN1+QKV, self-attention, out-projection, LN2+cross-Q,
        cross-attention, cross-out, LN3+FFN1, FFN2), final LN + output layer, a grid barrier between stages.
        GEMMs: QKV / FFN1 / output units of BNQ / BNF / BNV rows, the others (residual producers) of PART."""
        pid = gl.program_id(0)
        P = gl.num_programs(0)
        M = R * S
        NPART: gl.constexpr = D // PART
        sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        w0 = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
        w1 = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
        xbuf = gl.allocate_shared_memory(gl.float16, [512, 64], sl)
        # scratch for the diagonal sums: the second half of xbuf (the B side of a GEMM is dead by then, and the
        # output layer's, kept for all chunks, uses only the first half)
        scratch = xbuf._reinterpret(gl.float32, [2, 64, 128], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])).index(
            1
        )
        mb = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
        mbarrier.init(mb.index(0), count=1)
        mbarrier.init(mb.index(1), count=1)
        fence_async_shared()
        has_q = pid < 3 * D // BNQ
        has_p = pid < NPART
        has_f = pid < F // BNF
        nq = pid * BNQ
        npt = pid * PART
        nf = pid * BNF
        nv = (V + BNV - 1) // BNV  # output chunks: pid, pid + P, ...
        _issue_w(dqkv, nq, mb.index(0), w0, BNQ, D, has_q)
        _issue_w(do_, npt, mb.index(1), w1, PART, D, has_p)
        p0 = 0  # phases of the two slots' barriers
        p1 = 0
        _embed_g(pid, M, S, tok, offs, emb, pe, xscale, V, x, x2, st, D, PART, NPART, MT, NW)
        g = 1
        _garrive(bar)
        _gwait(bar, err, g * P, MAXS)
        for layer in range(L):
            last = layer + 1 == L
            # LN1 + QKV (slot 0) -> qkv; then LN2+cross-Q's slab
            if has_q:
                out = _gemm_g(x2, M, w0, xbuf, scratch, mb.index(0), p0, bqkv + layer * 3 * D, cqkv + layer * 3 * D, st,
                              eps, x, nq, 3 * D, D, BNQ, LBNQ, MT, LMT, True, False, D, NPART, PART, NW)  # fmt: skip
                p0 = p0 ^ 1
                _store16(out, qkv, 3 * D, nq, M, BNQ, MT, False, NW)
            g += 1
            _garrive(bar)
            _issue_w(dcq, layer * D + npt, mb.index(0), w0, PART, D, has_p)
            _gwait(bar, err, g * P, MAXS)
            # self-attention -> ob
            _self_attn_g(pid, P, R, S, layer, qkv, offs, kc, vc, C, new_kv, ob, D, H, DK, SP, BT, NW)
            g += 1
            _garrive(bar)
            _gwait(bar, err, g * P, MAXS)
            # out-projection + residual (slot 1); then cross-out's slab
            if has_p:
                out = _gemm_g(ob, M, w1, xbuf, scratch, mb.index(1), p1, bo + layer * D, bo, st, eps, x, npt, D,
                              D, PART, LPART, MT, LMT, False, True, D, NPART, PART, NW)  # fmt: skip
                p1 = p1 ^ 1
                _store_res(out, x, x2, st, npt, pid, M, PART, MT, D, NPART, NW)
            g += 1
            _garrive(bar)
            _issue_w(dco, layer * D + npt, mb.index(1), w1, PART, D, has_p)
            _gwait(bar, err, g * P, MAXS)
            # LN2 + cross-Q (slot 0) -> q2; then LN3+FFN1's slab
            if has_p:
                out = _gemm_g(x2, M, w0, xbuf, scratch, mb.index(0), p0, bcq + layer * D, ccq + layer * D, st, eps, x, npt,
                              D, D, PART, LPART, MT, LMT, True, False, D, NPART, PART, NW)  # fmt: skip
                p0 = p0 ^ 1
                _store16(out, q2, D, npt, M, PART, MT, False, NW)
            g += 1
            _garrive(bar)
            _issue_w(d1, layer * F + nf, mb.index(0), w0, BNF, D, has_f)
            _gwait(bar, err, g * P, MAXS)
            # cross-attention -> ob
            _cross_attn_g(pid, P, R, S, layer, q2, ck, cv, T, ob, xp, xml, xcnt, D, H, DK, SP, BT, NS, NW)
            g += 1
            _garrive(bar)
            _gwait(bar, err, g * P, MAXS)
            # cross-out + residual (slot 1); then FFN2's slab
            if has_p:
                out = _gemm_g(ob, M, w1, xbuf, scratch, mb.index(1), p1, bco + layer * D, bco, st, eps, x, npt, D,
                              D, PART, LPART, MT, LMT, False, True, D, NPART, PART, NW)  # fmt: skip
                p1 = p1 ^ 1
                _store_res(out, x, x2, st, npt, pid, M, PART, MT, D, NPART, NW)
            g += 1
            _garrive(bar)
            _issue_w(d2, layer * D + npt, mb.index(1), w1, PART, F, has_p)
            _gwait(bar, err, g * P, MAXS)
            # LN3 + FFN1 + ReLU (slot 0) -> hb; then the next layer's QKV slab (or the first output chunk)
            if has_f:
                out = _gemm_g(x2, M, w0, xbuf, scratch, mb.index(0), p0, b1 + layer * F, c1 + layer * F, st, eps, x, nf,
                              F, D, BNF, LBNF, MT, LMT, True, False, D, NPART, PART, NW)  # fmt: skip
                p0 = p0 ^ 1
                _store16(out, hb, F, nf, M, BNF, MT, True, NW)
            g += 1
            _garrive(bar)
            if last:
                _issue_w(dout, pid * BNV, mb.index(0), w0, BNV, D, pid < nv)
            else:
                _issue_w(dqkv, (layer + 1) * 3 * D + nq, mb.index(0), w0, BNQ, D, has_q)
            _gwait(bar, err, g * P, MAXS)
            # FFN2 + residual (slot 1); then the next layer's out-projection slab (or the second output chunk)
            if has_p:
                out = _gemm_g(hb, M, w1, xbuf, scratch, mb.index(1), p1, b2 + layer * D, b2, st, eps, x, npt, D,
                              F, PART, LPART, MT, LMT, False, True, D, NPART, PART, NW)  # fmt: skip
                p1 = p1 ^ 1
                _store_res(out, x, x2, st, npt, pid, M, PART, MT, D, NPART, NW)
            g += 1
            _garrive(bar)
            if last:
                _issue_w(dout, (pid + P) * BNV, mb.index(1), w1, BNV, D, pid + P < nv)
            else:
                _issue_w(do_, (layer + 1) * D + npt, mb.index(1), w1, PART, D, has_p)
            _gwait(bar, err, g * P, MAXS)
        # final LayerNorm + output layer: chunks pid, pid + P, ... through the two slots
        ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
        rr = gl.arange(0, BNV, layout=gl.SliceLayout(1, ol))
        tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
        _x_issue(x2, M, xbuf, D, BNV, LBNV, MT, LMT, True, D, NW)  # the B side once, for all chunks
        for c0 in range(pid, nv, 2 * P):
            for k in gl.static_range(2):
                c = c0 + k * P
                if c < nv:
                    n0 = c * BNV
                    if k == 0:
                        out = _gemm_g(x2, M, w0, xbuf, scratch, mb.index(0), p0, bout, cout, st, eps, x, n0, V, D,
                                      BNV, LBNV, MT, LMT, True, False, D, NPART, PART, NW, False)  # fmt: skip
                        p0 = p0 ^ 1
                        _issue_w(dout, (c + 2 * P) * BNV, mb.index(0), w0, BNV, D, c + 2 * P < nv)
                    else:
                        out = _gemm_g(x2, M, w1, xbuf, scratch, mb.index(1), p1, bout, cout, st, eps, x, n0, V, D,
                                      BNV, LBNV, MT, LMT, True, False, D, NPART, PART, NW, False)  # fmt: skip
                        p1 = p1 ^ 1
                        _issue_w(dout, (c + 2 * P) * BNV, mb.index(1), w1, BNV, D, c + 2 * P < nv)
                    gl.store(logits + tt[None, :] * V + n0 + rr[:, None], out,
                             mask=(tt < M)[None, :] & (n0 + rr < V)[:, None])  # fmt: skip
        mbarrier.invalidate(mb.index(0))
        mbarrier.invalidate(mb.index(1))
        _gexit(bar, done, P)


def _log2(n: int) -> int:
    return int(n).bit_length() - 1


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def slab_desc(w: torch.Tensor, bn: int):
    """TMA descriptor of weights [rows, K] viewed as [K / S / 64, S, rows, 64] (S = 64 / bn): one box = one
    program's slab of bn rows, landing as stacked A tiles (see _issue_w)."""
    rows, k = w.reshape(-1, w.shape[-1]).shape
    s = 64 // bn
    v = w.reshape(rows, s, k // s // 64, 64).permute(2, 1, 0, 3)
    lay = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=4)
    return TensorDescriptor.from_tensor(v, [k // s // 64, s, bn, 64], lay)


class GluonDecoder:
    """The decoder megakernel in Gluon for calls of up to MAX_TOKENS tokens (see the module docstring); built on
    the weights :class:`owsm_trt.megakernel.MegaDecoder` prepares (LayerNorms folded in)."""

    name = "megakernel-gluon"

    def __init__(self, mega, programs: int | None = None, num_warps: int = 4, maxs: int = 1 << 24):
        self.m = mega
        d, f = mega.D, mega.F
        dev = mega.device
        self.P = programs or torch.cuda.get_device_properties(dev).multi_processor_count
        self.part = max(8, 4096 // min(d, f))  # rows of a residual-producing unit: K / (64 / part) >= 64
        self.bnq = self.bnf = self.bnv = 32
        if 3 * d // self.bnq > self.P or d // self.part > self.P or f // self.bnf > self.P:
            raise ValueError("the model has more GEMM units than the GPU has SMs")
        self.nw, self.maxs = num_warps, maxs
        self.descs = (
            slab_desc(mega.wqkv, self.bnq), slab_desc(mega.wo, self.part), slab_desc(mega.wcq, self.part),
            slab_desc(mega.wco, self.part), slab_desc(mega.w1, self.bnf), slab_desc(mega.w2, self.part),
            slab_desc(mega.wout, self.bnv),
        )  # fmt: skip
        self.err = mega.err

    def check(self) -> None:
        self.m.check()

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, kind="decode"):
        m = self.m
        R, S = tokens.shape
        M = R * S
        if M > MAX_TOKENS:
            raise ValueError(f"the Gluon decoder kernel takes up to {MAX_TOKENS} tokens per call, not {M}")
        D, F, L, H, P = m.D, m.F, m.L, m.H, self.P
        dev = m.device
        mt = max(2, _pow2(M))
        ns = max(1, min(8, P // (R * H)))
        npart = D // self.part
        f16 = {"device": dev, "dtype": torch.float16}
        i32 = {"device": dev, "dtype": torch.int32}
        x = torch.zeros(M, D, device=dev)
        x2 = torch.zeros(M, 2, D, **f16)
        qkv = torch.zeros(M, 3 * D, **f16)
        ob, q2 = torch.zeros(M, D, **f16), torch.zeros(M, D, **f16)
        hb = torch.zeros(M, F, **f16)
        st = torch.zeros(2, npart, M, device=dev)
        xp = torch.zeros(ns, M, D, device=dev)
        xml = torch.zeros(ns, M, H, 2, device=dev)
        xcnt = torch.zeros(R * H, **i32)
        bar, done = torch.zeros(1, **i32), torch.zeros(1, **i32)
        dq, do_, dcq, dco, d1, d2, dout = self.descs
        C, T = self_k.shape[3], cross_k.shape[3]
        args = (
            tokens, offset, m.emb, m.pe, m.xscale,
            dq, m.bqkv, m.cqkv, do_, m.bo, dcq, m.bcq, m.ccq, dco, m.bco, d1, m.b1, m.c1, d2, m.b2, dout, m.bout, m.cout,
            self_k, self_v, C, cross_k, cross_v, T, logits, new_kv,
            x, x2, qkv, ob, q2, hb, st, xp, xml, xcnt, bar, self.err, done,
            R, S, L, m.V, m.eps,
        )  # fmt: skip
        const = dict(
            D=D, H=H, DK=D // H, F=F, MT=mt, LMT=_log2(mt), BNQ=self.bnq, LBNQ=_log2(self.bnq), BNF=self.bnf,
            LBNF=_log2(self.bnf), PART=self.part, LPART=_log2(self.part), BNV=self.bnv, LBNV=_log2(self.bnv),
            SP=max(16, _pow2(S)), BT=64, NS=ns, NW=self.nw, MAXS=self.maxs,
        )  # fmt: skip

        def run() -> None:
            _decoder_kernel_g[(P,)](*args, **const, num_warps=self.nw)

        run.buffers = (x, x2, qkv, ob, q2, hb, st, xp, xml, xcnt, bar, done)
        run.const = const
        run.writes_cache = True
        return run
