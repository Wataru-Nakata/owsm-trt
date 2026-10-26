"""One persistent Triton kernel per decoder call ("megakernel"), CUDA only.

Building blocks (this module grows into the whole decoder step):

* :func:`_grid_sync` -- a grid-wide barrier between the stages of the call. The grid has one program per SM, all
  resident at once; a monotonic counter in global memory counts arrivals, and the last program to leave the
  kernel resets it (:func:`_grid_exit`), so CUDA-graph replays need no memset. A barrier that waits too long
  sets an error flag; every later barrier then falls through, so a bug cannot hang the GPU.
* :func:`_gemm` -- ``y = x @ w.T + b`` for a few rows: weights are the MMA's M side ("swap AB"), every program
  owns a contiguous range of output features over the full K (no split-K), larger inputs are cut into token
  blocks.
"""

import logging

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # e.g. torch without Triton (Windows)
    triton = None

log = logging.getLogger(__name__)


def available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


if triton is not None:

    @triton.jit
    def _ld_acquire(p):
        return tl.inline_asm_elementwise(
            "ld.global.acquire.gpu.b32 $0, [$1];", "=r,l", [p], dtype=tl.int32, is_pure=False, pack=1
        )

    @triton.jit
    def _stamp(p):
        """Store the global nanosecond timer at ``p`` (profiling)."""
        t = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
        tl.store(p, t)

    @triton.jit
    def _grid_arrive(bar):
        """Arrive at a grid barrier (after this program's global writes, which the release publishes)."""
        tl.debug_barrier()
        tl.atomic_add(bar, 1, sem="release", scope="gpu")  # scalar atomics run on one thread

    @triton.jit
    def _grid_wait(bar, err, target, MAX_SPIN: tl.constexpr):
        """Wait until the barrier counter reaches ``target`` = (barriers so far + 1) x programs. A wait of
        more than MAX_SPIN polls sets ``err``; once it is set, barriers no longer wait."""
        e = tl.load(err, volatile=True)
        v = _ld_acquire(bar)
        n = tl.full([], 0, tl.int32)
        while (v < target) & (n < MAX_SPIN) & (e == 0):
            v = _ld_acquire(bar)
            n += 1
        if (v < target) & (e == 0):
            tl.atomic_xchg(err, 1, sem="relaxed", scope="gpu")
        tl.debug_barrier()

    @triton.jit
    def _grid_sync(bar, err, target, MAX_SPIN: tl.constexpr):
        _grid_arrive(bar)
        _grid_wait(bar, err, target, MAX_SPIN)

    @triton.jit
    def _grid_exit(bar, done, P):
        """The last program to get here resets the barrier counter for the next launch."""
        tl.debug_barrier()
        t = tl.atomic_add(done, 1, sem="acq_rel", scope="gpu")
        if t == P - 1:
            tl.atomic_xchg(done, 0, sem="relaxed", scope="gpu")
            tl.atomic_xchg(bar, 0, sem="relaxed", scope="gpu")

    @triton.jit
    def _ln_load(st, NU_R, M, tok, tmask, NUP: tl.constexpr):
        """The producers' partial LayerNorm statistics of tokens ``tok``: st = [2, NU_R, M] holds, per range
        nu of features, their mean and sum of squared deviations."""
        nu = tl.arange(0, NUP)
        msk = (nu < NU_R)[:, None] & tmask[None, :]
        mu_p = tl.load(st + nu[:, None] * M + tok[None, :], mask=msk, other=0.0, cache_modifier=".cg")
        m2_p = tl.load(
            st + (NU_R + nu[:, None]) * M + tok[None, :], mask=msk, other=0.0, cache_modifier=".cg"
        )
        return mu_p, m2_p

    @triton.jit
    def _ln_finish(mu_p, m2_p, NU_R, D: tl.constexpr, eps, NUP: tl.constexpr):
        """Mean and 1/std per token from :func:`_ln_load`'s partials (exact parallel combination)."""
        nu = tl.arange(0, NUP)
        cnt = tl.where(nu < NU_R, ((nu + 1) * D // NU_R - nu * D // NU_R).to(tl.float32), 0.0)
        mean = tl.sum(mu_p * cnt[:, None], 0) / D
        dev = mu_p - mean[None, :]
        m2 = tl.sum(m2_p + cnt[:, None] * dev * dev, 0)
        return mean, 1.0 / tl.sqrt(m2 / D + eps)

    @triton.jit
    def _store_x(val, rn, nmask, tok, tmask, x, x2, st_out, nu, M, cnt, D: tl.constexpr, NU: tl.constexpr):
        """Store residual-stream values val [BN features, BMT tokens]: fp32 in x [M, D], as an fp16 hi/lo
        pair in x2 [M, 2, D] (the LayerNorm GEMMs' operand) and the tile's partial statistics (its features
        must be all of unit nu's)."""
        omask = nmask[:, None] & tmask[None, :]
        tl.store(x + tok[None, :] * D + rn[:, None], val, mask=omask)
        hi = val.to(tl.float16)
        off2 = tok[None, :] * (2 * D) + rn[:, None]
        tl.store(x2 + off2, hi, mask=omask)
        tl.store(x2 + off2 + D, (val - hi.to(tl.float32)).to(tl.float16), mask=omask)
        mu = tl.sum(tl.where(omask, val, 0.0), 0) / cnt
        dev = tl.where(omask, val - mu[None, :], 0.0)
        tl.store(st_out + nu * M + tok, mu, mask=tmask)
        tl.store(st_out + (NU + nu) * M + tok, tl.sum(dev * dev, 0), mask=tmask)

    @triton.jit
    def _finish(acc, b, res, rn, nmask, tok, tmask, y, y_stride, y2, st_out, nu, M, cnt,
                YM: tl.constexpr, RELU: tl.constexpr, NU: tl.constexpr):  # fmt: skip
        acc += b[:, None]
        if RELU:
            acc = tl.maximum(acc, 0.0)
        if YM == 2:
            _store_x(acc + res, rn, nmask, tok, tmask, y, y2, st_out, nu, M, cnt, y_stride, NU)
        else:
            omask = nmask[:, None] & tmask[None, :]
            out = y + tok[None, :] * y_stride + rn[:, None]
            if YM == 0:
                tl.store(out, acc.to(tl.float16), mask=omask)
            else:
                tl.store(out, acc, mask=omask)

    @triton.jit
    def _kloop(w, x, x_stride, rn, nmask, xr, xm, k_lo, k_hi, K: tl.constexpr, BN: tl.constexpr,
               XC: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
        """sum over k in [k_lo, k_hi) of w[rn, k] * x[xr, k] -> [BN, XC] (fp32)."""
        acc = tl.zeros([BN, XC], dtype=tl.float32)
        for k0 in tl.range(k_lo, k_hi, BK, num_stages=STAGES):
            rk = k0 + tl.arange(0, BK)
            wt = tl.load(w + rn[:, None] * K + rk[None, :], mask=nmask[:, None], other=0.0)
            xt = tl.load(
                x + xr[None, :] * x_stride + rk[:, None], mask=xm[None, :], other=0.0, cache_modifier=".cg"
            )
            acc = tl.dot(wt, xt, acc)
        return acc

    @triton.jit
    def _gemm(
        pid, P, M,
        x, x_stride, st_in, NU_IN, cvec, eps,
        w, bias, N, K: tl.constexpr,
        y, y_stride, y2, st_out, part, cnt,
        LN: tl.constexpr, YM: tl.constexpr, RELU: tl.constexpr,
        BMT: tl.constexpr, TB: tl.constexpr, NU: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr, STAGES: tl.constexpr, NUP: tl.constexpr, KS: tl.constexpr,
        FLAT: tl.constexpr, tsp=None, TS: tl.constexpr = False,
    ):  # fmt: skip
        """y[m, n] = act(sum_k X[m, k] w[n, k] + bias[n]) for m < M, n < N, spread over P programs.

        Work units: TB token blocks of BMT rows x NU ranges of output features x KS slices of K (unit
        u = (tb * NU + nu) * KS + ks). With KS > 1 every slice stores its partial sum in ``part`` [KS, M, N]
        (fp32) and counts itself in ``cnt`` [TB * NU]; the last slice of a unit adds the partials up in a
        fixed order and finishes the unit.

        X = x (fp16 [M, K], row stride x_stride). With LN, X = LayerNorm(x_hi + x_lo) instead, x = [M, 2, K]
        the residual stream as an fp16 hi/lo pair, the LayerNorm folded into the weights: w = W * gamma,
        cvec = row sums of w, bias = b + W beta, so that y = rstd * (x . w - mean * cvec) + bias, with mean /
        rstd from the partial statistics ``st_in``. Every load a unit needs is issued before its K loop.
        YM=0: y = fp16 store; YM=1: fp32 store; YM=2: residual stream y (fp32) += result, also stored as
        the hi/lo pair y2 with the unit's partial statistics in ``st_out`` [2, NU, M] (YM=2 and KS > 1 need
        BN >= the unit's features).
        """
        KK: tl.constexpr = K // KS
        for u in range(pid, TB * NU * KS, P):
            ks = u % KS
            nu = (u // KS) % NU
            tb = u // (KS * NU)
            r0 = nu * N // NU
            r1 = (nu + 1) * N // NU
            cnt_f = (r1 - r0).to(tl.float32)
            tok = tb * BMT + tl.arange(0, BMT)
            tmask = tok < M
            if LN:
                mu_p, m2_p = _ln_load(st_in, NU_IN, M, tok, tmask, NUP)
            for n0 in tl.range(r0, r1, BN, flatten=FLAT):
                rn = n0 + tl.arange(0, BN)
                nmask = rn < r1
                omask = nmask[:, None] & tmask[None, :]
                b = tl.load(bias + rn, mask=nmask, other=0.0)
                res = tl.zeros([BN, BMT], dtype=tl.float32)
                if YM == 2:
                    if KS == 1:
                        res = tl.load(y + tok[None, :] * y_stride + rn[:, None], mask=omask, other=0.0,
                                      cache_modifier=".cg")  # fmt: skip
                if LN:
                    cv = tl.load(cvec + rn, mask=nmask, other=0.0)
                    xr = 2 * tb * BMT + tl.arange(0, 2 * BMT)  # rows of x viewed as [2M, K]: (token, hi/lo)
                    acc2 = _kloop(w, x, x_stride, rn, nmask, xr, xr < 2 * M, ks * KK, (ks + 1) * KK, K, BN,
                                  2 * BMT, BK, STAGES)  # fmt: skip
                    acc = tl.sum(tl.reshape(acc2, [BN, BMT, 2]), 2)
                else:
                    acc = _kloop(w, x, x_stride, rn, nmask, tok, tmask, ks * KK, (ks + 1) * KK, K, BN, BMT, BK,
                                 STAGES)  # fmt: skip
                if TS:
                    _stamp(tsp + 3)
                if LN:
                    mean, rstd = _ln_finish(mu_p, m2_p, NU_IN, K, eps, NUP)
                    acc = (acc - cv[:, None] * mean[None, :]) * rstd[None, :]
                if KS == 1:
                    _finish(
                        acc,
                        b,
                        res,
                        rn,
                        nmask,
                        tok,
                        tmask,
                        y,
                        y_stride,
                        y2,
                        st_out,
                        nu,
                        M,
                        cnt_f,
                        YM,
                        RELU,
                        NU,
                    )
                    if TS:
                        _stamp(tsp + 4)
                else:
                    pp = part + tok[None, :] * N + rn[:, None]
                    tl.store(pp + ks * M * N, acc, mask=omask)
                    tl.debug_barrier()
                    arrived = tl.atomic_add(cnt + tb * NU + nu, 1, sem="acq_rel", scope="gpu")
                    tl.debug_barrier()
                    if arrived == KS - 1:  # the last slice: add the partials up in order
                        if YM == 2:
                            res = tl.load(y + tok[None, :] * y_stride + rn[:, None], mask=omask, other=0.0,
                                          cache_modifier=".cg")  # fmt: skip
                        tot = tl.zeros([BN, BMT], dtype=tl.float32)
                        for j in tl.static_range(KS):
                            tot += tl.load(pp + j * M * N, mask=omask, other=0.0, cache_modifier=".cg")
                        tl.atomic_xchg(cnt + tb * NU + nu, 0, sem="relaxed", scope="gpu")
                        _finish(tot, b, res, rn, nmask, tok, tmask, y, y_stride, y2, st_out, nu, M, cnt_f, YM, RELU,
                                NU)  # fmt: skip
                        if TS:
                            _stamp(tsp + 4)

    @triton.jit
    def _embed(pid, P, M, S, tok, offs, emb, pe, x, x2, st, xscale, V,
               D: tl.constexpr, BMT: tl.constexpr, TB: tl.constexpr, NU: tl.constexpr, BN: tl.constexpr):  # fmt: skip
        """x = embedding * sqrt(d) + positional encoding (see _store_x)."""
        for u in range(pid, TB * NU, P):
            tb = u // NU
            nu = u % NU
            r0 = nu * D // NU
            r1 = (nu + 1) * D // NU
            t = tb * BMT + tl.arange(0, BMT)
            tm = t < M
            n = r0 + tl.arange(0, BN)
            nm = n < r1
            msk = nm[:, None] & tm[None, :]
            ids = tl.load(tok + t, mask=tm, other=0)
            ids = tl.minimum(tl.maximum(ids, 0), V - 1).to(tl.int64)
            pos = tl.load(offs + t // S, mask=tm, other=0) + t % S
            e = tl.load(emb + ids[None, :] * D + n[:, None], mask=msk, other=0.0).to(tl.float32)
            p = tl.load(pe + pos[None, :] * D + n[:, None], mask=msk, other=0.0)
            val = tl.where(msk, e * xscale + p, 0.0)
            _store_x(val, n, nm, t, tm, x, x2, st, nu, M, (r1 - r0).to(tl.float32), D, NU)

    @triton.jit
    def _online(q, kt, vt, ok, m_i, l_i, acc):
        """One block of keys kt [DK, BT] / values vt [BT, DK] into the running softmax of queries q."""
        sc = tl.where(ok, tl.dot(q, kt), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vt)
        return m_new, l_i, acc

    @triton.jit
    def _self_attn(pid, P, R, S, layer, qkv, offs, kc, vc, C, new_kv, o,
                   D: tl.constexpr, H: tl.constexpr, DK: tl.constexpr, SP: tl.constexpr, BT: tl.constexpr):  # fmt: skip
        """Per (row, head): the row's queries attend to its cache slots < offset and causally to its new
        keys (from registers); the new keys/values then go into the cache at the offset and to new_kv."""
        s = tl.arange(0, SP)
        d = tl.arange(0, DK)
        sm = (s < S)[:, None]
        for u in range(pid, R * H, P):
            b = u // H
            h = u % H
            off = tl.load(offs + b)
            rows = b * S + s
            qp = qkv + rows[:, None] * (3 * D) + h * DK + d[None, :]
            q = tl.load(qp, mask=sm, other=0.0, cache_modifier=".cg")
            kn = tl.load(qp + D, mask=sm, other=0.0, cache_modifier=".cg")
            vn = tl.load(qp + 2 * D, mask=sm, other=0.0, cache_modifier=".cg")
            cb = ((layer * R + b) * H + h).to(tl.int64) * C * DK
            q = (q.to(tl.float32) * (DK**-0.5)).to(tl.float16)
            m_i = tl.full([SP], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([SP], dtype=tl.float32)
            acc = tl.zeros([SP, DK], dtype=tl.float32)
            for t0 in range(0, off, BT):
                t = t0 + tl.arange(0, BT)
                tm = t < off
                kt = tl.load(kc + cb + t[None, :] * DK + d[:, None], mask=tm[None, :], other=0.0,
                             cache_modifier=".cg")  # fmt: skip
                vt = tl.load(vc + cb + t[:, None] * DK + d[None, :], mask=tm[:, None], other=0.0,
                             cache_modifier=".cg")  # fmt: skip
                m_i, l_i, acc = _online(q, kt, vt, tm[None, :] & (s >= 0)[:, None], m_i, l_i, acc)
            j = tl.arange(0, SP)
            ok = (j[None, :] <= s[:, None]) & (j < S)[None, :]
            m_i, l_i, acc = _online(q, tl.trans(kn), vn, ok, m_i, l_i, acc)
            tl.store(
                o + rows[:, None] * D + h * DK + d[None, :], (acc / l_i[:, None]).to(tl.float16), mask=sm
            )
            dst = (off + s)[:, None] * DK + d[None, :]
            tl.store(kc + cb + dst, kn, mask=sm)
            tl.store(vc + cb + dst, vn, mask=sm)
            nk = (((layer * 2) * R + b) * H + h).to(tl.int64) * S * DK + s[:, None] * DK + d[None, :]
            tl.store(new_kv + nk, kn, mask=sm)
            tl.store(new_kv + nk + R * H * S * DK, vn, mask=sm)

    @triton.jit
    def _cross_attn(pid, P, R, S, layer, q2, ck, cv, T, o, xp, xml, xcnt,
                    D: tl.constexpr, H: tl.constexpr, DK: tl.constexpr, SP: tl.constexpr, BT: tl.constexpr,
                    NS: tl.constexpr):  # fmt: skip
        """Per (row, head, slice of the T keys): with NS > 1 slices, every slice stores its partial softmax
        (running max, sum, output) in xml / xp and the last one of a (row, head) combines them."""
        s = tl.arange(0, SP)
        d = tl.arange(0, DK)
        sm = (s < S)[:, None]
        M = R * S
        tch = ((T + NS - 1) // NS + 15) // 16 * 16
        for u in range(pid, R * H * NS, P):
            sp = u % NS
            bh = u // NS
            b = bh // H
            h = bh % H
            rows = b * S + s
            q = tl.load(
                q2 + rows[:, None] * D + h * DK + d[None, :], mask=sm, other=0.0, cache_modifier=".cg"
            )
            q = (q.to(tl.float32) * (DK**-0.5)).to(tl.float16)
            cb = ((layer * R + b) * H + h).to(tl.int64) * T * DK
            t_lo = sp * tch
            t_hi = tl.minimum(T, t_lo + tch)
            m_i = tl.full([SP], float("-inf"), dtype=tl.float32)
            l_i = tl.zeros([SP], dtype=tl.float32)
            acc = tl.zeros([SP, DK], dtype=tl.float32)
            for t0 in range(t_lo, t_hi, BT):
                t = t0 + tl.arange(0, BT)
                tm = t < t_hi
                kt = tl.load(ck + cb + t[None, :] * DK + d[:, None], mask=tm[None, :], other=0.0,
                             cache_modifier=".cg")  # fmt: skip
                vt = tl.load(cv + cb + t[:, None] * DK + d[None, :], mask=tm[:, None], other=0.0,
                             cache_modifier=".cg")  # fmt: skip
                m_i, l_i, acc = _online(q, kt, vt, tm[None, :] & (s >= 0)[:, None], m_i, l_i, acc)
            out = o + rows[:, None] * D + h * DK + d[None, :]
            if NS == 1:
                tl.store(out, (acc / l_i[:, None]).to(tl.float16), mask=sm)
            else:
                pa = xp + (sp * M + rows)[:, None] * D + h * DK + d[None, :]
                pm = xml + ((sp * M + rows) * H + h) * 2
                tl.store(pa, acc, mask=sm)
                tl.store(pm, m_i, mask=s < S)
                tl.store(pm + 1, l_i, mask=s < S)
                tl.debug_barrier()
                arrived = tl.atomic_add(xcnt + bh, 1, sem="acq_rel", scope="gpu")
                tl.debug_barrier()
                if arrived == NS - 1:  # the last slice of (row, head): combine
                    mx = tl.full([SP], float("-inf"), dtype=tl.float32)
                    for j in tl.static_range(NS):
                        pmj = xml + ((j * M + rows) * H + h) * 2
                        mx = tl.maximum(mx, tl.load(pmj, mask=s < S, other=0.0, cache_modifier=".cg"))
                    num = tl.zeros([SP, DK], dtype=tl.float32)
                    den = tl.zeros([SP], dtype=tl.float32)
                    for j in tl.static_range(NS):
                        pmj = xml + ((j * M + rows) * H + h) * 2
                        wj = tl.exp(tl.load(pmj, mask=s < S, other=0.0, cache_modifier=".cg") - mx)
                        lj = tl.load(pmj + 1, mask=s < S, other=0.0, cache_modifier=".cg")
                        aj = tl.load(xp + (j * M + rows)[:, None] * D + h * DK + d[None, :], mask=sm, other=0.0,
                                     cache_modifier=".cg")  # fmt: skip
                        num += wj[:, None] * aj
                        den += wj * lj
                    tl.atomic_xchg(xcnt + bh, 0, sem="relaxed", scope="gpu")
                    tl.store(out, (num / den[:, None]).to(tl.float16), mask=sm)

    @triton.jit
    def _decoder_kernel(
        tok, offs, emb, pe, xscale,
        wqkv, bqkv, cqkv, wo, bo, wcq, bcq, ccq, wco, bco, w1, b1, c1, w2, b2, wout, bout, cout,
        kc, vc, C, ck, cv, T, logits, new_kv,
        x, x2, qkv, ob, q2, hb, st, part, cnt, xp, xml, xcnt, bar, err, done, ts,
        R, S, L, V, eps, g_lo, g_hi, nu_a, nu_b,
        D: tl.constexpr, H: tl.constexpr, DK: tl.constexpr, F: tl.constexpr,
        BMT: tl.constexpr, TB: tl.constexpr, NU: tl.constexpr, NU2: tl.constexpr, KS2: tl.constexpr,
        NUP: tl.constexpr, NS: tl.constexpr,
        BNQ: tl.constexpr, BKQ: tl.constexpr, SQ: tl.constexpr,
        BND: tl.constexpr, BKD: tl.constexpr, SD: tl.constexpr,
        BNF: tl.constexpr, BKF: tl.constexpr, SF: tl.constexpr,
        BN2: tl.constexpr, BK2: tl.constexpr, S2: tl.constexpr,
        BNV: tl.constexpr, BKV: tl.constexpr, SV: tl.constexpr,
        SP: tl.constexpr, BTS: tl.constexpr, BTX: tl.constexpr, SYNC: tl.constexpr, MAXS: tl.constexpr, TS: tl.constexpr = False,
        ONE: tl.constexpr = False,
    ):  # fmt: skip
        """Stages g_lo .. g_hi-1 of a decoder call: 0 = embedding, 1 + 8 * layer + i = stage i of a layer
        (LN1+QKV, self-attention, out-proj, LN2+cross-Q, cross-attention, cross-out, LN3+FFN1, FFN2),
        8 * L + 1 = final LayerNorm + output layer. With SYNC, a grid barrier separates the stages.

        Weights of the GEMMs after a LayerNorm come with it folded in (see _gemm). The residual stream x
        (fp32, with an fp16 hi/lo copy x2) gets LayerNorm statistics in NU parts from the embedding and the
        out-projections and NU2 parts from FFN2 (NU2 feature ranges x KS2 slices of K); nu_a / nu_b are NU / NU2
        again as run-time values for the consumers of those statistics: with a compile-time number of parts,
        Triton 3.8 miscompiles _ln_finish's vector division by it in some builds (wrong mean, NaN 1/std).

        Profiling: TS stamps the time per (stage, program) into ts [stages, P, 6] (start, GEMM loads done,
        GEMM epilogue done, arrival, barrier exit); ONE repeats stage g_lo g_hi - g_lo times."""
        pid = tl.program_id(0)
        P = tl.num_programs(0)
        M = R * S
        for gi in range(g_lo, g_hi):
            g = gi
            if ONE:
                g = g_lo + 0 * gi
            tsp = ts + (gi * P + pid) * 6
            if TS:
                _stamp(tsp)
            if g == 0:
                _embed(pid, P, M, S, tok, offs, emb, pe, x, x2, st, xscale, V, D, BMT, TB, NU, BND)
            elif g == 8 * L + 1:
                _gemm(pid, P, M, x2, D, st, nu_b, cout, eps, wout, bout, V, D, logits, V, x2, st, part, cnt,
                      True, 1, False, BMT, TB, NU, BNV, BKV, SV, NUP, 1, True, tsp, TS)  # fmt: skip
            else:
                layer = (g - 1) // 8
                i = (g - 1) % 8
                l64 = layer.to(tl.int64)
                if i == 0:
                    nu_in = tl.where(layer == 0, nu_a, nu_b)
                    _gemm(pid, P, M, x2, D, st, nu_in, cqkv + layer * 3 * D, eps, wqkv + l64 * 3 * D * D,
                          bqkv + layer * 3 * D, 3 * D, D, qkv, 3 * D, x2, st, part, cnt,
                          True, 0, False, BMT, TB, NU, BNQ, BKQ, SQ, NUP, 1, False, tsp, TS)  # fmt: skip
                elif i == 1:
                    _self_attn(pid, P, R, S, layer, qkv, offs, kc, vc, C, new_kv, ob, D, H, DK, SP, BTS)
                elif i == 2:
                    _gemm(pid, P, M, ob, D, st, NU, bo, eps, wo + l64 * D * D, bo + layer * D, D, D,
                          x, D, x2, st, part, cnt, False, 2, False, BMT, TB, NU, BND, BKD, SD, NUP, 1, False, tsp, TS)  # fmt: skip
                elif i == 3:
                    _gemm(pid, P, M, x2, D, st, nu_a, ccq + layer * D, eps, wcq + l64 * D * D, bcq + layer * D,
                          D, D, q2, D, x2, st, part, cnt, True, 0, False, BMT, TB, NU, BND, BKD, SD, NUP, 1,
                          False, tsp, TS)  # fmt: skip
                elif i == 4:
                    _cross_attn(pid, P, R, S, layer, q2, ck, cv, T, ob, xp, xml, xcnt, D, H, DK, SP, BTX, NS)
                elif i == 5:
                    _gemm(pid, P, M, ob, D, st, NU, bco, eps, wco + l64 * D * D, bco + layer * D, D, D,
                          x, D, x2, st, part, cnt, False, 2, False, BMT, TB, NU, BND, BKD, SD, NUP, 1, False, tsp, TS)  # fmt: skip
                elif i == 6:
                    _gemm(pid, P, M, x2, D, st, nu_a, c1 + layer * F, eps, w1 + l64 * F * D, b1 + layer * F,
                          F, D, hb, F, x2, st, part, cnt, True, 0, True, BMT, TB, NU, BNF, BKF, SF, NUP, 1,
                          False, tsp, TS)  # fmt: skip
                else:
                    _gemm(pid, P, M, hb, F, st, NU, b2, eps, w2 + l64 * D * F, b2 + layer * D, D, F,
                          x, D, x2, st, part, cnt, False, 2, False, BMT, TB, NU2, BN2, BK2, S2, NUP, KS2, False, tsp, TS)  # fmt: skip
            if TS:
                _stamp(tsp + 5)
            if SYNC:
                if gi < g_hi - 1:
                    _grid_arrive(bar)
                    _grid_wait(bar, err, (gi - g_lo + 1) * P, MAXS)
            if TS:
                _stamp(tsp + 1)
        if SYNC:
            _grid_exit(bar, done, P)


def _pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _gemm_tiles(bn: int, kk: int, bmt: int, xbytes: int, budget: int, bk_max: int) -> tuple[int, int]:
    """(BK, STAGES) for a GEMM unit of ``bn`` weight rows x ``kk`` columns: as few round trips as fit the
    shared memory budget (all of a unit's tiles in flight at once if possible)."""
    bk = min(kk, bk_max)
    while True:
        stages = 2 if kk // bk <= 2 else 3
        if stages * (bn * bk * 2 + bk * bmt * xbytes) <= budget or bk <= 64:
            return bk, stages
        bk //= 2


def kernel_config(rows: int, window: int, programs: int, d: int, f: int, heads: int, opts: dict) -> dict:
    """Constexprs of _decoder_kernel for a call of ``rows`` x ``window`` tokens: token blocks (BMT, TB) x
    feature ranges (NU; FFN2: NU2 ranges x KS2 slices of K), cross-attention slices (NS), tiles and
    pipeline depth per GEMM."""
    m = rows * window
    mp = max(16, _pow2(m))
    if mp <= 32:
        bmt, tb = mp, 1
    elif mp <= 64:
        bmt, tb = 32, -(-m // 32)
    else:
        bmt, tb = 64, -(-m // 64)
    nu = programs // tb
    ks2 = opts.get("KS2", 4 if tb == 1 else 1)
    nu2 = programs // (tb * ks2)
    ns = opts.get("NS", max(1, min(8, programs // (rows * heads))))
    budget, bk_max = opts.get("SMEM", 160 * 1024), opts.get("BK_MAX", 512)

    def bn(n, units):
        return min(128, max(16, _pow2(-(-n // units))))

    cfg = {
        "BMT": bmt,
        "TB": tb,
        "NU": nu,
        "NU2": nu2,
        "KS2": ks2,
        "NUP": max(16, _pow2(max(nu, nu2))),
        "NS": ns,
    }
    for key, n, units, kk, xb in (("Q", 3 * d, nu, d, 4), ("D", d, nu, d, 4), ("F", f, nu, d, 4),
                                  ("2", d, nu2, f // ks2, 2), ("V", 0, 0, d, 4)):  # fmt: skip
        b = opts.get("BNV", 64) if key == "V" else bn(n, units)
        bk, stages = _gemm_tiles(b, kk, bmt, xb, budget, bk_max)
        cfg[f"BN{key}"], cfg[f"BK{key}"], cfg[f"S{key}"] = (
            b,
            opts.get(f"BK{key}", bk),
            opts.get(f"S{key}", stages),
        )
    return cfg


class MegaDecoder:
    """The decoder of an encoder-decoder model as one persistent kernel per call (fp16 weights).

    :meth:`step_fn` has the contract of the backends' ``step_fn`` (fills ``logits`` and ``new_kv``) and also
    writes the new keys/values into the cache at each row's offset, as the callers do afterwards
    (``step_fn(...).writes_cache``).
    """

    name = "megakernel"

    @torch.no_grad()
    def __init__(self, decoder, device: torch.device, programs: int | None = None, **opts):
        layers = decoder.decoders

        def f32(t):
            return t.detach().to(device=device, dtype=torch.float32)

        def stack(get):
            return torch.stack([f32(get(layer)) for layer in layers])

        def qkv(att, what):
            if hasattr(att, "linear_qkv"):
                return getattr(att.linear_qkv, what)
            return torch.cat(
                [getattr(att.linear_q, what), getattr(att.linear_k, what), getattr(att.linear_v, what)]
            )

        def folded(w, b, norm):
            """A linear layer after a LayerNorm: (W * gamma (fp16), row sums of that, b + W beta) (see _gemm)."""
            g, beta = norm
            wf = (w * g.unsqueeze(-2)).half()
            c = wf.float().sum(-1)
            bf = b + (w @ beta.unsqueeze(-1)).squeeze(-1)
            return wf.contiguous(), bf.contiguous(), c.contiguous()

        def norms(i):
            return stack(lambda m: (m.norm1, m.norm2, m.norm3)[i].weight), stack(
                lambda m: (m.norm1, m.norm2, m.norm3)[i].bias
            )

        self.wqkv, self.bqkv, self.cqkv = folded(
            stack(lambda m: qkv(m.self_attn, "weight")), stack(lambda m: qkv(m.self_attn, "bias")), norms(0)
        )
        self.wcq, self.bcq, self.ccq = folded(
            stack(lambda m: m.src_attn.linear_q.weight), stack(lambda m: m.src_attn.linear_q.bias), norms(1)
        )
        self.w1, self.b1, self.c1 = folded(
            stack(lambda m: m.feed_forward.w_1.weight), stack(lambda m: m.feed_forward.w_1.bias), norms(2)
        )
        self.wout, self.bout, self.cout = folded(
            f32(decoder.output_layer.weight), f32(decoder.output_layer.bias),
            (f32(decoder.after_norm.weight), f32(decoder.after_norm.bias)),
        )  # fmt: skip
        self.wo = stack(lambda m: m.self_attn.linear_out.weight).half().contiguous()
        self.bo = stack(lambda m: m.self_attn.linear_out.bias).contiguous()
        self.wco = stack(lambda m: m.src_attn.linear_out.weight).half().contiguous()
        self.bco = stack(lambda m: m.src_attn.linear_out.bias).contiguous()
        self.w2 = stack(lambda m: m.feed_forward.w_2.weight).half().contiguous()
        self.b2 = stack(lambda m: m.feed_forward.w_2.bias).contiguous()
        self.eps = float(decoder.after_norm.eps)
        self.emb = decoder.embed[0].weight.detach().to(device=device, dtype=torch.float16).contiguous()
        self.pe = f32(decoder.pe).contiguous()
        self.xscale = float(decoder.xscale)
        self.L = len(layers)
        self.D = self.wo.shape[1]
        self.F = self.w1.shape[1]
        self.H = layers[0].self_attn.h
        self.V = self.wout.shape[0]
        self.device = device
        if programs is None:
            programs = torch.cuda.get_device_properties(device).multi_processor_count
        self.programs = programs
        self.opts = {"num_warps": 8, "MAXS": 1 << 24}
        self.opts.update(opts)
        self.err = torch.zeros(1, dtype=torch.int32, device=device)
        # per (stage, program): 6 time stamps (opts TS=True, profiling only; see _decoder_kernel)
        n_ts = (8 * self.L + 2 + 64) * programs * 6 if self.opts.get("TS") else 1
        self.ts = torch.zeros(n_ts, dtype=torch.int64, device=device)

    def check(self) -> None:
        """Raise if a barrier of an earlier call timed out (its results were garbage)."""
        if self.err.item():
            self.err.zero_()
            raise RuntimeError(
                "the decoder megakernel timed out at a grid barrier (were all its programs resident?)"
            )

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, kind="decode",
                interpret=False):  # fmt: skip
        R, S = tokens.shape
        M = R * S
        D, F, L, P, H = self.D, self.F, self.L, self.programs, self.H
        dev = self.device
        o = self.opts
        cfg = kernel_config(R, S, P, D, F, H, o)
        const = dict(
            D=D, H=H, DK=D // H, F=F, **cfg, SP=max(16, _pow2(S)), BTS=64, BTX=64,
            MAXS=o["MAXS"], TS=bool(o.get("TS")) and not interpret,
        )  # fmt: skip
        nu, ns = max(cfg["NU"], cfg["NU2"]), cfg["NS"]
        f16 = {"device": dev, "dtype": torch.float16}
        i32 = {"device": dev, "dtype": torch.int32}
        x = torch.zeros(M, D, device=dev)
        x2 = torch.zeros(M, 2, D, **f16)
        qkv = torch.zeros(M, 3 * D, **f16)
        ob, q2 = torch.zeros(M, D, **f16), torch.zeros(M, D, **f16)
        hb = torch.zeros(M, F, **f16)
        st = torch.zeros(2, nu, M, device=dev)
        part = torch.zeros(cfg["KS2"], M, D, device=dev)
        cnt = torch.zeros(cfg["TB"] * cfg["NU2"], **i32)
        xp = torch.zeros(ns, M, D, device=dev)
        xml = torch.zeros(ns, M, H, 2, device=dev)
        xcnt = torch.zeros(R * H, **i32)
        bar, done = torch.zeros(1, **i32), torch.zeros(1, **i32)
        C, T = self_k.shape[3], cross_k.shape[3]
        args = (
            tokens, offset, self.emb, self.pe, self.xscale,
            self.wqkv, self.bqkv, self.cqkv, self.wo, self.bo, self.wcq, self.bcq, self.ccq, self.wco, self.bco,
            self.w1, self.b1, self.c1, self.w2, self.b2, self.wout, self.bout, self.cout,
            self_k, self_v, C, cross_k, cross_v, T, logits, new_kv,
            x, x2, qkv, ob, q2, hb, st, part, cnt, xp, xml, xcnt, bar, self.err, done, self.ts,
            R, S, L, self.V, self.eps,
        )  # fmt: skip
        nus = (
            cfg["NU"],
            cfg["NU2"],
        )  # after g_lo, g_hi (run-time numbers of statistics parts, see the kernel)
        n_stages = 8 * L + 2
        nw = o["num_warps"]

        def run() -> None:
            if interpret:
                for g in range(n_stages):
                    _decoder_kernel[(P,)](*args, g, g + 1, *nus, SYNC=False, **const, num_warps=nw)
            else:
                _decoder_kernel[(P,)](*args, 0, n_stages, *nus, SYNC=True, **const, num_warps=nw)

        def one(g: int, reps: int) -> None:  # profiling: stage g, reps times (see _decoder_kernel)
            _decoder_kernel[(P,)](*args, g, g + reps, *nus, SYNC=True, ONE=True, **const, num_warps=nw)

        def upto(g: int) -> None:  # debugging: stages 0 .. g-1 only
            _decoder_kernel[(P,)](*args, 0, g, *nus, SYNC=True, **const, num_warps=nw)

        run.one = one
        run.upto = upto
        run.buffers = (x, x2, qkv, ob, q2, hb, st, part, cnt, xp, xml, xcnt, bar, done)  # keep them alive
        run.const = const
        run.writes_cache = True
        return run


class MegaBackend:
    """A backend whose decoder calls (decode, verify) of up to ``max_tokens`` tokens (rows x window) and
    ``max_rows`` rows run as a megakernel (``kernel.step_fn``); everything else (encoder, prompt prefill, larger
    calls) is the wrapped backend's (TensorRT). ``rows_kernel`` = (kernel, max_tokens, max_rows): another
    megakernel (and its limits) for the calls of several rows (kernel None: the wrapped backend's).

    With ``trial``, every call on a kernel runs once when it is made: if that raises (a compile error, out of
    memory, ...), leaves non-finite logits or times out at a grid barrier, the call is the wrapped backend's
    instead (logged), and so is every later one on that kernel."""

    def __init__(self, inner, kernel, max_tokens: int | None = None, max_rows: int | None = None,
                 rows_kernel: tuple | None = None, trial: bool = False):  # fmt: skip
        self.inner, self.kernel = inner, kernel
        self.max_tokens, self.max_rows = max_tokens, max_rows
        self.rows_kernel = rows_kernel
        self.trial = trial
        self.failed: set[int] = set()  # id() of kernels whose trial failed

    def __getattr__(self, name):  # encode, encode_fn, trimmed, cfg, name, ...
        return getattr(self.inner, name)

    def _kernels(self):
        return [k for k in (self.kernel, (self.rows_kernel or (None,))[0]) if k is not None]

    def supports_shared_cross(self, rows: int) -> bool:
        kernel, tokens, limit = self.kernel, self.max_tokens, self.max_rows
        if rows > 1 and self.rows_kernel is not None:
            kernel, tokens, limit = self.rows_kernel
        return (bool(getattr(kernel, "shared_cross_kv", False)) and id(kernel) not in self.failed
                and (tokens is None or rows <= tokens) and (limit is None or rows <= limit))

    def _inner_step(self, args, kind, reason):
        # If a shared-KV kernel's trial failed, retain correct TRT fallback. The session refreshes
        # these expanded buffers after each encode, outside the per-token decoder loop.
        rows, cross_rows = args[0].shape[0], args[4].shape[1]
        prepare = None
        if rows != cross_rows:
            if cross_rows < 1 or rows % cross_rows:
                raise ValueError("cross KV rows must divide decoder rows")
            sources = args[4:6]
            expanded = [torch.empty(t.shape[0], rows, *t.shape[2:], dtype=t.dtype, device=t.device)
                        for t in sources]

            def prepare():
                for target, source in zip(expanded, sources):
                    target.copy_(source.repeat_interleave(rows // cross_rows, dim=1))

            prepare()
            args = (*args[:4], *expanded, *args[6:])
            reason += "; shared cross KV expanded after encode"
        run = self.inner.step_fn(*args, kind=kind)
        run.dispatch, run.fallback_reason = self.inner.name, reason
        run.prepare_cross = prepare
        return run

    @property
    def supports_finished_stop(self) -> bool:
        """Whether the single-row kernel can skip a whole completed plain decode call."""
        return bool(getattr(self.kernel, "supports_finished_stop", False)) and id(self.kernel) not in self.failed

    def supports_finished_groups(self, rows: int) -> bool:
        """Whether the actual multi-row decode route supports a uniform completed-group exit."""
        if rows < 2 or self.rows_kernel is None:
            return False
        kernel, tokens, limit = self.rows_kernel
        return (bool(getattr(kernel, "supports_finished_groups", False)) and id(kernel) not in self.failed
                and (tokens is None or rows <= tokens) and (limit is None or rows <= limit))

    @property
    def argmax(self) -> bool:
        """Whether step_fn takes ``pred``: calls on a kernel that can then write the logits' argmax there too
        (see writes_pred on the returned function; other calls ignore it). ``notime`` = (flags [R] bool,
        first timestamp token, last one): rows flagged get the argmax over the other tokens (the timestamp
        rules under <notimestamps>); the others' are not final then."""
        return any(getattr(k, "argmax", False) for k in self._kernels())

    @property
    def solo_differs(self) -> bool:
        """Whether a single-row call is a different call when no encoder runs alongside (step_fn's ``solo``)."""
        return getattr(self.kernel, "solo_differs", False) and id(self.kernel) not in self.failed

    def step_fn(self, tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv, kind="decode", pred=None,
                solo=False, notime=None, inp=None, stop=None):  # fmt: skip
        """``pred`` (and ``notime``): see argmax; ``solo``: nothing runs alongside the call (see solo_differs);
        ``inp`` = (the call's input rows [R, W + 4]: W tokens, offset, ...; W): a kernel reads the tokens and
        offsets there rather than from ``tokens`` / ``offset`` (``run.reads_inp``: no copies needed)."""
        args = (tokens, offset, self_k, self_v, cross_k, cross_v, logits, new_kv)
        kernel, max_tokens, max_rows = self.kernel, self.max_tokens, self.max_rows
        if tokens.shape[0] > 1 and self.rows_kernel is not None:
            kernel, max_tokens, max_rows = self.rows_kernel
        big = (max_tokens is not None and tokens.numel() > max_tokens) or (
            max_rows is not None and tokens.shape[0] > max_rows
        )
        unsupported_cross = cross_k.shape[1] != tokens.shape[0] and not getattr(kernel, "shared_cross_kv", False)
        if kind == "prefill" or big or kernel is None or id(kernel) in self.failed or unsupported_cross:
            reason = ("prefill" if kind == "prefill" else "shape limit" if big
                      else "no kernel for rows" if kernel is None else "kernel trial failed"
                      if id(kernel) in self.failed else "kernel lacks shared cross KV")
            return self._inner_step(args, kind, reason)
        kw = {"pred": pred, "notime": notime} if pred is not None and getattr(kernel, "argmax", False) else {}
        if solo and getattr(kernel, "solo_differs", False):
            kw["solo"] = True
        if stop is not None and kind == "decode" and tokens.shape[1] == 1:
            single = tokens.shape[0] == 1 and getattr(kernel, "supports_finished_stop", False)
            group = tokens.shape[0] > 1 and getattr(kernel, "supports_finished_groups", False)
            if single or group:
                kw["stop"] = stop
        kargs = args
        if inp is not None:  # (the kernels take strided tokens / offsets)
            rows, w = inp
            kargs = (rows[:, :w], rows[:, w], *args[2:])
        if not self.trial:
            run = kernel.step_fn(*kargs, kind=kind, **kw)
            run.reads_inp = inp is not None
            run.dispatch, run.fallback_reason = kernel.name, None
            return run
        try:
            run = kernel.step_fn(*kargs, kind=kind, **kw)
            run.reads_inp = inp is not None
            run()
            torch.cuda.synchronize(logits.device)
            kernel.check()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("non-finite logits")
            run.dispatch, run.fallback_reason = kernel.name, None
            return run
        except Exception as e:  # noqa: BLE001 (any failure: TensorRT instead)
            self.failed.add(id(kernel))
            log.warning(
                "decoder megakernel %s failed (%s: %s); TensorRT instead", kernel.name, type(e).__name__, e
            )
            return self._inner_step(args, kind, f"{kernel.name} trial: {type(e).__name__}: {e}")

    def check(self) -> None:
        self.kernel.check()
