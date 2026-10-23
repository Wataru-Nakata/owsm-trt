"""Gate (a), part 1: a GEMM stage unit in Gluon with its weights already in shared memory.

y[t, n0 + r] = sum_k X[t, k] W[n0 + r, k] + b for the BN weight rows of a program, MT (padded) tokens.
wgmma wants 64 rows on its A side, so the weight slab [BN, K] is cut into S = 64 / BN blocks of K' = K / S
columns stacked as A = [64, K'] (row s * BN + r = W[n0 + r, s K' : (s + 1) K']) and X goes to the B side as
[S * MT, K'] (row s * MT + t = X[t, s K' : (s + 1) K']). One wgmma chain over K' computes all block products;
y is the sum of the S diagonal blocks (s = s'). Nothing is padded in shared memory; the S-fold extra
MMA work is cheap.

Every program (one per SM) loads its slab once (TMA), then repeats: X from L2 (cp.async), the wgmma chain,
the diagonal sum, store y. Time per unit from %globaltimer.

python scripts/gluon_unit_bench.py [MODEL]
"""

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


@gluon.jit
def _now():
    return gl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=gl.int64, is_pure=False, pack=1
    )


@gluon.jit
def _unit(w_desc, bias, x, y, ts, REPS, M, N, K: gl.constexpr, BN: gl.constexpr, MT: gl.constexpr,
          LBN: gl.constexpr, LMT: gl.constexpr, NW: gl.constexpr):  # fmt: skip
    S: gl.constexpr = 64 // BN
    KP: gl.constexpr = K // S
    NKC: gl.constexpr = KP // 64
    NB: gl.constexpr = S * MT
    pid = gl.program_id(0)
    n0 = (pid * BN) % N
    sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    wbuf = gl.allocate_shared_memory(gl.float16, [NKC * 64, 64], sl)  # A tiles: chunk kc = rows kc*64 ..
    xbuf = gl.allocate_shared_memory(gl.float16, [NKC * NB, 64], sl)  # B tiles: chunk kc = rows kc*NB ..
    scl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    scratch = gl.allocate_shared_memory(gl.float32, [64, NB], scl)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    mbarrier.expect(bar, BN * K * 2)
    for kc in gl.static_range(NKC):
        for s in gl.static_range(S):
            tma.async_load(w_desc, [n0, s * KP + kc * 64], bar, wbuf.slice(kc * 64 + s * BN, BN))
    mbarrier.wait(bar, 0)

    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[NW, 1], instr_shape=[16, NB, 16]
    )
    xl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    rho = gl.arange(0, NKC * NB, layout=gl.SliceLayout(1, xl))  # (kc, s, t)
    kk = gl.arange(0, 64, layout=gl.SliceLayout(0, xl))
    t = rho & (MT - 1)
    s_ = (rho >> LMT) & (S - 1)
    kc_ = rho >> (LMT + 6 - LBN)
    xp = x + t[:, None] * K + (s_ * KP + kc_ * 64)[:, None] + kk[None, :]
    xm = (t < M)[:, None] & (kk >= 0)[None, :]
    ol: gl.constexpr = gl.BlockedLayout([1, 1], [32 // MT, MT], [NW, 1], [1, 0])
    rr = gl.arange(0, BN, layout=gl.SliceLayout(1, ol))
    tt = gl.arange(0, MT, layout=gl.SliceLayout(0, ol))
    b = gl.load(bias + n0 + rr).to(gl.float32)
    yp = y + tt[None, :] * N + n0 + rr[:, None]
    ym = (tt < M)[None, :] & (rr >= 0)[:, None]
    t0 = _now()
    for r in range(REPS):
        async_copy.async_load(xbuf, xp, mask=xm, cache_modifier=".cg")
        async_copy.commit_group()
        async_copy.wait_group(0)
        fence_async_shared()
        gl.barrier()
        acc = gl.zeros([64, NB], gl.float32, layout=mma)
        for kc in gl.static_range(NKC):
            acc = warpgroup_mma(
                wbuf.slice(kc * 64, 64), xbuf.slice(kc * NB, NB).permute((1, 0)), acc, is_async=True
            )
        acc = warpgroup_mma_wait(0, deps=[acc])
        scratch.store(acc)
        gl.barrier()
        out = gl.zeros([BN, MT], gl.float32, layout=ol)
        for s in gl.static_range(S):
            out += scratch.slice(s * BN, BN, dim=0).slice(s * MT, MT, dim=1).load(ol)
        gl.store(yp, (out + b[:, None]).to(gl.float16), mask=ym)
        gl.barrier()
    t1 = _now()
    gl.store(ts + pid, t1 - t0)
    mbarrier.invalidate(bar)


def log2(n):
    return int(n).bit_length() - 1


def main():
    P = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"
    reps = 200
    print(
        f"{P} programs, {reps} units each, weights in shared memory, X from L2; us per unit (median)",
        flush=True,
    )
    layout = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    for K, BN, N in ((1024, 16, 2048), (1024, 8, 1024), (1024, 32, 3072), (1024, 32, 4096), (4096, 8, 1024)):
        w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.03
        bias = torch.randn(N, device=dev, dtype=torch.float16)
        desc = TensorDescriptor.from_tensor(w, [BN, 64], layout)
        line = f"K={K} BN={BN}:"
        for M, MT in ((1, 8), (8, 8), (16, 16)):
            if K * MT * 2 > 64 * 1024:
                continue
            x = torch.randn(16, K, device=dev, dtype=torch.float16)
            y = torch.zeros(16, N, device=dev, dtype=torch.float16)
            ts = torch.zeros(P, dtype=torch.int64, device=dev)
            try:
                for _ in range(2):
                    _unit[(P,)](desc, bias, x, y, ts, reps, M, N, K=K, BN=BN, MT=MT, LBN=log2(BN), LMT=log2(MT),
                                NW=4, num_warps=4)  # fmt: skip
                torch.cuda.synchronize()
                us = ts.double().median().item() / reps / 1e3
                n_used = min(P * BN, N)
                ref = (x[:M].float() @ w.float().T + bias.float())[:, :n_used]
                err = (y[:M, :n_used].float() - ref).abs().max().item()
                line += f"  M={M} {us:.2f} us (err {err:.3f})"
            except Exception as e:  # noqa: BLE001
                line += f"  M={M} FAILED {type(e).__name__}: {str(e)[:300]}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
