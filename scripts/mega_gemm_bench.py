"""What does one small GEMM unit of the megakernel cost, without memory latency in the way? Every program (one
per SM) computes the same [BN rows x K] x [K x MT tokens] product many times over L2-resident data, no
barriers; time per unit from %globaltimer. Variants: tl.dot (2D), batched dot with K across warps,
CUDA-core FMAs (tl.sum), loads only.

python scripts/mega_gemm_bench.py [MODEL]
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _now():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _unit(w, x, out, ts, REPS, M, K: tl.constexpr, BN: tl.constexpr, MT: tl.constexpr, BK: tl.constexpr,
          STAGES: tl.constexpr, MODE: tl.constexpr, KSPLIT: tl.constexpr):  # fmt: skip
    pid = tl.program_id(0)
    rn = tl.arange(0, BN)
    tok = tl.arange(0, MT)
    tm = tok < M
    t0 = _now()
    tot = tl.zeros([BN, MT], dtype=tl.float32)
    for r in range(REPS):
        wb = w + (r % 4) * BN * K  # 4 slabs round robin (all in L2)
        if MODE == 0:  # 2D dot
            acc = tl.zeros([BN, MT], dtype=tl.float32)
            for k0 in tl.range(0, K, BK, num_stages=STAGES):
                rk = k0 + tl.arange(0, BK)
                wt = tl.load(wb + rn[:, None] * K + rk[None, :])
                xt = tl.load(
                    x + tok[None, :] * K + rk[:, None], mask=tm[None, :], other=0.0, cache_modifier=".cg"
                )
                acc = tl.dot(wt, xt, acc)
        elif MODE == 1:  # batched dot, K across warps
            KC: tl.constexpr = BK // KSPLIT
            j = tl.arange(0, KSPLIT)
            kc = tl.arange(0, KC)
            acc3 = tl.zeros([KSPLIT, BN, MT], dtype=tl.float32)
            for k0 in tl.range(0, K, BK, num_stages=STAGES):
                kidx = k0 + j[:, None] * KC + kc[None, :]
                wt = tl.load(wb + rn[None, :, None] * K + kidx[:, None, :])
                xt = tl.load(x + tok[None, None, :] * K + kidx[:, :, None], mask=tm[None, None, :], other=0.0,
                             cache_modifier=".cg")  # fmt: skip
                acc3 = tl.dot(wt, xt, acc3)
            acc = tl.sum(acc3, 0)
        elif MODE == 2:  # CUDA-core FMAs: [BN, MT, BK] products summed over K
            acc = tl.zeros([BN, MT], dtype=tl.float32)
            for k0 in tl.range(0, K, BK, num_stages=STAGES):
                rk = k0 + tl.arange(0, BK)
                wt = tl.load(wb + rn[:, None] * K + rk[None, :]).to(tl.float32)
                xt = tl.load(x + tok[:, None] * K + rk[None, :], mask=tm[:, None], other=0.0,
                             cache_modifier=".cg").to(tl.float32)  # fmt: skip
                acc += tl.sum(wt[:, None, :] * xt[None, :, :], 2)
        else:  # loads only
            acc = tl.zeros([BN, MT], dtype=tl.float32)
            for k0 in tl.range(0, K, BK, num_stages=STAGES):
                rk = k0 + tl.arange(0, BK)
                wt = tl.load(wb + rn[:, None] * K + rk[None, :]).to(tl.float32)
                xt = tl.load(x + tok[:, None] * K + rk[None, :], mask=tm[:, None], other=0.0,
                             cache_modifier=".cg").to(tl.float32)  # fmt: skip
                acc += tl.sum(wt, 1)[:, None] + tl.sum(xt, 1)[None, :]
        tot += acc
    t1 = _now()
    tl.store(out + pid * BN * MT + rn[:, None] * MT + tok[None, :], tot)
    tl.store(ts + pid, t1 - t0)


def main():
    P = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"
    reps = 200
    print(f"{P} programs, {reps} units each, data in L2; us per unit (median over programs)", flush=True)
    for K, BN in ((1024, 16), (1024, 32), (4096, 16)):
        w = torch.randn(4 * BN * K, device=dev, dtype=torch.float16) * 0.01
        for MT, M in ((16, 1), (16, 8), (16, 16), (32, 32)):
            x = torch.randn(MT * K, device=dev, dtype=torch.float16) * 0.01
            out = torch.zeros(P * BN * MT, device=dev)
            ts = torch.zeros(P, dtype=torch.int64, device=dev)
            line = f"K={K} BN={BN} M={M:2d}:"
            for name, mode, bk, st, ksp, nw in (
                ("dot bk512 s2", 0, 512, 2, 1, 8), ("dot bk256 s3", 0, 256, 3, 1, 8), ("dot bk128 s4", 0, 128, 4, 1, 8),
                ("dot bk256 s3 w4", 0, 256, 3, 1, 4), ("dot3d ks8", 1, 512, 2, 8, 8), ("dot3d ks4", 1, 256, 3, 4, 8),
                ("fma bk64", 2, 64, 3, 1, 8), ("fma bk32", 2, 32, 3, 1, 8), ("loads", 3, 256, 3, 1, 8),
            ):  # fmt: skip
                if mode == 2 and BN * MT * bk > 16384:
                    continue
                try:
                    args = (w, x, out, ts, reps, M)
                    kw = dict(K=K, BN=BN, MT=MT, BK=bk, STAGES=st, MODE=mode, KSPLIT=ksp, num_warps=nw)
                    _unit[(P,)](*args, **kw)
                    torch.cuda.synchronize()
                    _unit[(P,)](*args, **kw)
                    torch.cuda.synchronize()
                    us = ts.double().median().item() / reps / 1e3
                    line += f"  {name} {us:.2f}"
                except Exception as e:  # noqa: BLE001
                    line += f"  {name} FAILED({type(e).__name__})"
            print(line, flush=True)


if __name__ == "__main__":
    main()
