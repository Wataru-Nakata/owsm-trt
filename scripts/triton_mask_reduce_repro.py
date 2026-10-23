"""Minimal reproducer of a Triton 3.8 miscompile (found in the decoder megakernel, worked around there).

A [256, M] masked load whose mask is arange(0, 256) < N, weights computed under the same condition and a sum
over axis 0 give wrong results when N is a tl.constexpr that is not a power of 2 (132, 33); with N passed at run
time they are right. The integer counts alone are right. H200, Triton 3.8.0, torch 2.14.1+cu130 (job 2750010).

python scripts/triton_mask_reduce_repro.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def counts(out, n_rt, D: tl.constexpr, N_C: tl.constexpr, CONST: tl.constexpr):
    i = tl.arange(0, 256)
    n = N_C if CONST else n_rt
    cnt = (i + 1) * D // n - i * D // n
    tl.store(out + i, cnt)


@triton.jit
def masked_sum(out, mu_ptr, n_rt, D: tl.constexpr, N_C: tl.constexpr, CONST: tl.constexpr, M: tl.constexpr):
    """The pattern of the bug in the wild: [256, M] masked tile, weights from the counts, reduction."""
    i = tl.arange(0, 256)
    t = tl.arange(0, M)
    n = N_C if CONST else n_rt
    valid = i < n
    cnt = tl.where(valid, ((i + 1) * D // n - i * D // n).to(tl.float32), 0.0)
    mu = tl.load(mu_ptr + i[:, None] * M + t[None, :], mask=valid[:, None] & (t >= 0)[None, :], other=0.0)
    tl.store(out + t, tl.sum(mu * cnt[:, None], 0) / D)


def main():
    print(
        "triton",
        triton.__version__,
        "torch",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.get_device_name(),
        flush=True,
    )
    for D in (256, 1024):
        for N in (132, 128, 33):
            ref = torch.tensor([(i + 1) * D // N - i * D // N for i in range(256)])
            line = f"D={D} N={N}:"
            for const in (False, True):
                out = torch.zeros(256, dtype=torch.int32, device="cuda")
                counts[(1,)](out, N, D=D, N_C=N, CONST=const, num_warps=4)
                ok = torch.equal(out.cpu().long(), ref)
                line += f"  counts({'constexpr' if const else 'runtime'}) {'ok' if ok else 'WRONG'}"
            for M in (16, 32):
                mu = torch.randn(256, M, device="cuda")
                refm = (mu[:N].cpu() * ref[:N, None].float()).sum(0) / D
                for const in (False, True):
                    for nw in (4, 8):
                        o = torch.zeros(M, device="cuda")
                        masked_sum[(1,)](o, mu, N, D=D, N_C=N, CONST=const, M=M, num_warps=nw)
                        err = (o.cpu() - refm).abs().max().item()
                        line += f"  sum[M={M},{'c' if const else 'rt'},w{nw}] {'ok' if err < 1e-4 else f'WRONG({err:.2g})'}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
