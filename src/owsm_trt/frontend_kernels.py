"""Inference-only fp32 frontend pointwise kernels; STFT and mel GEMM stay in PyTorch."""

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None


def available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


if triton is not None:

    @triton.jit
    def _power(
        spec,
        out,
        N,
        F: tl.constexpr,
        T: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        S2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        f, t, b = i % F, (i // F) % T, i // (F * T)
        p = 2 * (b * S0 + f * S1 + t * S2)
        real = tl.load(spec + p, i < N, other=0)
        imag = tl.load(spec + p + 1, i < N, other=0)
        # Separate fp32 multiplication/addition matches the two PyTorch squares; no FMA.
        value = real * real + imag * imag
        tl.store(out + i, value, i < N)

    @triton.jit
    def _log_mvn(mel, mean, std, out, N, M: tl.constexpr, GLOBAL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(mel + i, i < N, other=1.0)
        x = tl.where(x < 1.0e-10, 1.0e-10, x)
        # Use CUDA libdevice's logf, not Triton's approximate log2-based logarithm.
        x = libdevice.log(x)
        if GLOBAL:
            mu = tl.load(mean + i % M)
            sigma = tl.load(std + i % M)
            x = tl.div_rn(x - mu, sigma)
        tl.store(out + i, x, i < N)


def power(spec: torch.Tensor) -> torch.Tensor:
    """Complex [B,F,T] -> contiguous fp32 [B,T,F], keeping frequency-major STFT values."""
    b, f, t = spec.shape
    out = torch.empty((b, t, f), dtype=torch.float32, device=spec.device)
    _power[(triton.cdiv(out.numel(), 256),)](
        torch.view_as_real(spec), out, out.numel(), f, t, *spec.stride(), 256, enable_fp_fusion=False
    )
    return out


def log_mvn(mel: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    """Contiguous fp32 [B,T,M] -> [B,T,M]; optional global mean/std normalization."""
    out = torch.empty_like(mel)
    _log_mvn[(triton.cdiv(mel.numel(), 256),)](
        mel,
        mean if mean is not None else mel,
        std if std is not None else mel,
        out,
        mel.numel(),
        mel.shape[-1],
        mean is not None,
        256,
        enable_fp_fusion=False,
    )
    return out
