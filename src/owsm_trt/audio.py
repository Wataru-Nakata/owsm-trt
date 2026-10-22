"""Audio loading and resampling without extra dependencies (soundfile + torch only)."""

import math
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

AudioLike = str | os.PathLike | np.ndarray | torch.Tensor


@lru_cache(maxsize=8)
def _sinc_kernel(orig: int, new: int, width: int = 6, rolloff: float = 0.99) -> torch.Tensor:
    """Hann-windowed sinc kernel [new, 1, K] (same construction as torchaudio's default resampler)."""
    base = min(orig, new) * rolloff
    w = math.ceil(width * orig / base)
    idx = torch.arange(-w, w + orig, dtype=torch.float64)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx
    t = (t * base).clamp(-width, width)
    window = torch.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0, dtype=torch.float64), torch.sin(t) / t)
    return (kernel * window * base / orig).float()  # [new, 1, K]


def resample(wav: torch.Tensor, orig: int, new: int) -> torch.Tensor:
    """Band-limited resampling of a 1-D waveform."""
    if orig == new:
        return wav
    g = math.gcd(orig, new)
    orig, new = orig // g, new // g
    kernel = _sinc_kernel(orig, new).to(wav.device)
    w = (kernel.shape[-1] - orig) // 2
    n = wav.shape[-1]
    x = F.pad(wav.reshape(1, 1, n).float(), (w, w + orig))
    y = F.conv1d(x, kernel, stride=orig)  # [1, new, frames]
    y = y.transpose(1, 2).reshape(-1)
    return y[: math.ceil(new * n / orig)]


def duration(audio: AudioLike, sr: int) -> float:
    """Length in seconds, read from the file header for paths (no decoding)."""
    if isinstance(audio, (str, os.PathLike)):
        import soundfile as sf

        info = sf.info(os.fspath(audio))
        return info.frames / info.samplerate
    shape = np.shape(audio)
    return max(shape) / sr if shape else 0.0


def load_audio(audio: AudioLike, sample_rate: int = 16000, sr: int | None = None) -> torch.Tensor:
    """Return a mono float32 waveform at ``sample_rate`` as a 1-D CPU tensor.

    ``audio`` is a file path, or an array/tensor sampled at ``sr`` (defaults to ``sample_rate``).
    """
    if isinstance(audio, (str, os.PathLike)):
        import soundfile as sf

        data, sr = sf.read(os.fspath(audio), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))
    else:
        wav = torch.as_tensor(audio, dtype=torch.float32).cpu()
        if wav.dim() == 2:  # [N, C] or [C, N]
            wav = wav.mean(dim=1 if wav.shape[0] > wav.shape[1] else 0)
        if wav.dim() != 1:
            raise ValueError(f"expected 1-D or 2-D audio, got shape {tuple(wav.shape)}")
    return resample(wav, sr or sample_rate, sample_rate)
