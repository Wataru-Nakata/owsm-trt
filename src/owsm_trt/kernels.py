"""Triton kernels for the work around a decoder call (CUDA only; the callers fall back to PyTorch).

* :func:`choose` -- the argmax after every window position under ESPnet's timestamp rules, in one pass
  over the logits. The allowed tokens are a few ranges of ids, so no [rows, vocab] mask is built.
* :func:`write_kv` -- each row's new keys/values into the KV cache at its own offset.
* :func:`accept` -- per row, how many drafted tokens the decoder accepted, and the token after them: all
  the host needs back from a call.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # e.g. torch without Triton (Windows): the PyTorch code paths are used
    triton = None


def available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


if triton is not None:

    @triton.jit
    def _choose(logits, inp, notime, pred, V, W: tl.constexpr, RULES: tl.constexpr, first_time, last_time, eos,
                BV: tl.constexpr):  # fmt: skip
        """One window position: inp row = [W tokens | offset | #timestamps | last timestamp | first position]."""
        r = tl.program_id(0)
        i = tl.program_id(1)
        row = inp + r * (W + 4)
        if RULES:
            count = tl.load(row + W + 1)
            last = tl.load(row + W + 2)
            for j in range(1, i + 1):  # timestamps the window itself adds before position i
                t = tl.load(row + j)
                is_t = (t >= first_time) & (t <= last_time)
                count += is_t.to(tl.int64)
                last = tl.where(is_t, tl.maximum(last, t), last)
            prev = tl.load(row + i)
            prev_time = (prev >= first_time) & (prev <= last_time)
            first = tl.load(row + W + 3) == i
            no_time = tl.load(notime + r) != 0
        best = tl.full([], float("-inf"), tl.float32)
        best_i = tl.full([], 0, tl.int64)
        base = logits + (r * W + i).to(tl.int64) * V
        for v0 in range(0, V, BV):
            v = v0 + tl.arange(0, BV)
            x = tl.load(base + v, mask=v < V, other=float("-inf"))
            if RULES:
                time = (v >= first_time) & (v <= last_time)
                open_pair = (v == eos) | (time & (v <= last))
                after_end = ((v < last) | (v > last_time)) & (v != eos)
                bad = tl.where(prev_time, after_end, True)
                bad = tl.where(count % 2 == 1, open_pair, bad)
                bad = tl.where(first, ~time, bad)
                bad = tl.where(no_time, time, bad)
                x = tl.where(bad, float("-inf"), x)
            m = tl.max(x, 0)
            best_i = tl.where(m > best, v0 + tl.argmax(x, 0).to(tl.int64), best_i)  # ties: the first one
            best = tl.maximum(best, m)
        tl.store(pred + r * W + i, best_i)

    @triton.jit
    def _write_kv(new_kv, k, v, offs, R, S, C, H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr):
        """new_kv [L, 2, R, H, S, D] -> k, v [L, R, H, C, D] at slots offs[r] .. offs[r] + S - 1."""
        p = tl.program_id(0)  # (layer, row, head)
        kv = tl.program_id(1)
        h = p % H
        r = (p // H) % R
        layer = p // (H * R)
        s = tl.arange(0, BS)
        d = tl.arange(0, D)
        m = (s < S)[:, None]
        src = new_kv + (((layer * 2 + kv) * R + r) * H + h) * S * D + s[:, None] * D + d[None, :]
        off = tl.load(offs + r)
        dst = p.to(tl.int64) * C * D + (off + s)[:, None] * D + d[None, :]
        x = tl.load(src, mask=m)
        if kv == 0:
            tl.store(k + dst, x, mask=m)
        else:
            tl.store(v + dst, x, mask=m)

    @triton.jit
    def _accept(pred, inp, res, eos, W: tl.constexpr, BW: tl.constexpr):
        """One row: inp = [W tokens | offset | #timestamps | last timestamp | first position]."""
        r = tl.program_id(0)
        row = inp + r * (W + 4)
        s = tl.maximum(tl.load(row + W + 3), 0)  # where the head (primer or last token) ends
        i = tl.arange(0, BW)
        inside = i < W - 1
        p = tl.load(pred + r * W + i, mask=inside, other=0)  # the decoder's choice after position i ...
        nxt = tl.load(row + 1 + i, mask=inside, other=0)  # ... for the token at i + 1
        ok = (inside & (p == nxt) & (nxt != eos)) | (i < s)
        end = tl.min(tl.where(ok, BW, i), 0)  # s + n: the first position whose choice is not accepted
        tl.store(res + r * 2, end - s)
        tl.store(res + r * 2 + 1, tl.load(pred + r * W + end))


def choose(logits: torch.Tensor, inp: torch.Tensor, notime: torch.Tensor, pred: torch.Tensor, rules) -> None:
    """``pred`` [R, W] = argmax of ``logits`` [R, W, V] under ``rules`` (None: plain argmax); ``inp``
    [R, W + 4] as in runtime._Verify, ``notime`` [R] bool."""
    R, W, V = logits.shape
    args = (rules.first_time, rules.last_time, rules.eos) if rules is not None else (0, -1, -1)
    _choose[(R, W)](logits, inp, notime, pred, V, W, rules is not None, *args, BV=2048, num_warps=4)


def write_kv(new_kv: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offs: torch.Tensor) -> None:
    """Write ``new_kv`` [L, 2, R, H, S, D] into the caches ``k``/``v`` [L, R, H, C, D] at ``offs`` [R]."""
    L, _, R, H, S, D = new_kv.shape
    C = k.shape[3]
    BS = max(16, triton.next_power_of_2(S))
    _write_kv[(L * R * H, 2)](new_kv, k, v, offs, R, S, C, H=H, D=D, BS=BS, num_warps=4)


def accept(pred: torch.Tensor, inp: torch.Tensor, res: torch.Tensor, eos: int) -> None:
    """``res`` [R, 2] = runtime._Verify.accept() for the argmax ``pred`` [R, W] of a call on ``inp``."""
    R, W = pred.shape
    _accept[(R,)](pred, inp, res, eos, W=W, BW=triton.next_power_of_2(max(W, 2)), num_warps=1)
