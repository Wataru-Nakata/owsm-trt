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
    def _choose(logits, inp, notime, pred, scores, V, W: tl.constexpr, RULES: tl.constexpr, first_time, last_time, eos,
                BV: tl.constexpr, SKIP_NOTIME: tl.constexpr = False, PARTS: tl.constexpr = 1):  # fmt: skip
        """One window position: inp row = [W tokens | offset | #timestamps | last timestamp | first position].
        SKIP_NOTIME: rows with <notimestamps> are left alone (their argmax is there already)."""
        r = tl.program_id(0)
        i = tl.program_id(1)
        if SKIP_NOTIME:
            if tl.load(notime + r) != 0:
                return
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
        start = tl.program_id(2) * BV if PARTS > 1 else 0
        stop = start + BV if PARTS > 1 else V
        for v0 in range(start, stop, BV):
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
        dest = (r * W + i) * PARTS + tl.program_id(2) if PARTS > 1 else r * W + i
        tl.store(pred + dest, best_i)
        if PARTS > 1:
            tl.store(scores + dest, best)

    @triton.jit
    def _choose_reduce(scores, indices, notime, pred, W: tl.constexpr, PARTS: tl.constexpr,
                       BP: tl.constexpr, SKIP_NOTIME: tl.constexpr):  # fmt: skip
        r, i = tl.program_id(0), tl.program_id(1)
        if SKIP_NOTIME:
            if tl.load(notime + r) != 0:
                return
        p = tl.arange(0, BP)
        at = (r * W + i) * PARTS + p
        val = tl.load(scores + at, p < PARTS, other=float("-inf"))
        best = tl.max(val, 0)
        # Select the first vocabulary block, then its original argmax. All-invalid stays token 0.
        part = tl.min(tl.where((p < PARTS) & (val == best), p, BP), 0)
        idx = tl.load(indices + (r * W + i) * PARTS + part, part < PARTS, other=0)
        tl.store(pred + r * W + i, tl.where(best > float("-inf"), idx, 0))

    @triton.jit
    def _choose_accept(scores, indices, notime, pred, inp, res, eos, W: tl.constexpr,
                       PARTS: tl.constexpr, BW: tl.constexpr, BP: tl.constexpr,
                       SKIP_NOTIME: tl.constexpr):  # fmt: skip
        r = tl.program_id(0)
        w, p = tl.arange(0, BW), tl.arange(0, BP)
        skip = tl.load(notime + r) != 0 if SKIP_NOTIME else False
        if skip:
            chosen = tl.load(pred + r * W + w, w < W, other=0)
        else:
            at = (r * W + w[:, None]) * PARTS + p[None, :]
            val = tl.load(scores + at, (w < W)[:, None] & (p < PARTS)[None, :], other=float("-inf"))
            best = tl.max(val, 1)
            part = tl.min(tl.where((p < PARTS)[None, :] & (val == best[:, None]), p[None, :], BP), 1)
            chosen = tl.load(indices + (r * W + w) * PARTS + part, (w < W) & (part < PARTS), other=0)
            chosen = tl.where(best > float("-inf"), chosen, 0)
            tl.store(pred + r * W + w, chosen, w < W)
        row = inp + r * (W + 4)
        first = tl.maximum(tl.load(row + W + 3), 0)
        nxt = tl.load(row + w + 1, w < W - 1, other=0)
        ok = ((w < W - 1) & (chosen == nxt) & (nxt != eos)) | (w < first)
        end = tl.min(tl.where(ok, BW, w), 0)
        last = tl.sum(tl.where(w == end, chosen, 0), 0)
        tl.store(res + r * 2, end - first)
        tl.store(res + r * 2 + 1, last)

    @triton.jit
    def _greedy_rank(
        logits,
        tok,
        notime,
        first,
        count,
        last,
        scores,
        indices,
        V,
        PARTS: tl.constexpr,
        RULES: tl.constexpr,
        ft,
        lt,
        eos,
        BV: tl.constexpr,
    ):
        r, p = tl.program_id(0), tl.program_id(1)
        v = p * BV + tl.arange(0, BV)
        x = tl.load(logits + r.to(tl.int64) * V + v, v < V, other=float("-inf"))
        if RULES:
            previous = tl.load(tok + r)
            end = tl.load(last + r)
            time = (v >= ft) & (v <= lt)
            open_pair = (v == eos) | (time & (v <= end))
            after_end = ((v < end) | (v > lt)) & (v != eos)
            bad = tl.where((previous >= ft) & (previous <= lt), after_end, True)
            bad = tl.where(tl.load(count + r) % 2 == 1, open_pair, bad)
            bad = tl.where(tl.load(first + r), ~time, bad)
            bad = tl.where(tl.load(notime + r), time, bad)
            x = tl.where(bad, float("-inf"), x)
        # PyTorch argmax prefers the first NaN, then the first maximal value. This is deliberately
        # separate from drafted choose's existing NaN behavior. Masked-out NaNs are already -inf.
        nan = x != x
        nan_index = tl.min(tl.where(nan, v, 2147483647), 0)
        best = tl.max(tl.where(nan, float("-inf"), x), 0)
        index = tl.min(tl.where(x == best, v, 2147483647), 0)
        index = tl.where(nan_index < V, nan_index, index)
        tl.store(scores + r * PARTS + p, tl.where(nan_index < V, float("nan"), best))
        tl.store(indices + r * PARTS + p, index.to(tl.int64))

    @triton.jit
    def _greedy_finish(
        scores,
        indices,
        tok,
        done,
        seq,
        pos,
        first,
        count,
        last,
        PARTS: tl.constexpr,
        BP: tl.constexpr,
        C: tl.constexpr,
        RULES: tl.constexpr,
        ft,
        lt,
        eos,
    ):
        r = tl.program_id(0)
        p = tl.arange(0, BP)
        val = tl.load(scores + r * PARTS + p, p < PARTS, other=float("-inf"))
        idx = tl.load(indices + r * PARTS + p, p < PARTS, other=2147483647)
        nan = val != val
        nan_index = tl.min(tl.where(nan, idx, 2147483647), 0)
        best = tl.max(tl.where(nan, float("-inf"), val), 0)
        chosen = tl.min(tl.where(val == best, idx, 2147483647), 0)
        chosen = tl.where(nan_index < 2147483647, nan_index, chosen)
        was_done = tl.load(done + r)
        nxt = tl.where(was_done, eos, chosen)
        tl.store(done + r, was_done | (nxt == eos))
        tl.store(seq + r.to(tl.int64) * C + tl.load(pos), nxt)
        tl.store(tok + r, nxt)
        if RULES:
            is_time = (nxt >= ft) & (nxt <= lt)
            tl.store(count + r, tl.load(count + r) + is_time.to(tl.int64))
            tl.store(last + r, tl.where(is_time, nxt, tl.load(last + r)))
            tl.store(first + r, False)

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


def choose(logits: torch.Tensor, inp: torch.Tensor, notime: torch.Tensor, pred: torch.Tensor, rules,
           skip_notime: bool = False, workspace: tuple | None = None) -> None:  # fmt: skip
    """``pred`` [R, W] = argmax of ``logits`` [R, W, V] under ``rules`` (None: plain argmax); ``inp``
    [R, W + 4] as in runtime._Verify, ``notime`` [R] bool; ``skip_notime``: only the rows without
    <notimestamps> (``pred`` holds the others' already)."""
    R, W, V = logits.shape
    args = (rules.first_time, rules.last_time, rules.eos) if rules is not None else (0, -1, -1)
    parts = workspace[0].shape[-1] if workspace is not None else 1
    scores, indices = workspace if workspace is not None else (logits, pred)
    _choose[(R, W, parts)](logits, inp, notime, indices, scores, V, W, rules is not None, *args,
                          BV=2048, SKIP_NOTIME=skip_notime, PARTS=parts, num_warps=4)  # fmt: skip
    if parts > 1:
        _choose_reduce[(R, W)](scores, indices, notime, pred, W, parts, triton.next_power_of_2(parts),
                              skip_notime, num_warps=1)  # fmt: skip
    elif workspace is not None:
        raise ValueError("choose workspace requires at least two vocabulary blocks")


def choose_accept(logits: torch.Tensor, inp: torch.Tensor, notime: torch.Tensor,
                  pred: torch.Tensor, rules, res: torch.Tensor, eos: int,
                  workspace: tuple, skip_notime: bool = False) -> None:  # fmt: skip
    """Block selection followed by one combined argmax/accept kernel; preserves pred for diagnostics."""
    R, W, V = logits.shape
    scores, indices = workspace
    parts = scores.shape[-1]
    args = (rules.first_time, rules.last_time, rules.eos) if rules is not None else (0, -1, -1)
    _choose[(R, W, parts)](logits, inp, notime, indices, scores, V, W, rules is not None, *args,
                          BV=2048, SKIP_NOTIME=skip_notime, PARTS=parts, num_warps=4)  # fmt: skip
    _choose_accept[(R,)](scores, indices, notime, pred, inp, res, eos, W, parts,
                        triton.next_power_of_2(max(W, 2)), triton.next_power_of_2(parts),
                        skip_notime, num_warps=4)  # fmt: skip


def choose_workspace(rows: int, window: int, vocab: int, device) -> tuple | None:
    """Private per-session [R,W,ceil(V/2048)] partial maxima and indices; graph-stable addresses."""
    parts = (vocab + 2047) // 2048
    if parts < 2:
        return None
    shape = (rows, window, parts)
    return (torch.empty(shape, device=device), torch.empty(shape, device=device, dtype=torch.long))


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


def greedy_select(logits, tok, done, seq, pos, notime, first, count, last, rules, eos, workspace):
    """Plain greedy mask/argmax and state advance; logits[R,1,V] -> tok[R,1]/seq[R,C]."""
    rows, _, vocab = logits.shape
    scores, indices = workspace
    parts = scores.shape[-1]
    args = (rules.first_time, rules.last_time, rules.eos) if rules is not None else (0, -1, -1)
    _greedy_rank[(rows, parts)](
        logits,
        tok,
        notime,
        first,
        count,
        last,
        scores,
        indices,
        vocab,
        parts,
        rules is not None,
        *args,
        BV=2048,
        num_warps=4,
    )
    _greedy_finish[(rows,)](
        scores,
        indices,
        tok,
        done,
        seq,
        pos,
        first,
        count,
        last,
        parts,
        triton.next_power_of_2(parts),
        seq.shape[1],
        rules is not None,
        args[0],
        args[1],
        eos,
        num_warps=1,
    )
