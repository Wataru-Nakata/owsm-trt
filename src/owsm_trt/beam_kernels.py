"""Beam bookkeeping kernels; log-softmax and global top-K remain PyTorch operations."""

import torch

from .kernels import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _reorder_state(top, indices, parent_out, token_out, valid_out, active, done, seq, score,
                       notime, first, count, last, pos_ptr, K: tl.constexpr, V: tl.constexpr,
                       C: tl.constexpr, RULES: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr,
                       BK: tl.constexpr, BC: tl.constexpr):
        b = tl.program_id(0)
        slot, column = tl.arange(0, BK), tl.arange(0, BC)
        row = b * K + slot
        index = tl.load(indices + row, slot < K, other=0)
        parent = b * K + index // V
        token = index % V
        value = tl.load(top + row, slot < K, other=float("-inf"))
        valid = tl.load(active + parent, slot < K, other=False) & ~tl.load(done + b)
        valid &= (value == value) & (tl.abs(value) != float("inf"))
        mask = (slot < K)[:, None] & (column < C + 2)[None, :]
        history = tl.load(seq + parent[:, None] * (C + 2) + column[None, :], mask, other=0)
        history = tl.where(column[None, :] == tl.load(pos_ptr), token[:, None], history)
        if RULES:
            old_notime = tl.load(notime + parent, slot < K, other=False)
            old_count = tl.load(count + parent, slot < K, other=0)
            old_last = tl.load(last + parent, slot < K, other=-1)
            is_time = (token >= FIRST) & (token <= LAST)
        # Each CTA owns an utterance and all its parents. Read every source before any in-place
        # write: duplicated parents and permutation cycles must not see another warp's new state.
        tl.debug_barrier()
        tl.store(seq + row[:, None] * (C + 2) + column[None, :], history, mask)
        tl.store(score + row, value, slot < K)
        tl.store(parent_out + row, parent, slot < K)
        tl.store(token_out + row, token, slot < K)
        tl.store(valid_out + row, valid, slot < K)
        if RULES:
            tl.store(notime + row, old_notime, slot < K)
            tl.store(first + row, False, slot < K)
            tl.store(count + row, old_count + is_time.to(tl.int64), slot < K)
            tl.store(last + row, tl.where(is_time, token, old_last), slot < K)

    @triton.jit
    def _reorder_kv(k, v, scratch, parent, pos_ptr, L: tl.constexpr, R: tl.constexpr,
                    H: tl.constexpr, C: tl.constexpr, D: tl.constexpr, PREFIX: tl.constexpr,
                    COPY_BACK: tl.constexpr, BLOCK: tl.constexpr, SKIP_IDENTITY: tl.constexpr):
        # One contiguous head tile. All parent reads finish before the separate copy-back launch.
        head_row, tile, kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i = tile * BLOCK + tl.arange(0, BLOCK)
        inside = i < C * D
        written = inside
        if PREFIX:
            # pos counts KV entries already written; the token selected at pos has no KV yet.
            written &= i < tl.load(pos_ptr) * D
        offset = head_row * C * D + i
        temporary = scratch + kv * L * R * H * C * D + offset
        if SKIP_IDENTITY:
            row = (head_row // H) % R
            if tl.load(parent + row) == row:
                # This row stays put and other rows only read it in the first launch. The
                # second launch must still clear its unwritten tail, including stale NaNs.
                if COPY_BACK and PREFIX:
                    if kv == 0:
                        tl.store(k + offset, 0, inside & ~written)
                    else:
                        tl.store(v + offset, 0, inside & ~written)
                return

        if COPY_BACK:
            value = tl.load(temporary, written, other=0)
            # Keep every unwritten V slot finite: some attention paths multiply masked zeros by V.
            if kv == 0:
                tl.store(k + offset, value, inside)
            else:
                tl.store(v + offset, value, inside)
        else:
            row = (head_row // H) % R
            layer = head_row // (R * H)
            head = head_row % H
            src_row = tl.load(parent + row)
            src = ((layer * R + src_row) * H + head) * C * D + i
            if kv == 0:
                value = tl.load(k + src, written, other=0)
            else:
                value = tl.load(v + src, written, other=0)
            tl.store(temporary, value, written)

    @triton.jit
    def _candidates(logits, logp, scores, active, done, notime, first, count, last, prev,
                    candidates, bad_blocks, V: tl.constexpr, K: tl.constexpr, BLOCKS: tl.constexpr,
                    RULES: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr, EOS: tl.constexpr,
                    BV: tl.constexpr, SHARED_PREFILL: tl.constexpr, LOGITS_STRIDE: tl.constexpr):
        row, block = tl.program_id(0), tl.program_id(1)
        ids = block * BV + tl.arange(0, BV)
        live = tl.load(active + row) & ~tl.load(done + row // K)
        source = row // K if SHARED_PREFILL else row
        raw = tl.load(logits + source * LOGITS_STRIDE + ids, ids < V, other=0)
        bad = (raw != raw) | (raw == float("inf"))
        tl.store(bad_blocks + row * BLOCKS + block, live & (tl.sum(bad.to(tl.int32), 0) > 0))
        lp = tl.load(logp + source * V + ids, ids < V, other=float("-inf"))
        allowed = live & (lp == lp) & (tl.abs(lp) != float("inf"))
        if RULES:
            t = (ids >= FIRST) & (ids <= LAST)
            last_t = tl.load(last + row)
            previous = tl.load(prev + row)
            open_pair = (ids == EOS) | (t & (ids <= last_t))
            after_end = ((ids < last_t) | (ids > LAST)) & (ids != EOS)
            forbidden = tl.where((previous >= FIRST) & (previous <= LAST), after_end, True)
            forbidden = tl.where(tl.load(count + row) % 2 == 1, open_pair, forbidden)
            forbidden = tl.where(tl.load(first + row), ~t, forbidden)
            forbidden = tl.where(tl.load(notime + row), t, forbidden)
            allowed &= ~forbidden
        value = tl.load(scores + row) + lp
        tl.store(candidates + row * V + ids, tl.where(allowed, value, float("-inf")), ids < V)

    @triton.jit
    def _finish(token, valid, score, done, active, seq, best_score, best_seq, best_end,
                ended_by_len, pos_ptr, step_ptr, last_step_ptr, upper_bound, tok, error,
                history_state, history_scores, B: tl.constexpr, K: tl.constexpr,
                C: tl.constexpr, EOS: tl.constexpr, M: tl.constexpr, THRESHOLD: tl.constexpr,
                HISTORY: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
        b = tl.program_id(0)
        k = tl.arange(0, BK)
        row = b * K + k
        token_k = tl.load(token + row, k < K, other=EOS)
        valid_k = tl.load(valid + row, k < K, other=False)
        score_k = tl.load(score + row, k < K, other=float("-inf"))
        old_done = tl.load(done + b)
        pos = tl.load(pos_ptr)
        step = tl.load(step_ptr)
        at_limit = step == tl.load(last_step_ptr)
        live = valid_k & ~old_done & (k < K)
        is_eos = token_k == EOS
        ended_score = tl.where((is_eos | at_limit) & live, score_k, float("-inf"))
        best_val = tl.max(ended_score, 0)
        # torch.max returns the first equal maximum; preserve that choice, including all -inf.
        best_k = tl.min(tl.where((ended_score == best_val) & (k < K), k, BK), 0)
        old_best = tl.load(best_score + b)
        better = best_val > old_best
        new_best = tl.where(better, best_val, old_best)
        new_end = tl.where(better, pos + 1, tl.load(best_end + b))
        tl.store(best_score + b, new_best)
        tl.store(best_end + b, new_end)
        if better:
            i = tl.arange(0, BC)
            best_tokens = tl.load(seq + (b * K + best_k) * (C + 2) + i, i < C + 2, other=0)
            tl.store(best_seq + b * (C + 2) + i, best_tokens, i < C + 2)
        length = pos + 1 + at_limit.to(tl.int64)
        old_length_best = tl.load(ended_by_len + b * (C + 3) + length)
        length_best = tl.maximum(old_length_best, best_val)
        tl.store(ended_by_len + b * (C + 3) + length, length_best)
        count_ended = tl.full((), 0, tl.int32)
        for m in tl.static_range(M):
            n = step - m
            value = tl.load(ended_by_len + b * (C + 3) + tl.maximum(n, 0))
            # Avoid depending on store/load forwarding for the just-updated history column.
            value = tl.where(n == length, length_best, value)
            count_ended += ((n >= 0) & (value > float("-inf")) &
                            (value - new_best < THRESHOLD)).to(tl.int32)
        live_after = live & ~is_eos & ~at_limit
        nlive = tl.sum(live_after.to(tl.int32), 0)
        max_live = tl.max(tl.where(live_after, score_k, float("-inf")), 0)
        finite_best = (new_best == new_best) & (tl.abs(new_best) != float("inf"))
        bound = finite_best & (new_best > max_live) & (nlive > 0)
        end_detect = count_ended == M
        bound_stop = bound & tl.load(upper_bound)
        reason = at_limit.to(tl.int64) | ((nlive == 0).to(tl.int64) * 2)
        reason |= end_detect.to(tl.int64) * 4 | bound_stop.to(tl.int64) * 8
        new_done = old_done | (reason != 0)
        tl.store(done + b, new_done)
        tl.store(active + row, live_after & ~new_done, k < K)
        tl.store(tok + row, token_k, k < K)
        if HISTORY:
            h = (step * B + b) * 9
            tl.store(history_state + h, pos)
            tl.store(history_state + h + 1, nlive)
            tl.store(history_state + h + 2, tl.sum((is_eos & live).to(tl.int32), 0) > 0)
            tl.store(history_state + h + 3, better)
            tl.store(history_state + h + 4, bound)
            tl.store(history_state + h + 5, new_done)
            tl.store(history_state + h + 6, reason)
            tl.store(history_state + h + 7, tl.load(error + b))
            tl.store(history_state + h + 8, new_end)
            tl.store(history_scores + (step * B + b) * 2, new_best)
            tl.store(history_scores + (step * B + b) * 2 + 1, max_live)


def candidates(session, logp: torch.Tensor, prefill_logits: torch.Tensor | None = None) -> torch.Tensor:
    """[B*K, V] log probabilities -> candidate scores; optional prefill reads shared [B, V] rows."""
    s = session
    v = logp.shape[-1]
    rules = s.rules
    first, last, eos = (rules.first_time, rules.last_time, rules.eos) if rules else (0, -1, s.eos)
    raw = s.logits if prefill_logits is None else prefill_logits
    _candidates[(s.rows, s._bad_blocks.shape[1])](
        raw, logp, s.score, s.active, s.done, s.t_notime, s.t_first, s.t_count, s.t_last, s.tok,
        s._candidates, s._bad_blocks, v, s.beam, s._bad_blocks.shape[1], rules is not None,
        first, last, eos, BV=1024, SHARED_PREFILL=prefill_logits is not None, LOGITS_STRIDE=raw.stride(0), num_warps=4,
    )
    s.numerical_error |= s._bad_blocks.view(s.batch, -1).any(dim=-1)
    return s._candidates


def reorder_kv(k: torch.Tensor, v: torch.Tensor, scratch: torch.Tensor, parent: torch.Tensor,
               pos: torch.Tensor, prefix: bool, skip_identity: bool = False) -> None:
    """[L, R, H, C, D] -> parent-gathered caches, using one fixed workspace for all buckets.

    Full mode preserves every slot exactly. Prefix mode preserves slots < pos and zeros the rest.
    Both launches use fixed addresses and no host read of pos; duplicate parents are safe.
    Optional identity skipping avoids copying an unchanged parent, but still clears unwritten slots.
    """
    layers, rows, heads, c, d = k.shape
    grid = (layers * rows * heads, triton.cdiv(c * d, 2048), 2)
    for copy_back in (False, True):
        _reorder_kv[grid](k, v, scratch, parent, pos, layers, rows, heads, c, d,
                         PREFIX=prefix, COPY_BACK=copy_back, BLOCK=2048, SKIP_IDENTITY=skip_identity, num_warps=4)


def reorder_state(session, top: torch.Tensor, indices: torch.Tensor):
    """Unchanged global top-K indices -> parent/token/valid and gathered token/timestamp state."""
    s = session
    rules = s.rules
    first, last = (rules.first_time, rules.last_time) if rules else (0, -1)
    parent, token, valid = s._selection_state
    _reorder_state[(s.batch,)](
        top, indices, parent, token, valid, s.active, s.done, s.seq, s.score,
        s.t_notime, s.t_first, s.t_count, s.t_last, s.pos,
        s.beam, s.logits.shape[-1], s.capacity, rules is not None, first, last,
        BK=triton.next_power_of_2(s.beam), BC=triton.next_power_of_2(s.capacity + 2), num_warps=4,
    )
    return parent, token, valid


def finish(session, token: torch.Tensor, valid: torch.Tensor, end_detect_m: int, end_detect_d: float) -> None:
    """Update one utterance per CTA; each CTA owns all its beam slots and finished history."""
    s = session
    history = s._beam_history
    _finish[(s.batch,)](
        token, valid, s.score, s.done, s.active, s.seq, s.best_score, s.best_seq, s.best_end,
        s.ended_by_len, s.pos, s.step, s.last_step, s.upper_bound, s.tok, s.numerical_error,
        history['state'] if history else s.seq, history['scores'] if history else s.score,
        s.batch, s.beam, s.capacity, s.eos, end_detect_m, end_detect_d, history is not None,
        BK=triton.next_power_of_2(s.beam), BC=triton.next_power_of_2(s.capacity + 2), num_warps=4,
    )
