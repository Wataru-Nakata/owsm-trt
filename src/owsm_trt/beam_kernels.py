"""Beam bookkeeping kernels; log-softmax and global top-K remain PyTorch operations."""

import torch

from .kernels import triton

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _candidates(logits, logp, scores, active, done, notime, first, count, last, prev,
                    candidates, bad_blocks, V: tl.constexpr, K: tl.constexpr, BLOCKS: tl.constexpr,
                    RULES: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr, EOS: tl.constexpr,
                    BV: tl.constexpr):
        row, block = tl.program_id(0), tl.program_id(1)
        ids = block * BV + tl.arange(0, BV)
        live = tl.load(active + row) & ~tl.load(done + row // K)
        raw = tl.load(logits + row * V + ids, ids < V, other=0)
        bad = (raw != raw) | (raw == float("inf"))
        tl.store(bad_blocks + row * BLOCKS + block, live & (tl.sum(bad.to(tl.int32), 0) > 0))
        lp = tl.load(logp + row * V + ids, ids < V, other=float("-inf"))
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


def candidates(session, logp: torch.Tensor) -> torch.Tensor:
    """[B*K, V] normalized log-probabilities -> masked cumulative scores, without a vocabulary mask."""
    s = session
    v = logp.shape[-1]
    rules = s.rules
    first, last, eos = (rules.first_time, rules.last_time, rules.eos) if rules else (0, -1, s.eos)
    _candidates[(s.rows, s._bad_blocks.shape[1])](
        s.logits, logp, s.score, s.active, s.done, s.t_notime, s.t_first, s.t_count, s.t_last, s.tok,
        s._candidates, s._bad_blocks, v, s.beam, s._bad_blocks.shape[1], rules is not None,
        first, last, eos, BV=1024, num_warps=4,
    )
    s.numerical_error |= s._bad_blocks.view(s.batch, -1).any(dim=-1)
    return s._candidates


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
