"""Decoder calls of CTC-drafted decoding, replayed on the CPU with an oracle in place of the decoder.

Given each utterance's draft and greedy output (scripts/dump_drafts.py), the replay makes exactly the
calls the GPU run makes: the decoder's prediction is only ever used after a correct prefix, and there it
is the next token of the greedy output (the timestamp rules included).

python scripts/draft_sim.py outputs/drafts.json

A dump of long-form windows (``dump_drafts.py --long``) has a primer, a token limit and the GPU's call
count per window; the replay checks that it makes the same number of calls.
"""

import argparse
import json
from collections import Counter

import torch

from owsm_trt.config import OWSMConfig
from owsm_trt.runtime import GreedySession, TimestampRules, TorchBackend, _Row

# Buffers only: the oracle replaces every decoder call.
TINY = OWSMConfig(
    vocab_size=8, d_model=8, n_heads=1, enc_layers=1, enc_ffn=8, cgmlp_units=8, cgmlp_kernel=3, merge_kernel=3,
    subsample="conv2d8", dec_layers=1, dec_ffn=8, dec_heads=1, n_mels=16, n_fft=64, win_length=48,
    hop_length=32, sample_rate=1600, speech_seconds=2.0, time_resolution=0.02,
)  # fmt: skip


class _NoDecoder(TorchBackend):
    def __init__(self):
        self.cfg = TINY

    def step_fn(self, *args, **kw):
        return lambda: None


class Oracle:
    """One (cache bucket, window) decoder call; row r holds utterance ``held[r]``."""

    def __init__(self, w: int, seqs: list[list[int]], held: dict[int, int]):
        self.w, self.seqs, self.held = w, seqs, held

    def __call__(self, inp: list[list[int]]) -> list[list[int]]:
        out = []
        for r, row in enumerate(inp):
            seq, o, pred, ok = self.seqs[self.held.get(r, 0)], row[self.w], [], True
            for i in range(self.w):
                ok = ok and o + i < len(seq) and row[i] == seq[o + i]
                pred.append(seq[o + i + 1] if ok and o + i + 1 < len(seq) else -2)
            out.append(pred)
        return out


def simulate(
    utts: list[dict],
    primer: list[int],
    eos: int,
    insertions: set[int],
    rows: int,
    give_up: int = 8,
    punctuation: frozenset[int] = frozenset(),
    max_new: int | None = None,
    time: tuple[int, int] | None = None,
) -> int:
    """Decoder calls for ``utts`` decoded ``rows`` at a time, rows refilled as in decode_stream.

    An utterance's own ``primer`` (long-form dumps) replaces the common one. ``time``: the first and
    last timestamp ids, for draft rules that look at timestamps (the oracle itself needs no rules)."""
    windows = tuple(w for w in (1, 2, 4, 8, 16, 32) if w * rows <= max(256, rows))
    rules = TimestampRules(-1, *time, eos, 1, torch.device("cpu")) if time else None
    sess = GreedySession(
        _NoDecoder(), TINY, rows, 512, eos, torch.float32, torch.device("cpu"), use_cuda_graph=False,
        buckets=[64, 128, 256, 512], draft_windows=windows, punctuation=punctuation, rules=rules,
    )  # fmt: skip
    primers = [u.get("primer", primer) for u in utts]
    seqs = [p + u["out"] + [eos] for p, u in zip(primers, utts)]
    held: dict[int, int] = {}
    sess._verify = {key: Oracle(key[1], seqs, held) for key in sess._verify}
    queue = list(range(len(utts)))

    def refill(r: int) -> _Row | None:
        if not queue:
            return None
        held[r] = queue.pop(0)
        return _Row(held[r], primers[held[r]], utts[held[r]]["draft"])

    limit = min(max_new or 512, 512 - max(map(len, primers)))
    out, calls = sess._drafted([None] * rows, refill, None, limit, insertions, give_up)
    bad = sum(out[i] != u["out"] for i, u in enumerate(utts))
    assert not bad, f"the replay disagrees with the GPU output on {bad} utterances"
    return calls


def replay_long(meta: dict, drafts: list[list[int]] | None = None, **kw) -> list[int]:
    """Calls per long-form window, each decoded on its own (batch 1, as transcribe_long), with its own
    draft or the given ones."""
    ins, punct = set(meta["insertions"]), frozenset(meta.get("punctuation", []))
    kw = {"insertions": ins, "punctuation": punct, "time": tuple(meta["time"]), **kw}
    calls = []
    for i, u in enumerate(meta["utterances"]):
        if drafts is not None:
            u = {**u, "draft": drafts[i]}
        args = {"give_up": u.get("give_up", 8), **kw}
        calls.append(simulate([u], meta["primer"], meta["eos"], rows=1, max_new=u.get("max_new"), **args))
    return calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("drafts")
    ap.add_argument("--rows", default="1,8,16")
    args = ap.parse_args()
    meta = json.load(open(args.drafts))
    utts, primer, eos, ins = meta["utterances"], meta["primer"], meta["eos"], set(meta["insertions"])
    punct = frozenset(meta.get("punctuation", []))
    if "calls" in utts[0]:  # long-form windows: the replay must make the GPU's calls
        calls = replay_long(meta)
        gpu = [u["calls"] for u in utts]
        same = sum(a == b for a, b in zip(calls, gpu))
        print(f"long-form: {len(utts)} windows, {sum(calls)} calls in the replay, {sum(gpu)} on the GPU "
              f"({same} windows the same)")  # fmt: skip
        return
    for rows in map(int, args.rows.split(",")):
        if rows == 1:
            per = [simulate([u], primer, eos, ins, 1, punctuation=punct) for u in utts]
            print(f"rows 1: {sum(per)} calls, {sum(per) / len(utts):.2f} per utterance")
            print("  calls per utterance:", dict(sorted(Counter(per).items())))
        else:
            print(
                f"rows {rows}: {simulate(utts, primer, eos, ins, rows, punctuation=punct)} calls (refilled rows)"
            )


if __name__ == "__main__":
    main()
