"""Where drafted long-form decoding spends its decoder calls, and what other drafts would save.

python scripts/long_drafts.py outputs/gpu_eval-<id>/drafts_long.json

Replays every window of a ``dump_drafts.py --long`` dump on the CPU (scripts/draft_sim.py, exact),
classifies each call by the token that ended it, and counts the calls of draft variants.
"""

import argparse
import json
from collections import Counter

import draft_sim

from owsm_trt import runtime
from owsm_trt.pipeline import ctc_collapse


class Recorder(draft_sim.Oracle):
    """The oracle, also noting why each call stopped accepting drafted tokens."""

    log: list[str] = []
    primer_len = 3

    def __call__(self, inp):
        preds = super().__call__(inp)
        for r, (row, pred) in enumerate(zip(inp, preds)):
            if r not in self.held and r != 0:
                continue
            seq, o, w = self.seqs[self.held.get(r, 0)], row[self.w], self.w
            s = self.primer_len - 1 if o == 0 else 0
            n = 0
            while s + n + 1 < w and pred[s + n] == row[s + n + 1] and row[s + n + 1] != self.eos:
                n += 1
            at = o + s + n + 1  # position of the emitted token in seq
            tok = seq[at] if at < len(seq) else None
            drafted = s + n + 1 < w and row[s + n + 1] != self.eos
            self.log.append(self.kind(seq, at, tok, drafted))
        return preds

    def kind(self, seq, at, tok, drafted) -> str:
        if tok == self.eos:
            return "eos"
        if self.first <= tok <= self.last:
            times = sum(self.first <= t <= self.last for t in seq[self.primer_len : at])
            if times == 0:
                return "ts-first"
            return ("ts-end" if times % 2 else "ts-start") + ("" if drafted else "/undrafted")
        return "word" if drafted else "undrafted"


def stats(meta: dict) -> None:
    first, last = meta["time"]
    outs = [u["out"] for u in meta["utterances"]]
    firsts = Counter(o[0] - first for o in outs if o)
    print("first timestamp of a window (steps):", firsts.most_common(6))
    gaps = Counter()
    for o in outs:
        times = [t for t in o if first <= t <= last]
        gaps.update(b - a for a, b in zip(times[1::2], times[2::2]))  # start - previous end
    total = sum(gaps.values())
    print(
        f"start - previous end (steps), {total} pairs:",
        [(g, round(c / total, 3)) for g, c in gaps.most_common(8)],
    )


def no_start_guess(advance):
    """``_Row.advance`` without its guess of a start time after an end time."""

    def advance_without_guess(self, *args):
        n = len(self.draft)
        advance(self, *args)
        if len(self.draft) > n:
            del self.draft[self.ptr]

    return advance_without_guess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    args = ap.parse_args()
    meta = json.load(open(args.dump))
    utts = meta["utterances"]
    first, last = meta["time"]
    Recorder.first, Recorder.last, Recorder.eos = first, last, meta["eos"]
    Recorder.primer_len = len(meta["primer"])
    oracle, draft_sim.Oracle = draft_sim.Oracle, Recorder
    dumped = draft_sim.replay_long(meta)
    draft_sim.Oracle = oracle
    gpu = [u["calls"] for u in utts]
    tokens = sum(len(u["out"]) + 1 for u in utts)
    print(f"{len(utts)} windows, {tokens} tokens: {sum(dumped)} calls replayed, {sum(gpu)} on the GPU "
          f"({sum(a == b for a, b in zip(dumped, gpu))} windows the same), {tokens / sum(dumped):.2f} tokens/call")  # fmt: skip
    print("calls by the token that ended them:", Counter(Recorder.log).most_common())
    stats(meta)

    # drafts from the CTC best paths, with and without guessed timestamps (replayed exactly: each replay
    # checks that it still produces the dumped outputs)
    punct = frozenset(meta["punctuation"])
    times = frozenset(range(first, last + 1))
    notime = set(meta["insertions"]) - times
    ctc = [ctc_collapse(u["ctc"], meta["blank"]) for u in utts]
    plain = {"insertions": notime | times, "punctuation": punct | times, "give_up": 8}  # never drafted
    guess = {"insertions": notime, "punctuation": punct | times, "give_up": 16}  # drafted / guessed
    base = draft_sim.replay_long(meta, ctc, **plain)
    print(f"  CTC drafts: {sum(base)} calls")
    variants = {
        "<0.00> first": [[first] + d for d in ctc],
        "<0.00> + draft + <30.00>": [[first] + d + [last] for d in ctc],
    }
    advance = runtime._Row.advance
    for start_guess in (False, True):
        runtime._Row.advance = advance if start_guess else no_start_guess(advance)
        for name, ds in variants.items():
            calls = draft_sim.replay_long(meta, ds, **guess)
            name += ", start = previous end" if start_guess else ""
            print(f"  {name}: {sum(calls)} calls ({100 * (sum(calls) / sum(base) - 1):+.1f}%)")
    runtime._Row.advance = advance


if __name__ == "__main__":
    main()
