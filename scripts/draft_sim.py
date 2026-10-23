"""Decoder calls of CTC-drafted decoding, replayed on the CPU with an oracle in place of the decoder.

Given each utterance's draft and greedy output (scripts/dump_drafts.py), the replay makes exactly the
calls the GPU run makes: the decoder's prediction is only ever used after a correct prefix, and there it
is the next token of the greedy output.

python scripts/draft_sim.py outputs/drafts.json
"""

import argparse
import json
from collections import Counter

import torch

from owsm_trt.config import OWSMConfig
from owsm_trt.runtime import GreedySession, TorchBackend, _Row

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
) -> int:
    """Decoder calls for ``utts`` decoded ``rows`` at a time, rows refilled as in decode_stream."""
    windows = tuple(w for w in (1, 2, 4, 8, 16, 32) if w * rows <= max(256, rows))
    sess = GreedySession(
        _NoDecoder(), TINY, rows, 512, eos, torch.float32, torch.device("cpu"), use_cuda_graph=False,
        buckets=[64, 128, 256, 512], draft_windows=windows, punctuation=punctuation,
    )  # fmt: skip
    seqs = [primer + u["out"] + [eos] for u in utts]
    held: dict[int, int] = {}
    sess._verify = {key: Oracle(key[1], seqs, held) for key in sess._verify}
    queue = list(range(len(utts)))

    def refill(r: int) -> _Row | None:
        if not queue:
            return None
        held[r] = queue.pop(0)
        return _Row(held[r], primer, utts[held[r]]["draft"])

    out, calls = sess._drafted([None] * rows, refill, None, 512 - len(primer), insertions, give_up)
    bad = sum(out[i] != u["out"] for i, u in enumerate(utts))
    assert not bad, f"the replay disagrees with the GPU output on {bad} utterances"
    return calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("drafts")
    ap.add_argument("--rows", default="1,8,16")
    args = ap.parse_args()
    meta = json.load(open(args.drafts))
    utts, primer, eos, ins = meta["utterances"], meta["primer"], meta["eos"], set(meta["insertions"])
    punct = frozenset(meta.get("punctuation", []))
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
