"""Decoder calls of CTC-drafted decoding, replayed on the CPU with an oracle in place of the decoder.

Given each utterance's draft and greedy output (scripts/dump_drafts.py), the replay makes exactly the
calls the GPU run makes: the decoder's prediction is only ever used after a correct prefix, and there it
is the next token of the greedy output.

python scripts/draft_sim.py outputs/drafts.json

``--time`` also measures the host's time per call (the drafting loop, filling the call's input and
reading its result; what the GPU waits for), with the oracle's answers recorded beforehand.
"""

import argparse
import json
import time
from collections import Counter

import torch

from owsm_trt.config import OWSMConfig
from owsm_trt.runtime import GreedySession, TorchBackend, _accepted, _Row

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

    def __init__(self, w: int, seqs: list[list[int]], held: dict[int, int], eos: int):
        self.w, self.seqs, self.held, self.eos = w, seqs, held, eos

    def __call__(self, flat: list[int], meanwhile=None) -> list[list[int]]:
        if meanwhile is not None:
            meanwhile()
        out, w = [], self.w
        for r in range(len(flat) // (w + 4)):
            row = flat[r * (w + 4) : (r + 1) * (w + 4)]
            seq, o, pred, ok = self.seqs[self.held.get(r, 0)], row[w], [], True
            for i in range(w):
                ok = ok and o + i < len(seq) and row[i] == seq[o + i]
                pred.append(seq[o + i + 1] if ok and o + i + 1 < len(seq) else -2)
            out.append(list(_accepted(pred, row[:w], max(row[w + 3], 0), self.eos)))  # as the GPU does
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
    sess._verify = {key: Oracle(key[1], seqs, held, eos) for key in sess._verify}
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


def host_time(
    utts: list[dict],
    primer: list[int],
    eos: int,
    insertions: set[int],
    rows: int,
    punctuation: frozenset[int],
) -> tuple[int, float]:
    """(calls, host seconds per call) of the drafting loop on the session's own call objects, whose GPU
    part (the graph replay) is replaced by copying in the oracle's recorded answer."""
    windows = tuple(w for w in (1, 2, 4, 8, 16, 32) if w * rows <= max(256, rows))
    sess = GreedySession(
        _NoDecoder(), TINY, rows, 512, eos, torch.float32, torch.device("cpu"), use_cuda_graph=False,
        buckets=[64, 128, 256, 512], draft_windows=windows, punctuation=punctuation,
    )  # fmt: skip
    seqs = [primer + u["out"] + [eos] for u in utts]
    held: dict[int, int] = {}
    answers: list[torch.Tensor] = []
    step = [0]
    for (_, w), bound in sess._verify.items():
        oracle = Oracle(w, seqs, held, eos)

        def body(b=bound.bufs, oracle=oracle) -> None:
            if step[0] == len(answers):  # first pass: ask the oracle
                answers.append(torch.tensor(oracle(b.host.view(-1).tolist())))  # the call's input
            b.res_host.copy_(answers[step[0]])  # (what the call's graph ends with)
            step[0] += 1

        bound.body = body
    best = float("inf")
    for _ in range(4):
        queue, step[0] = list(range(len(utts))), 0

        def refill(r: int) -> _Row | None:
            if not queue:
                return None
            held[r] = queue.pop(0)
            return _Row(held[r], primer, utts[held[r]]["draft"])

        t0 = time.perf_counter()
        out, calls = sess._drafted([None] * rows, refill, None, 512 - len(primer), insertions, 8)
        if answers and len(answers) == calls and step[0] == calls:
            best = min(best, time.perf_counter() - t0)
        assert all(out[i] == u["out"] for i, u in enumerate(utts))
    return calls, best / calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("drafts")
    ap.add_argument("--rows", default="1,8,16")
    ap.add_argument(
        "--time", action="store_true", help="host time per call (rows refilled as in decode_stream)"
    )
    args = ap.parse_args()
    meta = json.load(open(args.drafts))
    utts, primer, eos, ins = meta["utterances"], meta["primer"], meta["eos"], set(meta["insertions"])
    punct = frozenset(meta.get("punctuation", []))
    for rows in map(int, args.rows.split(",")):
        if args.time:
            calls, sec = host_time(utts, primer, eos, ins, rows, punct)
            print(f"rows {rows}: {calls} calls, host {1e6 * sec:.1f} us per call")
        elif rows == 1:
            per = [simulate([u], primer, eos, ins, 1, punctuation=punct) for u in utts]
            print(f"rows 1: {sum(per)} calls, {sum(per) / len(utts):.2f} per utterance")
            print("  calls per utterance:", dict(sorted(Counter(per).items())))
        else:
            print(
                f"rows {rows}: {simulate(utts, primer, eos, ins, rows, punctuation=punct)} calls (refilled rows)"
            )


if __name__ == "__main__":
    main()
