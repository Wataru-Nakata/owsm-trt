"""Dump each utterance's CTC draft and greedy output (token ids), for tuning drafted decoding offline
with scripts/draft_sim.py.

python scripts/dump_drafts.py --model .cache/models/owsm_v4_medium_1B \
    --data .cache/testdata/librispeech_test_clean/manifest.json --out outputs/drafts.json

With ``--long`` the data are long recordings (scripts/make_longform.py), decoded with
``transcribe_long``; every 30 s window is dumped (with timestamps in its output) along with its
offset, its draft (with the guessed timestamps, see OWSM._long_window), the CTC best path (frame-wise)
and the decoder calls it took.
"""

import argparse
import json

import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio
from owsm_trt.runtime import GreedySession


class LongRecorder:
    """Records every window ``transcribe_long`` decodes (while installed)."""

    def __init__(self, model: OWSM):
        self.windows: list[dict] = []
        self.recording = ""
        self.model, self._window, self._drafted = model, model._long_window, GreedySession.decode_drafted
        self._encode, self.paths = GreedySession.encode, None
        self.insertions, self.punctuation = model.tokenizer.not_drafted, model.tokenizer.punctuation
        rec = self

        def window(chunk, primer, beam=1, draft=False):
            rec.current = {"id": rec.recording, "offset": chunk.storage_offset(), "samples": len(chunk)}
            return rec._window(chunk, primer, beam, draft)

        def encode(sess, feats):
            rec.paths = rec._encode(sess, feats)  # the CTC best paths of the window
            return rec.paths

        def drafted(sess, primer, drafts, max_new_tokens, insertions=frozenset(), give_up=8):
            ctc = rec.paths[0].tolist()
            outs, calls = rec._drafted(sess, primer, drafts, max_new_tokens, insertions, give_up)
            rec.insertions, rec.punctuation = (
                insertions,
                sess.punctuation,
            )  # the rules the drafts were used with
            rec.windows.append({
                **rec.current, "primer": [int(x) for x in primer[0]], "draft": list(drafts[0]), "ctc": ctc,
                "out": list(outs[0]), "calls": calls, "max_new": max_new_tokens, "give_up": give_up,
            })  # fmt: skip
            return outs, calls

        model._long_window = window
        GreedySession.decode_drafted = drafted
        GreedySession.encode = encode

    def remove(self) -> None:
        del self.model._long_window
        GreedySession.decode_drafted = self._drafted
        GreedySession.encode = self._encode

    def meta(self) -> dict:
        t = self.model.tokenizer
        return {
            "primer": self.windows[0]["primer"] if self.windows else [], "eos": t.eos,
            "insertions": sorted(self.insertions), "punctuation": sorted(self.punctuation),
            "time": [t.first_time, t.last_time], "blank": t.blank, "utterances": self.windows,
        }  # fmt: skip


def dump_long(model: OWSM, items: list[dict], waves: list[torch.Tensor], out: str) -> list:
    """transcribe_long over ``waves`` (sequential), every window dumped to ``out``; returns the results."""
    rec = LongRecorder(model)
    results = []
    try:
        for it, w in zip(items, waves):
            rec.recording = it["id"]
            results.append(model.transcribe_long(w))
    finally:
        rec.remove()
    json.dump(rec.meta(), open(out, "w"))
    calls = sum(w["calls"] for w in rec.windows)
    print(f"{len(rec.windows)} windows of {len(items)} recordings ({calls} decoder calls) -> {out}")
    return results


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--long", action="store_true", help="long recordings, decoded with transcribe_long")
    args = ap.parse_args()
    items = json.load(open(args.data))
    model = OWSM.from_pretrained(args.model, backend=args.backend, device=args.device)
    if args.long:
        dump_long(model, items, [load_audio(it["path"]) for it in items], args.out)
        return
    t = model.tokenizer
    primer = model._primer("eng", "asr", False, None)
    rows = []
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        sess = model._session(len(chunk))
        paths = sess.encode(model._features([load_audio(it["path"]) for it in chunk]))
        drafts = model._ctc_drafts(paths)
        outs, _ = sess.decode_drafted(
            torch.tensor([primer] * len(chunk), device=model.device), drafts, model._max_new(), t.not_drafted
        )
        for it, d, o in zip(chunk, drafts, outs):
            rows.append({"id": it["id"], "draft": d, "out": o})
    meta = {"primer": primer, "eos": t.eos, "insertions": sorted(t.not_drafted), "utterances": rows}
    meta["punctuation"] = sorted(t.punctuation)
    json.dump(meta, open(args.out, "w"))
    print(f"{len(rows)} utterances -> {args.out}")


if __name__ == "__main__":
    main()
