"""Long-form WER / speed: ``OWSM.transcribe_long`` over recordings of minutes (scripts/make_longform.py).

python scripts/eval_long.py --model .cache/models/owsm_v4_medium_1B \
    --data .cache/longform/test_clean/manifest.json --save outputs/long_seq.json

``--batch-size 1`` is the sequential decoding (ESPnet's ``decode_long``, one recording at a time);
``--batch-size > 1`` the batched mode (windows cut at pauses, all recordings at once).
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from owsm_trt import OWSM
from owsm_trt.audio import load_audio


def normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def edit_distance(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance (vectorized over ``b``)."""
    if not a or not b:
        return len(a) + len(b)
    vocab: dict[str, int] = {}
    xs = [vocab.setdefault(w, len(vocab)) for w in a]
    ys = np.array([vocab.setdefault(w, len(vocab)) for w in b])
    ar = np.arange(len(ys) + 1)
    prev = ar.copy()
    for i, x in enumerate(xs, 1):
        tmp = np.empty_like(prev)
        tmp[0] = i
        np.minimum(prev[1:] + 1, prev[:-1] + (ys != x), out=tmp[1:])
        prev = np.minimum.accumulate(tmp - ar) + ar  # cur[j] = min(tmp[j], cur[j - 1] + 1)
    return int(prev[-1])


def wer(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    """(word errors, reference words)."""
    errs = sum(edit_distance(normalize(r), normalize(h)) for r, h in zip(refs, hyps))
    return errs, sum(len(normalize(r)) for r in refs)


class Counter:
    """Counts the windows that go through the encoder (``OWSM._features`` rows)."""

    def __init__(self, model: OWSM):
        self.windows = 0
        features = model._features

        def counted(waves):
            self.windows += len(waves)
            return features(waves)

        model._features = counted
        self.model = model

    def remove(self) -> None:
        del self.model._features


def run(model: OWSM, waves: list[torch.Tensor], batch_size: int, **kw) -> tuple[list, float]:
    sync = torch.cuda.synchronize if model.device.type == "cuda" else lambda: None
    sync()
    t0 = time.perf_counter()
    if batch_size > 1:
        results = model.transcribe_long(waves, batch_size=batch_size, **kw)
    else:
        results = [model.transcribe_long(w, **kw) for w in waves]
    sync()
    return results, time.perf_counter() - t0


def evaluate(
    model: OWSM,
    items: list[dict],
    waves: list[torch.Tensor],
    batch_size: int = 1,
    repeat: int = 1,
    save: str | None = None,
    compare: list[str] = (),
    label: str = "",
    warmup: bool = True,
    **kw,
) -> dict:
    """Warm-up pass, then ``repeat`` timed passes; prints WER, RTFx, decoder calls and agreement."""
    audio_s = sum(len(w) for w in waves) / model.cfg.sample_rate
    counter = Counter(model)
    if warmup:
        run(model, waves, batch_size, **kw)  # CUDA graphs for every window length
    model.draft_stats = [0, 0]
    counter.windows = 0
    times, results = [], None
    for _ in range(repeat):
        results, dt = run(model, waves, batch_size, **kw)
        times.append(dt)
    counter.remove()
    dt = sorted(times)[len(times) // 2]
    errs, words = wer([it["text"] for it in items], [r.text for r in results])
    tokens, calls = (x / repeat for x in model.draft_stats)
    stats = {
        "label": label, "batch": batch_size, "wer": 100 * errs / words, "errors": errs, "words": words,
        "seconds": dt, "audio_s": audio_s, "rtfx": audio_s / dt, "windows": counter.windows / repeat,
        "tokens": tokens, "calls": calls,
    }  # fmt: skip
    print(
        f"[{label}] batch={batch_size} {len(items)} recordings ({audio_s / 3600:.2f} h): WER {stats['wer']:.2f}% "
        f"({errs}/{words}), {dt:.2f}s (median of {repeat}) RTFx {stats['rtfx']:.0f}, "
        f"{stats['windows']:.0f} windows, {calls:.0f} decoder calls ({tokens / max(calls, 1):.2f} tokens/call)"
    )
    hyps = {
        it["id"]: {"text": r.text, "segments": [[s.start, s.end, s.text] for s in r.segments]}
        for it, r in zip(items, results)
    }
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        json.dump(hyps, open(save, "w"), indent=1, ensure_ascii=False)
    for path in compare:
        if not Path(path).exists():
            print(f"  (no {path} to compare against)")
            continue
        ref = json.load(open(path))
        keys = [k for k in hyps if k in ref]
        same_text = sum(hyps[k]["text"] == ref[k]["text"] for k in keys)
        same_segs = sum(hyps[k]["segments"] == ref[k]["segments"] for k in keys)
        e, n = wer([ref[k]["text"] for k in keys], [hyps[k]["text"] for k in keys])
        print(
            f"  vs {path}: identical text {same_text}/{len(keys)}, identical segments {same_segs}/{len(keys)}, "
            f"word differences {e} ({100 * e / max(n, 1):.2f}%)"
        )
        for k in [k for k in keys if hyps[k]["text"] != ref[k]["text"]][:3]:
            a, b = normalize(ref[k]["text"]), normalize(hyps[k]["text"])
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"    {k} first difference at word {i}:\n      ref : {' '.join(a[max(0, i - 5) : i + 10])}"
                  f"\n      this: {' '.join(b[max(0, i - 5) : i + 10])}")  # fmt: skip
    stats["hyps"] = hyps
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=".cache/longform/test_clean/manifest.json")
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, help="first N recordings only")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-draft", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--compare", nargs="*", default=[])
    args = ap.parse_args()
    items = json.load(open(args.data))[: args.limit]
    waves = [load_audio(it["path"]) for it in items]
    model = OWSM.from_pretrained(
        args.model, backend=args.backend, dtype=args.dtype, device=args.device, ctc_draft=not args.no_draft
    )
    evaluate(model, items, waves, args.batch_size, args.repeat, args.save, args.compare)


if __name__ == "__main__":
    main()
