"""Long-form test set: the utterances of each LibriSpeech chapter, concatenated in order into one recording.

python scripts/make_longform.py --data .cache/testdata/librispeech_test_clean/manifest.json \
    --out .cache/longform/test_clean

Writes <out>/<speaker>-<chapter>.flac and <out>/manifest.json with, per recording, its path, the
concatenated reference text and where each utterance starts (in samples), for scripts/eval_long.py.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".cache/testdata/librispeech_test_clean/manifest.json")
    ap.add_argument("--out", default=".cache/longform/test_clean")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chapters: dict[str, list[dict]] = defaultdict(list)
    for it in json.load(open(args.data)):
        chapters["-".join(it["id"].split("-")[:2])].append(it)
    rows = []
    for name, utts in sorted(chapters.items()):
        utts.sort(key=lambda it: int(it["id"].split("-")[2]))
        waves, starts, n = [], [], 0
        for it in utts:
            w, sr = sf.read(it["path"], dtype="int16")
            assert sr == 16000 and w.ndim == 1, it["path"]
            waves.append(w)
            starts.append(n)
            n += len(w)
        path = out / f"{name}.flac"
        sf.write(path, np.concatenate(waves), 16000, subtype="PCM_16")
        rows.append({
            "id": name, "path": str(path.resolve()), "seconds": round(n / 16000, 2),
            "text": " ".join(it["text"] for it in utts),
            "utterances": [{"id": it["id"], "start": s, "text": it["text"]} for it, s in zip(utts, starts)],
        })  # fmt: skip
    json.dump(rows, open(out / "manifest.json", "w"), indent=1)
    secs = [r["seconds"] for r in rows]
    print(f"{len(rows)} recordings, {sum(secs) / 3600:.2f} h, {min(secs):.0f}-{max(secs):.0f} s -> {out}")


if __name__ == "__main__":
    main()
