"""Manifests for the FP8 calibration study (scripts/fp8_calibrate.py, scripts/eval_wer.py --lang):

* FLEURS (google/fleurs parquet files, downloaded into --fleurs): <lang>_<split>.json per language/split,
  audio written next to them as .wav (the parquet's own WAV bytes)
* JVS (Japanese, --jvs): jvs_test.json (parallel100 of speakers 1-10) and jvs_calib.json (nonpara30 of
  speakers 91-100: other speakers, other sentences)
* LibriSpeech dev-other (--librispeech): librispeech_dev_other.json

Each entry: id, path, text, lang (OWSM language code). Calibration and test entries are disjoint.

python scripts/prepare_multilingual.py --out .cache/multilingual --fleurs .cache/fleurs \
    --jvs /path/to/jvs_ver1 --librispeech .cache/librispeech/LibriSpeech/dev-other
"""

import argparse
import json
from pathlib import Path

FLEURS = {"de_de": "deu", "es_419": "spa", "cmn_hans_cn": "zho", "ja_jp": "jpn", "fr_fr": "fra", "en_us": "eng",
          "ru_ru": "rus"}  # fmt: skip


def fleurs(root: Path, out: Path) -> None:
    import pyarrow.parquet as pq

    for f in sorted(root.glob("parquet-data/*/*.parquet")):
        lang, split = f.parent.name, f.name.split("-")[0]
        if lang not in FLEURS:
            continue
        audio_dir = out / "fleurs" / lang / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        table = pq.read_table(f, columns=["audio", "raw_transcription", "transcription"])
        for r in table.to_pylist():
            name = Path(r["audio"]["path"]).stem
            path = audio_dir / f"{name}.wav"
            if not path.exists():
                path.write_bytes(r["audio"]["bytes"])
            rows.append({"id": f"{lang}-{name}", "path": str(path.resolve()), "text": r["raw_transcription"],
                         "norm": r["transcription"], "lang": FLEURS[lang]})  # fmt: skip
        (out / f"fleurs_{lang}_{split}.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))
        print(f"{f}: {len(rows)}")


def jvs(root: Path, out: Path) -> None:
    for name, speakers, subset in (
        ("jvs_test", range(1, 11), "parallel100"),
        ("jvs_calib", range(91, 101), "nonpara30"),
    ):
        rows = []
        for spk in speakers:
            d = root / f"jvs{spk:03d}" / subset
            for line in (d / "transcripts_utf8.txt").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                utt, text = line.split(":", 1)
                path = d / "wav24kHz16bit" / f"{utt}.wav"
                if path.exists():
                    rows.append({"id": f"jvs{spk:03d}-{utt}", "path": str(path), "text": text, "lang": "jpn"})
        (out / f"{name}.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))
        print(f"{name}: {len(rows)}")


def librispeech(root: Path, out: Path) -> None:
    rows = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            rows.append(
                {
                    "id": utt,
                    "path": str((trans.parent / f"{utt}.flac").resolve()),
                    "text": text,
                    "lang": "eng",
                }
            )
    (out / "librispeech_dev_other.json").write_text(json.dumps(rows, indent=0))
    print(f"librispeech dev-other: {len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".cache/multilingual")
    ap.add_argument("--fleurs")
    ap.add_argument("--jvs")
    ap.add_argument("--librispeech")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fleurs:
        fleurs(Path(args.fleurs), out)
    if args.jvs:
        jvs(Path(args.jvs), out)
    if args.librispeech:
        librispeech(Path(args.librispeech), out)


if __name__ == "__main__":
    main()
