"""FP8 input ranges of the encoder, measured offline (see owsm_trt.fp8 scale files).

measure (GPU): the largest absolute input of every quantizable encoder layer, per recording, for the
recordings of the given manifests (scripts/prepare_multilingual.py; OWSM-CTC: with each recording's
language in the prefix). Recordings are zero-padded to the 30 s window, as the engines see them.

    python scripts/fp8_calibrate.py measure --model espnet/owsm_ctc_v4_1B --per-manifest 100 \
        --data a.json b.json --out rows.json

make (CPU): a scale file from such measurements, over a subset of the recordings (languages, how many).

    python scripts/fp8_calibrate.py make rows.json --out scales.json [--langs eng,deu] [--n 64 --seed 0]
"""

import argparse
import json
import random
from pathlib import Path

import torch


def measure(args) -> None:
    from owsm_trt import fp8, hub
    from owsm_trt.audio import load_audio
    from owsm_trt.config import load_espnet_config, parse_config
    from owsm_trt.modeling import load_model
    from owsm_trt.tokenizer import Tokenizer

    files = hub.resolve(args.model)
    raw = load_espnet_config(files.config)
    cfg = parse_config(raw)
    tok = Tokenizer(raw["token_list"], hub.resolve_relative(files, raw["bpemodel"]))
    net = load_model(cfg, str(files.checkpoint), "cuda", torch.float16)
    items = []
    for path in args.data:
        rows = json.load(open(path))
        random.Random(args.seed).shuffle(rows)
        items += rows[: args.per_manifest]
    n = cfg.num_samples
    batches, metas = [], []
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        x = torch.zeros(len(chunk), n, device="cuda")
        for row, it in zip(x, chunk):
            w = load_audio(it["path"], cfg.sample_rate)[:n]
            row[: len(w)] = w.cuda()
            metas.append(
                {"id": it["id"], "lang": it.get("lang", "eng"), "seconds": round(len(w) / cfg.sample_rate, 2)}
            )
        batches.append((net.frontend(x), [it.get("lang", "eng") for it in chunk]))
    na, asr = tok.na, tok.task_id("asr")

    def run(batch) -> None:
        feats, langs = batch
        if cfg.ctc_only:  # the prefix (language, task) and the prompt (none) feed the encoder too
            b = feats.shape[0]
            prefix = torch.tensor([[tok.lang_id(lang), asr] for lang in langs], device="cuda")
            net(feats, torch.tensor([[na]] * b, device="cuda"), prefix)
        else:
            net.encoder(feats)

    with torch.inference_mode():
        rows = fp8.calibrate_rows(net.encoder, run, batches)
    sha = fp8.checkpoint_sha256(files.checkpoint, hub.cache_dir() / "checkpoint_sha256.json")
    out = {"model": files.name, "checkpoint_sha256": sha, "recordings": metas, "rows": rows,
           "data": [str(p) for p in args.data]}  # fmt: skip
    Path(args.out).write_text(json.dumps(out))
    print(f"{len(metas)} recordings, {len(rows)} layers -> {args.out}")


def make(args) -> None:
    from owsm_trt import fp8

    data = json.load(open(args.rows))
    recs = data["recordings"]
    keep = [i for i, r in enumerate(recs) if not args.langs or r["lang"] in args.langs.split(",")]
    if args.n:
        random.Random(args.seed).shuffle(keep)
        keep = sorted(keep[: args.n])
    amax = {name: max(v[i] for i in keep) for name, v in data["rows"].items()}
    langs = {}
    for i in keep:
        langs[recs[i]["lang"]] = langs.get(recs[i]["lang"], 0) + 1
    about = {"recordings": len(keep), "languages": langs, "data": [Path(p).name for p in data["data"]],
             "seed": args.seed, "padding": "30 s windows"}  # fmt: skip
    fp8.save_scales(args.out, amax, data["model"], data["checkpoint_sha256"], about)
    print(f"{args.out}: {len(amax)} layers from {len(keep)} recordings {langs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", required=True)
    m.add_argument("--data", nargs="+", required=True)
    m.add_argument("--per-manifest", type=int, default=100)
    m.add_argument("--batch", type=int, default=8)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    k = sub.add_parser("make")
    k.add_argument("rows")
    k.add_argument("--out", required=True)
    k.add_argument("--langs", default="")
    k.add_argument("--n", type=int, default=0)
    k.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    measure(args) if args.cmd == "measure" else make(args)


if __name__ == "__main__":
    main()
