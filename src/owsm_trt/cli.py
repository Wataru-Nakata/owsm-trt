"""Command line interface: ``owsm-trt {build,transcribe,bench}``."""

import argparse
import json
import logging
import time


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="espnet/owsm_v4_medium_1B", help="HF repo id or local ESPnet model dir")
    p.add_argument("--backend", default="tensorrt", choices=["tensorrt", "torch"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--beam-size", type=int, default=1, help="1 = greedy; 5 = ESPnet's default beam search")
    p.add_argument("--no-ctc-draft", action="store_true", help="greedy without CTC-drafted token windows")
    p.add_argument(
        "--fp8", action="store_true", help="experimental: FP8 encoder with the scales the package ships"
    )


def _load(args, **kw):
    from . import OWSM

    return OWSM.from_pretrained(
        args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        ctc_draft=not args.no_ctc_draft,
        fp8=args.fp8,
        **kw,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="owsm-trt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build (or rebuild) and cache the TensorRT engines")
    _add_model_args(p)
    p.add_argument("--force", action="store_true", help="rebuild even if cached engines exist")

    p = sub.add_parser("transcribe", help="transcribe / translate audio files")
    _add_model_args(p)
    p.add_argument("audio", nargs="+")
    p.add_argument("--lang", default="eng", help="source language code, e.g. eng, jpn, deu")
    p.add_argument("--task", default="asr", help="asr, or st_<lang> for translation (e.g. st_eng)")
    p.add_argument("--timestamps", action="store_true")
    p.add_argument("--long", action="store_true", help="long-form decoding (audio > 30 s)")
    p.add_argument("--json", action="store_true", help="print JSON lines")

    p = sub.add_parser("bench", help="measure throughput on synthetic or given audio")
    _add_model_args(p)
    p.add_argument("audio", nargs="*")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=10)

    args = ap.parse_args(argv)
    # owsm_trt's progress, not the libraries' INFO (the ONNX optimizer's ~1000 lines per build, HTTP requests)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("owsm_trt").setLevel(logging.INFO)

    if args.cmd == "build":
        if args.backend != "tensorrt":
            ap.error("build only applies to --backend tensorrt")
        _load(args, rebuild=args.force)
        print("engines ready")
    elif args.cmd == "transcribe":
        model = _load(args)
        for path in args.audio:
            kw = {"lang": args.lang, "task": args.task, "beam_size": args.beam_size}
            if args.long:
                res = model.transcribe_long(path, **kw)
            else:
                res = model.transcribe(path, timestamps=args.timestamps, **kw)
            if args.json:
                segs = [s.__dict__ for s in res.segments]
                print(json.dumps({"audio": path, "text": res.text, "segments": segs}, ensure_ascii=False))
            else:
                print(f"{path}\t{res.text}")
                for s in res.segments:
                    print(f"  [{s.start:7.2f} - {s.end:7.2f}] {s.text}")
    elif args.cmd == "bench":
        import torch

        model = _load(args)
        audio = args.audio or [torch.randn(16000 * 10).numpy() * 0.01]
        batch = (audio * args.batch_size)[: max(args.batch_size, len(audio))]
        kw = {"batch_size": args.batch_size, "beam_size": args.beam_size}
        model.transcribe(batch, **kw)  # warm-up / graph capture
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = model.transcribe(batch, **kw)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        n_tok = sum(len(r.tokens) for r in out)
        print(f"{dt * 1000:.1f} ms per batch of {len(batch)} ({n_tok} tokens, {n_tok / dt:.0f} tok/s)")


if __name__ == "__main__":
    main()
