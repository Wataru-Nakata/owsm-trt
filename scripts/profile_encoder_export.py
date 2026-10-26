"""Fresh-process cost of the unused decoder ONNX export when decoder engines are reused."""

import argparse
import dataclasses
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from profile_beam import save, sha256, wall

from owsm_trt import OWSM, build, fp8, hub
from owsm_trt.modeling import load_model
from owsm_trt.runtime import TRTBackend


@torch.inference_mode()
def worker(args):
    model = OWSM.from_pretrained(args.model, ctc_draft=False)
    files = hub.resolve(args.model)
    checkpoint = sha256(files.checkpoint)
    net = load_model(model.cfg, str(files.checkpoint), model.device, model.dtype)
    opts = model.opts
    scales_hash = None
    if args.fp8:
        scales = fp8.packaged_scales(checkpoint)
        if scales is None:
            raise ValueError("checkpoint-matched packaged scales are required")
        fp8.quantize(net.encoder, fp8.load_scales(scales)["amax"])
        scales_hash = sha256(scales)
        opts = dataclasses.replace(opts, fp8_encoder=True, fp8_calibration="s" + fp8.scales_digest(scales))
    directory = args.out / "onnx"
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(73)
    fn = build.export_onnx if args.worker == "full" else build.export_encoder_onnx
    torch.cuda.reset_peak_memory_stats()
    _, elapsed = wall(lambda: fn(net, directory, opts))
    outputs = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in directory.iterdir() if p.is_file()}
    encoder = {k:v for k,v in outputs.items() if k.startswith("encoder.")}
    assert encoder and "encoder.onnx" in encoder
    assert ("decoder.onnx" in outputs) is (args.worker == "full")
    result = dict(mode=args.worker, export_ms=elapsed, files=outputs,
         encoder_files=encoder, checkpoint_sha256=checkpoint, scales_sha256=scales_hash,
         export_peak_cuda_bytes=torch.cuda.max_memory_allocated(), torch=torch.__version__,
         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), options=dataclasses.asdict(opts))
    if args.build_smoke:
        reference = hub.cache_dir() / files.name.replace("/", "--") / build.engine_tag(model.opts)
        target = args.out / "engine_smoke"

        def forbidden(*args, **kwargs):
            raise AssertionError("reused decoder unexpectedly exported")

        with patch.object(build, "export_decoder_onnx", forbidden):
            _, duration = wall(lambda: build.build_engines(net, target, opts, decoders_from=reference,
                                                           skip_decoder_export=True))
        for name in ("decoder.engine", "verify.engine"):
            assert sha256(target / name) == sha256(reference / name)
        backend = TRTBackend(model.cfg, target)
        shapes = []
        for batch in (1, 3):
            feats = model._features([torch.zeros(16000)] * batch)
            cfg = model.cfg
            shape = (cfg.dec_layers, batch, cfg.dec_heads, cfg.enc_frames, cfg.d_model // cfg.dec_heads)
            k = torch.empty(shape, dtype=model.dtype, device=model.device)
            v = torch.empty_like(k)
            ids = torch.empty(batch, cfg.enc_frames + 2, dtype=torch.int32, device=model.device)
            paths = backend.encode(feats, k, v, ids)
            assert bool(torch.isfinite(k).all()) and bool(torch.isfinite(v).all())
            assert paths.shape[0] == batch and bool(((paths >= 0) & (paths < cfg.vocab_size)).all())
            shapes.append(dict(features=list(feats.shape), kv=list(k.shape), ctc=list(paths.shape)))
        result["build_smoke"] = dict(wall_ms=duration, encoder_sha256=sha256(target / "encoder.engine"),
                                    copied_decoders_exact=True, shapes=shapes)
        del backend
        shutil.rmtree(target)
    save(args.out / "worker.json", result)
    # This directory is created by this worker under PBS_LOCALDIR; only hashes/small logs are published.
    shutil.rmtree(directory)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--worker", choices=["full", "encoder"])
    ap.add_argument("--build-smoke", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") or not os.environ.get("PBS_LOCALDIR"):
        ap.error("export/model work requires an allocated PBS compute job")
    if not args.out.resolve().is_relative_to(Path(os.environ["PBS_LOCALDIR"]).resolve()):
        ap.error("temporary ONNX exports must be on compute-local storage")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    samples, first = [], []
    expected = None
    for repeat in range(-1, args.repeat):
        for mode in (("full", "encoder") if repeat % 2 == 0 else ("encoder", "full")):
            out = args.out / f"{repeat}_{mode}"
            out.mkdir()
            cmd = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
                   "--worker", mode, "--out", str(out)]
            if args.fp8:
                cmd.append("--fp8")
            with (out / "worker.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            value = json.loads((out / "worker.json").read_text()) | {"repeat": repeat}
            fingerprint = (value["checkpoint_sha256"], value["scales_sha256"], value["encoder_files"])
            if expected is None:
                expected = fingerprint
            assert fingerprint == expected, "encoder ONNX/checkpoint/scales differ; inspect worker artifacts"
            (first if repeat < 0 else samples).append(value)
            save(args.out / "export.json", dict(samples=samples, first=first, encoder_exact=True))
            if path := os.environ.get("OWSM_PROFILE_PROGRESS"):
                with Path(path).open("a") as log:
                    log.write(f"encoder export {mode} cycle{repeat} complete\n")
    summary = {}
    for mode in ("full", "encoder"):
        values = [row["export_ms"] for row in samples if row["mode"] == mode]
        summary[mode] = dict(p50_ms=statistics.median(values), stdev_ms=statistics.pstdev(values),
                             min_ms=min(values), max_ms=max(values))
    smoke = None
    if args.build_smoke:
        out = args.out / "smoke"
        out.mkdir()
        cmd = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
               "--worker", "encoder", "--build-smoke", "--out", str(out)]
        if args.fp8:
            cmd.append("--fp8")
        with (out / "worker.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        smoke = json.loads((out / "worker.json").read_text())
        assert (smoke["checkpoint_sha256"], smoke["scales_sha256"], smoke["encoder_files"]) == expected
    save(args.out / "export.json", dict(samples=samples, first=first, encoder_exact=True, summary=summary, smoke=smoke,
         note="Fresh-process ONNX export component only. Excludes model load, TRT build, inference; shared OS/JIT caches warmed. Unchanged encoder ONNX/external data hashes are required."))


if __name__ == "__main__":
    main()
