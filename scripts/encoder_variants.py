"""Encoder engine variants, timed per batch size: the batch the optimization profile is tuned for,
CTC argmax inside the engine or not, features trimmed after the audio (the subsampled frames of
padding filled in) or not.

python scripts/encoder_variants.py --model .cache/models/owsm_v4_medium_1B
"""

import argparse
import json
import tempfile
from pathlib import Path

import tensorrt as trt
import torch
from torch import nn

from owsm_trt import hub
from owsm_trt.audio import load_audio
from owsm_trt.build import BuildOptions, _build
from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.modeling import frames_needed, load_model, padding_frame

DTYPES = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
}
B = 16
VARIANTS = {  # name: (CTC output, profile (min, opt, max) batch, trimmed features)
    "base": ("argmax", (1, B, B), False),
    "b1": ("argmax", (1, 1, 1), False),
    "b8": ("argmax", (1, 8, B), False),
    "logits": ("logits", (1, B, B), False),
    "noctc": ("none", (1, B, B), False),
    "trim": ("argmax", (1, B, B), True),
    "trim_b1": ("argmax", (1, 1, 1), True),
}


class Graph(nn.Module):
    def __init__(self, model, ctc: str):
        super().__init__()
        self.model, self.ctc = model, ctc

    def forward(self, feats: torch.Tensor):
        m = self.model
        enc = m.encoder(feats)
        kv = m.decoder.cross_kv(enc)
        if self.ctc == "argmax":
            return kv[:, 0], kv[:, 1], m.ctc(enc)
        if self.ctc == "logits":
            return kv[:, 0], kv[:, 1], m.ctc.ctc_lo(enc)
        return kv[:, 0], kv[:, 1]


class Runner:
    """An engine on fixed buffers, replayed as a CUDA graph."""

    def __init__(self, engine, feats: torch.Tensor):
        self.ctx = engine.create_execution_context()
        self.feats = feats
        self.ctx.set_input_shape("feats", tuple(feats.shape))
        self.ctx.set_tensor_address("feats", feats.data_ptr())
        self.bufs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.ctx.get_tensor_shape(name))
                self.bufs[name] = torch.empty(
                    shape, dtype=DTYPES[engine.get_tensor_dtype(name)], device="cuda"
                )
                self.ctx.set_tensor_address(name, self.bufs[name].data_ptr())
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            self.body()
            self.body()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.body()

    def body(self) -> None:
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if "ctc_logits" in self.bufs:
            self.bufs["ctc_ids"] = self.bufs["ctc_logits"].argmax(-1).int()

    def time(self, iters: int = 20) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.graph.replay()
        start.record()
        for _ in range(iters):
            self.graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def __call__(self, feats: torch.Tensor) -> dict:
        self.feats.copy_(feats)
        self.graph.replay()
        torch.cuda.synchronize()
        return {k: v.clone() for k, v in self.bufs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    args = ap.parse_args()
    files = hub.resolve(args.model)
    cfg = parse_config(load_espnet_config(files.config))
    model = load_model(cfg, str(files.checkpoint), "cuda", torch.float16)
    fill = padding_frame(model)

    # the shortest real recordings, padded to 30 s as the pipeline does, for the comparison with "base"
    waves = sorted((load_audio(it["path"]) for it in json.load(open(args.data))), key=len)[:B]
    speech = torch.zeros(len(waves), cfg.num_samples, device="cuda")
    for row, w in zip(speech, waves):
        row[: w.shape[0]] = w[: cfg.num_samples].cuda()
    feats = model.frontend(speech)
    need = max(frames_needed(cfg, w.shape[0]) for w in waves)
    print(f"{len(waves)} recordings, longest {max(w.shape[0] for w in waves) / cfg.sample_rate:.1f}s: "
          f"{need} of {cfg.num_frames} feature frames needed")  # fmt: skip

    tmp = Path(tempfile.mkdtemp())
    ref = None
    for name in args.variants.split(","):
        ctc, prof, trim = VARIANTS[name]
        model.encoder.fill = fill if trim else None
        onnx_path = tmp / name / "encoder.onnx"
        onnx_path.parent.mkdir()
        names = {"argmax": ["cross_k", "cross_v", "ctc_ids"], "logits": ["cross_k", "cross_v", "ctc_logits"]}
        dims = {0: torch.export.Dim("batch", min=1, max=B)}
        if trim:
            dims[1] = torch.export.Dim("time", min=64, max=cfg.num_frames)
        try:
            torch.onnx.export(
                Graph(model, ctc).eval(),
                (torch.randn(2, cfg.num_frames - 8 if trim else cfg.num_frames, cfg.n_mels, device="cuda"),),
                onnx_path,
                input_names=["feats"],
                output_names=names.get(ctc, ["cross_k", "cross_v"]),
                dynamic_shapes={"feats": dims},
                dynamo=True,
                external_data=True,
            )
        finally:
            model.encoder.fill = None
        engine_path = tmp / name / "encoder.engine"
        lengths = (64, 1024, cfg.num_frames) if trim else (cfg.num_frames,) * 3
        profile = {"feats": tuple((n, t, cfg.n_mels) for n, t in zip(prof, lengths))}
        _build(onnx_path, engine_path, [profile], BuildOptions())
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.read_bytes())

        times = []
        for b in (1, 8, 16):
            if not prof[0] <= b <= prof[2]:
                continue
            for t in (cfg.num_frames, 800) if trim else (cfg.num_frames,):
                ms = Runner(engine, torch.randn(b, t, cfg.n_mels, device="cuda")).time()
                times.append(
                    f"b{b}{'' if t == cfg.num_frames else f' T={t}'}: {ms:6.2f} ms ({ms / b:5.2f}/utt)"
                )
        print(f"== {name:11s} " + "  ".join(times), flush=True)

        b = min(len(waves), prof[2])
        x = feats[:b, :need] if trim else feats[:b]
        outs = Runner(engine, x.clone())(x)
        if ref is None:
            ref = outs
            continue
        k, r = outs["cross_k"].float(), ref["cross_k"][:, :b].float()  # [L, B, H, T, d_k]
        err = float((k - r).norm() / r.norm())
        msg = f"   vs {args.variants.split(',')[0]} on {b} recordings: cross_k rel err {err:.2e}"
        if "ctc_ids" in outs:
            same = (outs["ctc_ids"] == ref["ctc_ids"][:b]).float().mean()
            msg += f", CTC ids equal {float(same):.4f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
