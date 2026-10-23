"""Whole-encoder A/B of ``BuildOptions.dwconv_plugin``: the encoder engine as ``build_engines`` builds it
(same export, profiles and builder settings), with and without the depthwise-convolution plugin, timed
at batch 1/8/16 on 30 s of input, and the outputs of the two compared on real recordings.

python scripts/dwconv_encoder.py .cache/models/owsm_v4_medium_1B
"""

import argparse
import json
import statistics
import tempfile
import time
import traceback
from pathlib import Path

import tensorrt as trt
import torch

from owsm_trt import hub, plugins
from owsm_trt.audio import load_audio
from owsm_trt.build import MIN_FEATURE_FRAMES, BuildOptions, _build, export_onnx, trims_features
from owsm_trt.config import load_espnet_config, parse_config
from owsm_trt.modeling import DepthwiseConv1d, load_model

DTYPES = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
}


class Runner:
    """One encoder execution context on fixed buffers, replayed as a CUDA graph."""

    def __init__(self, engine, profile: int, feats: torch.Tensor):
        self.ctx = ctx = engine.create_execution_context()
        stream = torch.cuda.current_stream()
        ctx.set_optimization_profile_async(profile, stream.cuda_stream)
        stream.synchronize()
        self.feats = feats
        ctx.set_input_shape("feats", tuple(feats.shape))
        ctx.set_tensor_address("feats", feats.data_ptr())
        self.bufs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                self.bufs[name] = torch.empty(
                    shape, dtype=DTYPES[engine.get_tensor_dtype(name)], device="cuda"
                )
                ctx.set_tensor_address(name, self.bufs[name].data_ptr())
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            ctx.execute_async_v3(side.cuda_stream)
            ctx.execute_async_v3(side.cuda_stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)

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
    ap.add_argument("model")
    ap.add_argument("--data", default=".cache/testdata/librispeech_dummy/manifest.json")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--configs", type=json.loads, default=[],
        help='plugin engines (merge input concatenated by the plugin) with one kernel configuration each '
        'instead of TensorRT choosing among plugins.TACTICS, e.g. \'[{"v": 2, "tt": 16, "nt": 128, "nch": 1}]\'',
    )  # fmt: skip
    ap.add_argument("--no-plugin", action="store_true", help="skip the TensorRT-concatenation variant")
    args = ap.parse_args()
    files = hub.resolve(args.model)
    cfg = parse_config(load_espnet_config(files.config))
    model = load_model(cfg, str(files.checkpoint), "cuda", torch.float16)
    B = 16
    batches = [{"batch": (1, 1, 1)}, {"batch": (2, B, B)}]
    frames = (MIN_FEATURE_FRAMES, cfg.num_frames // 3, cfg.num_frames) if trims_features(cfg) else None
    frames = frames or (cfg.num_frames,) * 3
    shapes = [{"feats": tuple((n, t, cfg.n_mels) for n, t in zip(p["batch"], frames))} for p in batches]

    tmp = Path(tempfile.mkdtemp())
    engines = {}
    # name: (dwconv_plugin, plugin_cat, kernel configuration or None for TensorRT's choice)
    variants = {"conv": (False, False, None), "plugin": (True, False, None), "plugin_cat": (True, True, None)}
    if args.no_plugin:
        del variants["plugin"]
    for c in args.configs:
        variants["cat_" + "_".join(f"{k}{v}" for k, v in c.items())] = (True, True, c)
    exported = {}
    for name, (flag, cat, override) in variants.items():
        opts = BuildOptions(dwconv_plugin=flag)
        t0 = time.time()
        DepthwiseConv1d.plugin_cat = cat  # the merge convolution's input concatenated by the plugin
        plugins.override = override
        try:
            if (flag, cat) not in exported:
                exported[flag, cat] = export_onnx(model, tmp / name, opts)[0]
            _build(exported[flag, cat], tmp / name / "encoder.engine", shapes, opts)
        except Exception:  # noqa: BLE001
            print(f"{name}: export/build failed:\n{traceback.format_exc()}", flush=True)
            continue
        finally:
            DepthwiseConv1d.plugin_cat = True
            plugins.override = None
        data = (tmp / name / "encoder.engine").read_bytes()
        engines[name] = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(data)
        print(f"{name}: {engines[name].num_layers} layers, {len(data) / 2**20:.0f} MiB, "
              f"export + build {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    if "conv" not in engines or len(engines) < 2:
        return

    for b in (1, 8, 16):
        feats = torch.randn(b, cfg.num_frames, cfg.n_mels, device="cuda")
        runners = {name: Runner(e, 0 if b == 1 else 1, feats) for name, e in engines.items()}
        times = {name: [] for name in runners}
        for _ in range(args.rounds):  # interleaved, against clock/thermal drift
            for name, r in runners.items():
                times[name].append(r.time())
        med = {name: statistics.median(t) for name, t in times.items()}
        print(
            f"b{b:<2d} " + "  ".join(f"{n}: {m:6.2f} ms (min {min(times[n]):6.2f})" for n, m in med.items())
            + "  vs conv: " + " ".join(f"{n} {m / med['conv']:.3f}" for n, m in med.items() if n != "conv"),
            flush=True,
        )  # fmt: skip
        del runners

    # outputs on the shortest real recordings, padded to 30 s
    waves = sorted((load_audio(it["path"]) for it in json.load(open(args.data))), key=len)[:B]
    speech = torch.zeros(len(waves), cfg.num_samples, device="cuda")
    for row, w in zip(speech, waves):
        row[: w.shape[0]] = w[: cfg.num_samples].cuda()
    feats = model.frontend(speech)
    outs = {name: Runner(e, 1, feats.clone())(feats) for name, e in engines.items()}
    with torch.no_grad():  # PyTorch fp16, for scale
        ref = model.decoder.cross_kv(model.encoder(feats))[:, 0].float()
    r = outs["conv"]["cross_k"].float()
    print(f"conv vs PyTorch fp16: cross_k rel err {float((r - ref).norm() / ref.norm()):.2e}")
    for name, out in outs.items():
        if name == "conv":
            continue
        k = out["cross_k"].float()
        same = (out["ctc_ids"] == outs["conv"]["ctc_ids"]).float().mean()
        print(f"{name} vs conv on {len(waves)} recordings: cross_k rel err {float((k - r).norm() / r.norm()):.2e}, "
              f"CTC ids equal {float(same):.4f}; vs PyTorch fp16 {float((k - ref).norm() / ref.norm()):.2e}")  # fmt: skip


if __name__ == "__main__":
    main()
