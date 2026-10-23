"""Which of owsm_trt's speed-ups a model gets in this environment: package versions, the availability checks,
the TensorRT plugins inside its engines, the decoder megakernel, the stream lanes; and how long loading and the
first transcriptions take. (Clean-install QA, see scripts/fresh_install.sh.)

    python scripts/engaged.py espnet/owsm_v4_medium_1B speech.wav
"""

import collections
import importlib.metadata as md
import json
import logging
import sys
import time
import warnings

import torch

import owsm_trt
from owsm_trt import OWSM, attn_plugin, megakernel, megakernel_gluon, pipeline, plugins, runtime

PLUGINS = ("dwconv1d", "conv1_relu", "relu_flatten", "softmax_rows", "argmax_rows", "self_attn", "cross_attn")


def version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "-"


def engine_plugins(engine) -> dict[str, int]:
    """Plugin layers per plugin name (the engines are built with detailed layer information)."""
    import tensorrt as trt

    info = json.loads(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    )
    count = collections.Counter()
    for layer in info.get("Layers", []):  # (each layer once per optimization profile)
        if not isinstance(layer, dict) or "plugin" not in str(layer.get("LayerType", "")).lower():
            continue
        text = json.dumps(layer)
        count["plugin layers"] += 1
        for p in PLUGINS + ("dwconv1d_gated", "dwconv1d_cat"):
            if p in text and (p != "dwconv1d" or "dwconv1d_" not in text):
                count[p] += 1
    return dict(count)


def main() -> None:
    model_id, wav = sys.argv[1], sys.argv[2]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    dev = torch.device("cuda")
    print("python", sys.version.split()[0], "| owsm_trt", owsm_trt.__file__)
    for p in ("torch", "tensorrt", "tensorrt-cu13", "tensorrt-cu12", "triton", "cuda-bindings", "numpy", "onnx",
              "onnxscript", "huggingface-hub"):  # fmt: skip
        print(f"  {p} {version(p)}")
    print(
        "torch CUDA",
        torch.version.cuda,
        "|",
        torch.cuda.get_device_name(),
        torch.cuda.get_device_capability(),
    )
    print("available: nvrtc plugins", plugins.available(), "| attention plugins", attn_plugin.available(64),
          "| triton megakernel", megakernel.available(dev), "| gluon megakernel", megakernel_gluon.available(dev))  # fmt: skip
    print("lanes: encoder", runtime.ENCODER_LANES, "| ctc", pipeline.CTC_LANES)

    t0 = time.time()
    model = OWSM.from_pretrained(model_id)
    torch.cuda.synchronize()
    print(f"from_pretrained {time.time() - t0:.1f}s")
    be = model.backend
    print("backend", type(be).__name__, "->", type(getattr(be, "inner", be)).__name__)
    if isinstance(be, megakernel.MegaBackend):
        rows = (be.rows_kernel or (None,))[0]
        print(
            "  megakernel", type(be.kernel).__name__, "| rows kernel", type(rows).__name__ if rows else None
        )
    inner = getattr(be, "inner", be)
    engines = {k: v for k, v in vars(inner).items() if type(v).__name__ == "ICudaEngine"}
    for name, eng in engines.items():
        print(f"  {name}.engine plugins: {engine_plugins(eng)}")
    for i in range(3):
        t0 = time.time()
        res = model.transcribe(wav, lang="eng")
        torch.cuda.synchronize()
        print(f"transcribe #{i + 1} {time.time() - t0:.2f}s: {res.text[:80]}")
    if isinstance(be, megakernel.MegaBackend):
        print("  megakernel trials failed:", len(be.failed))


if __name__ == "__main__":
    main()
