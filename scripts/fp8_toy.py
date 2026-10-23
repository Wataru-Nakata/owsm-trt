"""Does TensorRT turn our FP8 quantize/dequantize pairs into FP8 kernels? Toy encoder-sized layers.

python scripts/fp8_toy.py
"""

import json
import tempfile
import time
from pathlib import Path

import tensorrt as trt
import torch
from torch import nn

from owsm_trt.build import BuildOptions, _build
from owsm_trt.fp8 import ONNX_TRANSLATIONS, QConv2d, QLinear


class Toy(nn.Module):
    """Two FFN-sized linears (as in an encoder layer at batch 1: 374 frames) and a subsampling conv."""

    def __init__(self, fp8: bool):
        super().__init__()
        torch.manual_seed(0)
        self.w1, self.w2 = nn.Linear(1024, 4096), nn.Linear(4096, 1024)
        self.conv = nn.Conv2d(1024, 1024, 3, 2)
        if fp8:
            self.w1, self.w2, self.conv = (
                QLinear(self.w1, 4.0),
                QLinear(self.w2, 4.0),
                QConv2d(self.conv, 4.0),
            )

    def forward(self, x: torch.Tensor, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.w2(torch.relu(self.w1(x))), self.conv(img)


def time_engine(path: Path, inputs: dict) -> tuple[float, list[str], dict]:
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(path.read_bytes())
    ctx = engine.create_execution_context()
    outs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(name, tuple(inputs[name].shape))
            ctx.set_tensor_address(name, inputs[name].data_ptr())
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(ctx.get_tensor_shape(name))
            outs[name] = torch.empty(shape, dtype=torch.float16, device="cuda")
            ctx.set_tensor_address(name, outs[name].data_ptr())
    stream = torch.cuda.current_stream()
    for _ in range(5):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 50 * 1000
    insp = engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    tactics = [f"{lay.get('LayerType', '')}: {lay.get('TacticName', '')[:90]}" for lay in info["Layers"]]
    return ms, tactics, outs


def main() -> None:
    x = torch.randn(4, 374, 1024, device="cuda", dtype=torch.float16)
    img = torch.randn(4, 1024, 750, 31, device="cuda", dtype=torch.float16)
    tmp = Path(tempfile.mkdtemp())
    with torch.no_grad():
        ref = Toy(False).cuda().eval()(x.float(), img.float())
    for fp8 in (False, True):
        model = Toy(fp8).cuda().half().eval()
        onnx_path, engine_path = tmp / f"toy{int(fp8)}.onnx", tmp / f"toy{int(fp8)}.engine"
        torch.onnx.export(
            model, (x, img), onnx_path, input_names=["x", "img"], output_names=["y", "z"], dynamo=True,
            custom_translation_table=ONNX_TRANSLATIONS if fp8 else None,
        )  # fmt: skip
        _build(onnx_path, engine_path, [], BuildOptions())
        ms, tactics, outs = time_engine(engine_path, {"x": x, "img": img})
        err = [float((outs[k].float() - r).norm() / r.norm()) for k, r in zip(("y", "z"), ref)]
        print(f"== {'fp8' if fp8 else 'fp16'}: {ms:.3f} ms per run (2 linears + 1 conv, batch 4)")
        print(f"   relative error vs fp32: linears {err[0]:.4f}, conv {err[1]:.4f}")
        for t in tactics:
            print("   ", t)


if __name__ == "__main__":
    main()
