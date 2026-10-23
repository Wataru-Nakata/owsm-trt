"""The depthwise conv1d plugin (owsm_trt.plugins) against TensorRT's grouped convolution with transposes,
one layer on its own: time per call, error against fp32 PyTorch, and the ONNX route (custom op -> ONNX
parser -> plugin) with an engine loaded in a process that never registered the plugin.

python scripts/dwconv_bench.py [model, ignored] [--quick]
"""

import argparse
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

from owsm_trt import plugins

T, C, K = 374, 2048, 31
LOGGER = trt.Logger(trt.Logger.WARNING)


def network_flags() -> int:
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if hasattr(trt.NetworkDefinitionCreationFlag, "PREFER_AOT_PYTHON_PLUGINS"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS)
    return flags


def build(kind: str, batch: int, w: np.ndarray, bias: np.ndarray, gated: bool = False) -> bytes:
    """kind: 'trt' (conv in [B, C, 1, T] between transposes, as the ONNX export gives it) or 'plugin'."""
    builder = trt.Builder(LOGGER)
    net = builder.create_network(network_flags())
    x = net.add_input("x", trt.float16, (-1, T, C))
    g = net.add_input("g", trt.float16, (-1, T, 2 * C)) if gated else None
    if kind == "plugin":
        wt = net.add_constant((K, C), trt.Weights(np.ascontiguousarray(w.T.astype(np.float32)))).get_output(0)
        bt = net.add_constant((C,), trt.Weights(bias.astype(np.float32))).get_output(0)
        if gated:
            op = trt.plugin.op.owsm.dwconv1d_gated(x, g, wt, bt)
        else:
            op = trt.plugin.op.owsm.dwconv1d(x, wt, bt)
        y = net.add_plugin(op, aot=True).get_output(0)
    else:
        tr = net.add_shuffle(x)
        tr.first_transpose = (0, 2, 1)
        tr.reshape_dims = (0, C, 1, T)
        conv = net.add_convolution_nd(
            tr.get_output(0), C, (1, K), trt.Weights(w.reshape(C, 1, 1, K)), trt.Weights(bias)
        )
        conv.num_groups = C
        conv.padding_nd = (0, K // 2)
        back = net.add_shuffle(conv.get_output(0))
        back.reshape_dims = (0, C, T)
        back.second_transpose = (0, 2, 1)
        y = back.get_output(0)
        if gated:
            sl = net.add_slice(g, (0, 0, 0), (1, T, C), (1, 1, 1))
            sl.set_input(2, net.add_shape(y).get_output(0))
            y = net.add_elementwise(sl.get_output(0), y, trt.ElementWiseOperation.PROD).get_output(0)
    net.mark_output(y)
    config = builder.create_builder_config()
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (batch, T, C), (batch, T, C), (batch, T, C))
    if gated:
        prof.set_shape("g", (batch, T, 2 * C), (batch, T, 2 * C), (batch, T, 2 * C))
    config.add_optimization_profile(prof)
    data = builder.build_serialized_network(net, config)
    if data is None:
        raise RuntimeError("build failed")
    return bytes(data)


def run(engine_bytes: bytes, inputs: dict, calls: int = 20, reps: int = 10) -> tuple[torch.Tensor, float]:
    """Output and microseconds per call (``calls`` executions captured in one CUDA graph)."""
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(engine_bytes)
    assert engine is not None, "engine did not deserialize"
    ctx = engine.create_execution_context()
    out = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(name, tuple(inputs[name].shape))
            ctx.set_tensor_address(name, inputs[name].data_ptr())
        else:
            out = torch.empty(tuple(ctx.get_tensor_shape(name)), dtype=torch.float16, device="cuda")
            ctx.set_tensor_address(name, out.data_ptr())
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) * 1000 / (calls * reps)


def reference(x: torch.Tensor, w: np.ndarray, bias: np.ndarray, g: torch.Tensor | None) -> torch.Tensor:
    wt = torch.tensor(w, device="cuda").float()[:, None]
    y = F.conv1d(
        x.float().transpose(1, 2), wt, torch.tensor(bias, device="cuda").float(), padding=K // 2, groups=C
    )
    y = y.transpose(1, 2)
    return y if g is None else y * g[..., :C].float()


def direct(
    cfg: dict,
    x: torch.Tensor,
    w: np.ndarray,
    bias: np.ndarray,
    g,
    cat: bool = False,
    sets: int = 1,
    calls: int = 24,
    reps: int = 10,
):
    """The kernel launched with the CUDA driver API (no TensorRT): output and microseconds per call.
    ``cat``: x given as two tensors (C // 2 + 2 channels and the rest). ``sets`` > 1: consecutive calls
    rotate over that many copies of the inputs and outputs, so that they find them in DRAM, not L2."""
    import ctypes

    from cuda.bindings import driver as cu

    def check(res):
        if res[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver: {res[0]}")
        return res[1] if len(res) > 1 else None

    ptx, _ = plugins.compile_ptx(K, gated=g is not None, cat=cat, **cfg)
    check(cu.cuInit(0))
    mod = check(cu.cuModuleLoadData(ptx.encode()))
    fn = check(cu.cuModuleGetFunction(mod, b"dwconv1d"))
    wt = torch.tensor(w.T.astype(np.float32), device="cuda").contiguous()
    bt = torch.tensor(bias.astype(np.float32), device="cuda")
    ptr, i32 = ctypes.c_void_p, ctypes.c_int
    c1 = C // 2 + 2
    keep, params, outs = [], [], []
    for i in range(sets):
        xi = x if i == 0 else x.clone()
        gi = g if g is None or i == 0 else g.clone()
        y = torch.empty_like(x)
        x1, x2 = xi[..., :c1].contiguous(), xi[..., c1:].contiguous()
        keep += [xi, gi, x1, x2]
        outs.append(y)
        if cat:
            p = (x1.data_ptr(), x2.data_ptr(), wt.data_ptr(), bt.data_ptr(), T, C, c1, y.data_ptr())
            params.append((p, (ptr, ptr, ptr, ptr, i32, i32, i32, ptr)))
        elif g is None:
            p = (xi.data_ptr(), wt.data_ptr(), bt.data_ptr(), T, C, y.data_ptr())
            params.append((p, (ptr, ptr, ptr, i32, i32, ptr)))
        else:
            p = (xi.data_ptr(), gi.data_ptr(), wt.data_ptr(), bt.data_ptr(), T, C, gi.shape[2], y.data_ptr())
            params.append((p, (ptr, ptr, ptr, ptr, i32, i32, i32, ptr)))
    v, nt = cfg["v"], cfg["nt"]
    grid = ((C // v + nt - 1) // nt, -(-T // (cfg["tt"] * cfg["nch"])), x.shape[0])

    def launch(i: int = 0):
        stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(fn, *grid, nt, 1, 1, 0, stream, params[i % sets], 0))

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            launch(i)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return outs[0], start.elapsed_time(end) * 1000 / (calls * reps)


class Toy(torch.nn.Module):
    """Both custom ops, for the ONNX route."""

    def __init__(self, w: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.register_buffer("w", torch.tensor(w.T.astype(np.float32)).contiguous())
        self.register_buffer("b", torch.tensor(bias.astype(np.float32)))

    def forward(self, x, g):
        y = torch.ops.owsm_trt.dwconv1d(x, self.w, self.b)
        return torch.ops.owsm_trt.dwconv1d_gated(y, g, self.w, self.b)


def onnx_route(w: np.ndarray, bias: np.ndarray, tmp: Path) -> None:
    toy = Toy(w, bias).cuda().eval()
    x = torch.randn(4, T, C, device="cuda", dtype=torch.float16)
    g = torch.randn(4, T, 2 * C, device="cuda", dtype=torch.float16)
    path = tmp / "toy.onnx"
    batch = torch.export.Dim("batch", min=1, max=16)
    torch.onnx.export(
        toy, (x, g), path, input_names=["x", "g"], output_names=["y"],
        dynamic_shapes={"x": {0: batch}, "g": {0: batch}}, dynamo=True,
        custom_translation_table=plugins.ONNX_TRANSLATIONS,
    )  # fmt: skip
    builder = trt.Builder(LOGGER)
    net = builder.create_network(network_flags())
    parser = trt.OnnxParser(net, LOGGER)
    if not parser.parse_from_file(str(path)):
        print("ONNX parse failed:", [str(parser.get_error(i)) for i in range(parser.num_errors)])
        return
    print("parsed layers:", [net.get_layer(i).type for i in range(net.num_layers)])
    config = builder.create_builder_config()
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (1, T, C), (4, T, C), (16, T, C))
    prof.set_shape("g", (1, T, 2 * C), (4, T, 2 * C), (16, T, 2 * C))
    config.add_optimization_profile(prof)
    data = builder.build_serialized_network(net, config)
    if data is None:
        print("ONNX route: build failed")
        return
    (tmp / "toy.engine").write_bytes(bytes(data))
    y, us = run(bytes(data), {"x": x, "g": g})
    with torch.no_grad():
        ref = toy(x, g).float()  # PyTorch implementation of the custom ops (fp16 between them)
    print(
        f"ONNX route: {us:.1f} us for both ops at B=4, max abs err vs PyTorch ops {float((y.float() - ref).abs().max()):.2e}"
    )
    torch.save({"x": x.cpu(), "g": g.cpu(), "ref": ref.cpu()}, tmp / "toy_io.pt")
    out = subprocess.run([sys.executable, __file__, "--child", str(tmp)], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip()[-2000:])


def child(tmp: Path) -> None:
    io = torch.load(tmp / "toy_io.pt")
    y, us = run((tmp / "toy.engine").read_bytes(), {"x": io["x"].cuda(), "g": io["g"].cuda()})
    err = float((y.float().cpu() - io["ref"]).abs().max())
    print(f"child (plugin never registered): loaded and ran, {us:.1f} us, max abs err {err:.2e}")


def cold(w: np.ndarray, bias: np.ndarray) -> None:
    """Each configuration at B=16 with its inputs in L2 (the same buffers every call) and in DRAM (calls
    rotating over 6 copies, ~0.5 GB): inside the encoder, the inputs come from the layers before."""
    x = torch.randn(16, T, C, device="cuda", dtype=torch.float16)
    g = torch.randn(16, T, 2 * C, device="cuda", dtype=torch.float16)
    extras = [
        dict(v=1, tt=16, nt=128, nch=1, minb=4),
        dict(v=1, tt=32, nt=128, nch=1, minb=4),
        dict(v=1, tt=8, nt=128, nch=1, minb=6),
        dict(v=2, tt=64, nt=128, nch=1, minb=3),
        dict(v=2, tt=16, nt=64, nch=1),
    ]
    for kind in ("plain", "gated", "cat"):
        ref = reference(x, w, bias, g if kind == "gated" else None)
        for cfg in list(plugins.TACTICS) + extras:
            try:
                args = (cfg, x, w, bias, g if kind == "gated" else None, kind == "cat")
                y, warm = direct(*args)
                _, cold_us = direct(*args, sets=6)
            except Exception:  # noqa: BLE001
                print(f"{kind} {cfg}: failed:\n{traceback.format_exc()}", flush=True)
                continue
            err = float((y.float() - ref).abs().max())
            print(
                f"B=16 {kind:5s} {cfg}: L2 {warm:6.1f} us, DRAM {cold_us:6.1f} us, max abs err {err:.2e}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cold", action="store_true", help="direct launches only, inputs in L2 and in DRAM")
    ap.add_argument("--child")
    args = ap.parse_args()
    if args.child:
        return child(Path(args.child))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((C, K)) * 0.1).astype(np.float16)
    bias = (rng.standard_normal(C) * 0.1).astype(np.float16)
    if args.cold:
        return cold(w, bias)
    plugins.override = dict(v=2, tt=32, nt=128, nch=1)  # replaced per configuration below
    plugins.register()
    tmp = Path(tempfile.mkdtemp())

    configs = [
        dict(v=2, tt=16, nt=128, nch=1),
        dict(v=2, tt=32, nt=128, nch=1),
        dict(v=2, tt=32, nt=64, nch=1),
        dict(v=2, tt=32, nt=256, nch=1),
        dict(v=2, tt=64, nt=128, nch=1),
        dict(v=2, tt=8, nt=128, nch=1),
        dict(v=2, tt=16, nt=128, nch=4),
        dict(v=2, tt=32, nt=128, nch=2),
        dict(v=1, tt=16, nt=128, nch=1),
        dict(v=1, tt=32, nt=128, nch=1),
        dict(v=1, tt=32, nt=256, nch=1),
        dict(v=1, tt=32, nt=128, nch=1, minb=4),
    ]
    print("== direct launches (CUDA driver API, no TensorRT)", flush=True)
    for batch in (16, 1):
        x = torch.randn(batch, T, C, device="cuda", dtype=torch.float16)
        g = torch.randn(batch, T, 2 * C, device="cuda", dtype=torch.float16)
        for kind in ("plain", "gated", "cat"):
            ref = reference(x, w, bias, g if kind == "gated" else None)
            for cfg in plugins.TACTICS:
                try:
                    y, us = direct(cfg, x, w, bias, g if kind == "gated" else None, cat=kind == "cat")
                except Exception:  # noqa: BLE001
                    print(f"B={batch} direct {kind} {cfg}: failed:\n{traceback.format_exc()}", flush=True)
                    continue
                err = float((y.float() - ref).abs().max())
                print(f"B={batch:2d} direct {kind} {cfg}: {us:7.1f} us, max abs err {err:.2e}", flush=True)
    if args.quick:
        configs = configs[:1]
    print("== TensorRT engines", flush=True)
    for batch in (16, 1):
        x = torch.randn(batch, T, C, device="cuda", dtype=torch.float16)
        g = torch.randn(batch, T, 2 * C, device="cuda", dtype=torch.float16)
        for gated in (False, True):
            ref = reference(x, w, bias, g if gated else None)
            inputs = {"x": x, "g": g}
            rows = [("TensorRT conv + transposes" + (" + slice-mul" if gated else ""), None)]
            rows += [(str(c), c) for c in (configs if not gated else configs[:3])]
            for name, cfg in rows:
                try:
                    if cfg is not None:
                        plugins.override = cfg
                    y, us = run(build("plugin" if cfg else "trt", batch, w, bias, gated), inputs)
                except Exception:  # noqa: BLE001
                    print(
                        f"B={batch} {'gated ' if gated else ''}{name}: failed:\n{traceback.format_exc()}",
                        flush=True,
                    )
                    continue
                err = float((y.float() - ref).abs().max())
                print(
                    f"B={batch:2d} {'gated ' if gated else ''}{name}: {us:7.1f} us per call, max abs err {err:.2e}",
                    flush=True,
                )
    plugins.override = dict(v=2, tt=32, nt=128, nch=1)
    try:
        onnx_route(w, bias, tmp)
    except Exception:  # noqa: BLE001
        print(f"ONNX route failed:\n{traceback.format_exc()[-3000:]}", flush=True)


if __name__ == "__main__":
    main()
