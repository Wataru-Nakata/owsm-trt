import pytest
import torch
from torch import nn

from owsm_trt import plugins
from owsm_trt.modeling import DepthwiseConv1d


class _Net(nn.Module):
    """The uses in the E-Branchformer: x + conv(cat(x1, x2)) (merge), gate[..., :C] * conv(x) (CSGU), and
    a plain convolution."""

    def __init__(self, c: int, k: int):
        super().__init__()
        self.a, self.b, self.c = (DepthwiseConv1d(c, k) for _ in range(3))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self.c(self.b(self.a((x1, x2), residual=True), gate=gate))


def _plugin_mode(net: nn.Module, *args, cat: bool = True) -> torch.Tensor:
    DepthwiseConv1d.plugin, DepthwiseConv1d.plugin_cat = True, cat
    try:
        return net(*args)
    finally:
        DepthwiseConv1d.plugin, DepthwiseConv1d.plugin_cat = False, True


@torch.no_grad()
def test_plugin_ops_match_conv():
    torch.manual_seed(0)
    net = _Net(24, 7).eval()
    x1, x2, gate = torch.randn(2, 30, 10), torch.randn(2, 30, 14), torch.randn(2, 30, 48)
    ref = net(x1, x2, gate)
    for cat in (True, False):
        assert torch.allclose(_plugin_mode(net, x1, x2, gate, cat=cat), ref, atol=1e-5)
    x = torch.cat([x1, x2], -1)
    x_r, x_g = gate.chunk(2, dim=-1)
    assert torch.equal(net.b(x_g, gate=gate), x_r * net.b(x_g))
    assert torch.equal(net.a((x1, x2), residual=True), x + net.a(x))


@pytest.mark.skipif(not plugins.available(), reason="needs CUDA, TensorRT's plugin API and NVRTC")
@pytest.mark.parametrize("c,k", [(96, 31), (75, 7)])  # odd channel count: one channel per thread
def test_tensorrt_plugin(tmp_path, c, k):
    import tensorrt as trt

    from owsm_trt.build import BuildOptions, _build

    torch.manual_seed(0)
    net = _Net(c, k).cuda().half().eval()
    for m in (net.a, net.b, net.c):
        nn.init.normal_(m.weight, std=0.2)
    T, c1 = 50, c // 2 + 2  # c = 96: two channels per thread in each part
    x1 = torch.randn(3, T, c1, device="cuda", dtype=torch.float16)
    x2 = torch.randn(3, T, c - c1, device="cuda", dtype=torch.float16)
    gate = torch.randn(3, T, 2 * c, device="cuda", dtype=torch.float16)
    plugins.register()
    batch = torch.export.Dim("batch", min=1, max=4)
    DepthwiseConv1d.plugin = True
    try:
        torch.onnx.export(
            net, (x1, x2, gate), tmp_path / "net.onnx", input_names=["x1", "x2", "gate"], output_names=["y"],
            dynamic_shapes={"x1": {0: batch}, "x2": {0: batch}, "gate": {0: batch}}, dynamo=True,
            custom_translation_table=plugins.ONNX_TRANSLATIONS,
        )  # fmt: skip
    finally:
        DepthwiseConv1d.plugin = False
    inputs = {"x1": x1, "x2": x2, "gate": gate}
    profile = {n: ((1, *t.shape[1:]), tuple(t.shape), (4, *t.shape[1:])) for n, t in inputs.items()}
    _build(tmp_path / "net.onnx", tmp_path / "net.engine", [profile], BuildOptions())
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine((tmp_path / "net.engine").read_bytes())
    layers = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    assert "dwconv1d" in layers  # the plugin, not TensorRT's convolution
    ctx = engine.create_execution_context()
    y = torch.empty(3, T, c, device="cuda", dtype=torch.float16)
    for name, t in inputs.items():
        ctx.set_input_shape(name, tuple(t.shape))
        ctx.set_tensor_address(name, t.data_ptr())
    ctx.set_tensor_address("y", y.data_ptr())
    stream = torch.cuda.current_stream()
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    with torch.no_grad():
        ref = net.float()(x1.float(), x2.float(), gate.float())
    torch.testing.assert_close(y.float(), ref, atol=2e-2, rtol=1e-2)
