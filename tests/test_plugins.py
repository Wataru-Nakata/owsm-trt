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


def test_padded_vocab_is_exact():
    """The CTC export pads the vocabulary: tokens that never win and add nothing to the conditioning."""
    import dataclasses

    from test_model import TINY

    from owsm_trt.build import _padded_vocab
    from owsm_trt.modeling import CTC, OWSMCTCModel

    cfg = dataclasses.replace(
        TINY, dec_layers=0, dec_ffn=0, dec_heads=0, ctc_only=True, prompt_d=16, prompt_heads=2,
        prompt_ffn=32, prompt_layers=2, interctc_layers=(1, 2), cross_attn_layers=(1,),
    )  # fmt: skip
    torch.manual_seed(3)
    m = OWSMCTCModel(cfg).eval().requires_grad_(False)
    m.frontend.melmat.uniform_(0, 1)
    with torch.no_grad():
        m.ctc.ctc_lo.weight.mul_(20)  # confident posteriors, as a trained model's
    feats = m.frontend(torch.randn(3, cfg.num_samples))
    args = feats, torch.tensor([[2]] * 3), torch.tensor([[4, 5]] * 3)
    prefix = m.embed_proj(m.embed(args[2]))
    memory = m.prompt_proj(m.prompt_encoder(m.embed(args[1])))
    ref, enc = m(*args), m.encoder(feats, prefix, memory, m.ctc)
    with _padded_vocab(m, 64):
        assert m.ctc.ctc_lo.weight.shape[0] == 64 and m.encoder.conditioning_layer.weight.shape[1] == 64
        assert torch.equal(m(*args), ref)
        torch.testing.assert_close(m.encoder(feats, prefix, memory, m.ctc), enc, rtol=0, atol=1e-5)
        assert (m.ctc.ctc_lo(enc).float().softmax(-1)[..., cfg.vocab_size :] == 0).all()
        CTC.plugin = True  # the plugin ops (here their PyTorch implementations) on time-major rows
        try:
            assert torch.equal(m(*args), ref)
            torch.testing.assert_close(m.encoder(feats, prefix, memory, m.ctc), enc, rtol=0, atol=1e-5)
        finally:
            CTC.plugin = False
    assert m.ctc.ctc_lo.weight.shape[0] == cfg.vocab_size
    assert m.encoder.conditioning_layer.weight.shape[1] == cfg.vocab_size
    # FP8 encoders (opt-in): the conditioning layer is an fp8.QLinear
    from owsm_trt import fp8

    def run(f):
        return m(f, *args[1:])

    fp8.quantize(m.encoder, fp8.calibrate(m.encoder, run, [feats]))
    assert isinstance(m.encoder.conditioning_layer, fp8.QLinear)
    ref, enc = m(*args), m.encoder(feats, prefix, memory, m.ctc)
    with _padded_vocab(m, 64):
        assert isinstance(m.encoder.conditioning_layer, fp8.QLinear)
        assert torch.equal(m(*args), ref)
        torch.testing.assert_close(m.encoder(feats, prefix, memory, m.ctc), enc, rtol=0, atol=1e-5)


@pytest.mark.skipif(not plugins.available(), reason="needs CUDA, TensorRT's plugin API and NVRTC")
@pytest.mark.parametrize("tactic", [None, 1, 4])  # None: TensorRT's pick
def test_ctc_row_plugins(tmp_path, tactic, monkeypatch):
    """softmax_rows / argmax_rows as TensorRT plugins against PyTorch, ties included."""
    import tensorrt as trt

    monkeypatch.setattr(plugins, "rows_override", tactic)

    from owsm_trt.build import BuildOptions, _build

    class Net(nn.Module):
        def forward(self, x):
            return plugins.softmax_rows(x), plugins.argmax_rows(x)

    torch.manual_seed(0)
    B, T, V = 3, 7, 1000
    x = (torch.randn(B, T, V, device="cuda") * 4).half()
    x[0, 0, 10] = x[0, 0, 900] = 30.0  # a tie: the first index wins
    x[1, 2] = -3e4
    x[1, 2, 999] = 0.0
    plugins.register_ctc()
    batch = torch.export.Dim("batch", min=1, max=4)
    time = torch.export.Dim("time", min=1, max=16)
    torch.onnx.export(
        Net(), (x,), tmp_path / "net.onnx", input_names=["x"], output_names=["p", "i"],
        dynamic_shapes={"x": {0: batch, 1: time}}, dynamo=True,
        custom_translation_table=plugins.CTC_ONNX_TRANSLATIONS,
    )  # fmt: skip
    _build(
        tmp_path / "net.onnx",
        tmp_path / "net.engine",
        [{"x": ((1, 1, V), (B, T, V), (4, 16, V))}],
        BuildOptions(),
    )
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine((tmp_path / "net.engine").read_bytes())
    layers = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    assert "softmax_rows" in layers and "argmax_rows" in layers
    ctx = engine.create_execution_context()
    p = torch.empty(B, T, V, device="cuda", dtype=torch.float16)
    i = torch.empty(B, T, device="cuda", dtype=torch.int32)
    ctx.set_input_shape("x", tuple(x.shape))
    for name, t in (("x", x), ("p", p), ("i", i)):
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.current_stream()
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(i, x.argmax(-1).int()) and i[0, 0] == 10 and i[1, 2] == 999
    ref = x.float().softmax(-1)
    torch.testing.assert_close(p.float(), ref, rtol=1e-3, atol=1e-6)
    assert (p[1, 2, :999] == 0).all() and p[1, 2, 999] == 1


def test_frozen_plugin_weights_are_graph_constants(tmp_path):
    """Frozen (as by the encoder export), the plugins' taps and biases are initializers of the ONNX
    graph: TensorRT computed them from the parameters on every call otherwise (one kernel per plugin)."""
    pytest.importorskip("onnx")
    import onnx

    torch.manual_seed(0)
    net = _Net(2048, 31).eval()  # sizes of OWSM v4 1B (the exporter itself folds small constants)
    x1, x2, gate = torch.randn(2, 40, 1000), torch.randn(2, 40, 1048), torch.randn(2, 40, 4096)
    ref = _plugin_mode(net, x1, x2, gate)
    for m in (net.a, net.b, net.c):
        m.freeze_plugin_weights()
    assert torch.equal(_plugin_mode(net, x1, x2, gate), ref)
    DepthwiseConv1d.plugin = True
    try:
        torch.onnx.export(
            net, (x1, x2, gate), tmp_path / "net.onnx", input_names=["x1", "x2", "gate"], output_names=["y"],
            dynamo=True, custom_translation_table=plugins.ONNX_TRANSLATIONS,
        )  # fmt: skip
    finally:
        DepthwiseConv1d.plugin = False
        for m in (net.a, net.b, net.c):
            m.freeze_plugin_weights(False)
    graph = onnx.load(str(tmp_path / "net.onnx")).graph
    constants = {t.name for t in graph.initializer}
    constants |= {o for n in graph.node if n.op_type == "Constant" for o in n.output}
    nodes = [n for n in graph.node if n.op_type.startswith("dwconv1d")]
    assert len(nodes) == 3
    for n in nodes:
        assert set(n.input[-2:]) <= constants, (n.op_type, n.input, [x.op_type for x in graph.node])
    assert not hasattr(net.a, "_plugin_frozen") or net.a._plugin_frozen is None


def test_fusable_silu():
    """The SiLU form the encoder export uses (TensorRT fuses it into the GEMM): the same function."""
    import torch.nn.functional as F

    from owsm_trt.modeling import FusableSiLU

    x = torch.cat([torch.linspace(-200, 200, 40001), torch.tensor([0.0, -0.0, 1e-30, -1e-30])])
    torch.testing.assert_close(FusableSiLU()(x), F.silu(x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        FusableSiLU()(x.half()).float(), F.silu(x.half()).float(), rtol=1e-3, atol=1e-3
    )


def test_conv1_plugin_op_matches_subsampling():
    """Conv2dSubsampling with its first convolution + ReLU as the conv1_relu op (its PyTorch implementation),
    weights frozen or not: the same output."""
    from test_model import TINY

    from owsm_trt.modeling import Conv2dSubsampling, OWSMModel

    torch.manual_seed(0)
    embed = OWSMModel(TINY).eval().requires_grad_(False).encoder.embed
    x = torch.randn(2, 101, TINY.n_mels)
    ref = embed(x)
    Conv2dSubsampling.plugin = True
    try:
        torch.testing.assert_close(embed(x), ref, rtol=1e-5, atol=1e-5)
        embed.freeze_plugin_weights()
        torch.testing.assert_close(embed(x), ref, rtol=1e-5, atol=1e-5)
    finally:
        Conv2dSubsampling.plugin = False
        embed.freeze_plugin_weights(False)


@pytest.mark.skipif(not plugins.available(), reason="needs CUDA, TensorRT's plugin API and NVRTC")
@pytest.mark.parametrize("c", [64, 1024])
def test_conv1_tensorrt_plugin(tmp_path, c):
    """conv1_relu as a TensorRT plugin (HWC8 output) feeding TensorRT's convolution, against PyTorch."""
    import tensorrt as trt

    from owsm_trt.build import BuildOptions, _build

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2 = nn.Conv2d(1, c, 3, 2), nn.Conv2d(c, c, 3, 2)

        def forward(self, x):
            w = self.c1.weight.float().reshape(c, 9).t().contiguous()
            return self.c2(plugins.conv1_relu(x, w, self.c1.bias.float()))

    torch.manual_seed(0)
    net = Net().cuda().half().eval()
    x = torch.randn(3, 301, 80, device="cuda", dtype=torch.float16)
    plugins.register_conv1()
    batch, time = torch.export.Dim("batch", min=1, max=4), torch.export.Dim("time", min=64, max=512)
    torch.onnx.export(
        net, (x,), tmp_path / "net.onnx", input_names=["x"], output_names=["y"],
        dynamic_shapes={"x": {0: batch, 1: time}}, dynamo=True,
        custom_translation_table=plugins.CONV1_ONNX_TRANSLATIONS,
    )  # fmt: skip
    _build(
        tmp_path / "net.onnx",
        tmp_path / "net.engine",
        [{"x": ((1, 64, 80), (3, 301, 80), (4, 512, 80))}],
        BuildOptions(),
    )
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine((tmp_path / "net.engine").read_bytes())
    layers = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    assert "conv1_relu" in layers
    ctx = engine.create_execution_context()
    ctx.set_input_shape("x", tuple(x.shape))
    y = torch.empty(tuple(ctx.get_tensor_shape("y")), device="cuda", dtype=torch.float16)
    for name, t in (("x", x), ("y", y)):
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.current_stream()
    assert ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    with torch.no_grad():
        ref = net.c2.float()(torch.relu(net.c1.float()(x.float().unsqueeze(1))))
    torch.testing.assert_close(y.float(), ref, rtol=2e-2, atol=2e-2)
