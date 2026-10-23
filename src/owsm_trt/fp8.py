"""FP8 (E4M3) quantization of the encoder for TensorRT (experimental).

Linear layers and the convolutions of the subsampling get quantize/dequantize pairs on their input
(per-tensor scale from calibration) and weight (per-output-channel scale). Exported to ONNX as
QuantizeLinear/DequantizeLinear, they let TensorRT run those layers as FP8 GEMMs/convolutions.
"""

import torch
import torch.nn.functional as F
from torch import nn

E4M3_MAX = 448.0


@torch.library.custom_op("owsm_trt::fp8_qdq", mutates_args=())
def fp8_qdq(x: torch.Tensor, scale: torch.Tensor, axis: int) -> torch.Tensor:
    """Round ``x / scale`` to FP8 E4M3 and back. ``scale`` is a scalar, or per slice along ``axis``."""
    if scale.dim() == 1:
        shape = [1] * x.dim()
        shape[axis] = -1
        scale = scale.view(shape)
    return ((x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(x.dtype) * scale).contiguous()


@fp8_qdq.register_fake
def _(x: torch.Tensor, scale: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.empty_like(x)


def _onnx_fp8_qdq(x, scale, axis: int):
    import onnx
    from onnxscript import opset20 as op

    zero = op.Constant(value=onnx.helper.make_tensor("zero", onnx.TensorProto.FLOAT8E4M3FN, [], [0.0]))
    if len(scale.shape) == 1:
        zero = op.Expand(zero, op.Shape(scale))
        q = op.QuantizeLinear(x, scale, zero, axis=axis)
        return op.DequantizeLinear(q, scale, zero, axis=axis)
    return op.DequantizeLinear(op.QuantizeLinear(x, scale, zero), scale, zero)


ONNX_TRANSLATIONS = {torch.ops.owsm_trt.fp8_qdq.default: _onnx_fp8_qdq}


class QLinear(nn.Module):
    """``nn.Linear`` with FP8 input (per tensor) and weight (per output channel)."""

    def __init__(self, lin: nn.Linear, input_amax: float):
        super().__init__()
        self.weight, self.bias = lin.weight, lin.bias
        w_amax = lin.weight.detach().abs().amax(dim=1).float().clamp(min=1e-8)
        self.register_buffer("w_scale", (w_amax / E4M3_MAX).to(lin.weight.dtype))
        self.register_buffer(
            "x_scale", torch.tensor(max(input_amax, 1e-8) / E4M3_MAX, dtype=lin.weight.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(fp8_qdq(x, self.x_scale, 0), fp8_qdq(self.weight, self.w_scale, 0), self.bias)


class QConv2d(nn.Module):
    """``nn.Conv2d`` with FP8 input (per tensor) and weight (per output channel)."""

    def __init__(self, conv: nn.Conv2d, input_amax: float):
        super().__init__()
        self.conv = conv
        w_amax = conv.weight.detach().abs().amax(dim=(1, 2, 3)).float().clamp(min=1e-8)
        self.register_buffer("w_scale", (w_amax / E4M3_MAX).to(conv.weight.dtype))
        self.register_buffer(
            "x_scale", torch.tensor(max(input_amax, 1e-8) / E4M3_MAX, dtype=conv.weight.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv
        w = fp8_qdq(c.weight, self.w_scale, 0)
        return F.conv2d(fp8_qdq(x, self.x_scale, 0), w, c.bias, c.stride, c.padding, c.dilation, c.groups)


def _quantizable(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    # the subsampling's d -> d convolutions (not the 1-channel input one, not depthwise ones)
    return isinstance(module, nn.Conv2d) and module.in_channels > 1 and module.groups == 1


@torch.no_grad()
def calibrate(root: nn.Module, run, batches) -> dict[str, float]:
    """Largest absolute input of every quantizable layer under ``root`` while ``run(batch)`` runs."""
    amax: dict[str, float] = {}
    hooks = []
    for name, module in root.named_modules():
        if _quantizable(module):

            def hook(mod, args, name=name):
                amax[name] = max(amax.get(name, 0.0), float(args[0].detach().abs().max()))

            hooks.append(module.register_forward_pre_hook(hook))
    try:
        for batch in batches:
            run(batch)
    finally:
        for h in hooks:
            h.remove()
    return amax


def quantize(root: nn.Module, amax: dict[str, float]) -> int:
    """Replace the calibrated layers under ``root`` by their FP8 versions; returns how many."""
    n = 0
    for name, a in amax.items():
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        module = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
        q = QLinear(module, a) if isinstance(module, nn.Linear) else QConv2d(module, a)
        if attr.isdigit():
            parent[int(attr)] = q
        else:
            setattr(parent, attr, q)
        n += 1
    return n
