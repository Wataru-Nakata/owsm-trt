"""FP8 (E4M3) quantization of the encoder for TensorRT (experimental).

Linear layers and the convolutions of the subsampling get quantize/dequantize pairs on their input
(per-tensor scale from calibration) and weight (per-output-channel scale). Exported to ONNX as
QuantizeLinear/DequantizeLinear, they let TensorRT run those layers as FP8 GEMMs/convolutions.

The input ranges come from calibration audio (:func:`calibrate`), or from a scale file computed once
offline (:func:`save_scales`; the package ships some in ``owsm_trt/fp8_scales``, keyed by checkpoint).
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

E4M3_MAX = 448.0
SCALES_DIR = Path(__file__).parent / "fp8_scales"  # scale files shipped with the package
log = logging.getLogger(__name__)


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
    rows = calibrate_rows(root, run, batches)
    return {name: max(v) for name, v in rows.items()}


@torch.no_grad()
def calibrate_rows(root: nn.Module, run, batches) -> dict[str, list[float]]:
    """Largest absolute input of every quantizable layer under ``root`` per batch row (recording), in
    the order ``run(batch)`` sees them: the amax of any subset of the recordings is the max over it."""
    rows: dict[str, list[float]] = {}
    hooks = []
    for name, module in root.named_modules():
        if _quantizable(module):

            def hook(mod, args, name=name):
                x = args[0].detach()
                rows.setdefault(name, []).extend(x.abs().flatten(1).amax(1).float().tolist())

            hooks.append(module.register_forward_pre_hook(hook))
    try:
        for batch in batches:
            run(batch)
    finally:
        for h in hooks:
            h.remove()
    return rows


def quantize(root: nn.Module, amax: dict[str, float], margin: float = 1.0, skip: str = "") -> int:
    """Replace the calibrated layers under ``root`` by their FP8 versions; returns how many. ``margin``
    multiplies the input ranges; layers whose name matches the regular expression ``skip`` stay fp16."""
    n = 0
    for name, a in amax.items():
        if skip and re.search(skip, name):
            continue
        a = a * margin
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


# --- scale files ------------------------------------------------------------------------------------


def save_scales(
    path: str | os.PathLike, amax: dict[str, float], model: str, checkpoint: str, about: dict
) -> None:
    """A scale file: the input ranges of the encoder's quantizable layers for one checkpoint (its sha256)."""
    data = {"format": 1, "model": model, "checkpoint_sha256": checkpoint, "calibration": about,
            "amax": {k: float(v) for k, v in sorted(amax.items())}}  # fmt: skip
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def load_scales(path: str | os.PathLike) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("format") != 1 or not isinstance(data.get("amax"), dict):
        raise ValueError(f"{path} is not an owsm_trt FP8 scale file")
    return data


def scales_digest(path: str | os.PathLike) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def packaged_scales(checkpoint: str) -> Path | None:
    """The scale file shipped with the package for the checkpoint with this sha256, if any."""
    for p in sorted(SCALES_DIR.glob("*.json")):
        try:
            if load_scales(p).get("checkpoint_sha256") == checkpoint:
                return p
        except (ValueError, json.JSONDecodeError):
            log.warning("ignoring the malformed FP8 scale file %s", p)
    return None


def checkpoint_sha256(path: str | os.PathLike, cache: Path | None = None) -> str:
    """The checkpoint's sha256: the blob name of a Hugging Face cache file (free), else hashed (once per
    file: remembered in ``cache`` by path, size and mtime)."""
    real = Path(os.path.realpath(path))
    if real.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", real.name):
        return real.name
    st = real.stat()
    key = f"{real}:{st.st_size}:{st.st_mtime_ns}"
    known = {}
    if cache is not None and cache.exists():
        try:
            known = json.loads(cache.read_text())
        except json.JSONDecodeError:
            known = {}
    if key in known:
        return known[key]
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    known[key] = h.hexdigest()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=0))
    return known[key]
