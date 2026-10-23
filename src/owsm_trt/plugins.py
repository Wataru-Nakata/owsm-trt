"""The E-Branchformer depthwise convolution over time as a TensorRT plugin (CUDA C, compiled with NVRTC).

TensorRT runs ``DepthwiseConv1d`` on [B, T, C] activations as a grouped convolution in [B, C, T], with
transposes before and after it that cost twice the convolution. :func:`DWCONV_SOURCE` computes it in
[B, T, C] directly: each thread owns ``V`` adjacent channels and ``TT`` consecutive output frames, holds
the channels' ``K`` taps in registers, loads the ``TT + K - 1`` input frames once and accumulates in fp32.
"""

import functools
import logging
import traceback
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

DWCONV_SOURCE = r"""
// x [B, T, C] fp16, w [K, C] fp32, bias [C] fp32 -> y [B, T, C] fp16 (each thread: V channels, NCH chunks
// of TT frames). GATED: y = g[b, t, c] * conv, g [B, T, G] fp16 (its first C channels). CAT: the input is
// x [B, T, C1] and x2 [B, T, C - C1] concatenated along channels.
#ifndef GATED
#define GATED 0
#endif
#ifndef CAT
#define CAT 0
#endif
#ifndef MINB
#define MINB 1
#endif
#ifndef ILP
#define ILP 1
#endif
typedef unsigned short u16;
__device__ __forceinline__ float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }

template <int N> struct Vec;
template <> struct Vec<1> { typedef u16 T; };
template <> struct Vec<2> { typedef unsigned T; };
template <> struct Vec<4> { typedef uint2 T; };
template <> struct Vec<8> { typedef uint4 T; };

template <int N> __device__ __forceinline__ void unpack(const typename Vec<N>::T& v, float* f);
template <> __device__ __forceinline__ void unpack<1>(const u16& v, float* f) { f[0] = h2f(v); }
template <> __device__ __forceinline__ void unpack<2>(const unsigned& v, float* f) {
  f[0] = h2f((u16)(v & 0xffff)); f[1] = h2f((u16)(v >> 16)); }
template <> __device__ __forceinline__ void unpack<4>(const uint2& v, float* f) {
  unpack<2>(v.x, f); unpack<2>(v.y, f + 2); }
template <> __device__ __forceinline__ void unpack<8>(const uint4& v, float* f) {
  unpack<2>(v.x, f); unpack<2>(v.y, f + 2); unpack<2>(v.z, f + 4); unpack<2>(v.w, f + 6); }

template <int N> __device__ __forceinline__ typename Vec<N>::T pack(const float* f);
template <> __device__ __forceinline__ u16 pack<1>(const float* f) { return f2h(f[0]); }
template <> __device__ __forceinline__ unsigned pack<2>(const float* f) {
  return (unsigned)f2h(f[0]) | ((unsigned)f2h(f[1]) << 16); }
template <> __device__ __forceinline__ uint2 pack<4>(const float* f) { return make_uint2(pack<2>(f), pack<2>(f + 2)); }
template <> __device__ __forceinline__ uint4 pack<8>(const float* f) {
  return make_uint4(pack<2>(f), pack<2>(f + 2), pack<2>(f + 4), pack<2>(f + 6)); }

typedef typename Vec<V>::T vec;

extern "C" __global__ void __launch_bounds__(NT, MINB) dwconv1d(
    const vec* __restrict__ x,
#if GATED
    const vec* __restrict__ g,
#endif
#if CAT
    const vec* __restrict__ x2,
#endif
    const float* __restrict__ w, const float* __restrict__ bias, int T, int C,
#if GATED
    int G,
#endif
#if CAT
    int C1,
#endif
    vec* __restrict__ y) {
  const int P = (K - 1) / 2;
  const int CV = C / V;  // vectors per frame
  const int cv = blockIdx.x * NT + threadIdx.x;
  if (cv >= CV) return;
  const int c = cv * V;
  const long frame0 = (long)blockIdx.z * T;
  const vec* __restrict__ src = x;  // this thread's input: vector sv of SV per frame
  int SV = CV, sv = cv;
#if CAT
  if (c < C1) SV = C1 / V; else { src = x2; SV = (C - C1) / V; sv = cv - C1 / V; }
#endif
  float wr[K][V], b[V];
#pragma unroll
  for (int k = 0; k < K; k++)
#pragma unroll
    for (int j = 0; j < V; j++) wr[k][j] = w[k * C + c + j];
#pragma unroll
  for (int j = 0; j < V; j++) b[j] = bias[c + j];
  for (int ch = 0; ch < NCH; ch++) {
    const int t0 = (blockIdx.y * NCH + ch) * TT;
    if (t0 >= T) break;
#if GATED  // loaded up front: issued at the end, the loads' latency is exposed
    vec gv[TT];
#pragma unroll
    for (int i = 0; i < TT; i++) {
      if (t0 + i < T) gv[i] = g[(frame0 + t0 + i) * (G / V) + cv]; else gv[i] = vec();
    }
#endif
    float a[TT + K - 1][V];
#pragma unroll
    for (int i = 0; i < TT + K - 1; i++) {
      const int t = t0 - P + i;
      vec v;
      if (t >= 0 && t < T) v = src[(frame0 + t) * SV + sv]; else v = vec();
      unpack<V>(v, a[i]);
    }
    // ILP outputs at a time, their taps interleaved: ILP * V independent chains of multiply-adds, each
    // summed in the same order as one output at a time (the compiler would not interleave them itself)
#pragma unroll
    for (int i0 = 0; i0 < TT; i0 += ILP) {
      float s[ILP][V];
#pragma unroll
      for (int u = 0; u < ILP; u++)
#pragma unroll
        for (int j = 0; j < V; j++) s[u][j] = b[j];
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
        for (int u = 0; u < ILP; u++)
#pragma unroll
          for (int j = 0; j < V; j++) s[u][j] = fmaf(wr[k][j], a[i0 + u + k][j], s[u][j]);
#pragma unroll
      for (int u = 0; u < ILP; u++) {
        const int i = i0 + u;
        if (t0 + i < T) {
#if GATED
          float gf[V];
          unpack<V>(gv[i], gf);
#pragma unroll
          for (int j = 0; j < V; j++) s[u][j] *= gf[j];
#endif
          y[(frame0 + t0 + i) * CV + cv] = pack<V>(s[u]);
        }
      }
    }
  }
}
"""


@functools.lru_cache(maxsize=None)
def compile_ptx(
    k: int, v: int = 2, tt: int = 32, nt: int = 128, nch: int = 1, minb: int = 1, ilp: int = 1,
    gated: bool = False, cat: bool = False, arch: str | None = None, cubin_log: bool = False,
) -> tuple[str, str]:  # fmt: skip
    """PTX of the kernel ``dwconv1d`` for one configuration (and the compile log). ``arch``: e.g.
    ``compute_90`` (default: the current GPU's); ``sm_90`` compiles to a cubin, for the ptxas log only."""
    from cuda.bindings import nvrtc

    def check(res):
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC: {res[0]}")
        return res[1] if len(res) > 1 else None

    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"compute_{major}{minor}"
    prog = check(nvrtc.nvrtcCreateProgram(DWCONV_SOURCE.encode(), b"dwconv1d.cu", 0, [], []))
    if tt % ilp:
        raise ValueError(f"tt={tt} is not a multiple of ilp={ilp}")
    defs = dict(K=k, V=v, TT=tt, NT=nt, NCH=nch, MINB=minb, ILP=ilp, GATED=int(gated), CAT=int(cat))
    opts = [f"-D{n}={x}".encode() for n, x in defs.items()]
    opts += [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
    if cubin_log:
        opts.append(b"--ptxas-options=-v")
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    log = b" " * check(nvrtc.nvrtcGetProgramLogSize(prog))
    check(nvrtc.nvrtcGetProgramLog(prog, log))
    log = log.decode(errors="replace").rstrip("\x00").strip()
    if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC compile failed: {log}")
    if arch.startswith("sm_"):
        return "", log
    ptx = b" " * check(nvrtc.nvrtcGetPTXSize(prog))
    check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)
    return ptx.decode().rstrip("\x00"), log


# --- PyTorch ops standing for the plugin in the exported graph ---------------------------------------


@torch.library.custom_op("owsm_trt::dwconv1d", mutates_args=())
def dwconv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Depthwise convolution over time, 'same' padding: x [B, T, C], weight [K, C] (fp32), bias [C] (fp32)."""
    k, c = weight.shape
    y = F.conv1d(x.transpose(1, 2).float(), weight.t().unsqueeze(1), bias, padding=(k - 1) // 2, groups=c)
    return y.transpose(1, 2).to(x.dtype).contiguous()


@dwconv1d.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


@torch.library.custom_op("owsm_trt::dwconv1d_gated", mutates_args=())
def dwconv1d_gated(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``gate[..., :C] * dwconv1d(x, weight, bias)``, in fp32 before the one rounding to x.dtype."""
    k, c = weight.shape
    y = F.conv1d(x.transpose(1, 2).float(), weight.t().unsqueeze(1), bias, padding=(k - 1) // 2, groups=c)
    return (gate[..., :c].float() * y.transpose(1, 2)).to(x.dtype).contiguous()


@dwconv1d_gated.register_fake
def _(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


@torch.library.custom_op("owsm_trt::dwconv1d_cat", mutates_args=())
def dwconv1d_cat(x: torch.Tensor, x2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``dwconv1d(torch.cat([x, x2], -1), weight, bias)``."""
    return dwconv1d(torch.cat([x, x2], -1), weight, bias)


@dwconv1d_cat.register_fake
def _(x: torch.Tensor, x2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[1])


def _onnx_op(name: str, inputs: tuple[str, ...]):
    """Translation of a custom op to an ONNX node TensorRT's parser turns into the plugin of that name."""

    def translate(*args):
        from onnx.defs import OpSchema
        from onnxscript import values

        param = OpSchema.FormalParameter
        schema = OpSchema(
            name,
            NAMESPACE,
            1,
            inputs=[param(n, "W" if n in ("weight", "bias") else "T") for n in inputs],
            outputs=[param("y", "T")],
            type_constraints=[
                ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
                ("W", ["tensor(float)"], ""),
            ],  # fmt: skip
            attributes=[
                OpSchema.Attribute(a, OpSchema.AttrType.STRING, "")
                for a in ("plugin_namespace", "plugin_version")
            ],
        )
        op = values.Op(values.Opset(NAMESPACE, 1), name, schema)
        return op(*args, plugin_namespace=NAMESPACE, plugin_version="1")

    return translate


NAMESPACE = "owsm"
ONNX_TRANSLATIONS = {
    torch.ops.owsm_trt.dwconv1d.default: _onnx_op("dwconv1d", ("x", "weight", "bias")),
    torch.ops.owsm_trt.dwconv1d_gated.default: _onnx_op("dwconv1d_gated", ("x", "gate", "weight", "bias")),
    torch.ops.owsm_trt.dwconv1d_cat.default: _onnx_op("dwconv1d_cat", ("x", "x2", "weight", "bias")),
}


# --- TensorRT plugins ---------------------------------------------------------------------------------

# Kernel configurations TensorRT times for every optimization profile (tactic i + 1 = TACTICS[i]): channels
# per thread, frames per thread, threads per block, frame chunks per thread. Alone, at B=16, T=374, C=2048,
# K=31 on an H200: 37-40 us for tt=16..64 (TensorRT's convolution with its transposes: 160 us); at B=1
# 4.8 us for tt=8, 5.0 for tt=16 (TensorRT: 20 us).
TACTICS = (
    dict(v=2, tt=8, nt=128, nch=1),
    dict(v=2, tt=16, nt=128, nch=1),
    dict(v=2, tt=32, nt=128, nch=1),
    dict(v=2, tt=64, nt=128, nch=1),
    dict(v=2, tt=32, nt=128, nch=2),
)
override: dict | None = None  # one configuration instead of TACTICS (benchmarks)
_registered = False


def available() -> bool:
    """Whether the plugins can be built here (TensorRT's Python plugin API, NVRTC, a GPU)."""
    try:
        import tensorrt.plugin  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _aot(trtp, kind: str, x, x2, w, tactic: int):
    """Kernel and launch of the ``kind`` plugin (``x2``: the gate or the second input, if any)."""
    try:
        cfg = dict(override or TACTICS[max(tactic, 1) - 1])
        k, c = int(w.shape[0]), int(w.shape[1])
        c2 = int(x2.shape[2]) if x2 is not None else 0
        if c % cfg["v"] or c2 % cfg["v"]:
            cfg["v"] = 1
        ptx, _ = compile_ptx(k, gated=kind == "gated", cat=kind == "cat", **cfg)
        v, nt = cfg["v"], cfg["nt"]
        launch = trtp.KernelLaunchParams()
        launch.grid_x = (c // v + nt - 1) // nt
        launch.grid_y = trtp.cdiv(x.shape_expr[1], cfg["tt"] * cfg["nch"])
        launch.grid_z = x.shape_expr[0]
        launch.block_x = nt
        extra = trtp.SymIntExprs(2 if x2 is None else 3)
        extra[0] = trtp.SymInt32(x.shape_expr[1])
        extra[1] = trtp.SymInt32(c)
        if x2 is not None:  # the gate's channels / the first input's
            extra[2] = trtp.SymInt32(c2 if kind == "gated" else c - c2)
        return "dwconv1d", ptx, launch, extra
    except Exception:  # TensorRT reports only "launch param computation failed"
        log.error("%s plugin:\n%s", kind, traceback.format_exc())
        raise


def register() -> None:
    """Register ``owsm::dwconv1d[_gated|_cat]`` with TensorRT (for building engines; running them needs nothing:
    the kernels are compiled into the engine)."""
    global _registered
    if _registered:
        return
    import tensorrt.plugin as trtp

    def tactics() -> list[int]:
        return list(range(1, (1 if override else len(TACTICS)) + 1))

    @trtp.register(f"{NAMESPACE}::dwconv1d")
    def _desc(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        return x.like()

    @trtp.autotune(f"{NAMESPACE}::dwconv1d")
    def _tune(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]
              ) -> List[trtp.AutoTuneCombination]:  # fmt: skip
        return [trtp.AutoTuneCombination("FP16, FP32, FP32, FP16", "LINEAR", tactics())]

    @trtp.aot_impl(f"{NAMESPACE}::dwconv1d")
    def _impl(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
              tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        return _aot(trtp, "plain", x, None, w, tactic)

    @trtp.register(f"{NAMESPACE}::dwconv1d_gated")
    def _desc_g(x: trtp.TensorDesc, g: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc
                ) -> Tuple[trtp.TensorDesc]:  # fmt: skip
        return x.like()

    @trtp.autotune(f"{NAMESPACE}::dwconv1d_gated")
    def _tune_g(x: trtp.TensorDesc, g: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc,
                outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:  # fmt: skip
        return [trtp.AutoTuneCombination("FP16, FP16, FP32, FP32, FP16", "LINEAR", tactics())]

    @trtp.aot_impl(f"{NAMESPACE}::dwconv1d_gated")
    def _impl_g(x: trtp.TensorDesc, g: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc,
                outputs: Tuple[trtp.TensorDesc], tactic: int
                ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        return _aot(trtp, "gated", x, g, w, tactic)

    @trtp.register(f"{NAMESPACE}::dwconv1d_cat")
    def _desc_c(x: trtp.TensorDesc, x2: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc
                ) -> Tuple[trtp.TensorDesc]:  # fmt: skip
        y = x.like()
        y.shape_expr[2] = w.shape_expr[1]
        return y

    @trtp.autotune(f"{NAMESPACE}::dwconv1d_cat")
    def _tune_c(x: trtp.TensorDesc, x2: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc,
                outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:  # fmt: skip
        return [trtp.AutoTuneCombination("FP16, FP16, FP32, FP32, FP16", "LINEAR", tactics())]

    @trtp.aot_impl(f"{NAMESPACE}::dwconv1d_cat")
    def _impl_c(x: trtp.TensorDesc, x2: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc,
                outputs: Tuple[trtp.TensorDesc], tactic: int
                ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        return _aot(trtp, "cat", x, x2, w, tactic)

    _registered = True
