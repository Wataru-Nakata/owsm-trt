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
#pragma unroll
    for (int i = 0; i < TT; i++) {
      float s[V];
#pragma unroll
      for (int j = 0; j < V; j++) s[j] = b[j];
#pragma unroll
      for (int k = 0; k < K; k++)
#pragma unroll
        for (int j = 0; j < V; j++) s[j] = fmaf(wr[k][j], a[i + k][j], s[j]);
      if (t0 + i < T) {
#if GATED
        float gf[V];
        unpack<V>(gv[i], gf);
#pragma unroll
        for (int j = 0; j < V; j++) s[j] *= gf[j];
#endif
        y[(frame0 + t0 + i) * CV + cv] = pack<V>(s);
      }
    }
  }
}
"""


@functools.lru_cache(maxsize=None)
def compile_ptx(
    k: int, v: int = 2, tt: int = 32, nt: int = 128, nch: int = 1, minb: int = 1, gated: bool = False,
    cat: bool = False, arch: str | None = None, cubin_log: bool = False,
) -> tuple[str, str]:  # fmt: skip
    """PTX of the kernel ``dwconv1d`` for one configuration (and the compile log). ``arch``: e.g.
    ``compute_90`` (default: the current GPU's); ``sm_90`` compiles to a cubin, for the ptxas log only."""
    defs = dict(K=k, V=v, TT=tt, NT=nt, NCH=nch, MINB=minb, GATED=int(gated), CAT=int(cat))
    return nvrtc_ptx(DWCONV_SOURCE, "dwconv1d.cu", tuple(defs.items()), arch, cubin_log)


@functools.lru_cache(maxsize=None)
def nvrtc_ptx(
    source: str,
    name: str,
    defs: tuple[tuple[str, int], ...],
    arch: str | None = None,
    cubin_log: bool = False,
) -> tuple[str, str]:
    """PTX of CUDA C ``source`` compiled with the macros ``defs`` (and the compile log); see compile_ptx."""
    from cuda.bindings import nvrtc

    def check(res):
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC: {res[0]}")
        return res[1] if len(res) > 1 else None

    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"compute_{major}{minor}"
    prog = check(nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], []))
    opts = [f"-D{n}={x}".encode() for n, x in defs]
    opts += [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
    if cubin_log:
        opts.append(b"--ptxas-options=-v")
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    # NVRTC writes into these buffers, so they must be bytearrays: b" " * 1 (an empty log is its NUL) is
    # Python's shared b" ", and writing it made every later single-space read in the process a NUL (which
    # broke the next checkpoint's unpickling)
    log = bytearray(check(nvrtc.nvrtcGetProgramLogSize(prog)))
    check(nvrtc.nvrtcGetProgramLog(prog, log))
    log = log.decode(errors="replace").rstrip("\x00").strip()
    if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC compile failed: {log}")
    if arch.startswith("sm_"):
        return "", log
    ptx = bytearray(check(nvrtc.nvrtcGetPTXSize(prog)))
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


# --- OWSM-CTC: softmax and argmax over the vocabulary ---------------------------------------------------

ROWS_SOURCE = r"""
// Row-wise operations on x [rows, V] fp16 (V a multiple of 8, rows in any number of leading dimensions):
// one block of NT threads per row, thread t reading the vectors of 8 halves t, t + NT, ... of the row.
//   softmax_rows: y [rows, V] fp16 = softmax(x) computed in fp32 (each value rounded once). Two passes over
//                 the row: running maximum and sum (online softmax), then the output (the row again: from
//                 L2). Few registers, so that several blocks share an SM and overlap their passes.
//   argmax_rows:  y [rows] int32 = the first index of the largest value.
typedef unsigned short u16;
#define NV (V / 8)
#define NW (NT / 32)
#define LOG2E 1.4426950408889634f
#define LOWEST -3.0e38f  // a start for maxima (not -inf: -inf - -inf would make NaNs)
__device__ __forceinline__ float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) { f[2 * i] = h2f((u16)(w[i] & 0xffff)); f[2 * i + 1] = h2f((u16)(w[i] >> 16)); }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  unsigned w[4];
#pragma unroll
  for (int i = 0; i < 4; i++) w[i] = (unsigned)f2h(f[2 * i]) | ((unsigned)f2h(f[2 * i + 1]) << 16);
  return make_uint4(w[0], w[1], w[2], w[3]);
}
// (m, s): s = sum of exp(a - m) over the values a seen; merged with (m2, s2)
__device__ __forceinline__ void merge(float& m, float& s, float m2, float s2) {
  const float mm = fmaxf(m, m2);
  s = s * exp2f((m - mm) * LOG2E) + s2 * exp2f((m2 - mm) * LOG2E);
  m = mm;
}

extern "C" __global__ void __launch_bounds__(NT) softmax_rows(const uint4* __restrict__ x, uint4* __restrict__ y) {
  __shared__ float sh_m[NW], sh_s[NW];
  const long row = (long)blockIdx.y * gridDim.x + blockIdx.x;
  x += row * NV;
  y += row * NV;
  float m = LOWEST, s = 0.f;
#pragma unroll 4
  for (int j = threadIdx.x; j < NV; j += NT) {
    float f[8];
    unpack8(x[j], f);
    float mv = f[0];
#pragma unroll
    for (int k = 1; k < 8; k++) mv = fmaxf(mv, f[k]);
    float sv = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) sv += exp2f((f[k] - mv) * LOG2E);
    merge(m, s, mv, sv);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
  if ((threadIdx.x & 31) == 0) { sh_m[threadIdx.x >> 5] = m; sh_s[threadIdx.x >> 5] = s; }
  __syncthreads();
  m = sh_m[0];
  s = sh_s[0];
#pragma unroll
  for (int w = 1; w < NW; w++) merge(m, s, sh_m[w], sh_s[w]);
  const float inv = 1.f / s;
#pragma unroll 4
  for (int j = threadIdx.x; j < NV; j += NT) {
    float f[8];
    unpack8(x[j], f);
#pragma unroll
    for (int k = 0; k < 8; k++) f[k] = exp2f((f[k] - m) * LOG2E) * inv;
    y[j] = pack8(f);
  }
}

extern "C" __global__ void __launch_bounds__(NT) argmax_rows(const uint4* __restrict__ x, int* __restrict__ y) {
  __shared__ float sh_v[NW];
  __shared__ int sh_i[NW];
  const long row = (long)blockIdx.y * gridDim.x + blockIdx.x;
  x += row * NV;
  float best = LOWEST;
  int arg = 0x7fffffff;
#pragma unroll 4
  for (int j = threadIdx.x; j < NV; j += NT) {
    float f[8];
    unpack8(x[j], f);
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (f[k] > best) { best = f[k]; arg = j * 8 + k; }  // indices increase: the first maximum stays
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float b2 = __shfl_xor_sync(0xffffffffu, best, o);
    const int a2 = __shfl_xor_sync(0xffffffffu, arg, o);
    if (b2 > best || (b2 == best && a2 < arg)) { best = b2; arg = a2; }
  }
  if ((threadIdx.x & 31) == 0) { sh_v[threadIdx.x >> 5] = best; sh_i[threadIdx.x >> 5] = arg; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < NW; w++)  // (warp 0's is already in best / arg)
      if (sh_v[w] > best || (sh_v[w] == best && sh_i[w] < arg)) { best = sh_v[w]; arg = sh_i[w]; }
    y[row] = arg;
  }
}
"""

# Threads per row (tactic i + 1 = ROW_THREADS[i]); a 50048-wide fp16 row is 6256 vectors of 8. (softmax_rows
# alone at 16 x 376 x 50048 on an H200: 0.43 ms; a version holding each row in the registers of a cluster
# of 2-8 blocks, read once: 0.45-0.57 ms.)
ROW_THREADS = (128, 256, 512, 1024)
rows_override: int | None = None  # one tactic of the row plugins instead of all (tests, benchmarks)
VOCAB_MULTIPLE = 64  # CTC export: vocabulary padded to a multiple of this (see build._padded_vocab)


@torch.library.custom_op("owsm_trt::softmax_rows", mutates_args=())
def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension, computed in fp32 and rounded once to x.dtype."""
    return x.float().softmax(-1).to(x.dtype)


@softmax_rows.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


@torch.library.custom_op("owsm_trt::argmax_rows", mutates_args=())
def argmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Index (int32) of the largest value along the last dimension; the first one of equal values."""
    return x.argmax(-1).to(torch.int32)


@argmax_rows.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape[:-1], dtype=torch.int32)


def _onnx_rows_op(name: str, out_type: str):
    def translate(x):
        from onnx.defs import OpSchema
        from onnxscript import values

        param = OpSchema.FormalParameter
        schema = OpSchema(
            name,
            NAMESPACE,
            1,
            inputs=[param("x", "T")],
            outputs=[param("y", "U")],
            type_constraints=[
                ("T", ["tensor(float16)"], ""),
                ("U", [out_type], ""),
            ],  # fmt: skip
            attributes=[
                OpSchema.Attribute(a, OpSchema.AttrType.STRING, "")
                for a in ("plugin_namespace", "plugin_version")
            ],
        )
        op = values.Op(values.Opset(NAMESPACE, 1), name, schema)
        return op(x, plugin_namespace=NAMESPACE, plugin_version="1")

    return translate


CTC_ONNX_TRANSLATIONS = {
    torch.ops.owsm_trt.softmax_rows.default: _onnx_rows_op("softmax_rows", "tensor(float16)"),
    torch.ops.owsm_trt.argmax_rows.default: _onnx_rows_op("argmax_rows", "tensor(int32)"),
}
_ctc_registered = False


def _rows_aot(trtp, kernel: str, x, tactic: int):
    try:
        vocab = int(x.shape[2])
        if vocab % 8:
            raise ValueError(f"{kernel}: the vocabulary ({vocab}) must be a multiple of 8")
        nt = ROW_THREADS[max(tactic, 1) - 1]
        ptx, _ = nvrtc_ptx(ROWS_SOURCE, "rows.cu", (("V", vocab), ("NT", nt)))
        launch = trtp.KernelLaunchParams()
        launch.grid_x = x.shape_expr[1]
        launch.grid_y = x.shape_expr[0]
        launch.block_x = nt
        return kernel, ptx, launch, trtp.SymIntExprs(0)
    except Exception:  # TensorRT reports only "launch param computation failed"
        log.error("%s plugin:\n%s", kernel, traceback.format_exc())
        raise


def register_ctc() -> None:
    """Register ``owsm::softmax_rows`` and ``owsm::argmax_rows`` (inputs [B, T, V] fp16) with TensorRT."""
    global _ctc_registered
    if _ctc_registered:
        return
    import tensorrt as trt
    import tensorrt.plugin as trtp

    def tactics() -> list[int]:
        return [rows_override] if rows_override is not None else list(range(1, len(ROW_THREADS) + 1))

    @trtp.register(f"{NAMESPACE}::softmax_rows")
    def _desc_s(x: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        return x.like()

    @trtp.autotune(f"{NAMESPACE}::softmax_rows")
    def _tune_s(x: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP16, FP16", "LINEAR", tactics())]

    @trtp.aot_impl(f"{NAMESPACE}::softmax_rows")
    def _impl_s(x: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        return _rows_aot(trtp, "softmax_rows", x, tactic)

    @trtp.register(f"{NAMESPACE}::argmax_rows")
    def _desc_a(x: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        return trtp.from_shape_expr((x.shape_expr[0], x.shape_expr[1]), dtype=trt.int32)

    @trtp.autotune(f"{NAMESPACE}::argmax_rows")
    def _tune_a(x: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP16, INT32", "LINEAR", tactics())]

    @trtp.aot_impl(f"{NAMESPACE}::argmax_rows")
    def _impl_a(x: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        return _rows_aot(trtp, "argmax_rows", x, tactic)

    _ctc_registered = True


# --- The subsampling's first convolution ----------------------------------------------------------------

CONV1_SOURCE = r"""
// relu(conv2d(x, 3x3, stride 2, one input channel)) for x [B, T, F] fp16 (LINEAR), w [9, C] fp32 (tap
// kh * 3 + kw), bias [C] fp32 -> y [B, C, T1, F1] fp16 in TensorRT's HWC8 format: memory [B][T1][F1][C],
// the layout the next (TensorRT) convolution reads. Block (t1, b): the 3 input rows of output row t1 in
// shared memory, thread g computes channels 8g .. 8g + 7 (weights in registers) at every f1 of the row, a
// warp writing 512 consecutive bytes per f1. Memory-bound on the output (C / 8 x the input).
typedef unsigned short u16;
__device__ __forceinline__ float h2f(u16 h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ u16 f2h(float f) { u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }

extern "C" __global__ void __launch_bounds__(NT) conv1_relu(
    const u16* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias, int T, uint4* __restrict__ y) {
  constexpr int F1 = (F - 3) / 2 + 1, G = C / 8;
  __shared__ float xs[3][F];
  const int t1 = blockIdx.x, b = blockIdx.y, g = threadIdx.x;
  const int T1 = (T - 3) / 2 + 1;
  for (int i = threadIdx.x; i < 3 * F; i += NT) xs[i / F][i % F] = h2f(x[((long)b * T + 2 * t1 + i / F) * F + i % F]);
  float wr[9][8], br[8];
  if (g < G) {
#pragma unroll
    for (int k = 0; k < 9; k++)
#pragma unroll
      for (int j = 0; j < 8; j++) wr[k][j] = w[k * C + 8 * g + j];
#pragma unroll
    for (int j = 0; j < 8; j++) br[j] = bias[8 * g + j];
  }
  __syncthreads();
  if (g >= G) return;
  uint4* out = y + ((long)b * T1 + t1) * F1 * G + g;
  for (int f1 = 0; f1 < F1; f1++) {
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] = br[j];
#pragma unroll
    for (int kh = 0; kh < 3; kh++)
#pragma unroll
      for (int kw = 0; kw < 3; kw++) {
        const float v = xs[kh][2 * f1 + kw];
#pragma unroll
        for (int j = 0; j < 8; j++) a[j] = fmaf(wr[kh * 3 + kw][j], v, a[j]);
      }
    unsigned p[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
      p[i] = (unsigned)f2h(fmaxf(a[2 * i], 0.f)) | ((unsigned)f2h(fmaxf(a[2 * i + 1], 0.f)) << 16);
    out[(long)f1 * G] = make_uint4(p[0], p[1], p[2], p[3]);
  }
}
"""


@torch.library.custom_op("owsm_trt::conv1_relu", mutates_args=())
def conv1_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """relu(conv2d(x[:, None], 3x3, stride 2)) computed in fp32: x [B, T, F], weight [9, C] (tap kh * 3 + kw,
    fp32), bias [C] (fp32) -> [B, C, T1, F1] in x.dtype."""
    c = weight.shape[1]
    y = F.conv2d(x.unsqueeze(1).float(), weight.t().reshape(c, 1, 3, 3), bias, stride=2)
    return y.relu().to(x.dtype)


@conv1_relu.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    b, t, f = x.shape
    return x.new_empty(b, weight.shape[1], (t - 3) // 2 + 1, (f - 3) // 2 + 1)


CONV1_ONNX_TRANSLATIONS = {
    torch.ops.owsm_trt.conv1_relu.default: _onnx_op("conv1_relu", ("x", "weight", "bias")),
}
_conv1_registered = False


def register_conv1() -> None:
    """Register ``owsm::conv1_relu`` with TensorRT (output in HWC8, for the convolution after it)."""
    global _conv1_registered
    if _conv1_registered:
        return
    import tensorrt as trt
    import tensorrt.plugin as trtp

    @trtp.register(f"{NAMESPACE}::conv1_relu")
    def _desc(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        t1 = (x.shape_expr[1] - 3) // 2 + 1
        f1 = (x.shape_expr[2] - 3) // 2 + 1
        return trtp.from_shape_expr((x.shape_expr[0], w.shape_expr[1], t1, f1), dtype=trt.float16)

    @trtp.autotune(f"{NAMESPACE}::conv1_relu")
    def _tune(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]
              ) -> List[trtp.AutoTuneCombination]:  # fmt: skip
        return [trtp.AutoTuneCombination("FP16, FP32, FP32, FP16", "LINEAR, LINEAR, LINEAR, HWC8", [1])]

    @trtp.aot_impl(f"{NAMESPACE}::conv1_relu")
    def _impl(x: trtp.TensorDesc, w: trtp.TensorDesc, b: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc],
              tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        try:
            c, f = int(w.shape[1]), int(x.shape[2])
            if c % 8:
                raise ValueError(f"conv1_relu: {c} channels (a multiple of 8 needed)")
            nt = -(-(c // 8) // 32) * 32
            ptx, _ = nvrtc_ptx(CONV1_SOURCE, "conv1.cu", (("C", c), ("F", f), ("NT", nt)))
            launch = trtp.KernelLaunchParams()
            launch.grid_x = (x.shape_expr[1] - 3) // 2 + 1
            launch.grid_y = x.shape_expr[0]
            launch.block_x = nt
            extra = trtp.SymIntExprs(1)
            extra[0] = trtp.SymInt32(x.shape_expr[1])
            return "conv1_relu", ptx, launch, extra
        except Exception:
            log.error("conv1_relu plugin:\n%s", traceback.format_exc())
            raise

    _conv1_registered = True


# --- The subsampling's last ReLU and flattening ----------------------------------------------------------

FLATTEN_SOURCE = r"""
// z[b, t, c * F + f] = max(y[b, c, t, f], 0) for y [B, C, T, F] fp16 in TensorRT's HWC8 format (memory
// [B][T][F][C], as the convolution before writes it) -> z [B, T, C * F] fp16 (LINEAR): the ReLU and the
// (channel, frequency) flattening the subsampling's linear layer reads, in one pass. Block (t, b, channel
// chunk of CB): its F x CB tile through shared memory, read along channels, written along the output.
typedef unsigned short u16;
__device__ __forceinline__ u16 relu_h(u16 h) { return (h & 0x8000) ? 0 : h; }  // (negative zero too)

extern "C" __global__ void __launch_bounds__(NT) relu_flatten(const u16* __restrict__ y, int T, u16* __restrict__ z) {
  __shared__ u16 tile[F][CB + 2];
  const int t = blockIdx.x, b = blockIdx.y, c0 = blockIdx.z * CB;
  const u16* src = y + ((long)b * T + t) * F * C;
  for (int i = threadIdx.x; i < F * CB; i += NT) {
    const int f = i / CB, c = i % CB;
    if (c0 + c < C) tile[f][c] = src[f * C + c0 + c];
  }
  __syncthreads();
  u16* dst = z + ((long)b * T + t) * C * F + (long)c0 * F;
  const int n = min(CB, C - c0) * F;
  for (int i = threadIdx.x; i < n; i += NT) dst[i] = relu_h(tile[i % F][i / F]);
}
"""


@torch.library.custom_op("owsm_trt::relu_flatten", mutates_args=())
def relu_flatten(y: torch.Tensor) -> torch.Tensor:
    """relu(y) [B, C, T, F] -> [B, T, C * F] (channel-major flattening)."""
    b, c, t, f = y.shape
    return y.relu().transpose(1, 2).reshape(b, t, c * f).contiguous()


@relu_flatten.register_fake
def _(y: torch.Tensor) -> torch.Tensor:
    b, c, t, f = y.shape
    return y.new_empty(b, t, c * f)


FLATTEN_ONNX_TRANSLATIONS = {torch.ops.owsm_trt.relu_flatten.default: _onnx_op("relu_flatten", ("y",))}
FLATTEN_CHANNELS = 256  # channels per block
_flatten_registered = False


def register_flatten() -> None:
    """Register ``owsm::relu_flatten`` with TensorRT (input in HWC8, as the convolution before writes it)."""
    global _flatten_registered
    if _flatten_registered:
        return
    import tensorrt as trt
    import tensorrt.plugin as trtp

    @trtp.register(f"{NAMESPACE}::relu_flatten")
    def _desc(y: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        s = y.shape_expr
        return trtp.from_shape_expr((s[0], s[2], s[1] * s[3]), dtype=trt.float16)

    @trtp.autotune(f"{NAMESPACE}::relu_flatten")
    def _tune(y: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP16, FP16", "HWC8, LINEAR", [1])]

    @trtp.aot_impl(f"{NAMESPACE}::relu_flatten")
    def _impl(y: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
              ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:  # fmt: skip
        try:
            c, f = int(y.shape[1]), int(y.shape[3])
            if c % 8:
                raise ValueError(f"relu_flatten: {c} channels (a multiple of 8 needed)")
            cb, nt = min(FLATTEN_CHANNELS, c), 256
            ptx, _ = nvrtc_ptx(FLATTEN_SOURCE, "flatten.cu", (("C", c), ("F", f), ("CB", cb), ("NT", nt)))
            launch = trtp.KernelLaunchParams()
            launch.grid_x = y.shape_expr[2]
            launch.grid_y = y.shape_expr[0]
            launch.grid_z = (c + cb - 1) // cb
            launch.block_x = nt
            extra = trtp.SymIntExprs(1)
            extra[0] = trtp.SymInt32(y.shape_expr[2])
            return "relu_flatten", ptx, launch, extra
        except Exception:
            log.error("relu_flatten plugin:\n%s", traceback.format_exc())
            raise

    _flatten_registered = True
