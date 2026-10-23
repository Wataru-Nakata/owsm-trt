"""Decoder attention as TensorRT plugins: Triton kernels compiled ahead of time.

For S > 1 new tokens per row, TensorRT's fused attention wants its own layouts: every layer it copies
``cat([cache, new])`` for self-attention and the whole cross-attention keys/values (transposed) before
the attention kernel. These kernels read everything where it is instead:

* :func:`self_attn`: the fused Q/K/V projection [B, S, 3·H·D] and the KV cache [B, H, C, D] of each row
  up to its own ``offset``, causal over the new tokens -> [B, S, H·D]
* :func:`cross_attn`: queries [B, S, H·D] and the cross-attention keys/values [B, H, T, D] -> [B, S, H·D]

Both are torch custom ops (with a PyTorch reference implementation) that export to ONNX nodes of domain
``owsm``; the TensorRT ONNX parser maps those to the plugins :func:`register_plugins` defines. The
plugins are ahead-of-time: the engine holds the kernels' PTX, so loading it needs neither Triton nor
the registration.
"""

import functools
import math
from typing import List, Tuple, Union

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the plugins cannot be built; the decoder is exported without them
    triton = None

NAMESPACE = "owsm"
# Kernel configurations (BS queries per program -- tl.dot needs >= 16 --, BN keys per step, warps,
# pipeline stages), offered to TensorRT as plugin tactics: it times them and keeps the fastest.
TACTICS = {
    "self_attn": [(16, 64, 4, 2), (16, 32, 4, 2), (16, 128, 4, 2), (16, 64, 4, 3), (32, 64, 4, 2), (16, 64, 8, 2)],
    "cross_attn": [(16, 64, 4, 2), (16, 128, 4, 2), (16, 64, 4, 3), (16, 128, 8, 2), (32, 64, 4, 2), (16, 32, 4, 3)],
}  # fmt: skip
TUNE = True  # False: only the first configuration (no tactic timing)
_SHAPE = {"heads": 0, "head_dim": 0}  # of the model being built (set by register_plugins)
_REGISTERED = False


def available(head_dim: int) -> bool:
    """Whether the plugins can be built here (Triton, TensorRT's ahead-of-time Python plugins, an Ampere or
    newer GPU) for this head size."""
    return head_dim in (16, 32, 64, 128) and unavailable() is None


@functools.cache
def unavailable() -> str | None:
    """Why this environment cannot build the plugins (None: it can)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.cuda.get_device_capability() < (8, 0):
        return "they need an Ampere or newer GPU"
    if triton is None:
        return "Triton is not installed (it comes with PyTorch on Linux)"
    try:
        import tensorrt
        import tensorrt.plugin as trtp
    except ImportError as e:
        return f"TensorRT's Python plugin API does not import ({e})"
    if not hasattr(trtp, "aot_impl"):
        return f"TensorRT {tensorrt.__version__} has no ahead-of-time Python plugins (owsm-trt needs TensorRT >= 10.16)"
    return None


# --- PyTorch reference implementations, exported as ONNX nodes of domain "owsm" ---


def _merge(x: torch.Tensor) -> torch.Tensor:  # [B, H, S, D] -> [B, S, H·D]
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


@torch.library.custom_op("owsm_trt::self_attn", mutates_args=())
def self_attn(
    qkv: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Self-attention of S new tokens per row over [cache slots < offset[b] | new tokens (causal)].

    qkv [B, S, 3·H·D] (queries, keys, values; heads in order), k_cache/v_cache [B, H, C, D],
    offset [B] int64 -> [B, S, H·D]"""
    b, s, _ = qkv.shape
    h, c, d = k_cache.shape[1:]
    q, k, v = qkv.view(b, s, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)
    new = torch.arange(s, device=qkv.device)
    cache_ok = (torch.arange(c, device=qkv.device) < offset[:, None])[:, None, None, :].expand(b, 1, s, c)
    mask = torch.cat([cache_ok, (new[None, :] <= new[:, None]).expand(b, 1, s, s)], -1)
    scores = (q.float() * d**-0.5) @ torch.cat([k_cache, k], 2).float().transpose(-1, -2)
    p = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return _merge((p @ torch.cat([v_cache, v], 2).float()).to(qkv.dtype))


@self_attn.register_fake
def _(qkv, k_cache, v_cache, offset):
    return qkv.new_empty(qkv.shape[0], qkv.shape[1], k_cache.shape[1] * k_cache.shape[3])


@torch.library.custom_op("owsm_trt::cross_attn", mutates_args=())
def cross_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention of q [B, S, H·D] to k/v [B, H, T, D] -> [B, S, H·D]."""
    b, s, _ = q.shape
    h, d = k.shape[1], k.shape[3]
    qh = q.view(b, s, h, d).transpose(1, 2)
    p = ((qh.float() * d**-0.5) @ k.float().transpose(-1, -2)).softmax(-1)
    return _merge((p @ v.float()).to(q.dtype))


@cross_attn.register_fake
def _(q, k, v):
    return torch.empty_like(q)


def _onnx_op(name: str, n_in: int):
    import onnxscript
    from onnx.defs import OpSchema

    types = [f"tensor({t})" for t in ("float16", "bfloat16", "float", "int64", "int32")]
    schema = OpSchema(
        name,
        NAMESPACE,
        1,
        inputs=[OpSchema.FormalParameter(f"input{i}", f"T{i}") for i in range(n_in)],
        outputs=[OpSchema.FormalParameter("output", "T0")],
        type_constraints=[(f"T{i}", types, "") for i in range(n_in)],
        attributes=[
            OpSchema.Attribute("plugin_namespace", OpSchema.AttrType.STRING, "TensorRT plugin namespace"),
            OpSchema.Attribute("plugin_version", OpSchema.AttrType.STRING, "TensorRT plugin version"),
        ],
    )
    op = onnxscript.values.Op(onnxscript.values.Opset(NAMESPACE, 1), name, schema)

    def translate(*args):
        return op(*args, plugin_namespace=NAMESPACE, plugin_version="1")

    return translate


def onnx_translations() -> dict:
    """``custom_translation_table`` entries for ``torch.onnx.export``."""
    ops = torch.ops.owsm_trt
    return {
        ops.self_attn.default: _onnx_op("self_attn", 4),
        ops.cross_attn.default: _onnx_op("cross_attn", 3),
    }


# --- Triton kernels (AOT calling convention: inputs, scalar arguments, outputs, then constexprs) ---

if triton is not None:

    @triton.jit
    def _attend(q, k_ptrs, v_ptrs, valid, ok, m_i, l_i, acc, QK_SCALE: tl.constexpr):
        """One block of keys/values [BN, D] into the running (max, sum, output) of an online softmax.
        ``valid`` [BN, 1]: keys that exist; ``ok`` [BS or 1, BN]: keys each query attends to. Every
        query row must attend to at least one key of the first block it sees."""
        k = tl.load(k_ptrs, mask=valid, other=0.0)
        s = tl.dot(q, tl.trans(k)) * QK_SCALE  # log2 units
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        v = tl.load(v_ptrs, mask=valid, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        return m_new, l_i * alpha + tl.sum(p, 1), acc

    @triton.jit
    def _self_attn_kernel(qkv, kc, vc, offs, S, C, y,
                          H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr,
                          QK_SCALE: tl.constexpr):  # fmt: skip
        bh = tl.program_id(0)  # (row, head)
        sb = tl.program_id(1)  # block of queries
        b = bh // H
        h = bh % H
        HD: tl.constexpr = H * D
        # clamped: TensorRT times the plugin on arbitrary input data while building
        off = tl.minimum(tl.maximum(tl.load(offs + b), 0), C).to(tl.int32)
        s = sb * BS + tl.arange(0, BS)
        sm = (s < S)[:, None]
        d = tl.arange(0, D)[None, :]
        q = tl.load(qkv + (b * S + s).to(tl.int64)[:, None] * (3 * HD) + h * D + d, mask=sm, other=0.0)
        m_i = tl.full([BS], float("-inf"), tl.float32)
        l_i = tl.zeros([BS], tl.float32)
        acc = tl.zeros([BS, D], tl.float32)
        cache = (b * H + h).to(tl.int64) * C * D
        for j0 in range(0, off, BN):  # the row's cache: slots < off
            j = j0 + tl.arange(0, BN)
            ptr = cache + j[:, None] * D + d
            m_i, l_i, acc = _attend(
                q, kc + ptr, vc + ptr, (j < off)[:, None], (j < off)[None, :], m_i, l_i, acc, QK_SCALE
            )
        for j0 in range(0, (sb + 1) * BS, BS):  # the new tokens, causal
            j = j0 + tl.arange(0, BS)
            ok = (j < S)[None, :] & (j[None, :] <= s[:, None])
            new = qkv + (b * S + j).to(tl.int64)[:, None] * (3 * HD) + h * D + d
            m_i, l_i, acc = _attend(q, new + HD, new + 2 * HD, (j < S)[:, None], ok, m_i, l_i, acc, QK_SCALE)
        out = y + (b * S + s).to(tl.int64)[:, None] * HD + h * D + d
        tl.store(out, (acc / l_i[:, None]).to(y.dtype.element_ty), mask=sm)

    @triton.jit
    def _cross_attn_kernel(q_in, kx, vx, S, T, y,
                           H: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr,
                           QK_SCALE: tl.constexpr):  # fmt: skip
        bh = tl.program_id(0)
        sb = tl.program_id(1)
        b = bh // H
        h = bh % H
        HD: tl.constexpr = H * D
        s = sb * BS + tl.arange(0, BS)
        sm = (s < S)[:, None]
        d = tl.arange(0, D)[None, :]
        rows = (b * S + s).to(tl.int64)[:, None] * HD + h * D + d
        q = tl.load(q_in + rows, mask=sm, other=0.0)
        m_i = tl.full([BS], float("-inf"), tl.float32)
        l_i = tl.zeros([BS], tl.float32)
        acc = tl.zeros([BS, D], tl.float32)
        base = (b * H + h).to(tl.int64) * T * D
        for j0 in range(0, T, BN):
            j = j0 + tl.arange(0, BN)
            ptr = base + j[:, None] * D + d
            m_i, l_i, acc = _attend(
                q, kx + ptr, vx + ptr, (j < T)[:, None], (j < T)[None, :], m_i, l_i, acc, QK_SCALE
            )
        tl.store(y + rows, (acc / l_i[:, None]).to(y.dtype.element_ty), mask=sm)


def _qk_scale(d: int) -> float:
    return d**-0.5 * math.log2(math.e)


def self_attn_triton(
    qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    offset: torch.Tensor,
    config: tuple[int, int, int, int] = TACTICS["self_attn"][0],
) -> torch.Tensor:
    """The :func:`self_attn` kernel launched directly, for tests and timing."""
    b, s, _ = qkv.shape
    h, c, d = k_cache.shape[1:]
    bs, bn, warps, stages = config
    y = qkv.new_empty(b, s, h * d)
    _self_attn_kernel[(b * h, triton.cdiv(s, bs))](
        qkv, k_cache, v_cache, offset, s, c, y, H=h, D=d, BS=bs, BN=bn, QK_SCALE=_qk_scale(d),
        num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return y


def cross_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: tuple[int, int, int, int] = TACTICS["cross_attn"][0],
) -> torch.Tensor:
    """The :func:`cross_attn` kernel launched directly, for tests and timing."""
    b, s, _ = q.shape
    h, t, d = k.shape[1:]
    bs, bn, warps, stages = config
    y = torch.empty_like(q)
    _cross_attn_kernel[(b * h, triton.cdiv(s, bs))](
        q, k, v, s, t, y, H=h, D=d, BS=bs, BN=bn, QK_SCALE=_qk_scale(d), num_warps=warps, num_stages=stages
    )
    return y


# --- TensorRT plugins ---

_COMPILED: dict = {}
_KERNELS = {  # plugin -> kernel, its arguments (inputs, scalars, outputs)
    "self_attn": ("_self_attn_kernel", ["qkv", "kc", "vc", "offs", "S", "C", "y"]),
    "cross_attn": ("_cross_attn_kernel", ["q_in", "kx", "vx", "S", "T", "y"]),
}


def _config(kind: str, tactic: int) -> tuple[int, int, int, int]:
    configs = TACTICS[kind]
    return configs[max(tactic - 1, 0) % len(configs)]  # tactic 0: no autotuning


def _compile(kind: str, tactic: int, dtype: str):
    """(name, PTX, warps, shared memory, queries per program) of a kernel for the model being built."""
    h, d = _SHAPE["heads"], _SHAPE["head_dim"]
    bs, bn, warps, stages = _config(kind, tactic)
    key = (kind, h, d, dtype, bs, bn, warps, stages)
    if key not in _COMPILED:
        name, args = _KERNELS[kind]
        types = {"offs": "*i64", "S": "i32", "C": "i32", "T": "i32"}
        consts = {"H": h, "D": d, "BS": bs, "BN": bn, "QK_SCALE": _qk_scale(d)}
        signature = {a: types.get(a, f"*{dtype}") for a in args} | dict.fromkeys(consts, "constexpr")
        # TensorRT and torch buffers are 16-byte aligned: lets loads/stores be vectorized
        aligned = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(args) if signature[a][0] == "*"}
        src = triton.compiler.ASTSource(
            fn=globals()[name], signature=signature, constexprs=consts, attrs=aligned
        )
        k = triton.compile(src, options={"num_warps": warps, "num_stages": stages})
        # a name of its own per configuration (entry and parameter names start with it)
        unique = f"{k.metadata.name}_{bs}_{bn}_{warps}_{stages}"
        ptx = k.asm["ptx"].replace(k.metadata.name, unique)
        _COMPILED[key] = (unique, ptx, k.metadata.num_warps, k.metadata.shared, bs)
    return _COMPILED[key]


def register_plugins(heads: int, head_dim: int) -> None:
    """Define the TensorRT plugins (once per process) for a model with ``heads`` x ``head_dim``."""
    global _REGISTERED
    _SHAPE.update(heads=heads, head_dim=head_dim)
    if _REGISTERED:
        return
    import tensorrt as trt
    import tensorrt.plugin as trtp

    Desc = trtp.TensorDesc
    AOT = Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]
    Tune = List[trtp.AutoTuneCombination]
    names = {trt.float16: ("FP16", "fp16"), trt.bfloat16: ("BF16", "bf16")}

    def launch(kind: str, x: Desc, tactic: int, extras: list):
        rows, tokens = x.shape_expr[0], x.shape_expr[1]
        name, ptx, warps, shared, bs = _compile(kind, tactic, names[x.dtype][1])
        params = trtp.KernelLaunchParams()
        params.grid_x = rows * _SHAPE["heads"]
        params.grid_y = trtp.cdiv(tokens, bs)
        params.block_x = warps * 32
        params.shared_mem = shared
        args = trtp.SymIntExprs(len(extras))
        for i, e in enumerate(extras):
            args[i] = trtp.SymInt32(e)
        return name, ptx, params, args

    def tune(kind: str, x: Desc, int_inputs: tuple = ()) -> Tune:
        t, n = names[x.dtype][0], len(_KERNELS[kind][1]) - 2  # inputs + outputs
        types = ", ".join("INT64" if i in int_inputs else t for i in range(n))
        tactics = range(1, len(TACTICS[kind]) + 1) if TUNE else [1]
        return [trtp.AutoTuneCombination(types, "LINEAR", list(tactics))]

    @trtp.register(f"{NAMESPACE}::self_attn")
    def _self_desc(qkv: Desc, k_cache: Desc, v_cache: Desc, offset: Desc) -> Tuple[Desc]:
        b, s = qkv.shape_expr[0], qkv.shape_expr[1]
        return (trtp.from_shape_expr((b, s, k_cache.shape_expr[1] * k_cache.shape_expr[3]), qkv.dtype),)

    @trtp.autotune(f"{NAMESPACE}::self_attn")
    def _self_tune(qkv: Desc, k_cache: Desc, v_cache: Desc, offset: Desc, outputs: Tuple[Desc]) -> Tune:
        return tune("self_attn", qkv, (3,))

    @trtp.aot_impl(f"{NAMESPACE}::self_attn")
    def _self_aot(
        qkv: Desc, k_cache: Desc, v_cache: Desc, offset: Desc, outputs: Tuple[Desc], tactic: int
    ) -> AOT:
        return launch("self_attn", qkv, tactic, [qkv.shape_expr[1], k_cache.shape_expr[2]])

    @trtp.register(f"{NAMESPACE}::cross_attn")
    def _cross_desc(q: Desc, k: Desc, v: Desc) -> Tuple[Desc]:
        return (q.like(),)

    @trtp.autotune(f"{NAMESPACE}::cross_attn")
    def _cross_tune(q: Desc, k: Desc, v: Desc, outputs: Tuple[Desc]) -> Tune:
        return tune("cross_attn", q)

    @trtp.aot_impl(f"{NAMESPACE}::cross_attn")
    def _cross_aot(q: Desc, k: Desc, v: Desc, outputs: Tuple[Desc], tactic: int) -> AOT:
        return launch("cross_attn", q, tactic, [q.shape_expr[1], k.shape_expr[2]])

    _REGISTERED = True
