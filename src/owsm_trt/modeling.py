"""Pure-PyTorch reimplementation of the OWSM models (ESPnet ``s2t`` and ``s2t_ctc``).

* :class:`OWSMModel`: encoder-decoder (OWSM v1-v4, OWLS, POWSM). The encoder is an E-Branchformer or
  a Transformer, the decoder a Transformer.
* :class:`OWSMCTCModel`: OWSM-CTC, an E-Branchformer encoder conditioned on a prompt, read with CTC.

Parameter names mirror ESPnet's so official checkpoints load with ``strict=True``. Inference-only:
inputs are always padded/trimmed to the fixed training length (30 s), so no padding masks exist.

The decoder is written with an explicit, static KV cache (``step``) instead of ESPnet's
re-projection of every past hidden state, and the cross-attention keys/values are projected once
per utterance (``cross_kv``). Both functions are export-friendly (pure tensor ops, no Python
control flow on tensor values) so the same code serves the eager path and the TensorRT export.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from . import attn_plugin
from .config import OWSMConfig

LN_EPS = 1e-12


def sinusoidal_positions(n: int, d: int) -> torch.Tensor:
    """[n, d] table identical to ESPnet's ``PositionalEncoding``."""
    pos = torch.arange(0, n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class LayerNorm(nn.LayerNorm):
    """ESPnet's LayerNorm (eps=1e-12).

    No explicit fp32 casts: CUDA kernels and the exported ONNX LayerNormalization (stash_type=1) already
    compute mean/variance in fp32, so casting around it only adds kernels.
    """

    def __init__(self, d: int):
        super().__init__(d, eps=LN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == self.weight.dtype:
            return super().forward(x)
        # an fp32 residual stream through fp16 weights (see Encoder.forward)
        return F.layer_norm(
            x, self.normalized_shape, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps
        )


class Frontend(nn.Module):
    """STFT -> power -> log-mel -> global or per-utterance mean normalisation. Always runs in fp32."""

    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        self.n_fft, self.hop, self.win = cfg.n_fft, cfg.hop_length, cfg.win_length
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)
        # Filled from the checkpoint ("frontend.logmel.melmat", "normalize.mean/std").
        self.register_buffer("melmat", torch.zeros(cfg.n_fft // 2 + 1, cfg.n_mels))
        self.global_mvn = cfg.normalize == "global_mvn"
        if self.global_mvn:
            self.register_buffer("mean", torch.zeros(cfg.n_mels))
            self.register_buffer("std", torch.ones(cfg.n_mels))

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        # speech: [B, N] -> feats: [B, T, n_mels]
        spec = torch.stft(
            speech.float(),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # [B, F, T]
        power = (spec.real**2 + spec.imag**2).transpose(1, 2)  # [B, T, F]
        mel = torch.matmul(power, self.melmat).clamp(min=1e-10).log()
        if self.global_mvn:
            return (mel - self.mean) / self.std
        return mel - mel.mean(dim=1, keepdim=True)  # utterance MVN, means only


class Conv2dSubsampling(nn.Module):
    """ESPnet's Conv2dSubsampling{,2,8}: 3x3 convolutions with the given strides, then a linear layer.

    ``plugin`` (export option): the first convolution and its ReLU as a TensorRT plugin
    (plugins.conv1_relu)."""

    plugin = False

    def __init__(self, n_mels: int, d: int, strides: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        f = n_mels
        for i, s in enumerate(strides):
            layers += [nn.Conv2d(1 if i == 0 else d, d, 3, s), nn.ReLU()]
            f = (f - 3) // s + 1
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(d * f, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, T', d]
        if Conv2dSubsampling.plugin:
            from . import plugins

            x = self.conv[2:](plugins.conv1_relu(x, *self.plugin_weights()))
        else:
            x = self.conv(x.unsqueeze(1))  # [B, d, T', F']
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))

    def plugin_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The first convolution's fp32 taps [9, C] (tap kh * 3 + kw) and bias [C] for plugins.conv1_relu;
        constants of an exported graph after :meth:`freeze_plugin_weights`."""
        frozen = getattr(self, "_plugin_frozen", None)
        if frozen is not None:
            return frozen
        conv = self.conv[0]
        return conv.weight.float().reshape(conv.out_channels, 9).t(), conv.bias.float()

    @torch.no_grad()
    def freeze_plugin_weights(self, frozen: bool = True) -> None:
        self._plugin_frozen = None
        if frozen:
            self._plugin_frozen = tuple(t.contiguous() for t in self.plugin_weights())


class MultiHeadedAttention(nn.Module):
    explicit = False  # export option: attention written out (see explicit_attention)

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.d_k = n_heads, d // n_heads
        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)

    def heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d] -> [B, H, T, d_k]
        return x.reshape(x.shape[0], x.shape[1], self.h, self.d_k).transpose(1, 2)

    @torch.no_grad()
    def fuse_qkv(self) -> None:
        """Replace the three projections by one [3d, d] GEMM (call after loading weights)."""
        d = self.h * self.d_k
        lin = nn.Linear(d, 3 * d, device=self.linear_q.weight.device, dtype=self.linear_q.weight.dtype)
        lin.weight.copy_(torch.cat([self.linear_q.weight, self.linear_k.weight, self.linear_v.weight]))
        lin.bias.copy_(torch.cat([self.linear_q.bias, self.linear_k.bias, self.linear_v.bias]))
        self.linear_qkv = lin.requires_grad_(False)
        del self.linear_q, self.linear_k, self.linear_v

    def qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [B, T, d] -> 3 x [B, H, T, d_k]
        if not hasattr(self, "linear_qkv"):
            return self.heads(self.linear_q(x)), self.heads(self.linear_k(x)), self.heads(self.linear_v(x))
        b, t = x.shape[:2]
        q, k, v = self.linear_qkv(x).reshape(b, t, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, d_k] -> [B, T, d]
        return self.linear_out(x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.h * self.d_k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(self.attend(*self.qkv(x)))

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return explicit_attention(q, k, v) if self.explicit else F.scaled_dot_product_attention(q, k, v)

    def cross(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Attention from ``x`` [B, T, d] to ``memory`` [B, M, d]."""
        q, k, v = (
            self.heads(self.linear_q(x)),
            self.heads(self.linear_k(memory)),
            self.heads(self.linear_v(memory)),
        )
        return self.merge(self.attend(q, k, v))


def cached_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Attention over [cache | new] without materialising the concatenated keys/values.

    q, k, v: [B, H, S, d_k]; k_cache, v_cache: [B, H, C, d_k]; mask: [B, 1, S, C+S] (True = attend)
    Only the score rows are concatenated, so the cache is read in place instead of copied every step.
    """
    cap = k_cache.shape[2]
    q = q * (q.shape[-1] ** -0.5)
    scores = torch.cat([q @ k_cache.transpose(-1, -2), q @ k.transpose(-1, -2)], dim=-1)  # [B, H, S, C+S]
    scores = scores.float().masked_fill(~mask, float("-inf"))
    p = scores.softmax(-1).to(q.dtype)
    return p[..., :cap] @ v_cache + p[..., cap:] @ v  # [B, H, S, d_k]


def explicit_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """softmax(q k^T / sqrt(d_k)) v with the scale on ``q``, for export: the ONNX exporter's
    ``scaled_dot_product_attention`` also scales ``k``, which TensorRT then computes in full on every
    multi-token decoder call (the whole cross-attention keys and KV cache of every layer)."""
    scores = (q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v


class FusableSiLU(nn.Module):
    """SiLU written x / (1 + exp(-x)), the same function: TensorRT fuses this form into the GEMM before it
    (x * sigmoid(x) becomes a kernel of its own). For export."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


class FeedForward(nn.Module):
    def __init__(self, d: int, units: int, activation: nn.Module):
        super().__init__()
        self.w_1 = nn.Linear(d, units)
        self.w_2 = nn.Linear(units, d)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


class DepthwiseConv1d(nn.Conv1d):
    """Depthwise convolution over time of [B, T, C] input ('same' padding, ``nn.Conv1d`` weights).

    ``x`` may be a pair of tensors, concatenated along channels. ``residual``: returns ``x + conv(x)``;
    ``gate``: returns ``gate[..., :C] * conv(x)``. With ``plugin`` set (TensorRT export), the convolution
    is a custom op that becomes a TensorRT plugin working in [B, T, C] (see :mod:`owsm_trt.plugins`),
    with the concatenation, residual and gate inside it.
    """

    plugin = False
    plugin_cat = True  # the plugin concatenates a pair of inputs itself

    def __init__(self, channels: int, kernel: int):
        super().__init__(channels, channels, kernel, 1, (kernel - 1) // 2, groups=channels)

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        residual: bool = False,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple) and not (DepthwiseConv1d.plugin and DepthwiseConv1d.plugin_cat):
            x = torch.cat(x, dim=-1)
        if DepthwiseConv1d.plugin:
            from . import plugins

            w, b = self.plugin_weights(residual)
            if isinstance(x, tuple):
                return plugins.dwconv1d_cat(*x, w, b)
            if gate is not None:
                return plugins.dwconv1d_gated(x, gate, w, b)
            return plugins.dwconv1d(x, w, b)
        y = super().forward(x.transpose(1, 2)).transpose(1, 2)
        if residual:
            y = x + y
        if gate is not None:
            y = gate[..., : y.shape[-1]] * y
        return y

    def plugin_weights(self, residual: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """The plugin's fp32 taps [K, C] (``residual``: x + conv(x), one more tap of weight 1 at the centre)
        and bias [C]. After :meth:`freeze_plugin_weights` they are constants of an exported graph (not
        computed from the parameters by TensorRT on every call)."""
        frozen = getattr(self, "_plugin_frozen", None)
        if frozen is not None:
            return frozen[residual], frozen[2]
        w = self.weight.float()[:, 0].t()  # [K, C]
        if residual:
            w = w + (torch.arange(w.shape[0], device=w.device) == self.padding[0]).to(w)[:, None]
        return w, self.bias.float()

    @torch.no_grad()
    def freeze_plugin_weights(self, frozen: bool = True) -> None:
        self._plugin_frozen = None
        if frozen:
            w, b = self.plugin_weights(False)
            self._plugin_frozen = (w.contiguous(), self.plugin_weights(True)[0].contiguous(), b.contiguous())


class ConvolutionalSpatialGatingUnit(nn.Module):
    def __init__(self, units: int, kernel: int):
        super().__init__()
        c = units // 2
        self.norm = LayerNorm(c)
        self.conv = DepthwiseConv1d(c, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_g = x.chunk(2, dim=-1)[1]
        return self.conv(self.norm(x_g), gate=x)  # x_r * conv(norm(x_g)), x_r = x[..., :C]


class ConvolutionalGatingMLP(nn.Module):
    def __init__(self, d: int, units: int, kernel: int):
        super().__init__()
        self.channel_proj1 = nn.Sequential(nn.Linear(d, units), nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(units, kernel)
        self.channel_proj2 = nn.Linear(units // 2, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_proj2(self.csgu(self.channel_proj1(x)))


class EBranchformerLayer(nn.Module):
    def __init__(self, cfg: OWSMConfig, cross_attn: bool = False):
        super().__init__()
        d = cfg.d_model
        self.attn = MultiHeadedAttention(d, cfg.n_heads)
        self.cgmlp = ConvolutionalGatingMLP(d, cfg.cgmlp_units, cfg.cgmlp_kernel)
        self.feed_forward = FeedForward(d, cfg.enc_ffn, nn.SiLU())
        self.feed_forward_macaron = FeedForward(d, cfg.enc_ffn, nn.SiLU())
        self.norm_ff = LayerNorm(d)
        self.norm_ff_macaron = LayerNorm(d)
        self.norm_mha = LayerNorm(d)
        self.norm_mlp = LayerNorm(d)
        self.norm_final = LayerNorm(d)
        k = cfg.merge_kernel
        self.depthwise_conv_fusion = DepthwiseConv1d(2 * d, k)
        self.merge_proj = nn.Linear(2 * d, d)
        if cross_attn:  # OWSM-CTC: attention to the prompt after the feed-forward module
            self.cross_attn = MultiHeadedAttention(d, cfg.n_heads)
            self.norm_cross_attn = LayerNorm(d)

    @property
    def self_attn(self) -> MultiHeadedAttention:
        return self.attn

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        # x may be an fp32 residual stream around fp16 modules; the casts are no-ops otherwise
        dt, r = self.merge_proj.weight.dtype, x.dtype
        x = x + 0.5 * self.feed_forward_macaron(self.norm_ff_macaron(x).to(dt)).to(r)
        x1 = self.attn(self.norm_mha(x).to(dt))
        x2 = self.cgmlp(self.norm_mlp(x).to(dt))
        x_concat = (x1, x2)  # [B, T, 2d], concatenated by the convolution
        x = x + self.merge_proj(self.depthwise_conv_fusion(x_concat, residual=True)).to(r)
        x = x + 0.5 * self.feed_forward(self.norm_ff(x).to(dt)).to(r)
        if memory is not None and hasattr(self, "cross_attn"):
            x = x + self.cross_attn.cross(self.norm_cross_attn(x).to(dt), memory).to(r)
        return self.norm_final(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm self-attention + ReLU feed-forward (ESPnet ``EncoderLayer``)."""

    def __init__(self, d: int, n_heads: int, units: int):
        super().__init__()
        self.self_attn = MultiHeadedAttention(d, n_heads)
        self.feed_forward = FeedForward(d, units, nn.ReLU())
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        dt, r = self.feed_forward.w_1.weight.dtype, x.dtype
        x = x + self.self_attn(self.norm1(x).to(dt)).to(r)
        return x + self.feed_forward(self.norm2(x).to(dt)).to(r)


class Encoder(nn.Module):
    """Speech encoder: convolutional subsampling, positional encoding, E-Branchformer or Transformer layers.

    For OWSM-CTC, ``prefix`` [B, 2, d] (language and task embeddings) is prepended to the subsampled
    frames, layers in ``cfg.cross_attn_layers`` attend to ``memory`` (the encoded prompt), and the
    intermediate CTC posteriors after the layers in ``cfg.interctc_layers`` are added back through
    ``conditioning_layer``.
    """

    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        d = cfg.d_model
        self.embed = Conv2dSubsampling(cfg.n_mels, d, cfg.strides)
        if cfg.encoder == "transformer":
            self.encoders = nn.ModuleList(
                TransformerEncoderLayer(d, cfg.n_heads, cfg.enc_ffn) for _ in range(cfg.enc_layers)
            )
        else:
            self.encoders = nn.ModuleList(
                EBranchformerLayer(cfg, i in cfg.cross_attn_layers) for i in range(cfg.enc_layers)
            )
        self.after_norm = LayerNorm(d)
        self.xscale = math.sqrt(d)
        self.normalized_layers = cfg.encoder == "e_branchformer"  # each layer ends with a LayerNorm
        self.interctc_layers = cfg.interctc_layers
        if cfg.interctc_layers:
            self.conditioning_layer = nn.Linear(cfg.vocab_size, d)
        n = cfg.enc_frames + cfg.n_prefix
        self.register_buffer("pe", sinusoidal_positions(n, d), persistent=False)
        self.n_frames = cfg.enc_frames
        # The subsampled frame of padding only (see padding_frame); when set, ``feats`` may stop early.
        self.fill: torch.Tensor | None = None

    def forward(
        self,
        feats: torch.Tensor,
        prefix: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        ctc: "CTC | None" = None,
    ) -> torch.Tensor:
        # feats: [B, T, n_mels] (fp32) -> [B, (2 +) T', d]
        dt = self.embed.out.weight.dtype
        x = self.embed(feats.to(dt))
        if self.fill is not None:  # feats stop early: every later frame sees padding only
            x = torch.cat([x, self.fill.expand(x.shape[0], self.n_frames, -1)], dim=1)[:, : self.n_frames]
        if prefix is not None:
            x = torch.cat([prefix.to(dt), x], dim=1)
        # The residual stream starts in fp32 (the layers compute in the model dtype): the subsampled
        # features times sqrt(d) can exceed the fp16 range (OWSM-CTC v3.2: 4.5e5, v4 1B: 4.4e4).
        x = x.float() * self.xscale + self.pe[: x.shape[1]]
        for i, layer in enumerate(self.encoders):
            x = layer(x, memory)
            if self.normalized_layers:
                x = x.to(dt)  # LayerNorm output: back to the model dtype after the first layer
            if i + 1 in self.interctc_layers:
                x = x + ctc.condition(x.to(dt), self.conditioning_layer).to(x.dtype)
        return self.after_norm(x).to(dt)


@torch.no_grad()
def padding_frame(model: "OWSMModel | OWSMCTCModel") -> torch.Tensor:
    """The subsampled encoder input [1, 1, d] of a stretch of padding (zeros): the same for every
    recording, given global mean normalization. Frames whose receptive field holds padding only need
    not be computed."""
    cfg, enc = model.cfg, model.encoder
    dev = enc.embed.out.weight.device
    feats = model.frontend(torch.zeros(1, cfg.num_samples, device=dev))
    x = enc.embed(feats.to(enc.embed.out.weight.dtype))
    return x[:, x.shape[1] // 2 :][:, :1].clone()


def frames_needed(cfg: OWSMConfig, samples: int) -> int:
    """Feature frames to compute for ``samples`` of audio, so that every subsampled frame left out
    sees padding only (see :func:`padding_frame`)."""
    first_pad = (samples + cfg.n_fft // 2) // cfg.hop_length + 1  # the first frame of padding only
    field, step = 1, 1  # receptive field and step of a subsampled frame, in feature frames
    for s in cfg.strides:
        field, step = field + 2 * step, step * s
    return min(cfg.num_frames, first_pad + field + step)


class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, units: int):
        super().__init__()
        self.self_attn = MultiHeadedAttention(d, n_heads)
        self.src_attn = MultiHeadedAttention(d, n_heads)
        self.feed_forward = FeedForward(d, units, nn.ReLU())
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.norm3 = LayerNorm(d)


class Decoder(nn.Module):
    def __init__(self, cfg: OWSMConfig, max_positions: int = 5000):
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Sequential(nn.Embedding(cfg.vocab_size, d))
        self.decoders = nn.ModuleList(
            DecoderLayer(d, cfg.dec_heads, cfg.dec_ffn) for _ in range(cfg.dec_layers)
        )
        self.after_norm = LayerNorm(d)
        self.output_layer = nn.Linear(d, cfg.vocab_size)
        self.xscale = math.sqrt(d)
        self.register_buffer("pe", sinusoidal_positions(max_positions, d), persistent=False)
        # Self-attention over [cache | new] without concatenating them (faster in eager PyTorch). For a
        # single new token TensorRT fuses the concatenated form into one MHA kernel instead.
        self.split_attention = True
        # Export option: self- and/or cross-attention ("self", "cross") as attn_plugin ops, which
        # become TensorRT plugins.
        self.plugins: frozenset[str] = frozenset()

    def cross_kv(self, enc: torch.Tensor) -> torch.Tensor:
        """Project encoder output to every layer's cross-attention keys/values once.

        enc: [B, T, d] -> [L, 2, B, H, T, d_k]
        """
        kv = []
        for layer in self.decoders:
            att = layer.src_attn
            kv.append(torch.stack([att.heads(att.linear_k(enc)), att.heads(att.linear_v(enc))]))
        return torch.stack(kv)

    def step(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
        self_k: list[torch.Tensor],
        self_v: list[torch.Tensor],
        cross_k: list[torch.Tensor],
        cross_v: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``S`` new tokens per row through the decoder against a static KV cache.

        tokens:          [B, S] token ids; row b at positions offset[b] .. offset[b]+S-1
        offset:          [B] int64, number of tokens already in each row's cache
        self_k, self_v:  per layer [B, H, C, d_k] cache with capacity C (slots >= offset[b] are ignored)
        cross_k, cross_v: per layer [B, H, T, d_k] from :meth:`cross_kv`
        returns:         logits [B, S, V] (fp32), new keys/values [L, 2, B, H, S, d_k]

        The caller writes the new keys/values into the cache at ``offset``.
        """
        b, s = tokens.shape
        cap = self_k[0].shape[2]
        dtype = self.output_layer.weight.dtype
        new_idx = torch.arange(s, device=tokens.device)
        positions = offset.unsqueeze(1) + new_idx  # [B, S]
        x = self.embed[0](tokens) * self.xscale + self.pe[positions].to(dtype)  # [B, S, d]

        # Each row attends to its cache slots < offset[b] and causally to its new tokens.
        slot = torch.arange(cap, device=tokens.device)
        cache_ok = (slot < offset.unsqueeze(1))[:, None, None, :].expand(b, 1, s, cap)  # [B, 1, S, C]
        new_ok = (new_idx.unsqueeze(0) <= new_idx.unsqueeze(1)).expand(b, 1, s, s)  # [B, 1, S, S]
        mask = torch.cat([cache_ok, new_ok], dim=-1)  # [B, 1, S, C+S]

        new_kv = []
        for i, layer in enumerate(self.decoders):
            att = layer.self_attn
            if "self" in self.plugins:
                x = x + self._plugin_self_attention(att, layer.norm1(x), offset, self_k[i], self_v[i], new_kv)
            else:
                q, k, v = att.qkv(layer.norm1(x))
                new_kv.append(torch.stack([k, v]))
                if self.split_attention:
                    y = cached_attention(q, self_k[i], self_v[i], k, v, mask)
                else:
                    k_all = torch.cat([self_k[i], k], dim=2)  # [B, H, C+S, d_k]
                    v_all = torch.cat([self_v[i], v], dim=2)
                    y = explicit_attention(q, k_all, v_all, mask)
                x = x + att.merge(y)

            src = layer.src_attn
            if "cross" in self.plugins:
                q = src.linear_q(layer.norm2(x))
                x = x + src.linear_out(attn_plugin.cross_attn(q, cross_k[i], cross_v[i]))
            else:
                q = src.heads(src.linear_q(layer.norm2(x)))
                x = x + src.merge(src.attend(q, cross_k[i], cross_v[i]))

            x = x + layer.feed_forward(layer.norm3(x))

        logits = self.output_layer(self.after_norm(x)).float()  # [B, S, V]
        return logits, torch.stack(new_kv)

    def _plugin_self_attention(
        self, att: MultiHeadedAttention, x: torch.Tensor, offset: torch.Tensor, k_cache, v_cache, new_kv: list
    ) -> torch.Tensor:
        """Self-attention as one attn_plugin op on the fused Q/K/V projection (see ``plugins``)."""
        if hasattr(att, "linear_qkv"):
            qkv = att.linear_qkv(x)  # [B, S, 3d]
        else:
            qkv = torch.cat([att.linear_q(x), att.linear_k(x), att.linear_v(x)], dim=-1)
        b, s = x.shape[:2]
        new_kv.append(qkv.view(b, s, 3, att.h, att.d_k)[:, :, 1:].permute(2, 0, 3, 1, 4))  # [2, B, H, S, d_k]
        return att.linear_out(attn_plugin.self_attn(qkv, k_cache, v_cache, offset))


class CTC(nn.Module):
    """The CTC head: OWSM-CTC's output (and intermediate CTC), the encoder-decoder models' drafts.

    With ``plugin`` set (TensorRT export, OWSM-CTC), the argmax and the softmax over the vocabulary are
    custom ops that become TensorRT plugins (see :mod:`owsm_trt.plugins`), on time-major rows [T, B, V]:
    TensorRT keeps the encoder's residual stream time-major after the cross-attention layers, and would
    otherwise transpose the vocabulary-wide logits for the plugins (several ms at batch 16). The
    transposes are on the d-wide encoder states instead (free where they are time-major anyway)."""

    plugin = False

    def __init__(self, d: int, vocab_size: int):
        super().__init__()
        self.ctc_lo = nn.Linear(d, vocab_size)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        # [B, T, d] -> best-path ids [B, T] (int32; blank = 0, repeats not collapsed)
        if CTC.plugin:
            from . import plugins

            return plugins.argmax_rows(self.ctc_lo(enc.transpose(0, 1))).transpose(0, 1)
        return self.ctc_lo(enc).argmax(-1).to(torch.int32)

    def condition(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        """Intermediate CTC conditioning: ``layer`` (vocabulary -> d) of the posteriors of ``x`` [B, T, d],
        computed in fp32 and rounded to x.dtype."""
        if CTC.plugin:
            from . import plugins

            return layer(plugins.softmax_rows(self.ctc_lo(x.transpose(0, 1)))).transpose(0, 1)
        return layer(self.ctc_lo(x).float().softmax(-1).to(x.dtype))


class OWSMModel(nn.Module):
    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = Frontend(cfg)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.ctc = CTC(cfg.d_model, cfg.vocab_size)

    def fuse_projections(self) -> None:
        """Fuse Q/K/V of every self-attention (cross-attention Q and K/V run at different times)."""
        for layer in self.encoder.encoders:
            layer.self_attn.fuse_qkv()
        for layer in self.decoder.decoders:
            layer.self_attn.fuse_qkv()

    def encode(self, speech: torch.Tensor) -> torch.Tensor:
        # speech: [B, num_samples] -> cross_kv: [L, 2, B, H, T, d_k]
        return self.decoder.cross_kv(self.encoder(self.frontend(speech)))


class PromptEncoder(nn.Module):
    """OWSM-CTC's Transformer over the prompt tokens (ESPnet ``TransformerEncoder``, no input layer)."""

    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        self.encoders = nn.ModuleList(
            TransformerEncoderLayer(cfg.prompt_d, cfg.prompt_heads, cfg.prompt_ffn)
            for _ in range(cfg.prompt_layers)
        )
        self.after_norm = LayerNorm(cfg.prompt_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            x = layer(x)
        return self.after_norm(x)


class OWSMCTCModel(nn.Module):
    """OWSM-CTC: an encoder conditioned on a prompt and on language/task tokens, read with CTC."""

    def __init__(self, cfg: OWSMConfig, max_prompt: int = 5000):
        super().__init__()
        self.cfg = cfg
        self.frontend = Frontend(cfg)
        self.encoder = Encoder(cfg)
        self.prompt_encoder = PromptEncoder(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.prompt_d)  # shared by the prompt and the prefix
        same = cfg.prompt_d == cfg.d_model
        self.embed_proj = nn.Identity() if same else nn.Linear(cfg.prompt_d, cfg.d_model)
        self.prompt_proj = nn.Identity() if same else nn.Linear(cfg.prompt_d, cfg.d_model)
        self.ctc = CTC(cfg.d_model, cfg.vocab_size)
        self.prompt_xscale = math.sqrt(cfg.prompt_d)
        self.register_buffer("prompt_pe", sinusoidal_positions(max_prompt, cfg.prompt_d), persistent=False)

    def fuse_projections(self) -> None:
        for layer in list(self.encoder.encoders) + list(self.prompt_encoder.encoders):
            layer.self_attn.fuse_qkv()

    def forward(self, feats: torch.Tensor, prompt: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """feats [B, T, n_mels], prompt [B, P] and prefix [B, 2] (language, task) token ids
        -> CTC best-path ids [B, 2 + T'] (the first two frames belong to the prefix)."""
        p = self.embed(prompt)
        p = p * self.prompt_xscale + self.prompt_pe[: p.shape[1]].to(p.dtype)
        memory = self.prompt_proj(self.prompt_encoder(p))
        enc = self.encoder(feats, self.embed_proj(self.embed(prefix)), memory, self.ctc)
        return self.ctc(enc)


def convert_espnet_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map an ESPnet s2t checkpoint onto :class:`OWSMModel` parameter names."""
    out = {}
    for k, v in sd.items():
        if k.startswith("frontend.logmel."):
            k = "frontend." + k[len("frontend.logmel.") :]
        elif k.startswith("normalize."):
            k = "frontend." + k[len("normalize.") :]
        # Legacy subsampling layout: embed.out = Sequential(Linear, PositionalEncoding)
        k = k.replace("encoder.embed.out.0.", "encoder.embed.out.")
        if k.startswith("encoder.embed.out.1.") or k.endswith(".pe"):
            continue
        out[k] = v
    return out


def load_model(
    cfg: OWSMConfig,
    checkpoint: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    fuse: bool = True,
) -> OWSMModel | OWSMCTCModel:
    model = OWSMCTCModel(cfg) if cfg.ctc_only else OWSMModel(cfg)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(convert_espnet_state_dict(sd), strict=True)
    model.eval().requires_grad_(False)
    if fuse:
        model.fuse_projections()
    for name, module in model.named_children():
        if name == "frontend":
            module.to(device=device)  # stays fp32
        else:
            module.to(device=device, dtype=dtype)
    return model.to(device)
