"""Pure-PyTorch reimplementation of OWSM v3.1 / v4 (ESPnet ``s2t`` E-Branchformer models).

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


class Frontend(nn.Module):
    """STFT -> power -> log-mel -> global MVN. Always runs in fp32."""

    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        self.n_fft, self.hop, self.win = cfg.n_fft, cfg.hop_length, cfg.win_length
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)
        # Filled from the checkpoint ("frontend.logmel.melmat", "normalize.mean/std").
        self.register_buffer("melmat", torch.zeros(cfg.n_fft // 2 + 1, cfg.n_mels))
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
        return (mel - self.mean) / self.std


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_mels: int, d: int, n_convs: int):
        super().__init__()
        layers: list[nn.Module] = []
        f = n_mels
        for i in range(n_convs):
            layers += [nn.Conv2d(1 if i == 0 else d, d, 3, 2), nn.ReLU()]
            f = (f - 1) // 2
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(d * f, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, T', d]
        x = self.conv(x.unsqueeze(1))  # [B, d, T', F']
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))


class MultiHeadedAttention(nn.Module):
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
        return self.merge(F.scaled_dot_product_attention(*self.qkv(x)))


def cached_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Attention over [cache | new] without materialising the concatenated keys/values.

    q, k, v: [B, H, S, d_k]; k_cache, v_cache: [B, H, C, d_k]; mask: [S, C+S] (True = attend)
    Only the score rows are concatenated, so the cache is read in place instead of copied every step.
    """
    cap = k_cache.shape[2]
    q = q * (q.shape[-1] ** -0.5)
    scores = torch.cat([q @ k_cache.transpose(-1, -2), q @ k.transpose(-1, -2)], dim=-1)  # [B, H, S, C+S]
    scores = scores.float().masked_fill(~mask, float("-inf"))
    p = scores.softmax(-1).to(q.dtype)
    return p[..., :cap] @ v_cache + p[..., cap:] @ v  # [B, H, S, d_k]


class FeedForward(nn.Module):
    def __init__(self, d: int, units: int, activation: nn.Module):
        super().__init__()
        self.w_1 = nn.Linear(d, units)
        self.w_2 = nn.Linear(units, d)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.activation(self.w_1(x)))


class ConvolutionalSpatialGatingUnit(nn.Module):
    def __init__(self, units: int, kernel: int):
        super().__init__()
        c = units // 2
        self.norm = LayerNorm(c)
        self.conv = nn.Conv1d(c, c, kernel, 1, (kernel - 1) // 2, groups=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_g = x.chunk(2, dim=-1)
        x_g = self.conv(self.norm(x_g).transpose(1, 2)).transpose(1, 2)
        return x_r * x_g


class ConvolutionalGatingMLP(nn.Module):
    def __init__(self, d: int, units: int, kernel: int):
        super().__init__()
        self.channel_proj1 = nn.Sequential(nn.Linear(d, units), nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(units, kernel)
        self.channel_proj2 = nn.Linear(units // 2, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_proj2(self.csgu(self.channel_proj1(x)))


class EBranchformerLayer(nn.Module):
    def __init__(self, cfg: OWSMConfig):
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
        self.depthwise_conv_fusion = nn.Conv1d(2 * d, 2 * d, k, 1, (k - 1) // 2, groups=2 * d)
        self.merge_proj = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward_macaron(self.norm_ff_macaron(x))
        x1 = self.attn(self.norm_mha(x))
        x2 = self.cgmlp(self.norm_mlp(x))
        x_concat = torch.cat([x1, x2], dim=-1)  # [B, T, 2d]
        x_tmp = self.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
        x = x + self.merge_proj(x_concat + x_tmp)
        x = x + 0.5 * self.feed_forward(self.norm_ff(x))
        return self.norm_final(x)


class Encoder(nn.Module):
    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        self.embed = Conv2dSubsampling(cfg.n_mels, cfg.d_model, cfg.subsample_convs)
        self.encoders = nn.ModuleList(EBranchformerLayer(cfg) for _ in range(cfg.enc_layers))
        self.after_norm = LayerNorm(cfg.d_model)
        self.xscale = math.sqrt(cfg.d_model)
        self.register_buffer("pe", sinusoidal_positions(cfg.enc_frames, cfg.d_model), persistent=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, n_mels] (fp32) -> [B, T', d]
        x = self.embed(feats.to(self.embed.out.weight.dtype))
        x = x * self.xscale + self.pe[: x.shape[1]].to(x.dtype)
        for layer in self.encoders:
            x = layer(x)
        return self.after_norm(x)


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
        # Eager PyTorch is faster with split attention; TensorRT fuses the concat + SDPA form into one
        # MHA kernel, so the export switches this off.
        self.split_attention = True

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
        """Run ``S`` new tokens through the decoder against a static KV cache.

        tokens:          [B, S] token ids at positions offset .. offset+S-1
        offset:          [1] int64, number of tokens already in the cache
        self_k, self_v:  per layer [B, H, C, d_k] cache with capacity C (slots >= offset are ignored)
        cross_k, cross_v: per layer [B, H, T, d_k] from :meth:`cross_kv`
        returns:         logits of the last position [B, V] (fp32), new keys/values [L, 2, B, H, S, d_k]

        The caller writes the new keys/values into the cache at ``offset``.
        """
        s = tokens.shape[1]
        cap = self_k[0].shape[2]
        dtype = self.output_layer.weight.dtype
        new_idx = torch.arange(s, device=tokens.device)
        x = self.embed[0](tokens) * self.xscale + self.pe[offset + new_idx].to(dtype)  # [B, S, d]

        # Attend to the cache slots < offset and causally to the new tokens.
        slot = torch.arange(cap, device=tokens.device)
        cache_ok = (slot < offset).unsqueeze(0).expand(s, cap)  # [S, C]
        new_ok = new_idx.unsqueeze(0) <= new_idx.unsqueeze(1)  # [S, S]
        mask = torch.cat([cache_ok, new_ok], dim=1)  # [S, C+S]

        new_kv = []
        for i, layer in enumerate(self.decoders):
            att = layer.self_attn
            q, k, v = att.qkv(layer.norm1(x))
            new_kv.append(torch.stack([k, v]))
            if self.split_attention:
                y = cached_attention(q, self_k[i], self_v[i], k, v, mask)
            else:
                k_all = torch.cat([self_k[i], k], dim=2)  # [B, H, C+S, d_k]
                v_all = torch.cat([self_v[i], v], dim=2)
                y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
            x = x + att.merge(y)

            src = layer.src_attn
            q = src.heads(src.linear_q(layer.norm2(x)))
            x = x + src.merge(F.scaled_dot_product_attention(q, cross_k[i], cross_v[i]))

            x = x + layer.feed_forward(layer.norm3(x))

        logits = self.output_layer(self.after_norm(x[:, -1])).float()  # [B, V]
        return logits, torch.stack(new_kv)


class OWSMModel(nn.Module):
    def __init__(self, cfg: OWSMConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = Frontend(cfg)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def fuse_projections(self) -> None:
        """Fuse Q/K/V of every self-attention (cross-attention Q and K/V run at different times)."""
        for layer in self.encoder.encoders:
            layer.attn.fuse_qkv()
        for layer in self.decoder.decoders:
            layer.self_attn.fuse_qkv()

    def encode(self, speech: torch.Tensor) -> torch.Tensor:
        # speech: [B, num_samples] -> cross_kv: [L, 2, B, H, T, d_k]
        return self.decoder.cross_kv(self.encoder(self.frontend(speech)))


def convert_espnet_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map an ESPnet s2t checkpoint onto :class:`OWSMModel` parameter names."""
    out = {}
    for k, v in sd.items():
        if k.startswith("ctc."):
            continue  # CTC head is unused for attention decoding
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
) -> OWSMModel:
    model = OWSMModel(cfg)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(convert_espnet_state_dict(sd), strict=True)
    model.eval().requires_grad_(False)
    if fuse:
        model.fuse_projections()
    model.encoder.to(device=device, dtype=dtype)
    model.decoder.to(device=device, dtype=dtype)
    model.frontend.to(device=device)  # stays fp32
    return model
