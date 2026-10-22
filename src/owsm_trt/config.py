"""Model configuration parsed from an ESPnet ``s2t`` training config (``config.yaml``)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    _Loader = yaml.CSafeLoader
except AttributeError:  # libyaml not available
    _Loader = yaml.SafeLoader


@dataclass(frozen=True)
class OWSMConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    enc_layers: int
    enc_ffn: int
    cgmlp_units: int
    cgmlp_kernel: int
    merge_kernel: int
    subsample_convs: int  # 2 for "conv2d" (x4), 3 for "conv2d8" (x8)
    dec_layers: int
    dec_ffn: int
    dec_heads: int
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    sample_rate: int
    speech_seconds: float
    time_resolution: float  # seconds per timestamp token step

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.speech_seconds)

    @property
    def num_frames(self) -> int:
        # STFT with center=True
        return self.num_samples // self.hop_length + 1

    @property
    def enc_frames(self) -> int:
        t = self.num_frames
        for _ in range(self.subsample_convs):
            t = (t - 1) // 2
        return t


_SUBSAMPLE = {"conv2d": 2, "conv2d8": 3}


def _expect(cfg: dict, key: str, allowed: set, default=None) -> None:
    value = cfg.get(key, default)
    if value not in allowed:
        raise NotImplementedError(f"unsupported {key}={value!r} (supported: {sorted(map(str, allowed))})")


def load_espnet_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)


def parse_config(cfg: dict) -> OWSMConfig:
    """Build an :class:`OWSMConfig` from a loaded ESPnet s2t config dict.

    Only the architecture used by OWSM v3.1 / v4 is supported: default STFT/log-mel frontend,
    global MVN, E-Branchformer encoder with absolute positional encoding, Transformer decoder.
    """
    _expect(cfg, "frontend", {"default"})
    _expect(cfg, "normalize", {"global_mvn"})
    _expect(cfg, "encoder", {"e_branchformer"})
    _expect(cfg, "decoder", {"transformer"})
    _expect(cfg, "preencoder", {None})
    _expect(cfg, "postencoder", {None})

    enc = cfg["encoder_conf"]
    _expect(enc, "attention_layer_type", {"selfattn"}, "rel_selfattn")
    _expect(enc, "pos_enc_layer_type", {"abs_pos"}, "rel_pos")
    _expect(enc, "input_layer", set(_SUBSAMPLE), "conv2d")
    _expect(enc, "use_linear_after_conv", {False}, False)
    _expect(enc, "gate_activation", {"identity"}, "identity")
    _expect(enc, "positionwise_layer_type", {"linear"}, "linear")
    _expect(enc, "use_ffn", {True}, True)
    _expect(enc, "macaron_ffn", {True}, True)
    _expect(enc, "ffn_activation_type", {"swish"}, "swish")
    if enc.get("interctc_layer_idx") and enc.get("interctc_use_conditioning"):
        raise NotImplementedError("interctc conditioning is not supported")

    dec = cfg["decoder_conf"]
    _expect(dec, "input_layer", {"embed"}, "embed")
    _expect(dec, "normalize_before", {True}, True)
    _expect(dec, "concat_after", {False}, False)
    _expect(dec, "qk_norm", {False}, False)

    front = cfg.get("frontend_conf") or {}
    _expect(front, "window", {"hann"}, "hann")
    _expect(front, "center", {True}, True)
    # The mel filterbank itself is stored in the checkpoint, so only the STFT options matter here.
    _expect(front, "normalized", {False}, False)
    _expect(front, "onesided", {True}, True)
    fs = front.get("fs", 16000)
    if isinstance(fs, str):
        fs = int(float(fs.lower().rstrip("k")) * 1000) if fs.lower().endswith("k") else int(fs)

    pre = cfg.get("preprocessor_conf") or {}
    d_model = enc.get("output_size", 256)
    return OWSMConfig(
        vocab_size=len(cfg["token_list"]),
        d_model=d_model,
        n_heads=enc.get("attention_heads", 4),
        enc_layers=enc.get("num_blocks", 12),
        enc_ffn=enc.get("linear_units", 2048),
        cgmlp_units=enc.get("cgmlp_linear_units", 2048),
        cgmlp_kernel=enc.get("cgmlp_conv_kernel", 31),
        merge_kernel=enc.get("merge_conv_kernel", 3),
        subsample_convs=_SUBSAMPLE[enc.get("input_layer", "conv2d")],
        dec_layers=dec.get("num_blocks", 6),
        dec_ffn=dec.get("linear_units", 2048),
        dec_heads=dec.get("attention_heads", 4),
        n_mels=front.get("n_mels", 80),
        n_fft=front.get("n_fft", 512),
        win_length=front.get("win_length") or front.get("n_fft", 512),
        hop_length=front.get("hop_length", 128),
        sample_rate=fs,
        speech_seconds=float(pre.get("speech_length", 30)),
        time_resolution=float(pre.get("speech_resolution", 0.02)),
    )
