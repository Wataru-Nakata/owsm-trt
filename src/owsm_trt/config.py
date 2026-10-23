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
    subsample: str  # input layer: "conv2d" (x4), "conv2d2" (x2) or "conv2d8" (x8)
    dec_layers: int  # 0 for CTC-only models
    dec_ffn: int
    dec_heads: int
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    sample_rate: int
    speech_seconds: float
    time_resolution: float  # seconds per timestamp token step
    encoder: str = "e_branchformer"  # or "transformer" (OWSM v1-v3, OWLS)
    normalize: str = "global_mvn"  # or "utterance_mvn"
    # OWSM-CTC: no decoder; the encoder reads a prompt through a small Transformer, is prefixed with
    # the language and task embeddings, and feeds intermediate CTC posteriors back into itself.
    ctc_only: bool = False
    prompt_d: int = 0
    prompt_heads: int = 0
    prompt_ffn: int = 0
    prompt_layers: int = 0
    interctc_layers: tuple[int, ...] = ()  # 1-based: conditioning after these encoder layers
    cross_attn_layers: tuple[int, ...] = ()  # 0-based encoder layers that attend to the prompt

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
    def strides(self) -> tuple[int, ...]:
        """Time/frequency strides of the 3x3 convolutions of the input layer."""
        return _SUBSAMPLE[self.subsample]

    @property
    def enc_frames(self) -> int:
        """Encoder output frames for one input window (without the OWSM-CTC prefix)."""
        t = self.num_frames
        for s in self.strides:
            t = (t - 3) // s + 1
        return t

    @property
    def n_prefix(self) -> int:
        """Language + task embeddings an OWSM-CTC encoder prepends to the speech frames."""
        return 2 if self.ctc_only else 0


_SUBSAMPLE = {"conv2d": (2, 2), "conv2d2": (2, 1), "conv2d8": (2, 2, 2)}


def _expect(cfg: dict, key: str, allowed: set, default=None) -> None:
    value = cfg.get(key, default)
    if value not in allowed:
        raise NotImplementedError(f"unsupported {key}={value!r} (supported: {sorted(map(str, allowed))})")


def load_espnet_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)


def parse_config(cfg: dict) -> OWSMConfig:
    """Build an :class:`OWSMConfig` from a loaded ESPnet s2t / s2t_ctc config dict.

    Supported: the OWSM family as released by ESPnet -- encoder-decoder models with an E-Branchformer
    (OWSM v2-ebf, v3.1, v3.2, v4, POWSM) or Transformer encoder (OWSM v1-v3, OWLS) and a Transformer
    decoder, and the CTC-only OWSM-CTC models (v3.1, v3.2, v4, POWSM-CTC). All use the default
    STFT/log-mel frontend and absolute positional encodings.
    """
    ctc_only = cfg.get("model", "espnet") == "espnet_ctc"
    _expect(cfg, "model", {"espnet", "espnet_ctc"}, "espnet")
    _expect(cfg, "frontend", {"default"})
    _expect(cfg, "normalize", {"global_mvn", "utterance_mvn"})
    if cfg.get("normalize") == "utterance_mvn":
        _expect(cfg.get("normalize_conf") or {}, "norm_vars", {False}, False)
        _expect(cfg.get("normalize_conf") or {}, "norm_means", {True}, True)
    _expect(cfg, "encoder", {"e_branchformer_ctc"} if ctc_only else {"e_branchformer", "transformer"})
    _expect(cfg, "decoder", {None} if ctc_only else {"transformer"})
    _expect(cfg, "preencoder", {None})
    _expect(cfg, "postencoder", {None})

    enc = cfg["encoder_conf"]
    _expect(enc, "input_layer", set(_SUBSAMPLE), "conv2d8" if ctc_only else "conv2d")
    _expect(enc, "qk_norm", {False}, False)
    if cfg["encoder"] == "transformer":
        _expect(enc, "pos_enc_layer_type", {"abs_pos"}, "abs_pos")
        _expect(enc, "normalize_before", {True}, True)
        _expect(enc, "concat_after", {False}, False)
        _expect(enc, "positionwise_layer_type", {"linear"}, "linear")
        _expect(enc, "interctc_layer_idx", {None, ()}, None)
    else:
        _expect(enc, "attention_layer_type", {"selfattn"}, "rel_selfattn")
        _expect(enc, "pos_enc_layer_type", {"abs_pos"}, "rel_pos")
        _expect(enc, "use_linear_after_conv", {False}, False)
        _expect(enc, "gate_activation", {"identity"}, "identity")
        _expect(enc, "positionwise_layer_type", {"linear"}, "linear")
        _expect(enc, "use_ffn", {True}, False)
        _expect(enc, "macaron_ffn", {True}, False)
        _expect(enc, "ffn_activation_type", {"swish"}, "swish")
    interctc: tuple[int, ...] = ()
    cross: tuple[int, ...] = ()
    n_blocks = enc.get("num_blocks", 12)
    if ctc_only:
        _expect(enc, "interctc_use_conditioning", {True}, False)
        interctc = tuple(enc.get("interctc_layer_idx") or ())
        use_cross = enc.get("use_cross_attention", True)
        if isinstance(use_cross, bool):
            use_cross = [use_cross] * n_blocks
        cross = tuple(i for i, u in enumerate(use_cross) if u)
        _expect(cfg, "promptencoder", {"transformer"})
    elif enc.get("interctc_layer_idx") and enc.get("interctc_use_conditioning"):
        raise NotImplementedError("interctc conditioning in an encoder-decoder model is not supported")

    dec = cfg.get("decoder_conf") or {}
    if not ctc_only:
        _expect(dec, "input_layer", {"embed"}, "embed")
        _expect(dec, "normalize_before", {True}, True)
        _expect(dec, "concat_after", {False}, False)
        _expect(dec, "qk_norm", {False}, False)
    prompt = cfg.get("promptencoder_conf") or {}
    if ctc_only:
        _expect(prompt, "input_layer", {None}, None)  # the s2t_ctc task builds it with input_layer=None
        _expect(prompt, "normalize_before", {True}, True)
        _expect(prompt, "concat_after", {False}, False)
        _expect(prompt, "positionwise_layer_type", {"linear"}, "linear")

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
        enc_layers=n_blocks,
        enc_ffn=enc.get("linear_units", 2048),
        cgmlp_units=enc.get("cgmlp_linear_units", 2048),
        cgmlp_kernel=enc.get("cgmlp_conv_kernel", 31),
        merge_kernel=enc.get("merge_conv_kernel", 3),
        subsample=enc.get("input_layer", "conv2d8" if ctc_only else "conv2d"),
        dec_layers=0 if ctc_only else dec.get("num_blocks", 6),
        dec_ffn=0 if ctc_only else dec.get("linear_units", 2048),
        dec_heads=0 if ctc_only else dec.get("attention_heads", 4),
        n_mels=front.get("n_mels", 80),
        n_fft=front.get("n_fft", 512),
        win_length=front.get("win_length") or front.get("n_fft", 512),
        hop_length=front.get("hop_length", 128),
        sample_rate=fs,
        speech_seconds=float(pre.get("speech_length", 30)),
        time_resolution=float(pre.get("speech_resolution", 0.02)),
        encoder="e_branchformer" if ctc_only else cfg["encoder"],
        normalize=cfg["normalize"],
        ctc_only=ctc_only,
        prompt_d=prompt.get("output_size", 256) if ctc_only else 0,
        prompt_heads=prompt.get("attention_heads", 4) if ctc_only else 0,
        prompt_ffn=prompt.get("linear_units", 2048) if ctc_only else 0,
        prompt_layers=prompt.get("num_blocks", 6) if ctc_only else 0,
        interctc_layers=interctc,
        cross_attn_layers=cross,
    )
