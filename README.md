# owsm-trt

Fast inference for the [OWSM](https://www.wavlab.org/activities/2024/owsm/) family (Open Whisper-style Speech Models,
ESPnet) with TensorRT. You can `pip install` it, and it does not depend on ESPnet.

```bash
pip install owsm-trt
```

```python
from owsm_trt import OWSM

model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B")   # first call builds TensorRT engines (~5 min, cached)
print(model.transcribe("speech.wav", lang="eng").text)

model.transcribe(["a.wav", "b.wav", "c.wav"], batch_size=8)  # batched
model.transcribe("speech.wav", beam_size=5)                  # ESPnet's default beam search
model.transcribe("speech.wav", timestamps=True).segments     # utterance timestamps
model.transcribe("speech.wav", lang="eng", task="st_deu")    # speech translation
model.detect_language("speech.wav")                          # [("eng", 0.999), ...]
model.transcribe_long("podcast.wav").segments                # > 30 s, with timestamps

ctc = OWSM.from_pretrained("espnet/owsm_ctc_v4_1B")          # OWSM-CTC: same API, no decoder
ctc.transcribe(["a.wav", "b.wav"], lang="eng", batch_size=16)

# experimental: FP8 encoder (Hopper), calibrated on a few recordings like the ones you will transcribe
fast = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", fp8_calibration=["c1.wav", "c2.wav", "c3.wav"])
```

```bash
owsm-trt build --model espnet/owsm_v4_medium_1B          # optional: pre-build engines
owsm-trt transcribe --lang eng speech.wav
owsm-trt transcribe --beam-size 5 speech.wav
owsm-trt transcribe --long --json podcast.wav
owsm-trt bench --batch-size 8
```

## Speed

Test setup:
* model: `espnet/owsm_v4_medium_1B` (OWSM-CTC: `espnet/owsm_ctc_v4_1B`)
* GPU: one H200
* data: 73 LibriSpeech dev-clean utterances (481 s of audio)
* RTFx: seconds of audio transcribed per second (median of 3 runs)
* "same as ESPnet": transcripts identical to ESPnet `Speech2Text` in fp32 with the same search

| setup | batch | RTFx | WER | same as ESPnet |
|---|---:|---:|---:|---:|
| ESPnet `Speech2Text`, beam 5 (its default), fp32 | 1 | 15 | 4.00% | |
| ESPnet `Speech2Text`, greedy, fp32 | 1 | 25 | 3.91% | |
| **owsm-trt, greedy** | 1 | **989** | 3.74% | 72/73 |
| **owsm-trt, greedy** | 8 | **1842** | 3.91% | 73/73 |
| **owsm-trt, greedy** | 16 | **1913** | 3.91% | 73/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 1 | 985 | 3.91% | 69/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 8 | 2067 | 3.83% | 69/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 16 | 2132 | 3.83% | 70/73 |
| **owsm-trt, beam 5** | 1 | **103** | 4.09% | 72/73 |
| ESPnet OWSM-CTC `best_path`, fp32 | 1 | 131 | 4.43% | |
| **owsm-trt, OWSM-CTC** | 1 | **978** | 4.26% | 64/73 |
| **owsm-trt, OWSM-CTC** | 16 | **1623** | 4.52% | 69/73 |
| owsm-trt, OWSM-CTC, FP8 (experimental) | 1 | 1050 | 4.52% | 65/73 |
| owsm-trt, OWSM-CTC, FP8 (experimental) | 16 | 1981 | 4.52% | 62/73 |

For one stream that is 40× ESPnet's greedy decoding and 66× its default beam search, and 77× with batch 16; beam
search itself is 7× faster, and OWSM-CTC 7× to 12×.

Rows are TensorRT fp16 unless marked FP8. Where their transcripts differ from ESPnet's fp32 output, it is by a comma,
a casing or a single word (fp16 rounding at a near-tie), and the WER moves by at most 0.17 points. In fp32
(`backend="torch", dtype="float32"`) the transcripts are identical to ESPnet's: 73/73 for greedy, beam 5, greedy and
beam 5 with timestamps, and OWSM-CTC. (TensorRT fp16 greedy with timestamps: 71/73, RTFx 739.)

The FP8 rows were calibrated on the first 8 test utterances. On the other 65 their WER is within one word of ESPnet's
fp32: 4.00-4.11% vs 4.11% for the 1B model, 4.84% vs 4.74% for OWSM-CTC. That is one test set; check FP8 on your own
data before relying on it. FP8 pays off with batches; at batch 1 the 1B model gains nothing from it.

Single-utterance greedy decoding takes 5.6 ms for the first test utterance (6 s): 0.2 ms features, 3.5 ms encoder,
1.9 ms decoding. With batch 16, sixteen utterances take 98 ms, 45 of them in the encoder.

The first `from_pretrained` builds the engines: 4.7 minutes for the 1B model on an H200 (3.8 minutes with FP8),
cached afterwards.

## How it works

* **No ESPnet.** The models are reimplemented in plain PyTorch (`owsm_trt/modeling.py`) and the official checkpoints
  load with `strict=True`.
* **TensorRT engines are built on first use.** The encoder and a decoder call are exported to ONNX and compiled
  into strongly-typed fp16 engines, cached under `~/.cache/owsm_trt` (set `OWSM_TRT_CACHE` to move them). Engines
  depend on the GPU and the TensorRT version.
  * The encoder and the decoder have separate optimization profiles for batch 1 and for batches (and the decoder
    one for prompt prefill). Kernels tuned for batch 16 waste much of the GPU at batch 1 (encoder: 4.1 instead of
    5.0 ms).
  * Calls that check several drafted tokens (below) run on a second engine built from the same network at a higher
    TensorRT optimization level (4), which finds kernels about twice as fast for them (1.3 vs 2.3 ms on the 1B
    model).
  * The encoder input and its first residual connection are kept in fp32 (for Transformer encoders the whole residual
    stream): on some checkpoints the subsampled features times √d exceed the fp16 range (up to 4.5·10⁵ on OWSM-CTC
    v3.2).
  * The exported decoder writes attention out with the scale on the query. The ONNX exporter's
    `scaled_dot_product_attention` also scales the keys, which TensorRT then recomputes for all cross-attention keys
    and the whole KV cache in every multi-token call (40% of such a call at batch 16).
* **The padding is not subsampled.** Inputs are padded to 30 s as in ESPnet, but the encoder's convolutional
  subsampling (a third of its time at batch 16) only runs up to the end of the longest recording in the batch, plus
  its receptive field. Every later subsampled frame sees padding only, so it is the same for every recording: it is
  computed once when the engine is built and filled in. The encoder output is that of the full input (up to fp16
  rounding); 8 s recordings take 27-31% less encoder time at batch 8-16. This needs global mean normalization,
  which every OWSM model has except OWSM-CTC v3.2.
* **Decoding work is minimized.**
  * Cross-attention keys/values are projected once per utterance.
  * The self-attention KV cache is static and comes in length buckets (64/128/256/512), so a step only reads as
    much cache as the sequence needs.
  * Each decoding step (engine, cache write, token choice) is one CUDA-graph replay; so is the encoder (one graph per
    5 s of input length). The host checks for the end every 4 steps.
  * Batched inputs are sorted by duration, so an encoder batch stops early and a decoding batch does not wait on one
    long utterance.
* **CTC drafts (greedy).** The encoder-decoder models also have a CTC head. Its best path is a draft of the
  transcript, usually right down to casing and punctuation. One decoder call checks up to 31 drafted tokens per
  utterance and keeps those that equal the decoder's own choice, so the output is exactly the greedy output with far
  fewer calls (almost half of the test utterances need a single call). In a batch every utterance advances by its own
  amount, at its own cache offset. When the decoder and the draft disagree, the draft is re-anchored on the last few
  emitted tokens. The draft is used for `task="asr"` only (the CTC head is trained on transcripts), and dropped when
  it stops being accepted (POWSM's CTC head writes phones). `ctc_draft=False` turns it off. A call checks at most
  256 tokens (rows × window), so at batch 16 a row gets up to 15 drafted tokens per call: wider calls cost more than
  they save.
* **Rows are refilled (greedy ASR of more than one batch).** Utterances need from 1 to 25 decoder calls, and a fixed
  batch would wait for its slowest one. Instead a row whose utterance is done takes the next one for the following
  call, at its own cache offset (on the test set: 38 instead of 92 calls per pass at batch 8). The encoder works
  ahead on a second CUDA stream, one batch at a time, into two staging buffers the rows are filled from.
* **Beam search is ESPnet's.** `beam_size=k` runs ESPnet's batched beam search (sum of log-probabilities, no length
  normalization, its `end_detect` rule) on a batch × k grid of hypotheses, every step one CUDA graph.
* **Timestamp rules.** Like ESPnet, every search applies its `ScoreFilter`: no timestamps after `<notimestamps>`;
  otherwise the output starts with a timestamp, timestamps come in increasing pairs, and `<eos>` only after a
  complete pair.
* **OWSM-CTC** has no decoder: one engine runs the prompt encoder and the encoder (with language/task prefix,
  cross-attention to the prompt and intermediate-CTC conditioning) and returns the frame-wise best path. Inputs are
  padded to 30 s, as on the model card. `transcribe_long` decodes 30 s windows overlapping by 2 s on each side and
  keeps their middle frames, like ESPnet's `decode_long`.
* **FP8 encoder (experimental, opt-in).** `OWSM.from_pretrained(..., fp8_calibration=[recordings])` runs the
  encoder's linear layers and subsampling convolutions as FP8 (E4M3) GEMMs/convolutions on Hopper GPUs: inputs get a
  per-tensor scale measured on the given recordings, weights a per-channel scale. A few in-domain recordings are
  enough (8 LibriSpeech utterances above). It is not the default because the build needs that audio, and because
  the accuracy check above is one test set.
* **Same API without TensorRT.** `backend="torch"` runs the same decoding in PyTorch with CUDA graphs.

## Supported models

All ESPnet OWSM releases: the encoder-decoder models with an E-Branchformer or Transformer encoder, and the CTC-only
OWSM-CTC models. Each checkpoint below was checked two ways on LibriSpeech dev-clean:

* **ESPnet parity**: our fp32 PyTorch output (encoder output, first decoder step, greedy transcript; for OWSM-CTC the
  best path) against ESPnet `Speech2Text`, as identical transcripts / utterances (on CPU; the 73-utterance rows on
  GPU).
* **TensorRT**: fp16 engines on one H200 over the 73 utterances: RTFx with batch 1 / batch 8 (greedy with CTC
  drafts), and WER of our fp32 output → TensorRT fp16.

| model (`espnet/...`) | ESPnet parity | TensorRT RTFx b1 / b8 | WER fp32 → fp16 |
|---|---|---|---|
| `owsm_v4_medium_1B` | 73/73 | 989 / 1842 | 3.91% → 3.74% |
| `owsm_v4_small_370M` | 5/5 | 1052 / 1521 | 4.17% → 4.17% |
| `owsm_v4_base_102M` | 8/8 | 1846 / 3657 | 8.70% → 8.61% |
| `owsm_ctc_v4_1B` | 73/73 | 976 / 1436 | 4.43% → 4.26% |
| `owsm_ctc_v3.2_ft_1B` | 2/2 | 623 / 1318 | 5.13% → 4.87% |
| `owsm_ctc_v3.1_1B` | 2/2 | 717 / 1409 | 4.17% → 4.26% |
| `owsm_v3.2` | 3/3 | 902 / 1247 | 4.17% → 4.17% |
| `owsm_v3.1_ebf` | 3/3 | 541 / 626 | 3.74% → 3.65% |
| `owsm_v3.1_ebf_small` | 2/2 | 1087 / 1354 | 4.43% → 4.26% |
| `owsm_v3.1_ebf_small_lowrestriction` | 2/2 | (same architecture as `owsm_v3.1_ebf_small`) | |
| `owsm_v3.1_ebf_base` | 3/3 | 833 / 2352 | 5.83% → 5.74% |
| `owsm_v3` | 2/2 | 441 / 605 | 4.78% → 4.52% |
| `owsm_v2_ebranchformer` | 2/2 | 797 / 908 | 2.96% → 2.96% |
| `owsm_v2` | 2/2 | 698 / 825 | 4.17% → 4.17% |
| `owsm_v1` | 3/3 | 597 / 833 | 7.39% → 7.39% |
| `owls_4B_180K` | (as 0.25B / 1B) | 233 / 249 | 4.35% → 4.35% |
| `owls_2B_180K` | (as 0.25B / 1B) | 459 / 495 | 3.22% → 3.39% |
| `owls_1B_180K` | 2/2 | 416 / 504 | 4.87% → 4.78% |
| `owls_025B_180K` | 3/3 | 731 / 1611 | 7.65% → 7.74% |
| `powsm`, task `pr` (phones) | 3/3 | 189 / 641 | 72/73 identical to fp32 |
| `powsm`, task `asr` | 3/3 | 138 / 509 | see below |
| `powsm_ctc`, tasks `pr` / `asr` | 2/2 / 2/2 | 864 / 1823 (`pr`) | 72/73 identical to fp32 |

The first, third and fourth rows were measured with the current code. The others were measured earlier, before CTC
drafts worked in batches, rows were refilled, the padding was left out of the subsampling and multi-token calls got
their own engine; the current code is faster on them too (`owsm_v4_base_102M` went from 1401 / 2194 to 1846 / 3657).

With timestamps, greedy and beam-5 outputs are identical to ESPnet's too: `owsm_v3.1_ebf_base` 3/3 on CPU, and
`owsm_v4_medium_1B` 73/73 for both searches in fp32 on GPU (TensorRT fp16 greedy: 71/73, RTFx 739).

Notes:
* Language codes follow each model's vocabulary (`eng`, or `en` for OWSM v1/v2). POWSM uses 20 s windows.
* POWSM's `asr` output drops letters its vocabulary lacks (e.g. upper-case `M`, `I`, `S` become `<unk>`), exactly
  as ESPnet's does; it is a phone recognizer first.
* OWLS 9B and 18B use the same code path but were not tried: building their engines likely needs more time and
  memory than the 1-hour test jobs used here (the 4B engines took 6 min).

## Requirements

You need an NVIDIA GPU, Python ≥ 3.10, torch ≥ 2.6, and PyTorch and TensorRT built for the **same CUDA major
version**. The PyPI defaults (`torch`, `tensorrt`) are both CUDA 13. If your torch is a CUDA 12 build, run
`pip install tensorrt-cu12`; owsm-trt warns at load time when the two do not match.

Tested with torch 2.14 + TensorRT 11.4 on an H200. (The ONNX export of v0.1's graphs also worked with torch 2.6 and
2.8; v0.2's graphs have only been exported with torch 2.14.)

## Limitations

* CTC drafts speed up greedy ASR only; beam search, translation and POWSM take one decoder call per token. Rows are
  refilled only in greedy ASR; otherwise a batch runs until its slowest utterance is done.
* The first input of each length (in 5 s steps) and batch size captures a CUDA graph, which takes a few encoder
  runs; benchmark after a warm-up over the whole set.
* Beam search with timestamps is slow (RTFx ~18 on the 1B model, ESPnet ~2): ESPnet's `end_detect` rarely stops it
  early, and we keep its behavior.
* Within a batch, prompts must have the same length (always true unless `prompt=` text is used).
* OWSM-CTC: no timestamps, language identification or beam search (ESPnet's CTC prefix beam search is not ported).

## Development

```bash
uv sync                                   # dev environment
uv run pytest                             # CPU unit tests (search vs naive ESPnet, drafts, KV cache, shapes)
```

Scripts in `scripts/`:
* `parity_espnet.py`: fp32 parity against official ESPnet (any family; `--timestamps`, `--task`). It runs in
  `tools/espnet_ref`, which is only used for reference and is never a dependency.
* `eval_wer.py`: WER, speed, and transcript agreement for one configuration (`--beam`, `--timestamps`, `--no-draft`).
* `check_models.py`: TensorRT fp16 against PyTorch fp32 for several models.
* `profile_phases.py`: CUDA-event timing of the frontend, encoder, prefill, decode steps, and verify windows.
* `draft_report.py`: where CTC drafts and the decoder disagree.
* `trt_layers.py`: per-layer TensorRT profile plus an engine-inspector dump.
* `encoder_variants.py`: encoder engines built several ways (optimization profile, trimmed features, CTC argmax),
  timed per batch size.
* `bench_espnet.py`: ESPnet `Speech2Text` baseline.
* `smoke_features.py`: language ID, timestamps, translation, prompts, and long-form.

On a cluster with slow shared storage (e.g. Lustre), `scripts/build_env_tar.sh` builds a relocatable environment on
local disk and packs it into one `.tar.zst`. `scripts/gpu_eval.pbs` unpacks it into `$PBS_LOCALDIR` and runs the
stages listed in `STAGES`.
