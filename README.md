# owsm-trt

Fast inference for the [OWSM](https://www.wavlab.org/activities/2024/owsm/) family (Open Whisper-style Speech Models,
ESPnet) with TensorRT. You can `pip install` it, and it does not depend on ESPnet.

```bash
pip install owsm-trt
```

```python
from owsm_trt import OWSM

model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B")   # first call builds TensorRT engines (~4 min, cached)
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
| **owsm-trt, greedy** | 1 | **655** | 3.74% | 72/73 |
| **owsm-trt, greedy** | 8 | **1181** | 3.91% | 73/73 |
| **owsm-trt, greedy** | 16 | **1200** | 3.91% | 73/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 1 | 778 | 3.83% | 70/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 8 | 1441 | 3.83% | 69/73 |
| **owsm-trt, beam 5** | 1 | **102** | 4.00% | 73/73 |
| **owsm-trt, beam 5** | 3 | **202** | 4.00% | 73/73 |
| ESPnet OWSM-CTC `best_path`, fp32 | 1 | 131 | 4.43% | |
| **owsm-trt, OWSM-CTC** | 1 | **693** | 4.52% | 67/73 |
| **owsm-trt, OWSM-CTC** | 16 | **1492** | 4.26% | 67/73 |
| owsm-trt, OWSM-CTC, FP8 (experimental) | 1 | 875 | 4.52% | 65/73 |
| owsm-trt, OWSM-CTC, FP8 (experimental) | 16 | 1924 | 4.35% | 64/73 |

For one stream that is 26× ESPnet's greedy decoding and 44× its default beam search, and 48× with batch 16; beam
search itself is 7× (batch 1) to 13× (batch 3) faster, and OWSM-CTC 5× to 11×.

Rows are TensorRT fp16 unless marked FP8. Where their transcripts differ from ESPnet's fp32 output, it is by a comma,
a casing or a single word (fp16 rounding at a near-tie), and the WER moves by at most 0.17 points. In fp32
(`backend="torch", dtype="float32"`) the transcripts are identical to ESPnet's: 73/73 for greedy, beam 5, greedy and
beam 5 with timestamps, and OWSM-CTC. (TensorRT fp16 greedy with timestamps: 71/73, RTFx 536.)

The FP8 rows were calibrated on the first 8 test utterances. On the other 65 their WER is within one word of ESPnet's
fp32: 4.00% vs 4.11% for the 1B model, 4.63-4.84% vs 4.74% for OWSM-CTC. That is one test set; check FP8 on your own
data before relying on it.

Single-utterance greedy decoding takes 7 ms for the first test utterance (6 s): 0.2 ms features, 5.0 ms encoder,
1.8 ms decoding. With batch 8, eight utterances take 60 ms, 23 of them in the encoder.

The first `from_pretrained` builds the engines: 4.4 minutes for the 1B model on an H200 (3.3 minutes with FP8),
cached afterwards.

## How it works

* **No ESPnet.** The models are reimplemented in plain PyTorch (`owsm_trt/modeling.py`) and the official checkpoints
  load with `strict=True`.
* **TensorRT engines are built on first use.** The encoder and a decoder call are exported to ONNX and compiled
  into strongly-typed fp16 engines, cached under `~/.cache/owsm_trt` (set `OWSM_TRT_CACHE` to move them). Engines
  depend on the GPU and the TensorRT version.
  * The decoder has separate optimization profiles for batch-1 steps, batched steps, and prompt prefill. Kernels
    tuned for batch 16 waste most of the GPU at batch 1.
  * Calls that check several drafted tokens (below) run on a second engine built from the same network at a higher
    TensorRT optimization level (4), which finds kernels about twice as fast for them (1.3 vs 2.3 ms on the 1B
    model).
  * The encoder input and its first residual connection are kept in fp32 (for Transformer encoders the whole residual
    stream): on some checkpoints the subsampled features times √d exceed the fp16 range (up to 4.5·10⁵ on OWSM-CTC
    v3.2).
* **Decoding work is minimized.**
  * Cross-attention keys/values are projected once per utterance.
  * The self-attention KV cache is static and comes in length buckets (64/128/256/512), so a step only reads as
    much cache as the sequence needs.
  * Each decoding step (engine, cache write, token choice) is one CUDA-graph replay; so is the encoder. The host
    checks for the end every 4 steps.
  * Batched inputs are sorted by duration, so a batch does not wait on one long utterance.
* **CTC drafts (greedy).** The encoder-decoder models also have a CTC head. Its best path is a draft of the
  transcript, usually right down to casing and punctuation. One decoder call checks up to 31 drafted tokens per
  utterance and keeps those that equal the decoder's own choice, so the output is exactly the greedy output with far
  fewer calls (almost half of the test utterances need a single call). In a batch every utterance advances by its own
  amount, at its own cache offset. When the decoder and the draft disagree, the draft is re-anchored on the last few
  emitted tokens. The draft is used for `task="asr"` only (the CTC head is trained on transcripts), and dropped when
  it stops being accepted (POWSM's CTC head writes phones). `ctc_draft=False` turns it off.
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
* **TensorRT**: fp16 engines on one H200 over the 73 utterances: RTFx with batch 1 / batch 8 (greedy, CTC drafts at
  batch 1), and WER of our fp32 output → TensorRT fp16.

| model (`espnet/...`) | ESPnet parity | TensorRT RTFx b1 / b8 | WER fp32 → fp16 |
|---|---|---|---|
| `owsm_v4_medium_1B` | 73/73 | 529 / 785 | 3.91% → 3.74% |
| `owsm_v4_small_370M` | 5/5 | 1052 / 1521 | 4.17% → 4.17% |
| `owsm_v4_base_102M` | 8/8 | 1401 / 2194 | 8.70% → 8.70% |
| `owsm_ctc_v4_1B` | 73/73 | 691 / 1472 (b16) | 4.43% → 4.35% |
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

The RTFx in this table were measured before CTC drafts worked in batches and before multi-token calls got their own
engine; the current code is faster (with batched drafts and a level-5 build of the multi-token profiles, which runs
the same kernels as the final level-4 engine: `owsm_v3.1_ebf_base` 1829 / 3610, `owls_025B_180K` 880 / 1680).

With timestamps, greedy and beam-5 outputs are identical to ESPnet's too: `owsm_v3.1_ebf_base` 3/3 on CPU, and
`owsm_v4_medium_1B` 73/73 for both searches in fp32 on GPU (TensorRT fp16 greedy: 72/73, RTFx 424).

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

* CTC drafts speed up greedy ASR only; beam search, translation and POWSM take one decoder call per token. A batch
  runs until its slowest utterance is done, so one utterance with a poor draft slows its batch down.
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
* `bench_espnet.py`: ESPnet `Speech2Text` baseline.
* `smoke_features.py`: language ID, timestamps, translation, prompts, and long-form.

On a cluster with slow shared storage (e.g. Lustre), `scripts/build_env_tar.sh` builds a relocatable environment on
local disk and packs it into one `.tar.zst`. `scripts/gpu_eval.pbs` unpacks it into `$PBS_LOCALDIR` and runs the
stages listed in `STAGES`.
