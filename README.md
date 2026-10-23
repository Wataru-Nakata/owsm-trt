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
model.transcribe_long(["a.wav", "b.wav"], batch_size=16)     # several recordings at once
model.transcribe_long(["a.wav", "b.wav"], batch_size=16, cut_at_pauses=True)  # windows cut at pauses

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
| **owsm-trt, greedy** | 1 | **1389** | 3.91% | 71/73 |
| **owsm-trt, greedy** | 8 | **2732** | 3.91% | 72/73 |
| **owsm-trt, greedy** | 16 | **2857** | 3.91% | 72/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 1 | 1577 | 3.91% | 69/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 8 | 3036 | 3.83% | 70/73 |
| owsm-trt, greedy, FP8 encoder (experimental) | 16 | 2788 | 3.91% | 68/73 |
| **owsm-trt, beam 5** | 1 | **114** | 4.09% | 72/73 |
| **owsm-trt, beam 5** | 3 | **215** | 4.00% | 73/73 |
| ESPnet OWSM-CTC `best_path`, fp32 | 1 | 131 | 4.43% | |
| **owsm-trt, OWSM-CTC** | 1 | **981** | 4.43% | 64/73 |
| **owsm-trt, OWSM-CTC** | 16 | **1682** | 4.52% | 63/73 |

For one stream that is 56× ESPnet's greedy decoding and 93× its default beam search, and 114× with batch 16; beam
search itself is 8× (batch 1) to 14× (batch 3) faster, and OWSM-CTC 7× to 13×. On all of LibriSpeech test-clean
(2620 utterances, 5.4 h) greedy decoding runs at RTFx 1611 with batch 1 and 3588 with batch 16, with the WER of the
PyTorch backend (2.42%); with the FP8 encoder about 3900 with batch 16 (WER 2.44%, calibrated on 8 of the
utterances).

Rows are TensorRT fp16 unless marked FP8. Where their transcripts differ from ESPnet's fp32 output, it is by a comma,
a casing or a single word (fp16 rounding at a near-tie: rebuilding the same engine moves as many), and the WER moves
by at most 0.17 points. In fp32 (`backend="torch", dtype="float32"`) the transcripts are identical to ESPnet's: 73/73
for greedy, beam 5, greedy and beam 5 with timestamps, and OWSM-CTC. (TensorRT fp16 greedy with timestamps: 71/73,
RTFx 1053.)

The 1B model's FP8 rows were calibrated on 8 LibriSpeech test-clean utterances, so all 73 test utterances are held
out: WER 3.83-3.91% vs ESPnet's 3.91%; on all of test-clean 2.41-2.44% vs 2.42% in fp16. With OWSM-CTC, FP8 was 22%
faster at batch 16 in an earlier version, with WER within one word on held-out utterances. Check FP8 on your own data
before relying on it. Here it is 11-14% faster at batch 1 and 8; at batch 16 the difference is within the
run-to-run spread.

Long recordings (`transcribe_long`), on the 87 chapters of LibriSpeech test-clean concatenated (5.4 h, 17-497 s
each):

| mode | RTFx | WER |
|---|---:|---:|
| ESPnet's `decode_long` (30 s windows, one after another) | 1085 | 2.78% |
| same windows, 16 recordings at a time (`batch_size=16`) | 3558 | 2.70% |
| windows cut at pauses, 16 at a time (`cut_at_pauses=True`) | 6245 | 2.27% |

The first two decode the same windows; their WER differs by fp16 near-ties (rebuilding the engine alone moves the
first one between 2.67% and 2.81%). Cutting at pauses decodes every second of audio once instead of 1.5 times and is
not ESPnet's segmentation.

Single-utterance greedy decoding takes 4.6 ms for the first test utterance (6 s): 0.2 ms features, 2.9 ms encoder,
1.4 ms decoding. With batch 16, sixteen utterances take 59 ms, 38 of them in the encoder.

The first `from_pretrained` builds the engines: 5.5 minutes for the 1B model on an H200 (4.5 minutes with FP8),
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
  * The exported decoder and encoder write attention out with the scale on the query. The ONNX exporter's
    `scaled_dot_product_attention` also scales the keys, which TensorRT then recomputes for all cross-attention keys
    and the whole KV cache in every multi-token call (40% of such a call at batch 16; 3-4% of the encoder).
  * Custom kernels run inside the engines as TensorRT plugins, compiled ahead of time into the engine (running it
    needs neither Triton nor a compiler):
    * decoder self-attention over [cache | new tokens] and cross-attention (Triton), which read the KV cache and the
      encoder keys/values where they are, instead of TensorRT's copies of them in every layer: +8.5% at batch 1,
      +19% for beam search;
    * the E-Branchformer depthwise convolutions on [time, channel] activations (CUDA C, compiled with NVRTC), which
      drop the transposes around TensorRT's grouped convolution: 160 → 38 µs per convolution at batch 16, 8-12% off
      the encoder.

    Building them needs Triton and NVRTC (both come with PyTorch on Linux) and an Ampere or newer GPU; otherwise
    the engines are built without them.
* **The padding is not subsampled.** Inputs are padded to 30 s as in ESPnet, but the encoder's convolutional
  subsampling (a third of its time at batch 16) only runs up to the end of the longest recording in the batch, plus
  its receptive field. Every later subsampled frame sees padding only, so it is the same for every recording: it is
  computed once when the engine is built and filled in. The encoder output is that of the full input (up to fp16
  rounding); 8 s recordings take 27-31% less encoder time at batch 8-16. Likewise the CTC head (below) only reads
  the frames that see audio, plus the window's last frame, where OWSM's CTC head puts the final punctuation mark
  (-3 to -6% of encoder time). This needs global mean normalization,
  which every OWSM model has except OWSM-CTC v3.2. The attention layers do need the whole 30 s: padding only up to
  a multiple of 10 s raises the 1B model's WER on LibriSpeech test-clean from 2.42% to 9.0% (OWSM-CTC v4: 2.62% to
  2.77%).
* **Decoding work is minimized.**
  * Cross-attention keys/values are projected once per utterance.
  * The self-attention KV cache is static and comes in length buckets (64/128/256/512), so a step only reads as
    much cache as the sequence needs.
  * Around a multi-token call, two Triton kernels write each row's new keys/values into the cache at its offset and
    pick the next tokens under the timestamp rules in one pass over the logits (the allowed tokens are ranges of
    ids, so no vocabulary-sized mask is built): 16% off a batch-16 call.
  * Which drafted tokens a call accepts is also decided on the GPU, inside the call's CUDA graph: the host gets
    back two numbers per row, and its own work per call is 0.1 ms or less.
  * Each decoding step (engine, cache write, token choice) is one CUDA-graph replay; so is the encoder (one graph per
    5 s of input length). The host checks for the end every 4 steps.
  * Batched inputs are sorted by duration, so an encoder batch stops early and a decoding batch does not wait on one
    long utterance.
* **CTC drafts (greedy).** The encoder-decoder models also have a CTC head. Its best path is a draft of the
  transcript, usually right down to casing and punctuation. One decoder call checks up to 31 drafted tokens per
  utterance and keeps those that equal the decoder's own choice, so the output is exactly the greedy output with far
  fewer calls (almost half of the test utterances need a single call). In a batch every utterance advances by its own
  amount, at its own cache offset. When the decoder and the draft disagree, the draft is re-anchored on the last few
  emitted tokens; a token found nowhere near is taken to replace the drafted one, unless it is punctuation the
  draft lacks (CTC drafts often miss commas). Tuned by replaying the drafts and outputs of LibriSpeech test-clean
  (`scripts/draft_sim.py`), these rules make 18% fewer calls than before. The draft is used for `task="asr"` only
  (the CTC head is trained on transcripts), and dropped when 8 calls have accepted fewer drafted tokens than calls
  (POWSM's CTC head writes phones). `ctc_draft=False` turns it off. A call checks at most
  256 tokens (rows × window), so at batch 16 a row gets up to 15 drafted tokens per call: wider calls cost more than
  they save.
* **Rows are refilled (greedy ASR of more than one batch).** Utterances need from 1 to 25 decoder calls, and a fixed
  batch would wait for its slowest one. Instead a row whose utterance is done takes the next one for the following
  call, at its own cache offset (on the test set: 29 instead of 92 calls per pass at batch 8). The encoder works
  ahead on a second CUDA stream, one batch at a time, into two staging buffers the rows are filled from; a smaller
  last batch is encoded at its own size rather than padded with empty rows.
* **Long recordings.** `transcribe_long` follows ESPnet's `decode_long`: 30 s windows, each starting at the last
  complete utterance of the one before, decoded with timestamps. The drafts also guess the timestamps (the first
  one is `<0.00>` and the last the window's end in most windows, and a start time usually equals the end time
  before it): 26% fewer decoder calls, same output. Two options decode faster: `batch_size` decodes the same
  windows of several recordings together, and `cut_at_pauses=True` cuts every window at the quietest 0.2 s of its
  last 15 s, so no audio is decoded twice (not ESPnet's segmentation).
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
* **Decoder megakernel (experimental, opt-in, Hopper).** `from_pretrained(..., decoder_kernel=True)` (or
  `OWSM_TRT_MEGAKERNEL=1`) runs small decoder calls (at most 8 rows and 16 tokens) as one persistent Gluon kernel per
  call: all 18 layers and the output layer, with grid-wide barriers between stages and each stage's weights copied
  into shared memory (TMA) while the previous stage finishes. A one-token call takes 0.65 ms instead of 0.72-0.77 ms
  with TensorRT, and drafted windows of 2-8 tokens 0.66-0.76 instead of 1.1 ms. On the test set it makes beam search
  22% faster (and its output 73/73 ESPnet's), batch 8 5% faster, and greedy batch 1 4% slower: most of its calls are
  32-token windows, which stay on TensorRT, and it shares the GPU worse with the encoder working ahead.
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
| `owsm_v4_medium_1B` | 73/73 | 1389 / 2732 | 3.91% → 3.91% |
| `owsm_v4_small_370M` | 5/5 | 1052 / 1521 | 4.17% → 4.17% |
| `owsm_v4_base_102M` | 8/8 | 3615 / 9672 | 8.70% → 8.61% |
| `owsm_ctc_v4_1B` | 73/73 | 967 / 1465 | 4.43% → 4.43% |
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
drafts worked in batches, rows were refilled, the padding was left out of the subsampling, multi-token calls got their
own engine and the plugins came; the current code is faster on them too (`owsm_v4_base_102M` went from 1401 / 2194 to
3615 / 9672).

With timestamps, greedy and beam-5 outputs are identical to ESPnet's too: `owsm_v3.1_ebf_base` 3/3 on CPU, and
`owsm_v4_medium_1B` 73/73 for both searches in fp32 on GPU (TensorRT fp16 greedy: 71/73, RTFx 1053).

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
* `dump_drafts.py` + `draft_sim.py`: drafts and greedy outputs of a test set, and an exact CPU replay of drafted
  decoding on them (an oracle stands in for the decoder) to count decoder calls for other draft rules.
* `pad_experiment.py`: WER with inputs padded to less than 30 s.
* `make_longform.py` + `eval_long.py`: long recordings from LibriSpeech chapters, and long-form WER / speed.
* `dwconv_bench.py`, `dwconv_encoder.py`: the depthwise-convolution plugin alone and inside the encoder.
* `qsub_pool.sh`: GPU jobs on a cluster, at most N at a time (`gpu_eval.pbs` runs the stages listed in `STAGES`).
* `trt_layers.py`: per-layer TensorRT profile plus an engine-inspector dump.
* `encoder_variants.py`: encoder engines built several ways (optimization profile, trimmed features, CTC argmax),
  timed per batch size.
* `bench_espnet.py`: ESPnet `Speech2Text` baseline.
* `smoke_features.py`: language ID, timestamps, translation, prompts, and long-form.

On a cluster with slow shared storage (e.g. Lustre), `scripts/build_env_tar.sh` builds a relocatable environment on
local disk and packs it into one `.tar.zst`. `scripts/gpu_eval.pbs` unpacks it into `$PBS_LOCALDIR` and runs the
stages listed in `STAGES`.
