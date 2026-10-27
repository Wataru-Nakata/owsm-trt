# owsm-trt

Fast inference for the [OWSM](https://www.wavlab.org/activities/2024/owsm/) family (Open Whisper-style Speech Models,
ESPnet) with TensorRT. You can `pip install` it, and it does not depend on ESPnet.

```bash
pip install git+https://github.com/Wataru-Nakata/owsm-trt
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

# experimental: FP8 encoder (Hopper) with the packaged scales (owsm_v4_medium_1B, owsm_ctc_v4_1B): +10-12% at batch 16
fast = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", fp8=True)
# ... or calibrated on a few recordings like the ones you will transcribe (any model)
fast = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", fp8_calibration=["c1.wav", "c2.wav", "c3.wav"])
# TensorRT decoder only (no decoder megakernel; it is used by default on Hopper GPUs)
plain = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", decoder_kernel=False)
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
| **owsm-trt, greedy** | 1 | **1949** | 3.91% | 73/73 |
| **owsm-trt, greedy** | 8 | **2943** | 3.91% | 73/73 |
| **owsm-trt, greedy** | 16 | **3075** | 3.74% | 72/73 |
| **owsm-trt, beam 5** | 1 | **139** | 4.00% | 73/73 |
| **owsm-trt, beam 5** | 3 | **220** | 4.09% | 72/73 |
| owsm-trt, greedy, TensorRT decoder only | 1 | 1562 | 3.91% | 72/73 |
| owsm-trt, beam 5, TensorRT decoder only | 1 | 112 | 4.00% | 73/73 |
| ESPnet OWSM-CTC `best_path`, fp32 | 1 | 131 | 4.43% | |
| **owsm-trt, OWSM-CTC** | 1 | **1506** | 4.43% | 72/73 |
| **owsm-trt, OWSM-CTC** | 16 | **2264** | 4.35% | 69/73 |

For one stream that is 78× ESPnet's greedy decoding and 130× its default beam search, and 123× with batch 16; beam
search itself is 9× (batch 1) to 15× (batch 3) faster, and OWSM-CTC 11× to 17×. On all of LibriSpeech test-clean
(2620 utterances, 5.4 h) greedy decoding runs at RTFx 2085 with batch 1 and 3825 with batch 16, with the WER of the
PyTorch backend (2.42%); OWSM-CTC at 1603 with batch 1 and 2529 with batch 16 (WER 2.66-2.69%).

Rows are TensorRT fp16, with the decoder megakernel (below) unless marked "TensorRT decoder only". Where their
transcripts differ from ESPnet's fp32 output, it is by a comma, a casing or a single word (fp16 rounding at a near-tie:
rebuilding the same engine moves as many; for OWSM-CTC at batch 16 two builds of the same code gave the same
transcript for only 64 of the 73, at WER 4.26% and 4.43%), and the WER moves by at most 0.17 points. In fp32 (`backend="torch",
dtype="float32"`) the transcripts are identical to ESPnet's: 73/73 for greedy, beam 5, greedy and beam 5 with
timestamps, and OWSM-CTC. (TensorRT fp16 greedy with timestamps: 73/73, RTFx 1315; with the TensorRT decoder only
72/73, RTFx 1070.)

**FP8 encoder (experimental, opt-in).** `fp8=True` uses activation scales shipped with the package for
`owsm_v4_medium_1B` and `owsm_ctc_v4_1B` (measured on 900 recordings in 7 languages, none from the test sets below).
Batch 16, fp16 → FP8 (same node):

| test set | 1B: WER/CER | 1B: RTFx | OWSM-CTC: WER/CER | OWSM-CTC: RTFx |
|---|---|---|---|---|
| LibriSpeech test-clean (English, WER) | 2.42 → 2.42 | 3844 → 4310 | 2.66 → 2.71 | 2568 → 2837 |
| JVS parallel100, 10 speakers (Japanese, CER) | 7.93 → 7.91 | 3948 → 4409 | 7.15 → 7.22 | 2649 → 2942 |
| FLEURS test German (WER) | 7.70 → 7.73 | 5022 → 5514 | 8.27 → 8.24 | 4265 → 4651 |
| FLEURS test Spanish (WER) | 4.68 → 4.64 | 5199 → 5705 | 5.47 → 5.41 | 3952 → 4338 |
| FLEURS test Mandarin (CER, `lang="zho"`) | 12.65 → 12.56 | 4710 → 5168 | 10.93 → 10.97 | 3808 → 4179 |

It is not the default:
* OWSM-CTC's Japanese CER is consistently 0.03-0.09 higher across calibration variants.
* On the 1B model one Mandarin utterance falls into a decoder repetition loop under some encoder perturbations, FP8
  among them (CER 12.89 vs 12.65 at batch 1).
* At batch 1 the 1B model gains nothing from it, being bound by the decoder.
* A separate H200/1B beam5 timestamp evaluation is nearly neutral in speed. In Japanese translation
  B8, one of12 inputs repeats to374 tokens and whole-list time rises139→461ms with packaged scales.
  [Encoder/quality comparison and exact input](docs/runtime-encoder-variants.md).

Other models: `fp8_calibration=[recordings]` measures the scales on your own audio.

Long recordings (`transcribe_long`), on the 87 chapters of LibriSpeech test-clean concatenated (5.4 h, 17-497 s
each):

| mode | RTFx | WER |
|---|---:|---:|
| ESPnet's `decode_long` (30 s windows, one after another) | 1380 | 2.68% |
| same windows, 16 recordings at a time (`batch_size=16`) | 4680 | 2.80% |
| windows cut at pauses, 16 at a time (`cut_at_pauses=True`) | 6818 | 2.27% |
| OWSM-CTC (30 s windows overlapping by 2 s, as ESPnet), one recording per call | 6098 | 2.70% |
| OWSM-CTC, all recordings in one call | 6696 | 2.68% |

The first two decode the same windows; their WER differs by fp16 near-ties (rebuilding the engine alone moves the
first one between 2.67% and 2.81%). Cutting at pauses decodes every second of audio once instead of 1.5 times and is
not ESPnet's segmentation.

Single-utterance greedy decoding takes 4.1 ms for the first test utterance (6 s): 0.25 ms features, 2.8 ms encoder,
1.0 ms decoding (4.6 ms with the TensorRT decoder only). With batch 16, sixteen utterances take 56 ms, 35 of them in the
encoder.

The first `from_pretrained` builds the engines: 5.5 minutes for the 1B model on an H200, cached afterwards (an FP8
build after it takes about a minute: it reuses the decoder engines). The first transcription then compiles the
decoder megakernel, about 18 s, also cached. From a fresh `pip install`, the first `owsm-trt transcribe` with the 1B
model takes about 5.5-6 minutes in all (download, engine build, megakernel); later runs start in about 10 s.

## How it works

* **No ESPnet.** The models are reimplemented in plain PyTorch (`owsm_trt/modeling.py`) and the official checkpoints
  load with `strict=True`.
* **TensorRT engines are built on first use.** The encoder and a decoder call are exported to ONNX and compiled
  into strongly-typed fp16 engines, cached under `~/.cache/owsm_trt` (set `OWSM_TRT_CACHE` to move them). Engines
  depend on the GPU and the TensorRT version.
  * Builds reusing decoder engines (such as FP8 encoder builds) skip the unused decoder ONNX export.
    On H200/1B, the export component fell from50.58 to26.64s (three alternating pairs, identical encoder
    ONNX hashes); this is not a total-build or inference-speed measurement.
    [Validation and scope](docs/runtime-decoder-export.md).
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
      the encoder;
    * the subsampling's first convolution (one input channel, written in the layout the next convolution reads:
      2.5 → 1.2 ms at batch 16) and its last ReLU + flatten, which reads TensorRT's convolution output layout
      directly (0.57 → 0.18 ms);
    * the argmax of the CTC heads and OWSM-CTC's intermediate-CTC softmax, on a vocabulary padded from 50002 to 50048
      (the padded ids can never win) and on time-major rows, as TensorRT keeps the encoder states after
      cross-attention: otherwise it transposes the vocabulary-wide logits (several ms per OWSM-CTC call at batch
      16).

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
    1 s of input length: the subsampling stops at the next whole second). The host checks for the end every 4 steps.
  * Batched inputs are sorted by duration, so an encoder batch stops early and a decoding batch does not wait on one
    long utterance. With batch 1 the order alternates long and short utterances instead, so the encoder works ahead
    while long ones decode (+1.5-6%, same output).
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
  they save. `draft_token_budget=128` is an opt-in for H200/1B B8: timestamped throughput improved
  4.2% English/6.8% Japanese in the public-API replication. Smaller windows can change fp16 output and are not
  always faster; the default remains 256. [API validation](docs/runtime-results-2771267.md) and [all window settings/differences](docs/runtime-results-2771126.md).
* **Single-utterance frontend graphs.** STFT/mel/normalization use a bounded CUDA graph cache by default;
  `frontend_graph=False` restores eager execution. H200 measurements with identical engines found single
  greedy RTFx +4.3% for v4 medium 1B and +11.5% for base 102M (English 73, five alternating repeats), with exact
  features/tokens/segments. Larger batches run eagerly; graph buffers add cold work and memory. Pointwise
  frontend fusion remains disabled. A seven-repeat validation reproduced +5.0%/+12.1%, including lazy
  draft setup. CTC-only v4 1B also improved single-utterance RTFx 4.4% English/5.2% Japanese with exact
  features/output. [Runtime measurements](docs/runtime-results-2771009.md), [CTC measurements](docs/runtime-results-2771126.md).
* **Greedy vocabulary selection and acceptance.** `greedy_kernels=True` (default) splits vocabulary
  argmax across blocks and fuses the final reduction with draft acceptance. Against the same engines
  with frontend graphs already enabled, timestamped single-request RTFx improved 4.3% English/4.8%
  Japanese on v4 1B and 10.6% on 102M; a 110 s recording improved 5.6% (H200, 11 alternating repeats).
  All sampled tokens and segments matched. Plain greedy changes were small; these are not speedup
  claims for every workload/GPU. Short-primer fixed-batch requests also skip redundant state
  initialization: independently replicated incremental gains were 2.1–2.5% on 1B English and about
  5% on 102M, with exact sampled output. This initialization change does not apply to streaming or
  long prefill. `greedy_kernels=False` restores serial selection and full initialization. CPU, missing Triton
  and small vocabularies retain the reference path. [Measurements and limits](docs/runtime-greedy-selection.md).
  The same option also fuses mask/argmax/state updates for non-drafted greedy and translation.
  On H200, sampled 1B English/Japanese requests gained 8.5–9.3% RTFx and102M English gained
  26.6–27.1%, with exact tokens/segments (12 inputs per condition,11 alternating repeats,
  versus the previously optimized path). Drafted/beam paths have no corresponding speedup claim.
  [Plain selection measurements and cold costs](docs/runtime-plain-selection.md).
* **Completed plain decoder calls.** With `greedy_kernels=True`, single-row Ring calls skip decoder
  computation after EOS while keeping the existing host-check interval and graph replay count.
  H200/1B and102M English/Japanese B1 throughput improved1.0–5.6% across12 tested conditions,
  with exact tokens/segments. Some per-input times and102M plain-ASR p95 slightly regressed;
  B>1, other decoder kernels and drafted verification have no corresponding speedup claim.
  [Measurements and limitations](docs/runtime-finished-skip.md).
  Completed multi-row Gluon groups also skip the call when every row has ended; partial groups
  run the full decoder. On1B/H200, English B3/B8 whole-list RTFx improved1.92–3.66% with exact
  outputs; Japanese B8 was within noise. The102M multi-row TensorRT route is unchanged.
  [Group measurements and public-path verification](docs/runtime-finished-groups.md).
* **Existing102M Ring row option.** `decoder_kernel="ring:rr"` remains an opt-in numerical
  change. On H200, English73/Japanese12 plain B3 ASR gained35–47% whole-list RTFx and
  timestamp beam5 B1 gained10.5–11.3%, against `auto` with the same engines (nine alternating
  pairs). English plain WER increased8.61→8.70%; Japanese timestamp ASR CER8.45→8.61%.
  All sampled timestamp IDs matched, but some text/token outputs changed. Japanese translation
  repeated badly in both routes; its timing is not a quality recommendation. Auto is unchanged.
  [Measurements, full differences and numerical gate limitations](docs/runtime-102m-ring-rows.md).
* **Cached-engine model loading.** Frontend/decoder helpers privately map supported ZIP checkpoints
  and read only needed tensors. On H200 with node-local files and warmed OS/JIT caches, fresh-process
  load medians fell from6.75 to4.61 s for1B and1.87 to1.61 s for102M (five alternating pairs, exact
  English/Japanese public outputs). CPU peak RSS fell by2.33 GiB/196 MiB; GPU peak was unchanged.
  These are startup measurements, not steady-state gains or cold-disk/build timings. Legacy formats
  and shared-mapping configurations keep ordinary loading. [Measurements](docs/runtime-checkpoint-load.md).
  Decoder setup also skips random parameter initialization before strict checkpoint copying. With
  mapping already enabled, a separate five-pair comparison reduces1B load4.69→3.28s (30.06%) and
  102M1.59→1.40s (12.14%), with all728 public outputs matching and46 focused tests passing.
  GPU peak allocation is unchanged. [Direct allocation measurements](docs/runtime-decoder-loading.md).
* **Fixed-batch encoder lookahead (opt-in).** `encoder_lookahead=True` overlaps the next encoder batch
  with beam/plain decoding for multiple-batch requests. On H200/1B English73, an independent warm
  comparison measures about6% more RTFx for beam B1/B3 and3.6–8.3% for translation B8/B1, with exact
  full outputs. The tested102M median gains are4.7–8.3%, but the independent102M beam B1 repeat
  has worse p95 (968→1003ms) and3/11 slower pairs. Short Japanese B8 translation is neutral.
  Existing drafted greedy streaming and single-batch calls retain their paths. Extra buffers/captures
  increase memory/cold cost, so the default remains False. [Measurements and limits](docs/runtime-encoder-lookahead.md).
* **Unused draft setup is deferred.** `lazy_drafts=True` (default) creates verification buffers on first
  draft use and prepares each bucket/window/solo call and graph only when needed. With resident engines, session-cold translation/language identification was 33–36% shorter
  on H200/v4 1B and 102M, saving 45.4/26.1 MiB of unused buffers. Warm ASR work is unchanged; long-prompt
  cold measurements varied. Deferring individual calls also reduced resident-engine session-cold ASR
  time by 34–39% on 1B/102M and B8 streaming setup/request time by about 60–61% (12 English recordings,
  three reconstructions, exact outputs). New shapes still incur setup; fully warm timing changed
  -0.5% to +0.8% across the measured cases. `lazy_drafts=False` restores eager construction.
  [Per-shape cold and warm validation](docs/runtime-lazy-verify.md).
  Unused plain decoder graphs are also deferred until ordinary multi-token decode is needed.
  Independently replicated incremental session-cold reductions were about5.6% for1B ASR,
  3.8–6.8% for102M ASR and21–22% for102M B8 streaming; one-token language ID improved7–8%.
  These use resident engines and warmed JIT; steady-state timing was unchanged within variation.
  Translation still prepares its plain decoder on first use; no cold-speed claim applies to it.
  [Deferred plain setup measurements](docs/runtime-deferred-plain.md).
* **Rows are refilled (greedy ASR of more than one batch).** Utterances need from 1 to 25 decoder calls, and a fixed
  batch would wait for its slowest one. Instead a row whose utterance is done takes the next one for the following
  call, at its own cache offset (on the test set: 29 instead of 92 calls per pass at batch 8). The encoder works
  ahead on a second CUDA stream, one batch at a time, into two staging buffers the rows are filled from; a smaller
  last batch is encoded at its own size rather than padded with empty rows. With batch 1 the two staging buffers are
  encoded at the same time on two streams, each with its own TensorRT contexts (+6% with the decoder megakernel,
  same output; `OWSM_TRT_ENCODER_LANES=0` turns it off).
* **Long recordings.** `transcribe_long` follows ESPnet's `decode_long`: 30 s windows, each starting at the last
  complete utterance of the one before, decoded with timestamps. The drafts also guess the timestamps (the first
  one is `<0.00>` and the last the window's end in most windows, and a start time usually equals the end time
  before it): 26% fewer decoder calls, same output. Two options decode faster: `batch_size` decodes the same
  windows of several recordings together, and `cut_at_pauses=True` cuts every window at the quietest 0.2 s of its
  last 15 s, so no audio is decoded twice (not ESPnet's segmentation). With `batch_size`, all recordings' windows
  share one stream of decoder calls: a row whose window is done takes the next window of any recording whose
  previous window has finished (27-33% faster than decoding the recordings' windows round by round).
* **Beam search follows ESPnet's selection.** `beam_size=k` runs batched beam search (sum of log-probabilities,
  no length normalization, ESPnet's `end_detect` rule) on a batch × k grid, every step one CUDA graph.
  By default, it additionally stops when the finite best finished score strictly exceeds every valid live
  score; ties continue and the existing stopping rules remain active. Set
  `OWSM.from_pretrained(..., beam_upper_bound=False)` to disable the bound. Host completion is checked every
  step for bound-enabled batch 1, every four steps otherwise. On H200/1B/beam 5 with timestamps, all tokens,
  timestamps and segments matched the original implementation on 73 English and 12 Japanese inputs at
  B=1/3; the host-check sweep used English only. Invalid slots cannot become live/finished hypotheses;
  invalid live logits (NaN/+Inf) or exhaustion without a finished hypothesis raise `RuntimeError` after
  decoding. See [beam profiling and validation](docs/beam-search.md) and the
  [60-candidate ledger](docs/beam-candidates.tsv) for tested scope and remaining work.
  Candidate masking/scoring and finished-state updates use fused Triton kernels on CUDA by default;
  full-vocabulary log-softmax and global top-K stay in PyTorch. `beam_kernels=False` selects the reference
  operations, also used on CPU or without Triton. Against the already optimized stopping path,
  [same-engine H200/1B measurements](docs/beam-fusion.md) show an additional 20.3% RTFx at B=1 and 14.2%
  at B=3 on 73 English inputs, with identical tokens, timestamps, segments and decoder call counts.
  KV reordering now gathers the written prefix through fixed scratch storage in two launches, clearing
  the unused tail. [Further same-engine measurements](docs/beam-kv.md) show 3.9% / 8.7% additional RTFx
  at B=1 / B=3, with exact output agreement; `beam_kernels=False` also disables this path.
  On supported Gluon shapes, beam rows now share encoder cross-KV storage. A further
  [B=1 comparison](docs/beam-cross.md) measured 4.3% more RTFx and cross-KV buffers shrinking from
  165.4 to 27.6 MB, with identical outputs. B=3/K=5 retains expanded KV with the TensorRT decoder by default.
  The [combined KV comparison](docs/beam-final.md) measured 9.25% / 8.90% additional RTFx at B=1 / B=3
  against the already fused search path on English 73, and 9.46% / 9.70% on Japanese12, with exact outputs.
  Parent selection, token-history gather and timestamp state now share one kernel for K<=8 and
  capacity<=512. [The next same-engine comparison](docs/beam-state-fusion.md) adds4.30% /4.24% RTFx
  at B=1 / B=3 on English 73, and5.07% /4.02% on Japanese12, with identical outputs and275 GPU tests.
  For further speed, opt into `beam_grouped_cross=True` and `beam_compact_tail=True` in
  `OWSM.from_pretrained`. The [guarded H200/1B preset](docs/beam-wider-rows.md#verified-api-preset-2773443)
  uses grouped Gluon attention at B=2/3. With tail compaction, measured RTFx improves29.43%/32.42%
  at B=2 (English73/Japanese12), and25.53%/24.31% at B=3 versus the optimized default.
  These options remain off by default: two English segment ends shift20 ms, and silence text can
  change. Main text/WER/CER match;63 additional GPU/routing tests and eight production API/dispatch
  comparisons pass. Tail compaction adds12 graphs at B3; first real-shape setup/JIT/capture took23.40s
  versus0.63s reference in this run. [Earlier B1 measurements](docs/beam-fast.md) use a different baseline.
* **Timestamp rules.** Like ESPnet, every search applies its `ScoreFilter`: no timestamps after `<notimestamps>`;
  otherwise the output starts with a timestamp, timestamps come in increasing pairs, and `<eos>` only after a
  complete pair.
* **OWSM-CTC** has no decoder: one engine runs the prompt encoder and the encoder (with language/task prefix,
  cross-attention to the prompt and intermediate-CTC conditioning) and returns the frame-wise best path. Inputs are
  padded to 30 s, as on the model card. Batches hold utterances of similar durations, and the next batch is queued on
  the GPU before the host collapses the best paths of the one before. With batch 1, two engine calls are in flight
  at a time on two streams, each with its own TensorRT contexts (+13-17%, same output). `transcribe_long` decodes 30 s windows
  overlapping by 2 s on each side and keeps their middle frames, like ESPnet's `decode_long`; the windows of all
  recordings go through the encoder together.
* **FP8 encoder (experimental, opt-in).** `fp8=True` or `fp8_calibration=[recordings]` runs the encoder's linear
  layers and subsampling convolutions as FP8 (E4M3) GEMMs/convolutions on Hopper GPUs: inputs get a per-tensor scale,
  weights a per-channel scale. With `fp8=True` the input scales come from a file in the package, keyed by the
  checkpoint's sha256; unknown checkpoints build fp16 (logged). The calibration set barely matters: 8 recordings,
  English only or 900 in 7 languages give the same WER within ±0.1 on every test set above, and widening the scales
  by a margin only costs precision (`fp8_margin`). The FP8 build reuses the fp16 build's decoder engines.
* **Decoder megakernel (Hopper; on by default).** On an H100/H200 the decoder calls run as one persistent Gluon
  kernel each (`decoder_kernel="auto"`; `decoder_kernel=False` or `OWSM_TRT_MEGAKERNEL=0` keeps them on TensorRT): all
  layers and the output layer, with grid-wide barriers between stages, while a producer warp streams the next weight
  tiles into a shared-memory ring (TMA) across the barriers.
  * Single-row calls (greedy batch 1, windows of up to 32 drafted tokens) run on 32 SMs while the encoder works
    ahead on the others, and on 64 SMs when nothing runs alongside; the two give bitwise the same results.
  * Calls of up to 8 rows and 16 tokens (beam search) run on a second kernel over all SMs. Larger calls stay on
    TensorRT.
  * A one-token call takes 0.67 ms instead of 0.79 ms with TensorRT, a 32-token window 0.80 instead of 1.13 ms.
  * On the 1B model greedy batch 1 is 15-18% faster, with timestamps 22%, beam 5 24%, and the transcripts are 73/73
    ESPnet's in all three; batch 8 and 16 are unchanged (most of their calls stay on TensorRT).
  * Each kernel is checked against an fp32 decoder at load and on its first call. On any failure its calls go to
    TensorRT (logged).
  * It costs about 1.2 GB more GPU memory (1B), about 18 s of Triton compilation the first time (cached) and 3.5 s
    more load time. Other GPUs use TensorRT.
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
| `owsm_v4_medium_1B` | 73/73 | 1949 / 2943 | 3.91% → 3.91% |
| `owsm_v4_small_370M` | 5/5 | 3399 / 6226 | 4.17% → 4.17% |
| `owsm_v4_base_102M` | 8/8 | 4568 / 10607 | 8.70% → 8.70% |
| `owsm_ctc_v4_1B` | 73/73 | 1276 / 2080 | 4.43% → 4.43% |
| `owsm_ctc_v3.2_ft_1B` | 2/2 | 1191 / 1854 | 5.13% → 5.04% |
| `owsm_ctc_v3.1_1B` | 2/2 | 1306 / 2156 | 4.17% → 4.17% |
| `owsm_v3.2` | 3/3 | 902 / 1247 | 4.17% → 4.17% |
| `owsm_v3.1_ebf` | 3/3 | 1341 / 1938 | 3.74% → 4.00% |
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
| `powsm_ctc`, tasks `pr` / `asr` | 2/2 / 2/2 | 2480 / 3284 (`pr`) | 73/73 identical to fp32 |

The rows for `owsm_v4_medium_1B`, `owsm_v4_small_370M`, `owsm_v4_base_102M`, the three OWSM-CTC models,
`owsm_v3.1_ebf` and `powsm_ctc` were measured with the current code (the decoder megakernel on). The others were
measured earlier, before CTC drafts worked in batches, rows were refilled, the padding was left out of the
subsampling, multi-token calls got their own engine and the plugins and the megakernel came; the current code is
faster on them too (`owsm_v3.1_ebf` went from 541 / 626 to 1341 / 1938). Its 4.00% is one utterance of the 73 that
flips at a near-tie (3.65% with the TensorRT decoder only); on all of test-clean its WER is 2.55% with the megakernel
and 2.54% without, at the same speed: its conv2d front end gives the encoder twice the frames of v4's, so batch 1
is bound by the encoder.

With timestamps, greedy and beam-5 outputs are identical to ESPnet's too: `owsm_v3.1_ebf_base` 3/3 on CPU, and
`owsm_v4_medium_1B` 73/73 for both searches in fp32 on GPU (TensorRT fp16 greedy: 73/73, RTFx 1308).

Notes:
* Language codes follow each model's vocabulary (`eng`, or `en` for OWSM v1/v2). POWSM uses 20 s windows.
* POWSM's `asr` output drops letters its vocabulary lacks (e.g. upper-case `M`, `I`, `S` become `<unk>`), exactly
  as ESPnet's does; it is a phone recognizer first.
* OWLS 9B and 18B use the same code path but were not tried: building their engines likely needs more time and
  memory than the 1-hour test jobs used here (the 4B engines took 6 min).

## Requirements

You need an NVIDIA GPU (Ampere or newer for the custom kernels), Python ≥ 3.10, torch ≥ 2.10 and TensorRT ≥ 10.16;
Installing owsm-trt brings everything else (on Linux, torch brings Triton and NVRTC's `cuda-bindings`). The
PyPI defaults (`torch`, `tensorrt`) are both CUDA 13; a CUDA 12 torch with them works too (torch 2.10 cu12 +
TensorRT 11.4 was tested). Don't install `tensorrt-cu12` next to `tensorrt`: the two overwrite each other's files
(owsm-trt warns when it finds both).

The decoder megakernel (Hopper GPUs) needs Triton 3.8's Gluon, i.e. torch 2.14. With torch 2.10-2.13 the TensorRT
decoder is used (logged): about 18% slower at batch 1 on the 1B model, the same at batch 16. Any other slower
fallback (e.g. engines without the custom kernels) is logged with its reason too.

Tested with torch 2.14 + TensorRT 11.4 on an H200, both in the development environment and from a fresh install of
the wheel (same speed, every optimization engaged; `scripts/fresh_install.sh`, `scripts/engaged.py`). The export
and the custom kernels also work with torch 2.10-2.13 and TensorRT 10.16, 11.0 and 11.2.

## Limitations

* CTC drafts speed up greedy ASR only; beam search, translation and POWSM take one decoder call per token. Rows are
  refilled only in greedy ASR; otherwise a batch runs until its slowest utterance is done.
* The first input of each length (in 1 s steps) and batch size captures a CUDA graph, which takes a few encoder
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
* `profile_beam.py`: same-engine baseline/score-bound ablations, separate wall/diagnostic/eager-phase runs,
  token and segment comparisons, capture/dispatch/environment metadata. Run it on an allocated GPU node.
* `check_models.py`: TensorRT fp16 against PyTorch fp32 for several models.
* `profile_phases.py`: CUDA-event timing of the frontend, encoder, prefill, decode steps, and verify windows.
* `draft_report.py`: where CTC drafts and the decoder disagree.
* `dump_drafts.py` + `draft_sim.py`: drafts and greedy outputs of a test set, and an exact CPU replay of drafted
  decoding on them (an oracle stands in for the decoder) to count decoder calls for other draft rules.
* `pad_experiment.py`: WER with inputs padded to less than 30 s.
* `make_longform.py` + `eval_long.py`: long recordings from LibriSpeech chapters, and long-form WER / speed.
* `dwconv_bench.py`, `dwconv_encoder.py`: the depthwise-convolution plugin alone and inside the encoder.
* `qsub_pool.sh`: GPU jobs on a PBS cluster, at most N at a time (`gpu_eval.pbs` runs the stages listed in `STAGES`;
  the account comes from `OWSM_QSUB_GROUP` / `OWSM_QSUB_QUEUE` / `OWSM_QSUB_RTYPE` or `.cache/qsub.env`).
* `trt_layers.py`: per-layer TensorRT profile plus an engine-inspector dump.
* `encoder_variants.py`: encoder engines built several ways (optimization profile, trimmed features, CTC argmax),
  timed per batch size.
* `fp8_calibrate.py`, `prepare_multilingual.py`: the packaged FP8 scales (input ranges per recording on GPU, scale
  files for any subset on CPU) and the multilingual calibration/test data they were measured on.
* `latency.py`, `b1_timeline.py`: single-utterance latency, and a kernel timeline of a batch-1 call.
* `bench_espnet.py`: ESPnet `Speech2Text` baseline.
* `smoke_features.py`: language ID, timestamps, translation, prompts, and long-form.

On a cluster with slow shared storage (e.g. Lustre), `scripts/build_env_tar.sh` builds a relocatable environment on
local disk and packs it into one `.tar.zst`. `scripts/gpu_eval.pbs` unpacks it into `$PBS_LOCALDIR` and runs the
stages listed in `STAGES`.
