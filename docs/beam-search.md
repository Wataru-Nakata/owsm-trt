# Timestamp beam search: implementation and evaluation

Baseline: `892c29fb9b01b22c3e56e6848af818c5470c9288` (also remote main at the start on 2026-10-26).
Work branch: `beam-timestamp-profile`. The original `v0.2` branch is preserved.
The [candidate ledger](beam-candidates.tsv) tracks all 60 supplied proposals. No unmeasured speed estimate
is presented as a result. The first experiment is PBS job `2765885.pbs1` (one H200 allocation,
`gai51740`, `R9920261300`, `rt_HG`, one node/GPU, maximum one hour).

## Behavior

`BeamSession` rejects extensions from inactive parents and non-finite or timestamp-forbidden candidates.
The old finite `LOG_ZERO=-1e10` sentinel could fill unused top-K slots and revive them; candidates now
use `-inf` plus explicit parent validity. Global top-K still selects EOS alongside other tokens; no
extra live slots are replenished. Only valid candidates enter finished history, including at forced EOS.
An exhausted search returns the previous best finished hypothesis, or raises if none exists. NaN/+Inf
in live logits sets a device error flag and raises at the end of decoding, after existing synchronization.
All-negative-infinity rows count as exhausted, not numerical errors. Errors from inactive rows are ignored.

`OWSM.from_pretrained(..., beam_upper_bound=True)` or `BeamSession(..., upper_bound=True)` enables the
strict bound. `BeamSession.set_upper_bound(bool)` changes a device flag without rebuilding engines or
recapturing graphs. The default remains **OFF**. It is applied after best-finished updates, and ORed
with existing `end_detect`, exhaustion and length-limit rules. Equality does not stop the search.

The bound applies to the current retained beam's top-1 result: sums of log-probabilities cannot increase
under extension. It does not establish global search optimality or preserve an n-best list. Positive
rewards, length normalization and future rescoring need a new proof. Timestamp masking remains **after**
full-vocabulary log-softmax. The change does not change greedy decoding or introduce approximate search.

## Diagnostics

Construct `BeamSession(..., diagnostics=True)` and call `get_diagnostics()` after `decode()`.
The decode return value is unchanged. Diagnostics-disabled sessions allocate no history tensors and
perform no history writes. The numerical-error flag is always present for correctness.

History uses fixed device buffers indexed by the search step; it is copied to the host in bulk on request.
Each utterance records best-finished/max-live scores, live count, EOS, best updates, bound satisfaction,
logical completion, numerical errors, output end and stop mask. Non-finite scores serialize to JSON null.

* `prefill_calls=1`, `prefill_selections=1`; selection 0 does not count as a single-token decoder call.
* `logical_end_step` is a zero-based search-loop index; `decoder_calls` excludes prefill.
* `best_end_with_primer` is the sequence end including primer and any selected EOS, excluding the extra
  EOS implicitly appended at the length limit. `history.position` is the selected token's sequence index.
  `best_finished_length_with_primer` includes that forced EOS, matching the length used by end_detect.
* `host_observed_done_step` is the final batch-wide completion check, not an individual-row host check.
  `extra_decoder_calls` includes chunk overshoot and waiting for other utterances.
  `chunk_overshoot` is the distance to the first scheduled check after logical completion.
* Stop bits: `1` length limit, `2` no valid live candidates, `4` existing end_detect, `8` enabled score bound.
  Multiple bits can be set. The summary preserves the mask at the first logical completion.

## Reproduction

Only run inference, audio processing and the model regression tests in an allocated compute job.
The provided PBS script stages environment, source, model, engine, audio, compiler caches and detailed
logs on `$PBS_LOCALDIR`; completed artifacts are copied back. A source tar is made **before submission**
and includes the original runtime and exact source revision. It does not follow edits made while queued.
The initial resource settings were approved in the implementation plan. Additional jobs require approval.

```bash
# Inside the allocated compute environment, with the original runtime saved using:
# git show 892c29f:src/owsm_trt/runtime.py > baseline_runtime.py
uv run python scripts/profile_beam.py \
  --model .cache/models/owsm_v4_medium_1B \
  --manifest .cache/testdata/librispeech_dummy/manifest.json \
  --beam 5 --timestamps --batch 1 3 --repeat 5 --limit 12 \
  --upper-bound both --baseline-runtime baseline_runtime.py \
  --phases --trace --disk --extra-checks --long-audio .cache/testdata/long_110s.wav \
  --out outputs/beam-profile
```

Additional controls: `--prompt`, `--lang`, `--task`, `--backend`, `--decoder-kernel`, `--chunk`,
`--compare-chunks 2 1` (additional bound-ON modes on the same graphs), `--no-timestamps`.
This profiler requires K>=2 and B*K<=16, matching the default engine profiles.
Manifest paths follow the existing `eval_wer.py` format. Inputs longer than one model window are rejected
by the short-window benchmark; `--long-audio` uses the separate long-form compatibility check.

For CPU-only synthetic state verification (no audio or model inference):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  uv run pytest -q -p no:cacheprovider tests/test_beam.py
```

## Measurement contract and limitations

`wall.json` contains alternating baseline/fixed-OFF/fixed-ON measurements, reversing order on successive
repeats. All use the same loaded backend/checkpoint/engines. All batch/feature shapes are warmed first.
Wall sessions have no diagnostics, events or profiler. `transcribe_ms` includes features/H2D, encoder,
prefill, search, host synchronization and output formatting, on already loaded audio. `decode_ms` comes
from a separate decode with encoder outputs prepared outside the timed interval. Neither includes disk
I/O. `disk.json` times the public API on paths, explicitly with the OS file cache unflushed.

`diagnostics.json` is a separate graph run whose token outputs are checked against wall runs. `phases.json`
uses a separate eager session and CUDA events for features/H2D, encoder, cross-KV expansion, prefill,
decoder, selection, reorder, finish/history, and host-check wall time. Eager events contain launch gaps;
they are not kernel-only times and are not graph-phase measurements. Graph-internal numbers remain null.
With `--trace`, kernel-only decode durations are summed by matching launch correlations to host phase
ranges; this is eager diagnostic execution and excludes encoder, copies and launch gaps. The optional
PyTorch trace supports further inspection; Nsight
availability is recorded. Event sums must not be interpreted as concurrent end-to-end wall time.

`cold.json` records session construction/JIT/capture, warmed feature shapes and per-cache decoder dispatch.
`environment.json` records source revision, software/GPU/driver, checkpoint and engine SHA256, build
metadata, requested kernel, environment and load/build components. JIT not independently attributable
from session setup is explicitly unavailable. Initial kernel selection failures are in `runtime.log`;
shape/trial fallbacks on `MegaBackend` also carry closure metadata. Gluon's current route permits at most
8 rows (16 tokens); beam 5 B=3 exceeds that row limit and uses the underlying TensorRT decoder.

The p50/p95 samples are **batch call** latency, not independently measured latency for every row of B=3.
The dataset contains different durations; per-item repeated samples are retained to avoid confusing
input-length variance with timing noise. RTFx is processed seconds divided by total transcribe wall time.
Peak VRAM includes resident comparison sessions; it is not the incremental memory cost of one mode.
Extra compatibility case timings include cold capture and are not steady-state speed measurements.

Text WER/CER is computed only when textual references exist. Timestamp-token and segment agreement
with baseline is not reference-time accuracy. Reference timestamp errors remain unavailable. Japanese,
music, fast-speech and multilingual timestamp quality require additional data/evaluation.

## Validation status

The original code failed 10 of the 15 synthetic validity cases; the validity fix passes all 15.
With strict-bound/diagnostic tests added: **21 passed, 1 CUDA test skipped** on the login-node sanity
environment (Python 3.12, torch 2.8.0; this is below the project's declared torch minimum and does not
establish supported-runtime/GPU compatibility). Ruff, shell syntax and diff-whitespace checks passed.
The allocated job runs the synthetic CUDA graph test, existing naive-beam/greedy/timestamp comparisons
(bound ON and OFF), model tests and long-form tests using the packaged GPU environment.

The allocated GPU job passed **100 tests** in 23.73 s, including the CUDA graph diagnostics/toggle test
and both ON/OFF naive-beam comparisons. GPU environment: H200 (132 SM), CUDA 13.0, driver 595.71.05,
torch 2.14.1, Triton 3.8.0, TensorRT 11.4.0.106. Engines were loaded, not rebuilt.

## First GPU results (12 English utterances)

Raw artifacts: `outputs/beam-2765885/`. The committed [compact result record](beam-results-2765885.json)
includes hashes, environment, statistics and limitations. All variants used the same fp16 checkpoint,
engines and decoder backend. Each input was repeated five times.

| Batch | Mode | Call p50 / p95 (ms) | Decode p50 (ms) | Decoder calls p50 | RTFx |
|---|---|---:|---:|---:|---:|
| 1 | Original | 106.69 / 553.76 | 102.62 | 82.5 | 40.84 |
| 1 | Fixed, bound OFF | 113.34 / 582.91 | 109.29 | 82.5 | 38.74 |
| 1 | Fixed, bound ON | 41.81 / 148.70 | 38.24 | 28 | 194.68 |
| 3 | Original | 897.62 / 908.42 | 893.98 | 373 | 36.36 |
| 3 | Fixed, bound OFF | 921.79 / 941.15 | 922.53 | 373 | 35.47 |
| 3 | Fixed, bound ON | 108.95 / 223.91 | 100.29 | 52 | 261.21 |

**B=3 caveat:** the initial warm-up sorted the whole manifest, while timed calls used manifest-order
groups. Some first-repeat calls captured new feature shapes. The B=3 table excludes the entire first
repeat for **all** modes (16 calls/mode); raw data including those captures remains unchanged. B=1 uses
all 60 calls/mode. Commit `35f3989` warms the exact measured groups and raises if any timed call captures
a new encoder shape. This correction needs its own GPU rerun; the table is an explicitly filtered first result.

For B=1, within-input transcribe coefficient of variation has median 0.057% / 0.065% / 0.051%
(original/OFF/ON); the large across-input standard deviation reflects different input/decode lengths.
B=3 within-input variation is larger: median 0.032% / 2.44% / 1.87% after excluding repeat 0.
The guards and bound computation add cost when stopping is disabled: OFF RTFx is about 5.1% below
original at B=1, 2.5% below at B=3. The bound's benefit comes from fewer calls and smaller reached cache
buckets, not a faster decoder kernel. These results do not establish speed on other datasets or GPUs.

All 12 outputs match original **all token IDs, timestamp IDs, text and segments** for both B=1 and B=3,
ON and OFF. WER is 11/303 = 3.63% in every case. Each produces 13 segments, with zero empty segments,
invalid intervals, unpaired timestamps or nonmonotonic times. Extra no-timestamp, English-to-English
translation, long prompt, silence/noise, repeated-speech 30 s window and the available 110 s recording
also match original results; their cold-inclusive timings are not performance claims.

Original/OFF reaches 374 selections on 5/12 B=1 inputs; ON logically finishes at search steps 17–109,
always through the strict bound. B=1 has 0–3 extra decoder calls (mean 1.58). B=3 still waits up to 84
calls for another batch member after an utterance logically finishes. Candidate 29 is a small next
ablation; candidate 33 requires a larger state-management change.

Peak allocated memory, with comparison sessions resident: 6.29 GiB at B=1; B=3 original/OFF 8.41 GiB,
ON 8.27 GiB. These are not isolated per-mode memory measurements. GPU utilization sampled every 5 s
over the entire test/profile job averages 69.1% (0–100%), including setup; per-mode utilization is unavailable.
`from_pretrained` takes 11.48 s. Initial fixed B=1 session construction/JIT/capture takes 2.79 s; original
is constructed afterwards and benefits from already compiled kernels, so those cold times are not a fair ablation.

In the **eager diagnostic decode trace**, B=1 kernel sums are decoder 19.55 ms, selection 4.25 ms,
reorder 1.77 ms, finish/history 2.98 ms, prefill 1.12 ms; B=3 gives 48.32 / 8.91 / 8.14 / 5.80 / 1.10 ms.
Decoder accounts for about 65–66% of these kernel sums. History instrumentation and eager launches
distort other phase costs; no graph-phase fraction or projected optimization speedup is inferred.
Nsight tools were unavailable. B=1 dispatches Gluon, B=3 falls back to TensorRT because 15 rows exceed
the 8-row route limit. No decoder, KV or approximate-search redesign is adopted from this evidence.

A follow-up of English 73 inputs, Japanese 12 inputs and chunk 4/2/1 was approved with the same one-GPU,
one-hour settings. At the pre-submission check the account already held **192 GPUs**; the follow-up is
prepared but has not been submitted, respecting the user's account-wide 192-GPU cap.
