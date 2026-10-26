# Whole-runtime optimization, starting at1a94363

The measured scope expands beyond timestamp beam: direct greedy with/without CTC drafts,
timestamps, streaming greedy B1/B8/B16, streaming timestamps B8, no-timestamp beam, timestamp-beam
regression, translation, long prompt and a110-second recording. Main corpus is English73; Japanese12
covers direct greedy/TS, stream B1/B8 and no-timestamp beam. The 102M checkpoint is now measured too; CTC-only models and other GPUs remain pending. Engine/checkpoint hashes, warmup/capture/dispatch and all token/segment
outputs are saved. Single requests and whole-manifest throughput are separate metrics.

The first candidate is frontend pointwise fusion. STFT and full fp32 mel GEMM remain unchanged.
Power uses separate multiplication/addition with FMA disabled. Log/clamp/global-MVN uses CUDA
libdevice logf and round-to-nearest division, with FMA disabled. Per-utterance MVN keeps the original
mean reduction. Unsupported CPU/Triton or grad-enabled execution retains the PyTorch operations.
Both components are initially disabled and independently ablated (reference/power/post/both).
No cached audio/encoder output, reduced precision or changed audio padding is introduced.

GPU tests require exact feature tensors, graph replay, independent stream safety, strided complex
input, clamp boundaries and nonfinite values. Model tests and actual transcripts follow. The prior
beam eager event sample measured features/H2D0.885 ms, but that instrumented sample is only motivation;
it is not a greedy bottleneck or speed claim. New frontend events/traces and uninstrumented wall
measurements will decide adoption. Cold load/JIT/warmup are separated from steady-state samples.

## B1 frontend graph (measured and enabled)

`OWSM.from_pretrained(..., frontend_graph=True)` (default) captures STFT/mel/normalization on a dedicated
capture stream. Replays serialize frontend workspace use through an event, while encoder/decoder
work can still overlap. Each result is cloned before graph reuse, so another encoder stream cannot
observe overwritten features. Shape/mode keys use a bounded32-entry cache; larger batches, CPU, autocast or
grad-enabled paths run eagerly. Saturation falls back eagerly without eviction/recapture. Graph
inputs/outputs and intermediate private pools add memory, which must be measured before adoption.

The address-lifetime/private-pool contract follows the official
[PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html).
Tests retain outputs across streams, repeat cache hits, hit the capacity limit, and exercise B>1
fallback. The first allocation ran pointwise fusion separately; the second completed graph comparisons.
See [measured results](runtime-results-2770946.md). `frontend_graph=False` restores the eager path.

## First allocation: partial result, 2770849

98 GPU tests passed; B1/B8 features matched exactly for all four pointwise modes. All tokens,
timestamp tokens, text and segments matched across eleven completed English cases. The profiler
then passed an unsupported `prompt` argument to `transcribe_long`; this is fixed for the next run.
Long-form and Japanese were not measured in this allocation.

B8 frontend-event median was 0.2744 ms reference versus 0.1980 ms fused; feature/H2D wall median
was 0.7111 versus 0.5442 ms. Most whole-runtime RTFx differences were within roughly one percent;
pointwise fusion is not adopted as a default. B1 streaming showed 3.3–3.4% with individual fusion
components but only 1.0% combined, requiring replication before attribution. These partial results
are in `outputs/runtime-2770849/english`, including all alternating samples and token outputs.
The follow-up separates frontend graph alone from graph plus fusion, adds whole-runtime traces,
and includes the cached 102M checkpoint without an engine rebuild.

## Lazy draft setup (measured and enabled)

`OWSM.from_pretrained(..., lazy_drafts=True)` (default) defers verification buffers, backend closures and graph capture until the
first `decode_drafted`/`decode_stream` call. Translation and language identification can therefore
use a greedy session without preparing unused draft windows. Setup completes before timestamp
state/prefill because capture warmup writes self-KV. Repeat calls reuse the same setup. GPU validation passed in job2771009. False restores eager initialization; standalone `GreedySession`
retains eager construction by default and accepts the same explicit switch.
`scripts/profile_runtime_cold.py` compares session construction with resident engines/JIT, records
first observations separately, and checks mode switches on the same session. This optimization
targets cold latency and unused memory, not a steady-state decoder speedup.

## Throughput follow-up (not yet measured)

The whole-runtime trace shows decoder verification dominates B8; actual session routing includes
Gluon and TensorRT shape fallbacks. `scripts/profile_draft_windows.py` independently compares the
existing total verification-token budgets256/128/64/32 for B3/B8, with/without timestamps. This is
N-class: changing verification windows and dispatch can change fp16 outputs. All differences and
WER/CER are retained; no approximate acceptance or constraint change is made. Budget32/B8 without
timestamps also falls back to fixed batches because the four-token primer does not meet the current
stream condition; this scheduling change is explicitly recorded. Actual window/dispatch counts are
collected in a separate instrumented pass, outside wall timing.

CTC-only v4 1B is also added to the frontend-graph comparison using a locally cached checkpoint and
engine. Those results are pending. Multi-GPU, engine rebuilds, lower precision and altered audio
padding are outside these experiments.
