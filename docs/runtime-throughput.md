# Whole-runtime optimization, starting at1a94363

The measured scope expands beyond timestamp beam: direct greedy with/without CTC drafts,
timestamps, streaming greedy B1/B8/B16, streaming timestamps B8, no-timestamp beam, timestamp-beam
regression, translation, long prompt and a110-second recording. Main corpus is English73; Japanese12
covers direct greedy/TS, stream B1/B8 and no-timestamp beam. Other checkpoints, CTC-only models and
GPUs remain unmeasured. Engine/checkpoint hashes, warmup/capture/dispatch and all token/segment
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

## Prepared B1 frontend graph experiment (not yet measured)

An independent opt-in `model.frontend_graph=True` captures STFT/mel/normalization on a dedicated
capture stream. Replays serialize frontend workspace use through an event, while encoder/decoder
work can still overlap. Each result is cloned before graph reuse, so another encoder stream cannot
observe overwritten features. Shape/mode keys use a bounded32-entry cache; larger batches, CPU or
grad-enabled paths run eagerly. Saturation falls back eagerly without eviction/recapture. Graph
inputs/outputs and intermediate private pools add memory, which must be measured before adoption.

The address-lifetime/private-pool contract follows the official
[PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html).
Tests retain outputs across streams, repeat cache hits, hit the capacity limit, and exercise B>1
fallback. The first GPU allocation still runs only the separately frozen pointwise experiment.

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

## Prepared lazy draft setup (not yet measured)

`model.lazy_drafts=True` defers verification buffers, backend closures and graph capture until the
first `decode_drafted`/`decode_stream` call. Translation and language identification can therefore
use a greedy session without preparing unused draft windows. Setup completes before timestamp
state/prefill because capture warmup writes self-KV. Repeat calls reuse the same setup. The initial
default is False until GPU validation; standalone `GreedySession` has the same `lazy_drafts` switch.
`scripts/profile_runtime_cold.py` compares session construction with resident engines/JIT, records
first observations separately, and checks mode switches on the same session. This optimization
targets cold latency and unused memory, not a steady-state decoder speedup.
