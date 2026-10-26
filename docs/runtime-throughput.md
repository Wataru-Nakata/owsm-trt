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
