# Exact beam state update fusion

Reference4d8466a; adopted default on CUDA for K<=8/capacity<=512 after job2770337.
`BeamSession(..., fused_state=False)` disables it; `beam_kernels=False` also disables it through OWSM.
The profiler keeps explicit `--fused-state` for ablations.
This candidate combines index decomposition, active-parent validity, score copy, token-history
parent gather/write and timestamp-state updates into one kernel. Full-vocabulary log-softmax and
global PyTorch top-K remain unchanged, including their tie choices. Self-KV gathering and finished
hypothesis handling retain the previously validated kernels.

One CTA owns each utterance and all K source/destination rows. All parent history and timestamp
state loads precede a block barrier before in-place stores, covering duplicate parents and cycles.
The barrier contract is documented by [Triton](https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html).
There is no device-to-host state read or per-step allocation for the parent/token/valid outputs.
The initial specialization supports K<=8, capacity<=512 on CUDA with Triton; other cases keep the
existing operations. Diagnostics remain optional and the decode return type does not change.

Small interpreter checks cover K5/B3 identity/duplicates/cycles, timestamps/no timestamps, limits
0/15/511, repeated updates, inactive/done rows and invalid scores. Existing boundary tests and the
new state checks initially pass53 tests with3 GPU cases skipped; separate NaN/+Inf/all-inf cases
also pass. Graph/bucket tests plus a real decoder check are allocation-only. The GPU allocation passes all275 tests (221 regression,23 cross/backend,31 state).
The attention split sweep is a separate experiment, not enabled by this state option.

The state allocation compares English73/Japanese12 at B1/B3 on the same loaded engines as the
reference, including extra cases, separate eager profiling and exact diagnostics. The profiler now
attributes the fused metadata kernel separately as `state_update`; graph-internal timing remains
unavailable. CPU-only timings are not performance evidence.

## Same-engine result (H200,1B,K5,timestamps)

Against4d8466a, alternating reference/variant on the same loaded checkpoint and engines:

| Inputs | B | p50 before → after (ms) | RTFx gain |
| --- | --- | --- | --- |
| English73,5 repetitions | 1 | 25.66 → 24.58 | +4.30% |
| English73,5 repetitions | 3 | 49.73 → 48.01 | +4.24% |
| Japanese12,3 repetitions | 1 | 35.05 → 33.60 | +5.07% |
| Japanese12,3 repetitions | 3 | 77.22 → 73.85 | +4.02% |

All tokens, timestamp tokens, segments and decoder call counts match, including partial final
batches. Each English batch group (73/73 B1,25/25 B3) and each Japanese group is faster by paired
median. Additional no-timestamp, translation, long-prompt, silence/noise,30-second repeated speech
and110-second audio checks match. WER45/1150 and CER26/592 are unchanged; reference timing is absent.
B3 times are batch latency, not per-utterance latency. Instrumented eager phase measurements are
separate from the non-instrumented CUDA-graph wall benchmark.

See [results, environment hashes and p95](beam-results-2770337.json). Raw results are in
`outputs/beam-2770337`; reproduce the shared-cross reference command with `--fused-state` and
`--baseline-runtime` pointing to4d8466a. Cold load/JIT/capture are recorded separately.
