# B1 frontend graph comparison: 2770946

H200; identical checkpoint and TensorRT engines within each model; frozen commit9483faf.
English73 (some cases use12/6), Japanese12, and one110-second recording. Five alternating repeats.
101 tests passed. All compared features/tokens/timestamp IDs/text/segments matched exactly.
No reference boundary annotations; this verifies agreement, not timestamp accuracy.

| Dataset/model | Case | p50 ms eager → graph | p95 ms eager → graph | Aggregate RTFx change |
|---|---|---:|---:|---:|
| base102m | single_greedy | 1.985 → 1.744 | 3.179 → 2.962 | +11.46% |
| base102m | single_ts | 2.714 → 2.473 | 4.236 → 3.968 | +8.37% |
| base102m | stream_greedy_b1 | 99.100 → 92.293 | 100.284 → 92.471 | +7.36% |
| base102m | stream_greedy_b8 | 43.831 → 43.700 | 57.061 → 58.427 | +0.90% |
| base102m | beam_no_ts | 10.914 → 10.719 | 24.375 → 24.208 | +1.69% |
| english | single_greedy | 5.238 → 5.004 | 8.081 → 7.820 | +4.28% |
| english | single_ts | 6.799 → 6.624 | 10.006 → 9.831 | +2.87% |
| english | stream_greedy_b1 | 257.441 → 254.108 | 267.723 → 255.612 | +2.47% |
| english | stream_greedy_b8 | 173.169 → 170.899 | 175.569 → 178.567 | -0.18% |
| english | stream_greedy_b16 | 163.612 → 163.524 | 164.975 → 164.093 | +0.26% |
| english | stream_ts_b8 | 199.928 → 198.066 | 206.145 → 215.940 | -0.66% |
| english | plain_greedy | 23.736 → 23.532 | 88.433 → 88.226 | +0.56% |
| english | beam_no_ts | 22.930 → 22.721 | 50.429 → 50.220 | +0.73% |
| english | beam_ts | 28.432 → 28.236 | 107.872 → 107.756 | +0.50% |
| english | translation | 25.246 → 25.027 | 88.469 → 88.268 | +0.58% |
| english | prompt | 8.007 → 7.842 | 16.767 → 16.554 | +2.07% |
| english | long_greedy | 75.995 → 75.391 | 76.095 → 75.455 | +0.83% |
| japanese | single_greedy | 6.299 → 6.081 | 11.393 → 11.263 | +3.04% |
| japanese | single_ts | 8.122 → 7.919 | 13.730 → 13.519 | +2.30% |
| japanese | stream_greedy_b1 | 58.999 → 58.130 | 59.263 → 58.279 | +1.40% |
| japanese | stream_greedy_b8 | 44.721 → 44.851 | 45.357 → 45.653 | -1.10% |
| japanese | beam_no_ts | 30.903 → 31.360 | 76.111 → 75.993 | -0.16% |

Stream cases time the whole corpus, not a single utterance. Aggregate RTFx uses all samples;
paired per-input medians and variation are in `outputs/runtime-2770946/analysis.json`.
B1 English feature/H2D p50:0.2358→0.1238ms. Same-input median CV for single greedy:
1B0.59→0.47%, 102M2.57→1.73%; paired median speedup4.42%/13.51%, respectively.

Graph+pointwise fusion had large outliers:102M streaming B8 reached607ms (ordinary samples≈43ms),
1B streaming B16 reached242ms (ordinary≈164ms). Decoder call counts and outputs did not change.
The cause is unresolved; fusion stays disabled, and graph alone is repeated in the next allocation.
B8/B16 graph alone has little expected benefit because only a singleton tail uses the graph.

Graph caches retain private memory. English single-greedy cold state held18 graph entries across
the two graph modes, with22.02MB input/output buffers, plus unquantified private workspace/intermediates.
Steady-state peak allocated values include all warmed variant caches and cannot establish isolated
memory savings. The next run records additional allocator deltas during capture.

Raw per-case `wall.json`, `tokens.json`, `cold.json`, environments and phase/whole-runtime traces are
under `outputs/runtime-2770946/{english,japanese,base102m}`. Traces are instrumented observations;
their summed kernel or CPU durations are not the uninstrumented wall measurements above.

Reproduce using `scripts/profile_runtime.py --model <local model> --manifest <manifest>
--variants reference both graph graph_both --repeat 5 --out <directory>` inside a compute allocation.
