# Runtime adoption validation: 2771009

Frozen commit678cbc2, H200, 112 tests passed; seven alternating repeats with lazy draft setup enabled.
English73/Japanese12, 102M English73, long110s. All features and all output fields match.
The output records also match the previous eager-draft job2770946 for every case and mode, with
identical checkpoint/engine hashes. This validates the new high-level defaults:
`frontend_graph=True, lazy_drafts=True`. Both switches accept False for their original paths.

| Dataset | Case | p50 ms eager → graph | p95 ms eager → graph | RTFx change |
|---|---|---:|---:|---:|
| base102m | single_greedy | 1.990 → 1.733 | 3.220 → 2.994 | +12.06% |
| base102m | single_ts | 2.736 → 2.468 | 4.251 → 3.990 | +8.63% |
| base102m | stream_greedy_b1 | 99.214 → 92.654 | 99.769 → 92.850 | +7.17% |
| base102m | stream_greedy_b8 | 43.423 → 43.507 | 62.484 → 51.997 | +4.72% |
| base102m | beam_no_ts | 11.123 → 11.034 | 24.640 → 24.500 | +1.70% |
| english | single_greedy | 4.906 → 4.655 | 7.702 → 7.481 | +4.96% |
| english | single_ts | 6.550 → 6.275 | 9.724 → 9.490 | +3.64% |
| english | stream_greedy_b1 | 255.103 → 251.741 | 259.534 → 263.910 | +0.16% |
| english | stream_greedy_b8 | 171.384 → 171.682 | 176.395 → 174.999 | +0.13% |
| english | stream_greedy_b16 | 162.657 → 161.236 | 166.064 → 162.938 | +0.82% |
| english | stream_ts_b8 | 197.836 → 198.241 | 206.145 → 200.683 | +0.60% |
| english | plain_greedy | 23.334 → 23.038 | 87.642 → 87.364 | +0.74% |
| english | beam_no_ts | 22.732 → 22.452 | 50.193 → 49.852 | +0.92% |
| english | beam_ts | 28.178 → 27.885 | 107.565 → 107.365 | +0.65% |
| english | translation | 24.804 → 24.593 | 87.558 → 87.439 | +0.64% |
| english | prompt | 7.858 → 7.605 | 16.895 → 16.584 | +2.15% |
| english | long_greedy | 76.236 → 75.195 | 76.617 → 75.318 | +1.47% |
| japanese | single_greedy | 6.333 → 6.034 | 11.410 → 11.219 | +3.87% |
| japanese | single_ts | 8.039 → 7.849 | 13.711 → 13.541 | +2.67% |
| japanese | stream_greedy_b1 | 59.433 → 58.757 | 59.847 → 58.921 | +1.33% |
| japanese | stream_greedy_b8 | 45.400 → 45.067 | 45.669 → 46.024 | +0.16% |
| japanese | beam_no_ts | 31.490 → 31.258 | 76.877 → 76.636 | +0.61% |

Throughput gains in1B streaming B1 are not robust:0.16% here versus2.47% previously. Large-batch
gains are also generally small; the frontend remains eager except singleton tails. The large
graph+fusion outliers from2770946 are retained in that report, with no claimed explanation.

## Session-cold setup (resident backend, warmed JIT; three alternating repeats)

| Model | First request | Eager ms | Lazy ms | Resident memory saved MiB |
|---|---|---:|---:|---:|
| 1b | translation | 416.76 | 277.89 | 45.37 |
| 1b | lid | 397.43 | 264.13 | 45.37 |
| 1b | asr | 397.39 | 394.77 | -0.06 |
| 1b | prompt | 387.02 | 406.19 | -0.06 |
| 1b | stream | 845.28 | 845.22 | 0.00 |
| 102m | translation | 198.48 | 126.61 | 26.12 |
| 102m | lid | 189.54 | 120.45 | 26.12 |
| 102m | asr | 188.03 | 188.52 | 0.00 |
| 102m | prompt | 191.74 | 190.35 | 0.00 |
| 102m | stream | 384.71 | 384.07 | 0.00 |

These are session reconstruction costs, not model-load times or warm request latency. Lazy setup
does not reduce ASR verification work. The1B long-prompt cold median increased387→406ms in this
small sample; stream cold medians were unchanged. Translation/LID save unused graphs. Mode-switch
and reuse output checks passed. Load/first observations are separate in each `cold.json`.

Graph private buffers add memory/capture work. After the eager path was warm, the16 additional
English shapes increased allocated memory by531.5MiB and reserved memory by about810MiB on both
1B/102M (17 total entries). This is a material memory cost for the single-request latency gain.
Allocator deltas during warmup are recorded in each
`cold.json`, but can include deferred backend allocations. Steady-state peak memory includes all
warmed variants and must not be presented as isolated graph memory savings.

Artifacts: `outputs/runtime-2771009/analysis.json`, per-casewall/tokens/cold,
`lazy_vs_previous.json`, `cold_1b/cold.json`, `cold_102m/cold.json`.

Reproduce within a compute allocation:
```bash
uv run python scripts/profile_runtime.py --model <model> --manifest <manifest> --variants reference graph --lazy-drafts --repeat 7 --out <result-dir>
uv run python scripts/profile_runtime_cold.py --model <model> --manifest <manifest> --repeat 3 --out <cold-result-dir>
```

Timing scope: steady-state calls use resident models and already loaded audio. Disk load is recorded
separately as an aggregate, not as per-request disk-inclusive p50/p95. The long-form API returns
empty joined token lists; its comparison verifies text/segment timing, not hidden window token IDs.
