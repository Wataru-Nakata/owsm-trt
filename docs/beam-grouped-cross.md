# Grouped beam-query cross-attention experiment

Candidate20, opt-in `GluonDecoder(..., group_cross_queries=True)` and profiler
`--gluon-group-queries --share-cross-kv`. Reference `e32e48a` already shares cross-KV storage.

Previously each beam query used a separate SP=16 attention tile with one valid query row. The
candidate packs the sibling queries into the same tile, reusing each K/V load. It also prefetches
once per utterance/head/slice. Self-attention, logits, global top-K and beam state stay separate.
The cross-attention slice count NS remains the original value for R query rows; there is no new
softmax/reduction partition or precision change. The existing routing limit bounds the grouped
tile to16 queries. Float operation/layout equivalence still requires exact empirical checks.

Ten compute-only tests compare logits, KV, graph replay and beam state under both grouped and
individual layouts, with prefetch on/off, B1K5 and B2K3, duplicate parents, different prompts and
bucket growth. CPU orchestration checks pass (4 passed,10 skipped). Allocation regressions include
both modes and existing Ring/fallback tests (17 extra tests, plus221 base tests).

`EXPERIMENT=group` benchmarks B1/K5 on English73 and Japanese12, alternates reference/candidate,
keeps the same engines/weights, and includes the six English extra cases. This isolates query
packing from the already adopted storage-sharing change. GPU performance and adoption are pending.

## Result: not adopted

Job2769166 completed in4m02s with221 base and17 additional tests passing. All English73/Japanese12
outputs and decoder call counts match; all18 English extra-case comparisons match. Actual dispatch
is Gluon with shared KV in both modes, grouped queries only in the candidate, and no fallback.

| Input | Reference p50 → grouped (ms) | RTFx change | Paired median change |
|---|---:|---:|---:|
| English | 25.64 → 26.23 | -2.14% | -2.19% |
| Japanese | 34.94 → 35.73 | -2.14% | -2.13% |

The reduction in repeated loads did not produce a latency gain. Existing independent-query attention
is retained. The experimental implementation stays on the separate `beam-grouped-cross` branch;
it is excluded from the final adopted branch. No changed slice count or larger routing limit was
attempted. [Full results](beam-results-2769166.json) and `outputs/beam-2769166/` preserve the negative
result. Admission184+0+1=185; final capacity176allocated,0queued, no observed overage.
