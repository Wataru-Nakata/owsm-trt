# Multi-utterance frontend replay: isolated experiment

The adopted frontend graph currently handles B1 only. Existing measurements showed feature/H2D
work falling from0.236 to0.124ms atB1, with small/neutral effects on larger-batch workloads whose
frontend remains eager. `FrontendReplay(max_batch=8)` now prepares an opt-in B2..8 experiment.
The public pipeline retains a private batch limit1; no default expansion or speed claim is made.

The same STFT/mel/normalization at the same input shape is captured. The cache remains bounded
at32 entries per variant. Capture-stream/workspace isolation, cross-stream event ordering and
independent cloned outputs are retained. Tests extend the existing global/utterance MVN, fused /
unfused, cross-stream lifetime, graph-cap and autocast checks to mixedB3/B8 shapes; unsupported
larger batches remain eager. GPU correctness and the first performance comparison passed in2776301 below; the public limit remains1.

`profile_frontend_batches.py` first checks exact real-model features and separately measures
feature/H2D time and graph allocation changes atB3/B8. It then compares B3 timestamp plain/beam,
B8 translation and drafted timestamp ASR, English73/Japanese12 on1B/102M,11 alternating pairs.
B1 graph replay remains enabled in both modes. Each mode owns a separate frontend graph cache;
the warm capture guard now compares each mode to its own warmed count. Full token/time/text /
segment equality and no runtime decoder fallback are required. Call diagnostics are separate.

Graph private pools may add substantial memory and cold capture work. The recorded feature
allocation delta includes a retained output/library workspace; end-to-end peaks include both
resident variants. These are not isolated deployment-memory savings. Existing B1 graph gains
must not be reported as a new B3/B8 gain. No change is adopted unless measured request/throughput
benefits justify the added capture/memory cost.


## First result2776301: retain public limit1

Frozen317e58a completed with exit0;11 GPU/CPU tests passed, real features matched atB3/B8 for
both models/corpora, and all16 conditions matched full tokens/time IDs/segments/text and decoder
call totals. Eleven alternating whole-list repeats gave aggregate RTFx changes:

|Model/corpus|Plain TS B3|Translation B8|Beam5 TS B3|Drafted TS B8|
|---|---:|---:|---:|---:|
|1B English73|+0.81%|+0.62%|+0.42%|+2.49%|
|1B Japanese12|+0.78%|−0.53%|+0.40%|+1.82%|
|102M English73|+1.21%|+2.75%|+3.60%|+5.11%|
|102M Japanese12|+1.41%|+0.13%|+1.14%|+2.03%|

Several aggregate changes include large outliers:102M English translation/beam paired medians
are only+0.65%/+1.33%, versus aggregate+2.75%/+3.60%.1B English drafted B8 has9/11 positive
pairs, a−5.78% minimum and+1.24% paired median.102M drafted B8 has11/11 positive pairs in both
corpora, paired medians+4.77%/+1.67%; this narrower case merits separate replication.

Feature/H2D B3 medians fell0.332–0.387→0.233–0.287ms. B8 changed little
(0.625–0.710→0.596–0.712ms). A single captured feature shape added70.6–102.4MiB of allocated
memory, including one output/library workspace. Across the four cases the English candidate
retained21 graphs versus one reference graph; Japanese retained six versus zero. Input/output
cache bytes exclude private intermediate pools, so neither that count nor co-resident end-to-end
peaks establish isolated deployment VRAM. Cold order includes first JIT/TRT work and is biased.

No broad default expansion: the small/noisy gains,1B translation regression and graph storage do
not justify enabling every B2..8 request. Keep the existing B1 default; the private experiment
remains reproducible.102M drafted B8 is pending independent throughput and isolated graph-memory/
cold measurement. No reference translation or boundary quality is available.
[Environment, paired samples, memory and equality](runtime-frontend-batches-2776301.json);
raw artifacts in `outputs/runtime-2776301/`, reproduction `scripts/profile_frontend_batches.pbs`.

## Narrow replication prepared

`profile_frontend_batches_recheck.pbs` targets102M drafted timestamp B8 only, English73/Japanese12,
11 independent alternating pairs. The unchanged frontend's11 passing tests from2776301 are
retained with a source hash; they are not rerun. Feature/output agreement is checked again.
`--cache-memory` constructs the sorted B8 encoder supply's cache with only one variant resident
at a time, records retained and peak allocated deltas, then releases it before the next variant.
Model and eager expected features stay fixed. PyTorch allocator observations exclude non-PyTorch
library/device allocations; post-release deltas reveal residual workspace. Cold feature/H2D calls
are synchronized and timed separately from equality checks and whole-request measurements.
This replication is pending; broad public expansion remains disabled.
