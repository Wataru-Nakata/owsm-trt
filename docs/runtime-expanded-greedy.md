# Expanded multilingual greedy timing

This measures the existing public `greedy_kernels=False/True` switch using one checkpoint and
engine per model. It does not measure a whole-version change from892c29f or attribute a combined
switch's gain to one component. The larger [output audits](runtime-large-regression.md) separately
pass13,232 inputs at B1 and bulk B8 for both1B and102M.

The timing sample has32 speech inputs per language (224 total), balanced deterministically across
duration/dataset strata, excluding prior screening overlap, nonspeech and recordings beyond30seconds.
This is a balanced performance sample, not a corpus-weighted speed estimate or an independent quality
holdout. B1 measures each input separately; bulk sends the32 same-language inputs in one call with
batch8. Seven alternating OFF/ON repeats follow warmup of every input and verification shape. Full
outputs must remain stable within repeats and identical across variants; new captures invalidate
steady-state timing. Audio loading/resampling, cold startup and diagnostic dispatch counts are separate.

## 1B / H200

[Job2786368](runtime-expanded-greedy-2786368.json), frozen240287b, completed PBS F/exit0 and publication
exit0. All14 conditions,224 input IDs, paired timing records, full output sequences and resolved flags
pass validation. There are no OFF/ON draft-decoder-call differences. All224 B1 candidate outputs also
match their saved full-corpus B1 audit outputs (same runtime tree, checkpoint/engines and flags).
Torch's distribution version is recorded as2.14.1; the audit's runtime string includes`+cu130`;
both record CUDA13.0. TensorRT11.4.0.106 and Triton3.8.0 versions are in the artifact.

| Language | B1 RTFx change | B1 p50 ms OFF → ON | B1 p95 ms OFF → ON | Bulk RTFx change | Bulk positive paired repeats |
|---|---:|---:|---:|---:|---:|
| German |+6.31%|8.371 →7.892|15.165 →14.135|+0.27%|4/7|
| English |+6.69%|7.163 →6.692|21.297 →19.896|+0.74%|6/7|
| French |+6.93%|9.237 →8.646|23.530 →21.891|+0.72%|5/7|
| Japanese |+6.45%|7.010 →6.618|13.796 →12.890|+0.74%|5/7|
| Russian |+6.78%|9.202 →8.556|27.986 →26.112|+1.47%|6/7|
| Spanish |+6.12%|7.966 →7.528|17.604 →16.363|+1.41%|7/7|
| Chinese |+6.87%|8.695 →8.054|22.554 →20.790|+0.19%|4/7|

RTFx changes use pooled warm wall time. The p50/p95 values pool224 per-input timings per language
(32 inputs ×7 repeats), so they include duration variation and are not224 independent audio samples.
At B1, all seven paired repeat totals improve in every language and all224 input medians improve.
Bulk differences are much smaller; six languages have at least one negative paired repeat. Across
languages the bulk paired-repeat range spans−3.41% to+3.28%. This run does not establish a consistent
bulk gain for every language. Raw per-request samples, standard deviations and repeat totals remain
available; no outliers are removed.

Model load takes7.407seconds in this run, excluding environment/corpus staging. Per-condition warmup
and full verification capture costs remain separate in `cold.json`; they are not compared as isolated
deployment cold starts because the frontend/backend and existing caches are shared. CUDA allocation
peaks include both variant pools and exclude allocations outside PyTorch's allocator, such as TensorRT
workspace; five-second `nvidia-smi` samples additionally include loading,
capture and untimed diagnostics. Neither is a standalone per-variant deployment memory estimate.

Reproduce inside a compute allocation with the frozen source and staged environment/engines:

```bash
python scripts/profile_expanded_greedy.py --model MODEL --data LOCAL_CORPUS \
  --out LOCAL_RESULTS --per-language 32 --repeat 7
python scripts/summarize_expanded_greedy.py outputs/runtime-2786368 \
  --out docs/runtime-expanded-greedy-2786368.json
```

The PBS driver is `scripts/profile_expanded_greedy.pbs`; snapshot and source metadata are in
`outputs/runtime-expanded-greedy-1b-submit-01`. Raw results are in`outputs/runtime-2786368/profile`,
including selected corpus keys/hashes, wall samples, cold metadata, full outputs and separate dispatch
counts. `outputs/runtime-2786368/audit-output-control.json` records the additional B1 audit comparison.

## 102M / H200

[Job2786438](runtime-expanded-greedy-2786438.json) uses the same240287b source,224 selected input
IDs and seven-repeat protocol. PBS F/exit0, publication exit0,14 conditions, full output agreement,
repeat stability and paired timing validation all pass. No OFF/ON draft-call counts differ.
All224 B1 candidate outputs also match the earlier full-corpus B1 candidate outputs. This remains
a combined public-switch measurement; individual optimizations' gains are not separated or added.

| Language | B1 RTFx change | B1 p50 ms OFF → ON | B1 p95 ms OFF → ON | Bulk RTFx change | Bulk positive paired repeats |
|---|---:|---:|---:|---:|---:|
| German |+16.91%|4.550 →3.991|16.256 →13.673|+8.05%|7/7|
| English |+16.21%|2.733 →2.365|15.085 →12.594|+8.74%|7/7|
| French |+16.44%|5.487 →4.797|15.132 →13.227|+7.63%|7/7|
| Japanese |+15.00%|3.429 →2.945|9.638 →8.141|+7.82%|7/7|
| Russian |+17.53%|5.052 →4.281|25.047 →20.943|+8.53%|7/7|
| Spanish |+16.04%|3.689 →3.099|19.512 →16.385|+8.20%|7/7|
| Chinese |+15.85%|3.491 →3.057|10.407 →8.749|+6.92%|7/7|

All seven paired repeat totals improve for all14 conditions, and all224 single-input medians improve.
Every condition's pooled p95 decreases. Bulk paired changes vary from+5.38% to+17.82%; all samples
are retained. Model load takes3.470seconds, excluding staging. The same cold/shared-pool-memory
limitations apply as for1B. Results describe this frozen runtime's OFF/ON switch on H200; they
do not establish gains for another GPU, translation, long-form, nonspeech or other batch sizes.

Raw data and the B1 audit control are in`outputs/runtime-2786438/`; the immutable submission is
`outputs/runtime-expanded-greedy-102m-submit-01`. Reproduction uses the same commands above with
the102M checkpoint and result directory.

## Numerical budget comparison (1B)

`--comparison draft_budget` compares the existing public budget256/128 on bulk B8 only (seven
language conditions), holding the public greedy optimizations ON in both. Separate session pools
and the same warm/alternating protocol are retained. It records the resolved budgets, permits and
retains N output differences, and still rejects unstable repeated outputs or backend failures.
The summarizer recomputes agreement from saved full outputs; its explicit N mode does not weaken
the E-mode equality check. Ten small sampler/aggregation fixtures pass, and re-summarizing both
completed E jobs preserves all28 conditions' statistics exactly.

[Job2786772](runtime-expanded-budget-2786772.json), frozen e755b49, completed all seven conditions
with PBS/publication exit0. Its budget256 reference matches all224 saved budget256/greedy-ON outputs
from2786368, with identical inputs/groups, runtime tree, checkpoint/engines and software versions
(`outputs/runtime-2786772/audit-output-control.json`). This is a within-job256/128 comparison,
not a speed comparison between jobs.

| Language | Pooled RTFx change | Request p50 ms 256 →128 | Request p95 ms 256 →128 | Positive pairs |
|---|---:|---:|---:|---:|
| German |+7.60%|119.438 →110.437|120.830 →111.452|7/7|
| English |+10.26%|126.176 →114.413|126.325 →115.289|7/7|
| French |+0.08%|135.619 →125.476|137.038 →171.661|6/7|
| Japanese |+13.68%|111.154 →99.196|129.449 →112.971|7/7|
| Russian |+10.74%|163.349 →146.975|163.889 →148.729|7/7|
| Spanish |+6.41%|132.683 →124.292|133.129 →125.064|7/7|
| Chinese |−3.59%|130.976 →116.908|132.107 →196.435|5/7|

Each request contains32 inputs, internally streamed at B8. All medians improve, but French and
Chinese have worse p95 and large candidate outliers (190.851ms and219.547ms); none are removed.
Seven repeats are insufficient to establish stable tail behavior or explain these outliers.
Four inputs change tokens/text/segments (two Japanese, one Russian, one Chinese); all224 timestamp
token sequences match. Russian word errors increase65→66/648 words, Chinese character errors
decrease148→147/1282 characters; the other language error totals are unchanged. Each variant's
repeated output is stable. The separate [13,232-input audit](runtime-large-regression-2786743.json)
retains63 changed inputs including nine changed timestamp sequences; the smaller timing sample's
timestamp agreement must not be generalized.

Model load is7.740seconds, with cold captures and audio loading excluded from warm request time.
Draft-decoder calls increase in six languages, staying equal in English; smaller verification windows
trade work per call for call count. Shared-pool allocator peaks and broad five-second utilization
samples retain the limitations above. No new default, kernel, precision or decoder route is adopted.
Reproduce with `--comparison draft_budget` and the frozen source in
`outputs/runtime-expanded-budget-1b-submit-01`; all raw timing/output/dispatch artifacts are under
`outputs/runtime-2786772/profile`.

## Frontend batch graph comparison / 102M

`--comparison frontend_batches` compares public frontend graph limits1/8 on bulk B8, with budget256
and greedy optimizations ON in both. The E equality gate and stable-output/no-new-capture checks
remain mandatory. Each language gets fresh separate32-entry caches so earlier language shapes
cannot consume its cache capacity; the large corpus audit separately measures cache saturation.
Saved cache shapes and resolved limits must demonstrate a batched graph actually captured.
Ten existing small selection/aggregation checks pass; reaggregation preserves all35 previously
measured E/N condition statistics. No default changes.

[Job2786935](runtime-expanded-frontend-2786935.json), frozen7ac9f15, completed PBS/publication exit0.
All224 outputs and draft-decoder-call counts match across variants; the reference also matches the
saved102M public-ON outputs from2786438 with the same inputs/groups/runtime tree/checkpoint/engines
and software. The separate full-corpus graph comparison passes13,232 outputs, including cache
saturation ([audit](runtime-large-regression-2786900.json)).

| Language | RTFx change | Request p50 ms limit1 →8 | Request p95 ms limit1 →8 | Positive pairs |
|---|---:|---:|---:|---:|
| German |+2.23%|60.852 →59.876|63.829 →60.810|7/7|
| English |+3.83%|43.406 →42.364|46.406 →42.846|7/7|
| French |+1.95%|84.297 →83.335|87.775 →83.769|7/7|
| Japanese |+3.72%|34.981 →34.033|37.504 →34.496|7/7|
| Russian |+2.45%|77.012 →75.776|80.102 →76.090|7/7|
| Spanish |+2.23%|58.657 →57.659|61.452 →58.759|7/7|
| Chinese |+4.15%|45.434 →44.285|49.495 →45.314|7/7|

Each request streams32 inputs at B8. Every paired repeat improves in all seven languages, with lower
request p50/p95 and unchanged outputs. These gains are relative to the current optimized runtime
with frontend limit1, not cumulative gains over the original repository. The candidate captures four
B8 shapes per language with zero capacity fallbacks; the reference runs these B8 features eagerly.
Captured input/output buffers occupy44.85–60.67MiB per language, excluding graph intermediates and
workspaces. Model load is3.757seconds; capture/audio/cold work is separate. Both variant session pools
are resident during timing, so recorded allocation peaks are not isolated deployment peaks.
The full-corpus saturated cache requires more buffers and falls back on unseen shapes; this fresh-cache
performance sample does not establish saturated-cache throughput or another checkpoint/GPU's gain.

Use `OWSM.from_pretrained(..., frontend_graph_max_batch=8)` to opt in. Reproduce the timing with
`--comparison frontend_batches` and `outputs/runtime-expanded-frontend-102m-submit-01`'s frozen
source. Raw timings/cold/cache/output/dispatch data and the previous-output control are retained in
`outputs/runtime-2786935/`. The default remains1 for the memory/performance tradeoff and mixed
earlier1B measurements; the larger1B frontend audit is separate.

## Prepared budget tail diagnosis

`--host-diagnostics` records GC overlap, process CPU time, faults and context switches through the
existing `HostLatency` observer; it does not change GC thresholds or subtract collection time.
`--candidate-first` reverses warmup and the initial alternating order. With `HOST_ROTATED=1`, the PBS
driver runs the diagnostic and rotated ordinary measurement in separate Python processes, saving
`host_profile` and `profile` separately. The standard warm summarizer rejects instrumented metadata;
eleven small selection/aggregation fixtures pass, including that guard. The three observer callback/
exception/overlap tests run on the compute node before the diagnostic. These changes are prepared
for the budget256/128 tail excursions; no cause or new performance conclusion is established yet.
