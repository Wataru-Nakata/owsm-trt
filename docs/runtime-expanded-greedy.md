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
peaks include both variant pools; five-second `nvidia-smi` samples additionally include loading,
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

## Prepared numerical budget comparison

`--comparison draft_budget` compares the existing public budget256/128 on bulk B8 only (seven
language conditions), holding the public greedy optimizations ON in both. Separate session pools
and the same warm/alternating protocol are retained. It records the resolved budgets, permits and
retains N output differences, and still rejects unstable repeated outputs or backend failures.
The summarizer recomputes agreement from saved full outputs; its explicit N mode does not weaken
the E-mode equality check. Ten small sampler/aggregation fixtures pass, and re-summarizing both
completed E jobs preserves all28 conditions' statistics exactly. GPU budget timing is pending;
this adds no new default, kernel, precision or decoder route.
