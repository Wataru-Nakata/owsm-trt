# Existing ring execution settings

Job2770946's instrumented110s trace attributes47.57ms to67 ring decoder calls. After the greedy
selection improvements, the existing ring program budgets are the next measured component to tune.
This experiment changes settings of the existing kernel; it does not introduce a new megakernel.

Compare the1B/H200 default concurrent/solo32/64 against16/64,48/64,32/96. Concurrent settings affect
encoder overlap as well as decoder time, so both single requests and whole-manifest B1 streaming are
measured. The sole change in32/96 is the solo budget. Every persistent CTA still participates in the
existing barriers; GPU SM count must be at least96. Ring tile/warp/ring-capacity parameters are unchanged.

This is an N experiment. The current kernel derives cross-attention split count from the concurrent
budget:1/2/3 slices at16/32/48 programs for1B. Changing that count can alter fp16 outputs. Solo96 retains
the baseline split count2. Tiny-model tests compare logits/new-KV/cache and existing solo bitwise
checks; each real-checkpoint setting is also checked against the existing fp32 reference before
benchmarking. Any kernel failure/fallback or repeated-output nondeterminism aborts; N differences
are retained with WER/CER and time-token/segment comparisons rather than hidden.

All variants share checkpoint/engine/weights and have isolated captured sessions. Selection mode is
fixed across variants. Alternating7-repeat wall samples exclude warmup and instrumentation; a separate
pass counts actual cache/window/solo/programs/cross-split dispatch. Both session pools and all variant
pools are additional benchmark-resident memory, not deployment VRAM. Disk reads are an aggregate
outside request latency. Long-form has no public joined token IDs, so its comparison is text/segments.

`scripts/profile_ring_programs.pbs` stages the environment, source, model, cached engines and English73/
Japanese12/110s audio into node-local storage. It uses the established serial1GPU<=1h policy, with fresh
queue/group/duplicate/account-cap192 checks at submission. The measured outcome is below; no budget change was adopted.

## Measured outcome: retain32/64

Job2771706 (source2f0d4f8,H200) passed6 kernel/KV tests and every real-checkpoint check. English4/
Japanese3 runtime cases had identical sampled tokens/text/time segments for all settings. Baseline
32/64 already used fused selection. Seven alternating steady-state repeats gave these RTFx changes:

| Case |16/64|48/64|32/96|
|---|---:|---:|---:|
| English single greedy |-2.21%|+0.75%|-0.69%|
| English single timestamps |-2.64%|+1.29%|-0.79%|
| English stream B1 |-6.47%|-0.79%|+0.01%|
| English110s |-5.21%|+1.59%|-1.45%|
| Japanese single greedy |-2.62%|+1.41%|-0.54%|
| Japanese single timestamps |-3.68%|+1.59%|-0.82%|
| Japanese stream B1 |-11.54%|-0.98%|-0.99%|

48/64 trades a small single-request gain for slower streaming;16/64 and32/96 do not improve the
measured workload broadly. The default stays32/64. All settings already existed; no kernel addition
is claimed. The N classification remains despite exact outputs on these samples. Actual decoder
call counts, solo/concurrent selection, cross-split count and TensorRT fallbacks are in the per-case
`actual_dispatch_<variant>.json`; their histogram sizes are not total decoder call counts.

Artifacts: `outputs/runtime-2771706/analysis.json`, per-language shape_checks/environment/summary,
per-case wall/tokens/cold/dispatch, tests.log and edge_recipe.json. The separate edge experiment
checks exact fused-selection outputs on silence and noise; it is not an annotated quality score.
The job exited0 in6m50s and its monitor stopped. Subsequent work continues on the same authorized
serial1GPU settings with fresh capacity/duplicate checks.

## Weight-prefetch distance (prepared)

The existing `ahead` option prefetches weights for later GEMM stages into L2. An isolated sweep
(`profile_ring_programs.py --sweep prefetch`) compares distances0/1/2 at fixed32/64 programs,
warp count, cache layout, attention split count and fp16 arithmetic. This is E intent: all sampled
outputs must be exact; any difference aborts. It is not a newly implemented decoder kernel.
Synthetic tests compare logits, KV writes, predictions and repeated graph replay at windows1/8/32,
solo/concurrent shapes and two offsets. Actual end-to-end gain is unmeasured; default stays0.
Run only if the post-selection profile still identifies the ring decoder as a material component.

## Prefetch measured outcome: retain0

Job2771944 passed all6 exact-logit/KV/graph tests and all real1B concurrentR1W8/soloR1W32 checks.
English4/Japanese3 runtime cases matched all sampled outputs (long-form text/segments). Seven
alternating repeats, selection/short-primer optimization fixed across settings:

| Case | ahead1 | ahead2 |
|---|---:|---:|
| English single plain | +0.34% | +0.23% |
| English single timestamp | +0.71% | +0.59% |
| English B1 streaming | -1.34% | -2.62% |
| English110s | +0.25% | -0.47% |
| Japanese single plain | +0.39% | -0.21% |
| Japanese single timestamp | +0.74% | +0.40% |
| Japanese B1 streaming | -0.26% | -0.98% |

The small single-request changes do not justify the streaming regressions. Existing default ahead0
is retained. This is tuning an already implemented parameter, not a new kernel. Results, actual dispatch,
quality and cold data: `outputs/runtime-2771944/analysis.json` and per-language directories.
