# 102M serial Ring program budget

Adopted after independent public verification2778349: H200132SM/102M uses64serial CTAs,
with concurrent12and cross-attention NS2unchanged. Plain and CTC-drafted B1 RTFx improves
3.7–6.9% across the two64-budget measurements, and110s improves7.1–8.1%. All changed-condition
paired totals improve in11/11repeats in each run. B8auto uses unchanged TensorRT and has no gain claim.
All sampled outputs/cache states match; English73/Japanese12, with plain/translation limited to12
individual inputs per language. Final API checks and limitations appear below. Historical pilots retain
all samples, including GC-associated excursions; their results are not multiplied together.

The subsequent [expanded plain timestamp audit2781184](runtime-large-regression-2781184.json) validates
13,232 inputs across seven languages against24 CTAs with identical checkpoint/engines and software.
All104 chunks pass source/order/hash/output checks:13,232 text/segment pairs and13,191 available full
token/timestamp pairs match exactly;41 long-form inputs have no joined token IDs. Both observed B1
sessions activate `plain_solo` and retain Ring routes without fallback; their settings record24/64 CTAs.
This includes common baseline diagnostic failures; it does not establish that either variant is free
of repetition or invalid timestamps. Large drafted/translation/87-chapter and bulk/refill checks remain
pending. No new latency or speed gain is inferred from this regression run.

Candidate31, E intent against the current Ring kernel, independent of wide-row N routing.
Actual graph trace2777712 places the single-row Ring decoder at75.3%of102M English translation
GPU kernel sums. These instrumented sums motivate a small same-engine ablation; they cannot
predict wall savings. The previous1B32/64budget study does not establish a102M schedule.

The original102M schedule used12programs with concurrent encoder work and24for serial single-row calls. The
initial experiment compared24against32and48serial programs on H200132SM. Concurrent12,
row routing, weights, fp16/fp32 arithmetic, cache layout, CTA tile size and cross-attention NS2
remain unchanged. `RingDecoder` already accepts `solo_programs`; the experiments tune this existing
parameter. Final adoption changes the guarded serial default, not the kernel arithmetic. Additional CTAs may increase barrier overhead, so a speedup is
not assumed. Every full token/time/text/segment output must match; any difference aborts.

Six GPU synthetic checks exercise W1/8/32, cross length64/374, per-bucket graph reuse, offsets
0/63/0, cache state, argmax and whole-call completed-stop behavior. The actual checkpoint has
18strict same-Ring comparisons across three budgets, three windows and two offsets, retaining
all logit/KV/cache/argmax equality and unchanged split counts. This is an equality check against
the currently selected decoder, not a fresh claim of fp32 equivalence.

Eleven alternating warmed repetitions cover English73/Japanese12. Plain ASR with/without
timestamps and translation use12individual requests; drafted timestamp ASR uses individual
requests and whole-list B8throughput. English additionally includes110s long-form (joined
text/segments, no joined raw token IDs). All shapes/verification calls are warmed, new warm
captures/fallbacks are errors, and actual dispatch/call counts are recorded separately.
All three pools remain resident and compiler/engine/frontend initialization is shared; cold
and allocator deltas are not standalone deployment comparisons.

```sh
python scripts/profile_ring_solo.py --model /local/owsm_v4_base_102M \
  --manifest /local/manifest.json --long-audio /local/long_110s.wav \
  --repeat 11 --out /local/results
```

Run only on compute nodes via the staged `scripts/profile_ring_solo.pbs`. Historical first results below retained default24 until the final verification.


Pilot2778037 (0110369) passed6GPU tests and18actual-checkpoint equality checks per language.
All11conditions have exactly identical outputs and decoder-call totals across24/32/48; long-form
compares joined text/segments. Exit0 and complete artifacts; Stageout_status1 is recorded separately.

| Condition | English32 /48 RTFx change | Japanese32 /48 RTFx change |
|---|---:|---:|
| Plain ASR B1 | +1.54% /+4.22% | +1.51% /+4.51% |
| Plain timestamp ASR B1 | +2.61% /+3.55% | +1.15% /+3.48% |
| Translation B1 | +3.05% /+4.05% | +2.66% /+4.88% |
| CTC-drafted timestamp B1 | +2.69% /+0.40% | +3.21% /+5.00% |
| CTC-drafted timestamp B8 | +0.30% /+0.03% | +0.59% /+0.84% |
| 110s long-form | +3.93% /+6.25% | not measured |

Every paired total improved in all11repeats for plainB1 and long-form, and their pooledp95
improved. B8is mixed/noisy and has no gain claim. English draftedB1 with48 has one paired total
-26.785%, while the other10are positive; its aggregate+0.397%is therefore much smaller than
its p95improvement. That latency excursion is retained with largest raw samples in the report;
its cause is unresolved and no sample is discarded. No warm capture/fallback/output drift was
observed. An independent repeat is required before any default change.

The first pilot records calls, bound backend names and drafted program histograms. A later
profiler revision additionally counts actual graph replays and records each plain bucket's
program/split configuration in a separate pass; this is not added to warm wall timing.
[Complete first results and outliers](runtime-ring-solo-2778037.json).

Independent repeat2778131 (1ac8b90) passed five graph-wrapper checks and repeated all18actual
checkpoint comparisons per language. All11conditions remain output/call exact. Actual replay
counts confirm B1solo Ring24/32/48with NS2; B8uses TensorRT throughout and is an unchanged control.

| Condition | English32 /48 RTFx change | Japanese32 /48 RTFx change |
|---|---:|---:|
| Plain ASR B1 | +1.58% /+4.36% | +1.39% /+4.40% |
| Plain timestamp ASR B1 | +2.31% /+3.29% | +1.22% /+3.20% |
| Translation B1 | +2.86% /+3.26% | +2.71% /+4.75% |
| CTC-drafted timestamp B1 | +3.37% /+0.74% | +3.53% /+6.28% |
| CTC-drafted timestamp B8 (unchanged control) | -11.22% /+3.58% | +1.04% /+1.19% |
| 110s long-form | +3.52% /+6.54% | not measured |

The English solo48draftedB1 excursion recurred at the exact same repeat2/input1272-141231-0011:
80.763ms versus its1.778ms median, after79.098ms versus1.848ms in the pilot. All samples are
retained. B8also has large excursions despite identical dispatch and graph/call counts.
This motivates separate GC/processCPU/context-switch diagnostics and an uninstrumented rotated
variant-order run; a GC cause is not established by repeat/input coincidence alone. At that point, the production
default stayed24 pending diagnosis. [Repeat and raw outliers](runtime-ring-solo-2778131.json).

`--host-diagnostics` records GC start/stop intervals, process CPU time, page faults and context
switches for each measured call in `host.json`. These instrumented measurements are separate from
ordinary wall results and do not alter GC policy. Overlap is reported, never subtracted to manufacture
corrected timings. `--rotated-order` changes the starting order to48/24/32; alternating reversal still
applies. `scripts/profile_ring_solo_host.pbs` runs these two diagnostics in separate fresh processes.

Host diagnosis2778234 (5efd58a) passed8CPU checks and18actual-checkpoint comparisons per process.
All outputs match the frozen2778131reference; decoder call totals and actual graph replay counts match.
In the instrumented pass, a reference24draftedB1 request took86.077ms, overlapping a generation2GC
collection for83.204ms. The slow event therefore also occurs on the existing24budget. The original
79/81ms events were not instrumented; GC is the supported explanation, not a direct measurement
of those earlier events. No runtime GC policy changes, samples removed or GC-subtracted times.

The separate uninstrumented rotated48/24/32order pass retains all samples: English RTFx changes
for48vs24are +4.10%plainASR, +4.00%plain timestamp, +2.68%translation, +3.60%CTC timestampB1,
-0.61%unchangedB8control and +6.49%110s. Instrumented times are not used as adoption performance.
The observed isolated full-heap GC pause motivates retaining raw extrema alongside p50/p95 and
rotating order; neither callback instrumentation nor CPU process time measures GPU kernel duration.
[Diagnosis, rotated comparison and evidence hashes](runtime-ring-solo-host-2778234.json).

Before selecting the default, the same narrow E experiment extends to64/96serial CTAs, with48as
an intermediate control and24as reference (`--solo-programs 24 48 64 96`). This changes no arithmetic,
NS2or concurrent12budget. Six GPU graph/state tests now also check64/96; each corpus requires24actual
checkpoint cases and every output/call to agree. These budgets are unmeasured until their artifacts
are published. The production default was still24 at that point.

Extended pilot2778290 used four resident pools (its environment prose incorrectly says three;
the saved budget dictionary, samples and memory accounting include all four). This run (f7bf820) passed5CPU/6GPU checks and24actual-checkpoint comparisons per language.
All11conditions match frozen2778131outputs; actual graph replay counts and decoder calls match too.
No sample is dropped. Every64/96paired total improved in all11repeats for each changed condition.

| Condition | English48 /64 /96 RTFx change | Japanese48 /64 /96 RTFx change |
|---|---:|---:|
| Plain ASR B1 | +2.83% /+4.23% /+4.57% | +3.01% /+4.57% /+4.69% |
| Plain timestamp ASR B1 | +3.52% /+3.68% /+4.09% | +3.98% /+4.34% /+2.91% |
| Translation B1 | +4.45% /+4.44% /+5.33% | +4.39% /+5.21% /+4.12% |
| CTC-drafted timestamp B1 | +1.01% /+5.37% /+5.51% | +5.72% /+6.90% /+7.08% |
| CTC-drafted timestamp B8 (unchanged control) | +0.35% /+0.34% /+0.24% | -0.02% /+0.12% /+0.74% |
| 110s long-form | +6.75% /+8.15% /+7.88% | not measured |

64is the adoption candidate: it improves all changed conditions over24, while96gives back some
Japanese timestamp/translation benefit and does not improve110s over64. This is a measured compromise,
not proof that64is globally optimal. H200132SM/102M default64is prepared with explicit24rollback,
unchanged concurrent12/NS2and unchanged row dispatch. Independent timing/public-API verification
subsequently passed as recorded below. [Pilot results](runtime-ring-solo-more-2778290.json).

Public verification2778349 (9fa5996) passed20CPU/GPU tests,18actual-checkpoint comparisons per language,
and15API cases per language. `actual_public_default=64`/`public_variant=solo64` confirm the actual auto
backend was measured, against explicit24rollback and48control. All corpus outputs match the frozen
2778131reference, and decoder calls plus actual graph replay counts match. The15API cases include
plain/drafted/no-timestamp ASR, translation, B8, beam B1/B3, silence/noise,30s, long prompt, encoder
lookahead,110s and reuse. No fallback/capture drift or output difference was observed.

| Condition | English public64 RTFx change | Japanese public64 RTFx change |
|---|---:|---:|
| Plain ASR B1 | +5.09% | +5.32% |
| Plain timestamp ASR B1 | +4.31% | +5.56% |
| Translation B1 | +3.75% | +5.80% |
| CTC-drafted timestamp B1 | +4.38% | +5.89% |
| CTC-drafted timestamp B8 (unchanged control) | +0.14% | -0.28% |
| 110s long-form | +7.06% | not measured |

Every changed condition improves all11paired totals and pooledp95. B8remains an unchanged/noisy
control; Japanese B8p95increases3.16% in this repeat. The48control retains a large English draftedB1
excursion and aggregate-0.46%; no sample is discarded. These are loaded-input warmed measurements,
not isolated deployment VRAM/cold-load results. Raw distributions, capture/memory data, quality and
source hashes are in the [public report](runtime-ring-solo-api-2778349.json).

The default applies only to H200132SM, D384/F1536/H6/L6and four warps. Other model/GPU/warp schedules
retain prior defaults. An explicit `RingDecoder(..., solo_programs=24)` overrides it; the existing public
`decoder_kernel="ring:s24"` selects the prior serial budget. Concurrent12, NS2, multi-row dispatch,
encoder lookahead and engine precision are unchanged. This is same-Ring E validation, not a new
fp32-equivalence result. No human timestamp boundaries or reference translations are available;
long-form compares joined text/segments. No gain is claimed for1B, other GPUs or beam multi-row work.

```sh
python scripts/profile_ring_solo.py --model /local/owsm_v4_base_102M \
  --manifest /local/manifest.json --long-audio /local/long_110s.wav \
  --solo-programs 24 48 64 --rotated-order --baseline-results /local/frozen_2778131 \
  --repeat 11 --out /local/results
```

The staged public job is `scripts/profile_ring_solo_api.pbs` and additional API checks use
`scripts/validate_ring_solo_api.py --expected-programs 64` with the same model/manifest/long audio.
