# 102M serial Ring program budget (two measurements; default24 retained)

Candidate31, E intent against the current Ring kernel, independent of wide-row N routing.
Actual graph trace2777712 places the single-row Ring decoder at75.3%of102M English translation
GPU kernel sums. These instrumented sums motivate a small same-engine ablation; they cannot
predict wall savings. The previous1B32/64budget study does not establish a102M schedule.

102M uses12programs with concurrent encoder work and24for serial single-row calls. The
experiment compares existing24against32and48serial programs on H200132SM. Concurrent12,
row routing, weights, fp16/fp32 arithmetic, cache layout, CTA tile size and cross-attention NS2
remain unchanged. `RingDecoder` already accepts `solo_programs`; no new kernel or production
route/default is introduced. Additional CTAs may increase barrier overhead, so a speedup is
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

Run only on compute nodes via the staged `scripts/profile_ring_solo.pbs`. The first results are below; default24 remains unchanged.


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
variant-order run; a GC cause is not established by repeat/input coincidence alone. The production
default stays24 pending diagnosis. [Repeat and raw outliers](runtime-ring-solo-2778131.json).

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
are published. The production default is still24.

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
is pending before documenting it as adopted. [Pilot results](runtime-ring-solo-more-2778290.json).
