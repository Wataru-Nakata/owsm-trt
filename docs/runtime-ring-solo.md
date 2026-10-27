# 102M serial Ring program budget (first measurement; default24 retained)

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
