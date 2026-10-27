# 102M serial Ring program budget (prepared, unmeasured)

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

Run only on compute nodes via the staged `scripts/profile_ring_solo.pbs`. No performance or
GPU equality result exists yet, and default24 remains unchanged.
