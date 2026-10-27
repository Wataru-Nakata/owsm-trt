# Single-row Ring cross-attention split experiment (unmeasured N)

The current serial64 profile2778412 attributes17.9–20.9% of instrumented1B stage time and
20.6–23.1%of102M to cross-attention. These are diagnostic fractions, not predicted wall savings.
Candidate25compares existing2slices against4/8at fixed CTA budgets:1B32/64and102M12/64.
Single-row attention splits are the only arithmetic grouping change; all row routing, precision,
weights, tile sizes, barriers, encoder buffers and beam behavior remain as before.

`RingDecoder(single_cross_splits=4|8)` is an experimental constructor-only override, with default
None retaining the existing formula. It applies to allR1calls within that mode, both concurrent
and solo, so encoder timing cannot change the arithmetic grouping within a mode. R>1retains the
prior formula. This is N, not exact relative toNS2; token/time/quality differences must be retained.
No public decoder-kernel string or default behavior is added by this experiment.

Thirteen synthetic GPU checks cover width384/1024, W1/8/32, T64/374, offsets0/63/0, graphs,
cache bounds, repeated states and EOS skip. Within each split setting, concurrent/solo outputs
must match exactly. DefaultNone and explicit2must match exactly; changed settings use the existing
fp32/logit/argmax/KV criteria. A multi-row check verifies the override is ignored there. Four invalid
constructor checks and nine serial-default guards accompany them (26focused checks total).

Each actual checkpoint/language then compares7states (randomT64/fullT with W1/8/32, plus an actual
first-prefill state),3settings and2schedules. Existing fp32 gates remain unchanged; nonfinite state,
failed gates, fallback or nondeterministic repeats abort before accepting timing. Full differences
and per-layer KV errors are saved. No failure is silently accepted as a performance sample.

`scripts/profile_ring_cross_splits.pbs` stages all work locally and benchmarks the same engines
for11alternating repeats: English73/Japanese12 direct drafted ASR with/without timestamps,
plain/translation12inputs, B1streaming, B8control, and English110s. Raw wall samples, cold/JIT,
pooled VRAM (three resident pools), dispatch/split counts, all tokens/segments and WER/CER are
retained. Silence/noise,30s and long-prompt outputs are saved separately. There are no human
boundary/reference-translation annotations; long-form exposes joined text/segments only.

This prototype is not adopted or measured yet. Reproduction (allocated compute node only):

```sh
python scripts/profile_ring_cross_splits.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --long-audio /local/long_110s.wav \
  --repeat 11 --out /local/results
```
