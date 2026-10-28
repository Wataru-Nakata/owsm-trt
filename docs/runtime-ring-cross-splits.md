# Single-row Ring cross-attention splits (opt-in N)

The current serial64 profile2778412 attributes17.9–20.9% of instrumented1B stage time and
20.6–23.1%of102M to cross-attention. These are diagnostic fractions, not predicted wall savings.
Candidate25compares existing2slices against4/8at fixed CTA budgets:1B32/64and102M12/64.
Single-row attention splits are the only arithmetic grouping change; all row routing, precision,
weights, tile sizes, barriers, encoder buffers and beam behavior remain as before.

`RingDecoder(single_cross_splits=4|8)` is an experimental constructor-only override, with default
None retaining the existing formula. It applies to allR1calls within that mode, both concurrent
and solo, so encoder timing cannot change the arithmetic grouping within a mode. R>1retains the
prior formula. This is N, not exact relative toNS2; token/time/quality differences must be retained.
The original prototype adds no default behavior. The guarded public `ring:split` option now has
the102M drafted-timestamp B1 large-corpus validation below; broader public timing/shape coverage
remains incomplete.

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

Job2778533 (source254a2bb, H200132SM, Torch2.14.1/cu130, Triton3.8, TRT11.4.0.106) completed
with exit0,26checks passed and all168 actual-model numerical combinations passing unchanged gates.
Stageout_status1 is recorded separately; all expected artifacts were published. The full same-engine
results, raw output differences, hashes and environment are in
[runtime-ring-cross-splits-2778533.json](runtime-ring-cross-splits-2778533.json); raw samples are in
`outputs/runtime-2778533`. RTFx changes below retain all11alternating repeats:

| Model / language | Plain | Plain TS | Translation | Drafted | Drafted TS | Stream B1 | B8 control | 110s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B / English, NS4 | +1.42% | +2.20% | +2.09% | +0.69% | +1.08% | -2.68% | +0.48% | +1.77% |
| 1B / Japanese, NS4 | +1.61% | +2.23% | +1.56% | +1.30% | +1.54% | -2.69% | -0.44% | — |
| 102M / English, NS8 | +4.53% | +4.82% | +4.79% | +1.86% | +2.91% | -6.70% | +0.95% | +3.43% |
| 102M / Japanese, NS8 | +3.59% | +4.52% | +4.67% | +3.36% | +4.65% | -0.22% | +0.90% | — |

All changed direct-request cases above have11/11 positive paired totals and lower pooled p95.
NS4/1B agrees exactly on all corpus and extra outputs. NS8/102M changes an English plain-TS
boundary from19.22 to19.24s (1/12inputs); WER remains33/303. Its Japanese synthetic repeated30s
input changes a start boundary from12.20 to12.44s, with unchanged text. Other selected-mode
corpus/extra outputs and decoder call counts agree; WER/CER do not change. These boundary
changes are baseline differences, not annotation-based accuracy. NS8/1B is rejected: direct
cases regress3.0–6.6%, streaming regresses8.5–10.9%, and2Japanese translation outputs differ.
NS4/102M is slower thanNS8 on the direct cases (1.4–3.4% gains vs baseline).

B1 streaming regressions prevent changing defaults. The public opt-in under validation selects
NS4/1B orNS8/102M on H200132SM only, consistently for allR1calls; unsupported hardware/models
retain auto behavior, including numerical setup checks. B8 uses the unchanged row path, so its
small fluctuations are a control, not an optimization gain. Cold/JIT and VRAM include resident
comparison pools; no isolated deployment-memory or cold-start gain is claimed.

Reproduction (allocated compute node only):

```sh
python scripts/profile_ring_cross_splits.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --long-audio /local/long_110s.wav \
  --repeat 11 --out /local/results
```

The public repeat2778800 was stopped for the account192-GPU cap after another allocation surge.
PBS exit271 and incomplete published artifacts prevent accepting it as independent validation.
Large-corpus regression now takes priority; see [expanded evaluation](runtime-large-regression.md).

The subsequent public102M B1 drafted-timestamp audit2786533 completes13,232 inputs at frozen3976d7f,
with104 verified chunks and PBS/publication exit0. NS2 reference outputs exactly match the earlier
default B1 audit. NS8 differs on41 inputs (36 texts,3 timestamp sequences); speech edit deltas are
Japanese0, English−2, German−1, French0, Russian+1, Spanish−1 and Chinese+2. One comparable boundary
moves20ms, while37 incompatible text/segment outputs are excluded from boundary statistics.
All changed outputs and denominators are [retained](runtime-large-regression-2786533.json).
This verifies the specified public path's larger numerical/quality scope, not independent public
timing, other shapes or1B. The streaming regression still prevents default adoption.
