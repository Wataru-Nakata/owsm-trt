# Ring weight-buffer capacity: isolated scheduling experiment

Prior H200/1B Nsight observations recorded memory waits, low eligible warps and about153–172KiB
shared memory per CTA including the128KiB producer ring. Registers also limited occupancy to one
block per SM; reducing this buffer alone does not establish improved occupancy or speed. Existing
prefetch-distance/attention/chunk sweeps failed to improve requests broadly and remain rejected.

The next experiment changes only the existing `ring_kb` option128→64/32. Programs, warps, attention
splits, tiles, arithmetic and current plain solo dispatch remain fixed. This is E intent: each candidate
must preserve full outputs. Runtime defaults are unchanged; this is parameter evaluation, not a new
kernel implementation. Smaller rings trade less producer lookahead for less shared storage; actual
performance may regress. The completed measurement below rejects both smaller settings.

`scripts/profile_ring_buffer.pbs` runs six GPU graph/cache/logit/argmax tests at W1/8/32, both solo
and concurrent budgets, offsets7/63/7, repeated ring wraparound and scalar EOS/restart. It then runs
`profile_ring_programs.py --sweep ring_buffer --split-choose --fuse-accept` with1B/102M on English73
and Japanese12,11 alternating repeats. Existing fp32 sanity checks cover real weights at W8 and
solo W32; full token/time/text/segment agreement and no fallback are required in every timed case.
Plain/translation cases use the profiler's existing12-input subset, while drafted direct/streaming
use the full manifests. English also includes110s long-form (segments/text; no joined token IDs).

Warm wall measurements exclude setup/capture and diagnostic dispatch counts. This combines
separate per-request and whole-list conditions; their percentiles must not be conflated. All variant
pools are resident, so peak allocation is not standalone deployment VRAM. Shared-memory bytes
refer to the weight ring alone, not total kernel resource use. Keep the128KiB default unless the
same-engine output tests and repeatable request gains support a narrower validated choice.


## Measured2776387: keep128KiB

Frozen6951df5, H200, exit0, six exact graph/cache/logit/argmax/EOS tests passed. All26 workload
cases across both models/corpora had exact full outputs and matching decoder-call totals for
both alternatives (52 comparisons). All real-checkpoint fp32 sanity checks passed; no fallback.

|Model|64KiB RTFx change range|32KiB RTFx change range|
|---|---:|---:|
|1B|−10.27% to−4.13%|−37.22% to−18.87%|
|102M|−0.79% to+0.10%|−12.63% to−4.51%|

Every1B paired corpus total worsened for both alternatives.102M64KiB is mostly slightly worse,
with occasional tiny/noisy positives; it supplies no repeatable speed gain. Smaller shared storage
alone is not a latency improvement. Keep128KiB, the existing producer schedule and runtime routing.
No new runtime path or API flag is needed. The pre-existing low-level option remains available.

[Paired measurements, environments, quality and cold summaries](runtime-ring-buffer-2776387.json);
raw complete wall/token/dispatch/cold tables in `outputs/runtime-2776387/`. Reproduce with
`scripts/profile_ring_buffer.pbs` or `profile_ring_programs.py --sweep ring_buffer`. Shared weight-ring
capacity is per CTA; it is not process VRAM, and these measurements do not establish occupancy
changes.11 repeats alternate three variants; request/whole-list/long-form scopes remain separate.
