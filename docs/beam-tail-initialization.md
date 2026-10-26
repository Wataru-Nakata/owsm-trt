# Initialize unused beam KV once (measured opt-in)

The existing 1B B3 profile assigns 11.56ms of 271.61ms decoder-stream kernel sums to KV reorder
(job2773582). Separate measured decode positions/bucket capacities from job2773900 imply that
54–70% of reordered slots are unwritten across the eight model/corpus/B conditions. This is a
slot count, not measured HBM traffic or an expected latency improvement. Identity skipping still
clears these tails every step, so it does not eliminate that part of reorder.

`BeamSession(kv_clear_once=True)` / `profile_beam.py --kv-clear-once` is an opt-in, defaultFalse
experiment with prefix or identity reorder. Before every prefill, it zeros the selected cache
bucket, including the p+1 bucket when a prompt exactly fills the preceding bucket. At each grow,
it copies the old finite cache and zeros only the newly added tail. Reorder stores only the
written prefix; CTAs wholly beyond pos return without reading parents/scratch. Normal identity
skipping can be selected independently. There is no additional per-step host read or allocation.

The finite-zero invariant matters because masked attention may still multiply zero by V. Clearing
before prefill also removes stale NaNs left by previous requests/capture trials. Newly initialized
slots stay zero until written; duplicate parents cannot move unwritten values into the prefix.
The public path and low-level defaults remain unchanged, and CPU fallback disables this option.

Three tiny CPU initialization/validation tests and six bounded Triton-interpreter cases passed.
GPU tests are prepared for duplicate/identity/cyclic parent mappings, tile boundaries, reused
workspace, real-decoder graph/eager parity, poisoned caches before repeated requests, exact prompt
bucket boundaries/capacity limits, and compacted-tail growth. They passed in job2774062. Same-engine
alternating latency and complete token/time/segment/step comparisons are required before adoption.

Prepared runner: `scripts/profile_kv_tail_init.pbs`, beam5/timestamps B1/B3,1B/102M,
English73/Japanese12, eleven alternating pairs against frozen prefix-default runtime d16a15f.
The initial ablation changes only tail initialization; identity skipping stays disabled. Separate
GPU/graph/NaN tests precede model work. Full outputs, decoder counts and long/prompt/noise extras
must match. One1B B3 eager trace is kept separate from warmed graph wall timings; it cannot
attribute graph node times or establish isolated baseline/candidate kernel speedups.


## Measurement2774062: default remains OFF

Frozen27cbf12 passed316 focused tests and31 naive regressions. All8 model/corpus/B conditions
match full tokens/timestamps/segments/text, decoder-call counts and extra cases. Actual cold
configuration confirms prefix reorder in both modes and clear-once only in the candidate.

| Model/corpus | B1 RTFx change | B3 RTFx change |
|---|---:|---:|
|1B English|+0.627%|+0.610%|
|1B Japanese|+0.725%|+0.606%|
|102M English|-0.191%|+0.705%|
|102M Japanese|+0.474%|+0.597%|

Each comparison uses11 alternating pairs. All repeats improved for1B English B1/B3,1B Japanese
B3 and102M English B3. Other conditions have negative repeats;102M English B1 has per-input CV
2.90–3.27%, with repeat totals-2.08..+0.86%. The gains are small and not uniformly robust, so the
public/default path is unchanged. No persistent cache/workspace was added; shared-process memory
peaks do not isolate each variant. Identity skipping is OFF in both modes, so gains are not added.

The separate1B B3 eager trace (44 decoder calls, diagnostics enabled) contains55.93ms of GPU
kernel sums: decoder48.43ms (86.60%), selection4.16ms (7.43%), reorder1.44ms (2.57%). These are
not graph node times or wall time, and there is no matching baseline trace to claim isolated
kernel speedups. The next priority is decoder dispatch; the existing102M Ring row route will be
measured independently as N. ASR WER/CER is unchanged; no reference translations/times are available.

[Full timing/quality/environment evidence](beam-tail-init-results-2774062.json), raw artifacts
`outputs/runtime-2774062/`. The unverified identity+clear-once combination is not a measured gain.
