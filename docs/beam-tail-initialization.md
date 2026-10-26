# Initialize unused beam KV once (unverified experiment)

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
bucket boundaries/capacity limits, and compacted-tail growth. They have not run yet. Same-engine
alternating latency and complete token/time/segment/step comparisons are required before adoption.

Prepared runner: `scripts/profile_kv_tail_init.pbs`, beam5/timestamps B1/B3,1B/102M,
English73/Japanese12, eleven alternating pairs against frozen prefix-default runtime d16a15f.
The initial ablation changes only tail initialization; identity skipping stays disabled. Separate
GPU/graph/NaN tests precede model work. Full outputs, decoder counts and long/prompt/noise extras
must match. One1B B3 eager trace is kept separate from warmed graph wall timings; it cannot
attribute graph node times or establish isolated baseline/candidate kernel speedups.
