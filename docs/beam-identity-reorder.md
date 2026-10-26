# Candidate16: skip unchanged-parent KV copies (unverified experiment)

The existing H200/1B B3 trace in `outputs/runtime-2773582/lookahead_overlap.json` assigns11.56ms
of271.61ms of decoder-stream kernel sums to `_reorder_kv` (about4.3%). These are kernel sums,
not wall time or a prediction of the gain. Most time remains in decoder GEMMs/attention.

`BeamSession(kv_reorder="identity")` adds a comparison to the existing two-launch prefix reorder.
When a destination row selects itself, the gather launch leaves its scratch untouched and the
copy-back launch leaves its written prefix untouched. Copy-back still zeros every unwritten slot,
including stale NaNs; full-mode kernel tests also check no-op identity rows. Other destinations
may read that same identity row safely: all parent reads precede the separate copy-back launch.
Duplicate parents and permutation cycles continue through the original scratch path. No host
read, new graph, persistent buffer, cache-layout change or decoder arithmetic change is added.
The default remains `prefix`; this is not yet adopted or GPU timed.

Twelve small CPU Triton-interpreter cases passed (mixed identity/duplicate/cyclic parents,
written lengths0/1/7/8, full/prefix behavior). The GPU suite extends all existing fp16/fp32 cases,
tile boundaries/workspace reuse, and real-decoder graph bucket/repeated-request tests. These
extended GPU cases are prepared, not yet executed.

`profile_beam.py --kv-reorder identity --kv-identity-stats` enables separate diagnostic counters
for unchanged rows and prefix-position-weighted unchanged rows. Those counters exist only in a
profiler subclass; warmed wall sessions contain no counter work. Counts must equal the existing
selection count, including the post-prefill selection. They cover all copied slots, including
inactive ones; unwritten-tail zeroing is not included in the weighted prefix fraction. Full output
and step parity plus independent alternating timings are required before any adoption.


Prepared runner: `scripts/profile_kv_identity.pbs`. It first runs the expanded KV/state/finished-stop
suite and existing naive regressions, then validates the adopted finished-stop public factory on
1B/102M with `validate_greedy_api.py --finished-stop`. The independent KV ablation uses frozen
pre-identity runtime64cfddf as baseline, same checkpoint/engine, beam5/timestamps, B1/B3, English73/
Japanese12 and11 alternating repeats on both models. All shapes are warmed; wall and diagnostic
sessions are separate. Full outputs, decoder-call counts and extra cases must match. Low-level
identity-counter reset/selection-count consistency also passed two tiny CPU synthetic decodes.
