# Candidate16: skip unchanged-parent KV copies (measured opt-in)

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
The default remains `prefix`; GPU measurements below do not justify broad adoption.

Twelve small CPU Triton-interpreter cases passed (mixed identity/duplicate/cyclic parents,
written lengths0/1/7/8, full/prefix behavior). The GPU suite extends all existing fp16/fp32 cases,
tile boundaries/workspace reuse, and real-decoder graph bucket/repeated-request tests. These
extended GPU cases passed in job2773900.

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


## First measurement:2773900; prefix default retained

Job2773900 (frozend94138a) passed281 focused tests plus31 naive regressions. All8 model/corpus/B
conditions, extra outputs and decoder-call counts match exactly; every diagnostic reorder count
matches its selection count. Both models also passed the adopted finished-stop public API check.

| Model/corpus/B | RTFx change | Paired group median | Unchanged rows | Prefix-weighted unchanged |
|---|---:|---:|---:|---:|
|1B English B1|+0.472%|+0.661%|72.10%|79.37%|
|1B English B3|+0.902%|+0.505%|55.23%|50.14%|
|1B Japanese B1|+0.899%|+0.680%|78.47%|85.41%|
|1B Japanese B3|+0.979%|+0.698%|61.47%|56.25%|
|102M English B1|+0.292%|+0.106%|71.13%|76.58%|
|102M English B3|+0.128%|+0.058%|54.16%|48.21%|
|102M Japanese B1|+0.028%|-0.059%|67.66%|73.68%|
|102M Japanese B3|+0.480%|+0.303%|52.89%|48.47%|

All11 repeat-total comparisons improved for1B B3 (English+0.469..+1.482%, Japanese+0.422..+1.865%).
However1B English B1 had a-2.626% repeat despite its positive paired median;102M changes are small
and mixed relative to per-group CV. No broad default change is justified yet. Independent1B B3
replication is the next bounded check. Large identity fractions do not imply equal speedups: tail
zeroing, decoder compute and the extra branch remain. No extra persistent cache/graph is required,
but allocator peaks were measured with both variants in the same process, not isolated deployment.

[Timing, counter, output and API evidence](beam-identity-results-2773900.json).
Raw logs and frozen source metadata: `outputs/runtime-2773900/`. No reference timestamp annotation;
110s extras compare joined public segments, not unavailable per-window token sequences.


## Independent1B B3 replication:2774007; default remains prefix

Fifteen alternating same-engine pairs matched full outputs and decoder calls. English RTFx was
+0.673% (paired-group median+0.517%, repeat totals-0.207..+1.011%); Japanese was+0.710%
(paired-group median+0.807%, repeat totals-5.352..+1.906%). Thus the first all-positive repeat
result did not reproduce across every pair. The effect is small and102M was already mixed.
Keep the existing prefix default and retain identity skipping as a low-level opt-in; no automatic
model-specific route is introduced for this small gain. A separate tail-initialization experiment
now targets the still-repeated tail stores without combining gains. [Replication data](beam-identity-replication-2774007.json).
