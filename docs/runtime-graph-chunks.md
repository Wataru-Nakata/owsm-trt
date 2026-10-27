# Four-step graph ablation (profiler only; unmeasured)

Candidate28, E intent. The completed actual-graph diagnostic2777712 confirms decoder remains
substantial, but CUPTI launch API sums include overhead/waits and cannot predict savings. This
small ablation tests whether grouping the already-permitted four steps saves measurable warm
wall time without changing any kernels, shapes, stopping checks or arithmetic. No production
session/API/default is changed; the helper lives under `scripts/`.

For CUDA sessions with host-check interval4 and no diagnostics/tail compaction, `prepare()` captures
four consecutive existing `_body(c)` calls into an additional graph per sufficiently large cache
bucket. It runs cold before prefill; two warmups plus capture must fit the bucket. Original graphs
remain alive and handle shorter remainders, bucket edges and small test buckets. The loop still
checks done at exactly the previous positions. It does not cross a cache bucket or generation
limit, suppress EOS/history updates, change global top-K, or extend an interval1 beam request.
Default B1 beam is an unchanged control. CTC-drafted decoding and compact-tail modes are excluded.

Eight synthetic CPU checks cover scheduling and cycle-free session lifetime.48 compute-node state cases compare one/four-step graphs
for greedy/beam5, B1/B3, timestamp/notimestamp, partial/early/no natural EOS, fused/reference selection,
prompt lengths3/7/14/29, limits1/5/capacity, bucket edges and request reuse. Comparisons cover returned
outputs, stop/score/time state, valid sequence prefixes and written KV; uninitialized future slots
are not required to equal. Five correlation/restoration tests cover the diagnostic wrapper.

The benchmark warms separate reference/candidate session pools on the same engines/checkpoint and
uses11 alternating repeats. H2001B/102M, English73/Japanese12: B1 beam and translation use the first12
individual requests; B3 timestamp beam and B8 translation measure whole-corpus fixed batches,
including partial final batches. Full outputs/call totals must match, with no warm captures or
fallback. Extra graph counts, cold time and allocation deltas are recorded; both pools remain
resident, so peak memory is not standalone deployment memory. New graph/capture cost requires a
repeatable gain before adoption.

A separate diagnostic uses `perf_counter_ns` around replay calls without CUPTI or GPU events. It
records counts and CPU duration samples, including timer overhead/possible driver waits; it is
not part of paired warm wall samples or a sum of savings. The previous CUPTI trace is only context.

```sh
python scripts/profile_graph_chunks.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --repeat 11 --out /local/results
```

The PBS runner performs the GPU checks and stages all code, audio, environment, models and logs
on `$PBS_LOCALDIR`. Any performance result remains pending; no graph-chunk speedup is claimed.

First allocation2777825 exited1 after12 checks passed. A later reference-session capture failed
with CUDA stream-capture invalidation and graph-reset warnings, before any performance phase.
The experimental installer held two bound methods on its session, creating a reference cycle.
The revised installer uses normal subclass binding and an unbound fallback, allowing reference
counting to destroy session graphs outside subsequent captures. Two CPU lifetime guards and
GPU end-of-case lifetime assertions cover this defect; confirming the complete GPU run remains
pending. The failure is retained; no speed result was produced. Production runtime is unchanged.
