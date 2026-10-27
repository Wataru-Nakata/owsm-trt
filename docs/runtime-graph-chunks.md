# Four-step graph ablation (measured, no broad adoption)

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
on `$PBS_LOCALDIR`. The completed comparison and limits are below.

First allocation2777825 exited1 after12 checks passed. A later reference-session capture failed
with CUDA stream-capture invalidation and graph-reset warnings, before any performance phase.
The experimental installer held two bound methods on its session, creating a reference cycle.
The revised installer uses normal subclass binding and an unbound fallback, allowing reference
counting to destroy session graphs outside subsequent captures. Two CPU lifetime guards and
GPU end-of-case lifetime assertions cover this defect; the corrected run below passed the complete GPU matrix. The failure is retained; no speed result was produced. Production runtime is unchanged.


Job2777838 (115960e) completed61checks and all16conditions with identical complete outputs,
plain decoder calls and existing stopping boundaries. No warm capture or runtime fallback
occurred. Exit0; scheduler Stageout_status1 is recorded separately from the published results.
The cycle-free helper also passed immediate session-release assertions after every GPU case.

| Model / corpus | B1 beam unchanged control | B3 beam | B1 translation | B8 translation |
|---|---:|---:|---:|---:|
| 1B English | +0.605% | +0.414% | +0.327% | +0.204% |
| 1B Japanese | -0.129% | +0.049% | -0.439% | +0.413% |
| 102M English | -0.024% | -0.956% | +1.566% | -0.527% |
| 102M Japanese | +0.437% | -0.402% | +0.655% | -0.643% |

Values are aggregate warm RTFx changes, eleven alternating repeats. The unchanged B1 beam
control demonstrates that sub-percent aggregate shifts are not reliable gains by themselves.
102M English B3 was slower in all11pairs (paired median-0.878%); Japanese B3/B8 were slower
in10/11pairs. 102M B1 translation improved in all11pairs for both languages, paired medians
+1.306%/+0.585%; this narrow result has no independent replication and does not justify general
adoption. 1B translations/beam have mixed pair signs, including10/11slower Japanese B1translation.

Fewer graph replays did not guarantee wall improvement. For example, the separate English
102M B3 diagnostic records752→269decoder graph launches and CPU enqueue sum6.063→4.400ms,
yet warm wall time regressed. English102M B1translation records428→107launches and1.598→0.559ms
enqueue sum. These timer-instrumented sums are neither device time nor additive wall savings.
Remainders/bucket edges/partial final B1beam batches explain less than fourfold reductions.

Candidate sessions add4or8graphs per case (zero for the unchanged beamB1control), retain original
graphs and have additional capture work. Saved warmup/allocator deltas include the whole case,
shared engine/frontend/JIT initialization and resident pools; they do not isolate standalone
capture cost or deployment VRAM. First-reference cold ordering strongly biases those numbers.
Allocation-wide5sGPU samples also include all cold/idle/diagnostic phases.

Decision: no production API/default/runtime change. Keep the profiler helper as an isolated
experiment; any follow-up for102M B1translation needs independent paired timing and isolated
cold/memory evidence. Prioritize the larger measured decoder work. No throughput improvement
is claimed for four-step replay in general.

[Complete samples, environment, agreement, cold/dispatch diagnostics and failure history](runtime-graph-chunks-2777838.json).
Raw full token/segment outputs, wall samples and logs: `outputs/runtime-2777838/`.
