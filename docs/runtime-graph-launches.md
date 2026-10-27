# Actual CUDA-graph launch attribution

Candidate28 investigation, with no runtime optimization or new search semantics. Existing eager
phase events include host launch gaps and cannot quantify the cost of graph replay. Current plain
and B3 beam loops enqueue up to four one-step graphs before their existing done check; B1 bound
beam checks each step, so batching its replays would change that policy and is excluded from any
claim of an equivalent four-step schedule.

`scripts/profile_graph_launches.py` uses the current public API and default kernels, warms all used
shapes (plus optional draft dispatch variants), and records21 uninstrumented wall samples. In a
separate profiler pass it temporarily wraps existing graph objects to label `.replay()` calls for
frontend, encoder, plain/beam decoder and drafted verification. It preserves graph addresses and
restores every wrapper even if tracing fails. It does not execute eager `_body` in place of a graph,
change decoder mathematics, insert per-step synchronization, or time the wrappers as steady-state.

The trace retains CUDA runtime API durations, kernel names/durations and replay counts. GPU sums
are attributed only if their correlation ID matches a host launch wholly inside a replay annotation
on the same CPU thread. Missing or ambiguous IDs remain unattributed; absent GPU activity is null, not zero cost.
Five small synthetic tests verify these conditions and exception-safe restoration. Summed CPU API
time can include waits and overlap GPU work. Summed kernel time can overlap other kernels/copies.
Neither is a predicted wall-time saving or graph-node phase split. A profiler can also perturb gaps;
any proposed optimization still needs its own uninstrumented same-engine ablation.

Each1B/102M English/Japanese run traces five representative loaded-input cases: beam5 timestamps
B1/B3, translation B1/B8, and drafted timestamps B8. B3/B8 mix duration quantiles to expose tails;
B1 uses a median-duration recording. These are representative traces, not full-corpus quality or
throughput averages. Traced and all warm outputs must match, new warm captures and silent kernel
fallback fail the run. Metadata, engine/checkpoint hashes, actual dispatch, call counts, full outputs,
compressed traces and trace summaries are retained. No new multi-step graphs are implemented yet.

```sh
python scripts/profile_graph_launches.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --repeat 21 --out /local/results
```

The PBS runner stages audio, code, environment, models, unchanged engines and temporary trace files
on `$PBS_LOCALDIR`, then publishes compact results and compressed traces. No speed claim precedes
these observations; graph-internal costs that cannot be correlated remain explicitly unmeasured.

The allocation also repeats the independent singleton-header ablation in separate processes for
1B/102M/CTC1B. Those uninstrumented timing pairs are not graph-attribution results. Prior header
correctness/artifact hashes are retained; the five attribution tests run in the new allocation.


## Completed diagnostic2777712

Frozen0fbc04c, H200, F/exit0; five attribution/restoration tests pass. All20 graph cases finish,
with21 identical warm outputs and an identical separately traced output. CUDA graph launch
correlations assign kernel sums to frontend/encoder/decoder/draft replay ranges. Prefill, eager
work and missing correlations stay in the unattributed category. No graph-node phases are inferred.

Representative English observations (milliseconds):

|Model / request|Warm wall p50|Diagnostic decoder/draft kernel sum|Diagnostic encoder kernel sum|CUPTI graph-launch API sum|
|---|---:|---:|---:|---:|
|1B beam TS B1|25.207|19.102|2.860|2.070|
|1B beam TS B3 mixed|172.900|140.626|8.465|38.418|
|1B translation B1|17.749|11.760|2.860|0.928|
|1B translation B8 mixed|113.656|89.903|17.334|2.345|
|1B drafted TS B8 mixed|40.168|18.502|17.383|4.852|
|102M beam TS B1|12.057|8.370|0.656|3.159|
|102M beam TS B3 mixed|55.379|47.379|1.501|14.417|
|102M translation B1|6.023|3.432|0.624|0.435|
|102M translation B8 mixed|44.277|33.033|2.712|10.808|
|102M drafted TS B8 mixed|19.480|11.390|2.699|3.937|

The API sums include profiler overhead/possible driver waits and overlap device execution. They
must not be subtracted from warm wall time to predict savings. Decoder work remains substantial;
drafted1B B8 also has material encoder work. Japanese102M translation reaches374 tokens on the
selected long example; it matches the existing output and is not a new truncation/decoding policy.

A small profiler-only candidate28 ablation is prepared next: capture four existing `_body` calls
and replay them only where the unchanged host-check interval permits exactly four steps in the
same cache bucket. Original graphs cover boundaries and remainders; default B1 beam's interval1
is retained as a control. Extra capture/memory costs and full outputs must be measured before any
production change. A separate low-overhead CPU timer, without CUPTI, will characterize replay
call duration; steady wall measurement remains uninstrumented. No multi-step speedup is yet known.

[All20 traces' summaries, warm samples, dispatch, outputs, hashes and telemetry](runtime-graph-launches-2777712.json).
Compressed raw traces remain under `outputs/runtime-2777712/`. Stageout_status1 is separate from
exit0 and verified artifacts. The singleton-header comparison is a different experiment.
