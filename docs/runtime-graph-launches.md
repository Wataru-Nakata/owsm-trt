# Actual CUDA-graph launch attribution (pending measurement)

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
