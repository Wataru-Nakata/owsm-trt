# Deferred verification calls

Job2771808 constructed48 B1 verification graphs per session but actually invoked11 shapes in direct
ASR and18 in B1 streaming; B8 timestamp mode constructed24 and invoked7. Whole-manifest cold warmup
was3.85–7.81s on1B. That includes all cold work and requests; it is not a graph-capture-only duration.

High-level `lazy_drafts=True` enables deferred verification by default. Standalone
`GreedySession(lazy_verify_graphs=True)` retains per-window input/scratch buffers, but
constructs the decoder closure (including backend trial) and captures its graph only on first use
of that bucket/window/solo mode. No cache layout, arithmetic, window or dispatch limit changes.
`OWSM.greedy_lazy_verify_graphs` follows the public `lazy_drafts` option.

First use can happen after long prompt prefill or a live bucket migration. Warmup uses offset0,
so the written firstW self-KV positions are cloned and restored around trial/capture. Pinned inputs
are reset before trial; a cold-only event wait protects their lifetime before actual input is copied.
The warm path does no preservation or extra synchronization. Partial factories own backend/buffers,
not the session, to avoid introducing a session reference cycle. Unused graphs stay absent.

CPU artificial-state checks cover untouched KV and idempotent construction. GPU tests cover windows,
buckets16/32/64, long prefill, notime/timed/mixed mode changes, repeated graph identities and graph-off
execution. Actual inference, tests and cold timing run only in an allocated job.

`profile_runtime_cold.py --mode verify --cases asr prompt stream` compares fresh sessions using one
resident engine/backend; the first cycle is recorded separately and later cycles warm JIT. It checks
outputs after both cold calls and reuse across tasks. This measures session-cold costs, not full model
load/build or steady-state speed. Measured results and the validated scope follow.

## Initial B1 measurements

Job2771944 passed119 regression tests and18 selected GPU tests (4 deferred-state/capture,8 existing
mode/cache,6 Ring-prefetch tests). `verify_cold/cold.json` matched all cold and reused-request outputs.
With resident1B/H200 engines and warmed JIT, three session reconstructions gave median cold latency:
ASR334.18→218.64ms (-34.6%), long prompt340.61→223.68ms (-34.3%), three-input B1 streaming698.94→587.89ms
(-15.9%). These are not model-load/build or steady-state gains. Direct ASR retained27.0MiB fewer bytes;
streaming allocation varied by32MiB across repeats, so the raw samples should be used for memory claims.
Actual verification graphs were48→1/2/3 for these three initial requests; template count remains48.

`profile_runtime.py graph_lazy` records first-use time/counts, then explicitly prepares all potential verification shapes for a separate
steady-state comparison. The latter tests hot-path overhead and prevents timing a new capture when
encoder overlap changes solo dispatch; it is not a requirement that deployments prewarm every shape.
Memory in that multi-variant steady benchmark is not the cold memory saving or deployment peak.

## B8/102M replication and adoption

Job2771972 (source097e53d) passed all public/API and cold comparisons for1B/102M and12 English inputs
including B8 streaming, plus English7/Japanese translation warm-profile cases. All outputs match.
Session-cold medians (three reconstructions, warmed backend/JIT):

| Model/request | Eager verify | Deferred verify | Reduction |
|---|---:|---:|---:|
| 1B ASR B1 |365.64ms|242.52ms|33.67%|
| 1B stream B8,12 inputs |3986.77ms|1544.28ms|61.26%|
| 102M ASR B1 |174.80ms|106.69ms|38.96%|
| 102M long prompt |173.17ms|112.65ms|34.95%|
| 102M stream B8,12 inputs |2307.81ms|912.91ms|60.44%|

With all possible verification shapes explicitly warm, RTFx changes ranged-0.48% to+0.75% in the
English cases and-0.21% in Japanese translation; no steady-state speedup is claimed. Direct1B/102M
allocation deltas fell27.0/12.1MiB. Streaming allocator/context retention varied; raw samples remain
available instead of treating differences of medians as per-request guarantees.

High-level `lazy_drafts=True` now enables this behavior as well as deferring the whole unused draft
setup. False restores eager verification preparation. Standalone GreedySession retains explicit
`lazy_verify_graphs=True`. First use of an unseen shape, including solo/concurrent switching, has
setup cost; a warm model does not imply every shape is warm. KV preservation is only on cold setup.
Other GPUs/checkpoints are unmeasured. Artifact: `outputs/runtime-2771972` (`cold1b`, `cold102m`, API
records, `analysis-graph_flags.json`, raw wall/capture/trace/dispatch records).
