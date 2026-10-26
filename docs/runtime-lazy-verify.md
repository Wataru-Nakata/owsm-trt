# Deferred verification calls (experimental)

Job2771808 constructed48 B1 verification graphs per session but actually invoked11 shapes in direct
ASR and18 in B1 streaming; B8 timestamp mode constructed24 and invoked7. Whole-manifest cold warmup
was3.85–7.81s on1B. That includes all cold work and requests; it is not a graph-capture-only duration.

The opt-in `GreedySession(lazy_verify_graphs=True)` retains per-window input/scratch buffers, but
constructs the decoder closure (including backend trial) and captures its graph only on first use
of that bucket/window/solo mode. No cache layout, arithmetic, window or dispatch limit changes.
`OWSM.greedy_lazy_verify_graphs` is an internal experimental switch, initially False.

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
load/build or steady-state speed. No performance result or adoption is claimed yet.
