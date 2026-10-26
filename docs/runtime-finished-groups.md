# Skip completed multi-row plain decoder groups (experiment)

The measured single-row Ring guard does not skip multi-row decoder calls. In job2773841, English
translationB3 still entered33/712 already-EOS calls on1B and45/712 on102M; Japanese had3/176 and
5/818. These are measured calls, not a predicted speedup. B8 has not yet been measured for this guard.

`GreedySession(skip_finished_groups=True)` is an independent low-level experiment, defaultFalse.
It is eligible only when the actual multi-row backend route supports it (currently Gluon within
its existing token/row limits). The existing public single-row flag and route limits do not change.
It passes the session's per-row EOS flags to the kernel. Every CTA reduces the same immutable
vector before any shared-memory setup or persistent-grid barrier. Only when all rows have ended
does every CTA return. Partial completion keeps the full original computation. This does not skip
individual rows, refill groups, change host polling or alter model/search arithmetic.

The flags remain unchanged while a call is in flight, ordered by the existing CUDA stream.
Graph replay counts/pos/selection behavior remain unchanged. Skipped calls leave output/cache/
workspaces untouched and the next request resets flags. No extra normal per-step reduction kernel
or host read is added. Unsupported TensorRT routes and drafted verification retain their calls.
Both benchmark modes retain the already-adopted single-row guard, including a B1 tail.

Fourteen CPU tests passed for mixed row EOS lengths, bucket boundaries/length limits, lazy/eager
construction, repeated requests, actual-route limits and unsupported flag contracts; all18
single-row CPU regressions also passed. Six GPU tests are prepared for R3/R4/R8 with/without CUDA
graphs: partial flags must reproduce all active arithmetic/cache outputs, all-ended flags must
leave every public/workspace tensor unchanged, and resets must resume exactly. GPU compilation
and performance remain unverified.

`scripts/profile_finished_groups.pbs` runs these tests and existing greedy/naive regressions,
then `profile_finished_skip.py --group-skip` for1B/102M English73/Japanese12: plain/TS B3 and
translationB3/B8, nine alternating same-engine whole-list repeats, exact tokens/segments/steps,
separate already-EOS counters and English extras. No request-latency claim is made from grouped
whole-list percentiles. Each profile fails if actual dispatch silently loses the guard. The same
job independently repeats the previously positive1B B3 identity-KV experiment for15 pairs on
English/Japanese, in separate processes with no group-skip flag. Both experiments remain off in
the public defaults; their gains are not added together.
