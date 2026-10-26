# Combined beam optimization validation

Final branch: `beam-timestamp-optimized`. Original `v0.2` and all intermediate/experimental branches
are preserved. Remote `main` was rechecked on2026-10-26 and still points to892c29fb9b01b22c3e56e6848af818c5470c9288.

The final comparison reference is `4b3b581`: it already includes strict score-bound stopping,
completion-check tuning and candidate/finish kernel fusion. The candidate additionally uses
valid-prefix self-KV gathers and shared cross-KV on supported Gluon shapes. Decoder CTA/prefetch
settings and grouped sibling-query attention were measured but rejected. Grouped-query code is
kept only on `beam-grouped-cross`, not in this final runtime.

The combined one-GPU allocation uses the same loaded1B fp16 checkpoint and engines per comparison,
K5/timestamps, English73 (5 repetitions) and Japanese12 (3 repetitions), B1 thenB3, plus six English
edge/long-form cases. All measured shapes warm up before timing, and baseline/candidate order
alternates. Detailed traces/diagnostics are separate from uninstrumented wall samples. This final
comparison is needed because isolated speedups cannot be added or multiplied into a measured gain.

Standalone correctness coverage:221 base tests plus12 cross-KV/Ring/fallback tests. Public
`beam_kernels=False` selects reference candidate/finish operations, PyTorch KV reordering and
expanded cross-KV; `beam_upper_bound=False` separately disables strict-bound stopping. The scope
of the final ablation is KV changes; it does not re-measure original ESPnet or repeat earlier bound
and search-fusion comparisons. Results are pending before job completion.

Resources retain gai51740/R9920261300/rt_HG, one node/one GPU, at most1hour. Standing user approval
allows serial jobs within192 account GPUs; queue, duplicates and allocated+eligible queued+new
capacity are checked immediately before every submission. Temporary code/environment/audio/logs
stay on PBS_LOCALDIR, with compact phase progress and final artifacts on shared storage.
