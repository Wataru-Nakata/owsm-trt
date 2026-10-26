# Ring stage and hardware-counter diagnostics

The accepted greedy selection changes leave the Ring decoder as the largest measured GPU component.
SM-budget and existing L2-prefetch sweeps did not yield consistent end-to-end gains; their defaults
remain unchanged. `scripts/profile_ring_stages.py` profiles the existing kernel before further tuning.

Run on a compute node with the actual checkpoint. Four fixed synthetic decoder states cover one
plain token, eight-token verification with32/64 CTAs, and32-token solo verification. These are kernel
diagnostics, not speech/request benchmarks. Weights are real; self/cross KV are synthetic, cache128,
offset32, one row. Each configuration uses existing default arithmetic/routing settings.

CUDA-event samples around uninstrumented CUDA graphs are saved separately from the `TS=True`
variant. Globaltimer stage records include embedding, eight stages per decoder layer, output head,
and the extra argmax stage when enabled. Barrier/skew time is instrumented and cannot simply be
subtracted from an uninstrumented request. Exact logits, predictions and self-cache equality between
the two variants is required. No new performance claim is available yet.

The optional Nsight Compute invocation in `profile_plain_stages.pbs` brackets only warmed launches,
keeps clock/cache control disabled, and writes counters/reports locally before publication. Permission
errors, timeouts, or absent tools are retained as missing data; no system permission change is attempted.
Counters and instrumented-stage timing are not mixed with steady-state request wall measurements.
