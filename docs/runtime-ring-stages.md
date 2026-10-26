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
the two variants is required. Measured diagnostics follow below; this script changes no runtime defaults.

The optional Nsight Compute invocation in `profile_plain_stages.pbs` brackets only warmed launches,
keeps clock/cache control disabled, and writes counters/reports locally before publication. Permission
errors, timeouts, or absent tools are retained as missing data; no system permission change is attempted.
Counters and instrumented-stage timing are not mixed with steady-state request wall measurements.


## Measured diagnostics:2772119

All four instrumented variants preserve exact logits, predictions and self-cache. Uninstrumented
CUDA-graph event medians are662.70µs (plain R1/W1/P32),680.02µs (verify W8/P32),614.58µs (W8/P64),
and710.60µs (W32/P64). The TS variant is timed independently; these are synthetic-state kernel
measurements and do not imply speech speedups or justify changing dispatch by themselves.

Instrumented stage shares: cross attention16.6–20.9%, FFN2 15.1–19.1%, FFN1 11.2–12.6%,
output head4.3–5.5%. Time is distributed across layer operations and synchronization; the output
head is not the primary component in these conditions. Globaltimer resolution quantizes short
intervals, and these fractions include instrumentation.

Nsight Compute2025.3.1 succeeded for plain and W8 solo,11 replay passes per launch. Reported SM
throughput was3.28/4.62% of peak, DRAM20.68/22.06%, L2 26.87/27.14%; eligible warps/scheduler
0.142/0.099 and active-cycle issue rates13.86/9.75%. Kernels launch32/64 CTAs on132 SMs with255
registers/thread and about153/172kB allocated shared memory per block. Both registers and shared
memory limit occupancy to one block per SM. These diagnostics indicate available arithmetic/bandwidth
capacity but do not alone identify which waits can be removed. Warp-stall/source counters and the isolated plain host-check experiment were subsequently
completed in2772221; see the follow-up below. Existing SM-count
and prefetch sweeps remain rejected; no speculative kernel change is adopted from these counters.

Artifacts: outputs/runtime-2772119/ring_stages/{summary.json,environment.json}, ring_counters.csv,
ring_counters.ncu-rep, ring_ncu.log. CSV is wide: one units row followed by one row per profiled launch.
The counter replay durations are separate from uninstrumented CUDA-event samples.


## Follow-up counters and rejected changes

Job2772221's PC samples placed substantial long-scoreboard waits at grid polling; local-memory
load/store requests were zero in the two profiled kernels. The barrier category mostly sampled
warp-specialization wait/register-allocation machinery. Those samples are not critical-path wall
fractions. Job2772303 reduced static polling invalidations48→29 while retaining exact outputs,
but did not yield consistent request gains, so the runtime experiment was removed. See
[runtime-ring-poll.md](runtime-ring-poll.md) for counts, limitations and reproduction.

Greedy completion checks every1/2 steps also did not produce a broadly better latency/p95 result;
interval4 remains the default (runtime-greedy-checks.md). The next independent measurements use
existing attention-prefetch and weight-chunk settings (runtime-ring-memory.md), with N-class
accumulation grouping explicitly separated from E-intent scheduling.
