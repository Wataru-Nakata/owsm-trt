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


## 102M row attribution2775955

`profile_ring_stages.py --cases rows3 rows5 rows8 --manifest /local/manifest.json` now uses
actual audio, the same cached encoder/prefill engines and the first selected token of each
plain-ASR row. It times only the subsequent Ring decoder call, with fixed state; encoder,
prefill and loading are excluded. `profile_ring_row_stages.pbs` compares these diagnostic states
with separate synthetic cache128/offset32 states,15 repeats. The original single-row cases
and default settings remain available. The main runtime and production routing are unchanged.

Both instrumented and uninstrumented calls must have identical logits/cache. Stage clocks
contain instrumentation overhead; GPU-event values and graph replay samples remain separate.
This attribution targets the existing102M row route measured in2774140. It cannot establish
request latency, numerical accuracy versus fp32, later-step behavior or a gain for any proposed
parameter/kernel change. Those require separate same-route correctness and wall measurements.


Frozenf2f243d, job2775955 exited0 with all six instrumented/uninstrumented logit/cache comparisons
exact. The15-repeat uninstrumented CUDA-event medians (fixed state, not request latency) were:

|State|R3|R5|R8|
|---|---:|---:|---:|
|Actual audio first-prefill|199.08µs|274.99µs|276.82µs|
|Synthetic cache128/offset32|196.95µs|274.72µs|276.80µs|

Instrumented cross-attention accounted for24.7–25.0% atR3 and38.8–39.2% atR5/R8; self-attention
10.1–13.1%, FFN2 10.7–14.6%, output projection8.9–12.4%. These shares include instrumentation,
not a decomposition of uninstrumented wall time. The default24 CTAs process R*6=30/48 attention
heads atR5/R8, requiring some CTAs to handle a second head. A next isolated experiment will
increase existing row program budgets while retaining the original attention split count, tile,
precision and single-row route. No speed gain for that change has been measured yet.

[Environment and compact attribution](runtime-ring-row-stages-2775955.json); raw stage/event
samples in `outputs/runtime-2775955/{real,synthetic}/`. First-prefill states cover only the initial
step; synthetic offset32 is a separate diagnostic. Stageout_status1 was reported despite full
publication/exit0. The earlier broad group-guard regression was removed before this measurement.


## Subsequent measured decisions

The row attribution above led to the guarded H200/102M R5..8 head schedule, independently verified
and adopted with unchanged arithmetic/splits ([results](runtime-ring-row-programs.md)). Single-row
plain calls now select the existing solo budget only without encoder lookahead; independent1B/102M
request measurements support that dispatch change ([results](runtime-plain-solo.md)). These are
separate same-route comparisons, not multiplicative gains over the earlier counter measurements.
The existing weight-ring capacity128/64/32 comparison subsequently rejected both smaller settings
([results](runtime-ring-buffer.md)); all runtime buffer defaults remain unchanged.

## Reprofile the current serial route

After the H200/102M serial64 adoption, `plain_solo` explicitly profiles R1/S1 with argmax and
solo dispatch. The old `plain` diagnostic retains its concurrent budget and no argmax; it is not
silently relabelled as current plain-request execution. `scripts/profile_ring_serial_stages.pbs`
runs both1B and102M at their current64serial CTAs: one actual audio/first-prefill state and
separate synthetic W1/8/32,15repetitions each. Every instrumented/uninstrumented logit/cache/pred
must match. No model/kernel arithmetic or runtime routing changes in this profiling step.

Results follow below. CUDA-event graph averages include launch gaps and are not request latency;
globaltimer stage sums include instrumentation and must not be subtracted as predicted speedups.
The profile excludes feature/encoder/prefill and cannot establish corpus quality or later-step
behavior. Its purpose is to select the next measured decoder bottleneck after the accepted schedule.

Job2778412 (f854900) passed all8instrumented/reference output/cache/pred comparisons. Both models
use64serial CTAs and NS2. Reference CUDA-event p50 averages in microseconds:

| Model | Real first-prefill S1 | Synthetic S1 | Synthetic S8 | Synthetic S32 |
|---|---:|---:|---:|---:|
| 1B | 618.01 | 620.51 | 627.09 | 713.39 |
| 102M | 177.77 | 175.39 | 182.60 | 220.74 |

Cross-attention is the largest instrumented stage group:17.9–20.9% for1B and20.6–23.1%for102M.
FFN2accounts for15.2–17.2%and13.1–16.2%, respectively. Medians are aggregated from15stage samples;
these fractions include instrumentation and are not removable wall time. The next isolated N
experiment holds CTA budgets fixed while changing single-row cross-attention splits2/4/8.
The split count will remain identical within each variant for concurrent/solo calls, preserving
schedule-independent arithmetic within that variant. Current defaults remainNS2 and quality/latency
must be measured before any adoption. [Environment and attribution](runtime-ring-serial-stages-2778412.json).
