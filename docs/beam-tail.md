# Single active beam group experiment

Baselinec99a196 (validated state fusion enabled). Opt-in `BeamSession(..., compact_tail=True)` /
profiler `--compact-tail`. Initial target is H200/1B/K5/B3 with timestamps.

At existing host completion checks, one bulk read of B done flags replaces the scalar all-done
read. If exactly one utterance remains, the next chunk feeds only its K decoder rows. Search,
best hypotheses, timestamp state, self-KV parent gathering and lengths keep the original B*K
layout. Prefill and stop conditions are unchanged. Done rows never become candidates again.

The Gluon kernel supports a larger layer stride on an otherwise contiguous group view of self
and cross KV. No KV copy is added at transition. Cache growth retains the original full buffers;
all group/bucket graph addresses are fixed. Each group/bucket needs another graph and decoder
workspace. Unsupported kernels retain the original path with a recorded reason. A kernel trial
fallback receives refreshed expanded cross KV after each encode, including tail closures.

Changing R15 TensorRT calls into R5 Gluon calls can change numerical results: this is an N
experiment, not an exact-output guarantee. Repeated decode resets the active group and transition
record. Diagnostics report the transition and decoder row-call count; actual fallback dispatch and
extra capture shapes are recorded. Wall timing includes the existing host-check cadence.

Small CPU tests cover every last group, intervals1/4, partial and simultaneous ends, forced limit,
bucket growth, repeated decode and trial-failure cross-KV refresh. GPU checks add graph replay and
all3 strided group positions with attention prefetch ON/OFF against contiguous buffers. Speed,
real-model output differences and adoption are pending measurement.

The prepared job compares tail B3 and grouped/slice8 B1 separately against state-fused reference,
English73 and Japanese12, sharing engines within each alternating comparison. Grouped/slice8 is
included to verify its combination with the newly adopted state fusion; it remains an opt-in N mode.
