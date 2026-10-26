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
all3 strided group positions with attention prefetch ON/OFF against contiguous buffers. GPU job2770426 passes all292 tests. Results below retain this as an opt-in N mode.

The prepared job compares tail B3 and grouped/slice8 B1 separately against state-fused reference,
English73 and Japanese12, sharing engines within each alternating comparison. Grouped/slice8 is
included to verify its combination with the newly adopted state fusion; it remains an opt-in N mode.

## Same-engine measurements

Reference is the state-fused default c99a196. English73 uses5 alternating repetitions, Japanese12
uses3; load/JIT/capture and warmup are excluded.

| Mode | Inputs | B | p50 before → after (ms) | p95 before → after (ms) | RTFx gain |
| --- | --- | --- | --- | --- | --- |
| Tail only | English73 | 3 | 48.10 → 46.32 | 100.32 → 88.15 | +9.83% |
| Tail only | Japanese12 | 3 | 74.37 → 67.65 | 125.04 → 115.74 | +8.23% |
| Grouped8 only | English73 | 1 | 24.33 → 22.84 | 51.71 → 47.98 | +7.00% |
| Grouped8 only | Japanese12 | 1 | 33.16 → 30.85 | 78.37 → 72.47 | +7.74% |

Tail reduces English decoder row-calls from13270 to10210 (23.1%) and Japanese3075 to2545 (17.2%).
It activates in19/25 English batches and4/4 Japanese batches, without changing decoder step counts.
All text matches and WER45/1150 / CER26/592 are unchanged. English tail tokens/timestamps match71/73:
`1272-128104-0004` changes one segment end10.48→10.50 s; `1272-141231-0005` changes end5.80→5.82 s.
Japanese tail tokens match12/12. Main segment counts match with no malformed timestamps. These are
baseline differences, not errors against reference annotations (none are available).

Grouped8 main tokens match85/85, but the previously reported silence transcript difference remains.
Additional tail-job cases run at B1, so their agreement does not establish B3 compaction compatibility
on silence, long prompt, translation or long-form audio. Synthetic tests cover the switching logic.

B3 adds12 tail graphs (3 groups x4 cache buckets). First English variant construction took23.31 s
including new JIT/capture, versus0.66 s reference; with compilation cached, Japanese construction was
0.705 vs0.608 s. These are cold costs, not steady-state latency. Peak allocated memory observed in
the shared process is11.18 GB (English B3) /10.02 GB (Japanese B3); because both variants coexist,
these are not independent memory footprints. Limits remain Hopper/Gluon routing <=8 rows/16 tokens.

[Full results and hashes](beam-results-2770426.json); raw tokens and traces: `outputs/beam-2770426`.
Both modes remain off by default. Public API flags `beam_compact_tail=True` and
`beam_grouped_cross=True` are wired independently; the final combined comparison follows separately.
