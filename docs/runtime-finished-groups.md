# Skip completed multi-row plain decoder groups

The measured single-row Ring guard does not skip multi-row decoder calls. In job2773841, English
translationB3 still entered33/712 already-EOS calls on1B and45/712 on102M; Japanese had3/176 and
5/818. These are measured calls, not a predicted speedup. The completed 1B subset of job2773955 measures the group guard below.

`GreedySession(skip_finished_groups=True)` retains its low-level defaultFalse. The public factory
enables it with `greedy_kernels=True`, subject to actual backend capability. Public GPU
verification and an independent timing repeat passed in job2774007.
It is eligible only when the actual multi-row backend route supports it (currently Gluon within
its existing token/row limits). `greedy_kernels=False` disables both single-row and multi-row guards. Route limits do not change.
It passes the session's per-row EOS flags to the kernel. Every CTA reduces the same immutable
vector before any shared-memory setup or persistent-grid barrier. Only when all rows have ended
does every CTA return. Partial completion keeps the full original computation. This does not skip
individual rows, refill groups, change host polling or alter model/search arithmetic.

The flags remain unchanged while a call is in flight, ordered by the existing CUDA stream.
Graph replay counts/pos/selection behavior remain unchanged. Skipped calls leave output/cache/
workspaces untouched and the next request resets flags. No extra normal per-step reduction kernel
or host read is added. Unsupported TensorRT routes and drafted verification retain their calls.
Both benchmark modes retain the already-adopted single-row guard, including a B1 tail.

Job2773955 (commit356eb69, H200, same checkpoint/engines) passed64 focused GPU/state tests
and31 naive beam/greedy/timestamp regressions. All1B English73/Japanese12 outputs matched in
all eight conditions, including full tokens/timestamp tokens/text/segments and decoder call counts.
English silence/noise, long prompt, full window, drafted B1/B3, lookahead and110s long form also
matched (long form exposes joined segments, not raw window tokens). Reference ASR WER/CER is
unchanged; reference translations/timestamp boundaries are unavailable.

| 1B case | English RTFx change | Japanese RTFx change |
|---|---:|---:|
| plain B3 | +3.47% | +4.61% |
| timestamp B3 | +3.66% | +2.38% |
| translation B3 | +3.37% | +1.64% |
| translation B8 | +1.92% | +0.35% |

These are warmed whole-list median wall comparisons over nine alternating pairs, not request
latency percentiles. All pairs were positive except Japanese B8 (-1.60% to+3.31%); that condition
has no repeatable improvement established. Separate counters identify33/38/30/9 new skipped
multi-row calls on English and9/6/3/3 on Japanese. B1 tails already had the adopted Ring guard
in both modes and are excluded from these new skip counts. Replay/selection counts stay equal.
See [raw compact results](runtime-finished-groups-results-2773955.json) and
`outputs/runtime-2773955/` for timings, quality, environment and full output artifacts.

The job then stopped with exit1 at the102M preflight: its actual B3/B8 route is TensorRT
(`no kernel for rows`), not Gluon, so this optimization is inapplicable. No102M speed claim is
made. The later independent identity-KV replication was not reached. Unsupported route coverage,
public-option wiring and independent replication are the next verification steps; the failed
job is not reported as an overall success. Both optimizations remain off in public defaults at
this result commit. Neither speedups nor cold/shared-process memory observations are combined.


The follow-up runner validates the public factory and rollback on1B and102M, including plain
B3 timestamps and translation B8. It repeats the eight1B timing conditions, tests all eligible
R2..8 GPU flag/restart contracts, and completes the previously unexecuted independent KV
replication. The102M comparison is functional fallback coverage, not a group-guard speed test.


## Independent replication and public adoption:2774007

Frozen d16a15f passed86 focused tests (all R2..8 graph/eager flag/reset contracts included) and31
naive regressions. Both1B and102M public-default/rollback checks matched all12 request cases;
actual guard dispatch matched capability. The102M TensorRT row fallback remains unguarded.
The same engines/checkpoints as2773955 produced exact full outputs/steps in all eight1B conditions,
with unchanged English extras. No engine rebuild or different precision is involved.

| 1B case | English RTFx change | Japanese RTFx change |
|---|---:|---:|
| plain B3 | +3.35% | +4.54% |
| timestamp B3 | +3.46% | +2.58% |
| translation B3 | +3.53% | +1.61% |
| translation B8 | +1.78% | +1.47% |

Nine alternating whole-list pairs were positive for every condition except Japanese B8
(-0.98% to+2.15%, also mixed in the first job). This group guard is adopted through the existing
`greedy_kernels` option; no speed claim is made for every individual input, Japanese B8,102M
multi-row TRT, other GPUs or drafted verification. The host still makes the same replay checks.
See [replication evidence](runtime-finished-groups-results-2774007.json) and
`outputs/runtime-2774007/`. Shared-process VRAM/cold observations are not isolated improvements.


## Ring groups: prepared, GPU verification pending

The existing R1 Ring guard is extended to an immutable vector for S1 multi-row calls. All CTAs
reduce the same flags before producer/barrier setup; partial groups run the full decoder. The
scalar R1 path is retained. Strided/non-vector flags are rejected. This uses the existing public
`greedy_kernels` group capability; auto102M still routes multi-row calls to TensorRT.

`profile_ring_finished_groups.pbs` prepares D384/F1536 R2..8 active/partial/all-ended/restart,
argmax/logits, eager/CUDA-graph tests and public Ring rollback checks. It compares group guard
OFF/ON within the same explicit `decoder_kernel="ring:rr"` route, retaining the adopted B1 guard
in both modes. English73/Japanese12, B3 ASR/timestamp/translation and B8 translation use11
alternating whole-list pairs; English extras include silence/noise, prompts, full/long windows,
drafted decoding and lookahead. This is an E comparison within Ring; the separately measured
Ring-versus-auto numerical/quality differences still apply. No additional speed claim yet.
