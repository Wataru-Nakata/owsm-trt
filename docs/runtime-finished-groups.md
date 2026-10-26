# Skip completed multi-row plain decoder groups (experiment)

The measured single-row Ring guard does not skip multi-row decoder calls. In job2773841, English
translationB3 still entered33/712 already-EOS calls on1B and45/712 on102M; Japanese had3/176 and
5/818. These are measured calls, not a predicted speedup. The completed 1B subset of job2773955 measures the group guard below.

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
