# Singleton header lookup (pending GPU measurement)

Candidate40, E intent. The earlier node-local warm-file profiler2776797 measured whole-corpus
header lookup p50 values2.45–3.11ms for English73 and0.38–0.51ms for Japanese12 (about32–43µs per
input on average). These are isolated CPU observations, not an expected whole-request speedup.

Both CTC and encoder-decoder `transcribe` sorted singleton inputs by duration, opening their file
header before the real loader opened the file again. An empty/single-item request now keeps its
only possible order without asking for duration. Multi-item sorting, file decoding, resampling,
shape/error checks and all GPU computation remain. There is no persistent file/audio cache.

The tiny fixed-batch/order/lookahead regression suite passes29 checks on CPU. No new unit tests
mirror the trivial ordering guard. `profile_single_header.py` binds the frozen pre-change
`OWSM.transcribe` method and the current method to the same model, giving each its own warmed
session pool and identical engines/loader/settings. Every timed call receives a singleton file
list; independent dispatch instrumentation requires a reference header lookup and zero candidate
lookups. Full tokens/times/text/segments must match. Scalar file, singleton list and loaded-tensor
output forms, missing files and invalid tensor dimensions are checked separately.

The PBS runner stages everything locally and measures11 alternating repetitions on1B,102M and
CTC-v4-1B, English73/Japanese12. For encoder-decoder models it uses greedy timestamps on the full
corpus, plus beam timestamps and translation on the first12 inputs. CTC uses no-timestamp single
requests. All p50/p95 values are single requests, unlike fixed-prefetch whole-corpus results.
Audio is read inside the timed call with warm OS caches. Cold network/shared-storage behavior
and file mutation races are unmeasured. Engine/capture failures cannot become fallback timings.

```sh
python scripts/profile_single_header.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --baseline-pipeline /local/baseline_pipeline.py \
  --repeat 11 --out /local/results
```

No speed claim or other-optimization composition is established before the run completes.

The allocation also independently repeats the fixed-prefetch comparison in separate processes.
Those multi-item calls use the same duration ordering in both versions, so the singleton shortcut
does not enter their timings. Keep the two result sets and their gain claims separate.


## First result2777617: exact outputs, mixed timings; adoption pending

Frozen9f2c6ce, H200, F/exit0. All14 conditions match full outputs and decoder-call totals. The
reference performs one header lookup per singleton, the candidate performs zero. Scalar/list/
loaded-tensor forms and missing-file/invalid-dimension errors match. First11-repeat RTFx changes:

|Model / corpus|Greedy TS (CTC: no TS)|Beam TS first12|Translation first12|
|---|---:|---:|---:|
|1B English73|+0.956%|-0.264%|+1.383%|
|1B Japanese12|+1.042%|+0.599%|-0.331%|
|102M English73|+1.002%|-10.419%|+0.618%|
|102M Japanese12|+1.620%|+1.096%|-0.369%|
|CTC1B English73|+1.206%|not applicable|not applicable|
|CTC1B Japanese12|+1.196%|not applicable|not applicable|

102M English beam's aggregate−10.419% includes large candidate outliers in repeat2 (up to176.41ms);
its per-input paired median is−0.059%, only5/12 input medians improve. The cause is undetermined;
these samples are not removed. Japanese translation shows small regressions for both models.
CTC improves every repeat total in both corpora; other conditions include negative repeats.
Do not claim a universal speedup or adopt the broad shortcut on this single mixed result.
An independent same-engine repeat is prepared alongside the separate graph-attribution diagnosis.

[All samples, counts, errors, environment and coarse telemetry](runtime-single-header-2777617.json).
The allocation also ran fixed-prefetch timing in separate processes; its multi-input gains do not
include this singleton change. No graph-launch optimization is part of these measurements.
