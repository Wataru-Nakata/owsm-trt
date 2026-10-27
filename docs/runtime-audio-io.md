# File-inclusive audio preparation: first isolated experiment

Previous runtime comparisons preload waveforms. The saved102M English73 aggregate load observations
were120–125ms, while the warmed loaded-input B8 request median was about53ms. Those separate
observations include different initialization and overlap, so they cannot be added or treated as a
speedup estimate. They motivate measuring the file-inclusive path explicitly.

`scripts/profile_audio_io.py` compares the unchanged file loader, a profiler-only mono shortcut,
and a resident-tensor control, with identical engines/model/decoder settings. The candidate skips
`mean(axis=1)` when a file has one channel, returning a tensor that owns the decoded NumPy view.
Stereo averaging, sample-rate conversion and tensor/array inputs retain their current behavior.
Numerical waveform equality is required (IEEE signed-zero representation is not a byte identity
claim). Full returned text/tokens/timestamp IDs/segments and call totals must match; no runtime
loader change or speed claim is made before measurement.

Separate warmed CPU phases time headers and audio decode/mono/resampling, excluding checks.
Whole-request measurements include these operations for file modes, whereas the loaded control
excludes them. Files are staged into node-local storage and OS caches are warm. This is not a
cold-disk/network/Lustre bandwidth result, nor a resampling benchmark for other sample rates.
The real corpora use16kHz; focused compute-node tests cover16/44.1kHz mono/stereo PCM/FLOAT and
retained tensor lifetimes, with nonfile conversion/error behavior preserved.

`profile_audio_io.pbs` runs nine focused tests then1B/102M English73/Japanese12, timestamped single
requests and B8 drafted streaming,11 alternating repeats. All sources warm their model/session
shapes; no new warm capture or decoder fallback is allowed. Actual disk/input modes are recorded
alongside environment/engine/checkpoint hashes. Diagnostics are excluded from wall samples;
per-request and whole-list percentile scopes remain separate. Default B1 frontend graphs stay on.


## First result2776693: public integration pending

Frozen4a8bf02,H200, exit0. Nine compute-node tests passed; all eight workload conditions match
waveforms, full tokens/time IDs/text/segments and decoder-call totals across all three input modes.
Eleven alternating file-inclusive comparisons gave:

|Model/corpus|Single TS B1 mono gain|Drafted TS B8 mono gain|
|---|---:|---:|
|1B English73|+2.29%|+3.47%|
|1B Japanese12|+4.21%|+3.60%|
|102M English73|+4.75%|+8.81%|
|102M Japanese12|+6.16%|+24.15%|

Every B1 paired total improved. B8 is outlier-sensitive:102M Japanese individual pairs span
−11.50% to+180.36%; its aggregate24.15% is not a stable general speedup. Independent public-loader
validation is required. CPU decode/mono p50 fell128.20→110.92ms for1B English and92.58→81.70ms
for102M English; Japanese fell4.55→1.31ms /4.79→1.47ms. Headers were measured separately.

The loaded-tensor control quantifies avoiding file I/O entirely and must not be counted as an
implemented optimization. All file measurements use warmed node-local data, with different format/
length mixes between corpora; they establish neither cold-Lustre nor arbitrary-resampler speed.
[Environment, phases and paired results](runtime-audio-io-2776693.json); raw artifacts
`outputs/runtime-2776693/`. Integrate only the mono shortcut, keeping stereo/resampling/nonfile
behavior, then independently compare the public loader against an explicit preserved legacy loader.


The public loader's mono view is now implemented; independent verification is pending in
`validate_frontend_api.pbs`. The profiler preserves explicit legacy file averaging and compares it
against the actual imported public loader, so the baseline cannot silently become the candidate.
All multichannel/nonfile/resampling behavior remains covered by the compute-node tests.

## Public verification2776797: mono shortcut adopted

Frozen61fef6c passed43 focused tests (including9 real loader cases) and the independent public-path
ablation. All sampled waveforms, tokens/time IDs/text/segments and decoder-call totals match.
The comparator retains the old file-averaging implementation; the candidate calls the actual
public loader from9341c60. Eleven alternating warmed file-inclusive comparisons gave:

|Model/corpus|Single TS B1 RTFx|Drafted TS B8 RTFx|B8 paired median|
|---|---:|---:|---:|
|1B English73|+2.71%|+3.66%|+3.71%|
|1B Japanese12|+3.73%|+4.52%|+3.98%|
|102M English73|+3.17%|+5.66%|+9.55%|
|102M Japanese12|+7.63%|+13.74%|+12.34%|

Every paired corpus total improves except102M English: B1 improves8/11 and B8 improves10/11.
Its B8 pairs span−24.14..+13.31%, and whole-list p95 regresses9.73%; retain this outlier rather
than claiming uniform tail-latency improvement. Other B8 p95 improves4.45–17.10%; pooled
mixed-duration B1 p95 improves0.07–5.29%. Numeric waveform equality includes mono/stereo,
16/44.1kHz and PCM16/FLOAT; byte identity for signed zero is not claimed. Mono files bypass
one redundant average; stereo/resampling/nonfile conversion retains its prior implementation.

[Public loader results and all pairs](runtime-audio-api-2776797.json), raw
`outputs/runtime-2776797/*_audio/`; reproduce using `scripts/validate_frontend_api.pbs`.
These are warmed node-local files, not cold network/Lustre measurements or an arbitrary-rate
resampler benchmark. Loaded controls avoid I/O entirely and are not an implemented gain.
Bounded asynchronous file prefetch is the next separate experiment (`runtime-audio-prefetch.md`).
