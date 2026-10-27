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
