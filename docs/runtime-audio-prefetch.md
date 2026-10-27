# Bounded audio prefetch experiment

Candidate40, E/N intent; no speed or adoption claim yet. The earlier mono-file measurements
(`runtime-audio-io.md`) show a substantial file-inclusive gap to resident waveform controls.
This experiment overlaps CPU file decoding/resampling for the next batch with the existing
GPU streaming path. It does not preload timed inputs outside the request.

The private `_audio_prefetch_workers` switch defaults to0. Workers1/2 apply only to an all-file
list that enters drafted greedy streaming. Two batches at most are submitted, including the
current batch; result order and the existing duration-based scheduling are preserved. Each
read owns its file handle. Feature extraction, H2D and model calls remain on the calling thread.
Non-file/mixed inputs retain synchronous loading; CPU autocast also falls back because its
state is thread-local. Early termination or failure cancels queued work and joins running reads.
Speculative future read errors are observed only when that batch is consumed.

Independent file handles are supported by [SoundFile's thread-safety contract](https://github.com/bastibe/python-soundfile/blob/master/README.rst).
[CFFI releases the GIL during C calls](https://cffi.readthedocs.io/en/stable/overview.html),
but this does not establish a speedup or eliminate CPU/memory-bandwidth contention.
The implementation introduces no dependencies and no persistent background service.

18 small fake-loader tests passed on CPU (ordering, two-batch bound, early exit, reader/consumer
errors, worker cleanup, inference mode, lazy inputs and CPU autocast fallback). Eight real
mono/stereo16/44.1kHz decode/resample cases plus the existing nine loader tests run only on a
compute node. `scripts/profile_audio_prefetch.pbs` stages code, environment, files, engines and
logs under `$PBS_LOCALDIR`, then measures1B/102M on English73/Japanese12 with11 alternating
pairs for greedy B1, greedy B8 and timestamped greedy B8 streaming. Every timed mode consumes
file paths and includes executor startup/join, file reads and header/scheduling work. All use
the current public mono loader and frontend graph limit1. OS caches are warm after staging;
these are not cold Lustre/network measurements.

Same engines/checkpoint, output token/time IDs, segments/text, decoder call totals, actual
loader thread dispatch and new-capture/fallback checks are recorded. Changes in read timing
can affect concurrent decoder dispatch, so numerical equality must be measured. All variant
session pools remain resident; reported peak allocation is not standalone deployment memory.
Use per-pair changes and outliers as well as aggregate RTFx. Do not add this gain to mono-loader
or frontend gains without a combined end-to-end measurement.

Reproduction within an allocated compute node:

```sh
python scripts/profile_audio_prefetch.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --lang eng --repeat 11 --out /local/results
```

Only consider default/public exposure after repeatable file-inclusive improvement and separate
public API validation; keep regression cases explicit. Single-request latency, beam/translation,
CTC-only and long-form paths are outside this first experiment.
