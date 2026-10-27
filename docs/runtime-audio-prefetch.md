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

## First result2776872: prepare opt-in, retain default0

Frozen4f207ca, H200, F/exit0,35 tests passed. All12 cases across1B/102M and English73/Japanese12
match waveforms, full tokens/time IDs/text/segments and decoder-call totals. Actual loader-thread
dispatch is verified separately; no warm captures, fallback or leaked worker threads occurred.
Eleven alternating whole-list file-inclusive comparisons:

|Model/corpus|Greedy B1 workers1/2|Greedy B8 workers1/2|Timestamp B8 workers1/2|
|---|---:|---:|---:|
|1B English73|+8.90/+7.26%|+35.10/+33.50%|+37.51/+36.87%|
|1B Japanese12|+0.51/−0.43%|−0.54/−0.88%|−0.58/−1.00%|
|102M English73|+40.19/+43.50%|+25.21/+59.15%|+26.50/+64.29%|
|102M Japanese12|+4.31/+0.69%|+1.71/+0.57%|+0.05/−1.07%|

Every English paired comparison improves.1B timestamp B8 workers1 spans+35.93..+39.25%;102M
workers2 spans+49.07..+79.00%, with paired median+62.71% and whole-list p95−41.06%. Japanese B8
is mostly worse:102M greedy workers1's aggregate+1.71% hides a paired median−1.58% and only2/11
positive pairs, including one+33.68% outlier. Do not label that a repeatable improvement. Different
corpora also have different formats/length mixes; these are not language-specific routing rules.

Prepare a public worker count0/1/2, retaining default0 and all-file/streaming/autocast eligibility.
Thread startup/join stays inside request timing. More workers do not consistently help;1B English
prefers1, while102M English B8 benefits from2. Do not introduce automatic model/language policy
from these small corpora. Independent public API validation and a fresh timing replication remain
required. [All pairs, p95, environments and dispatch](runtime-audio-prefetch-2776872.json); raw
`outputs/runtime-2776872/`. Long-form/single-file/beam/translation remain unchanged and unmeasured
for speed. The following experiment uses a separate GPU job and will not merge gains with frontend
batch capture or change model precision/search.
