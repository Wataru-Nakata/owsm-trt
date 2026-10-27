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

## Public opt-in prepared

`OWSM.from_pretrained(..., audio_prefetch_workers=1)` (or2) now passes the validated count through
the public constructor; default0 is unchanged. Non-integer/bool/out-of-range values fail before model
loading.13 new option checks and the18 fake-loader tests pass on CPU. The independent compute-node
runner `validate_audio_prefetch_api.pbs` checks48 focused audio cases, then each1B/102M public worker
setting on15 API cases: file B1/B8 with/without timestamps, mixed/loaded inputs, plain/beam/translation,
long prompts, CPU autocast, empty input, silence/noise, stereo44.1kHz, an injected read error with
worker cleanup and subsequent reuse, and110s long form. Loader thread evidence verifies eligible
and excluded paths. It repeats the original same-engine file-inclusive timing ablation with11 pairs.
Full `from_pretrained` API equivalence and same-engine timing are separate passes; public validation
is pending. No serving-wide default or workload auto-detection is introduced.

The same allocation runs the separate LSE numerical/graph diagnostic after audio artifacts exist;
it does not enable LSE in these audio benchmarks. Failure of that diagnostic is reported separately.

## Partial public result2776966:102M CUDA failure, validation not complete

Frozen3a99ff2 exited139 after48 audio tests passed.1B completed all15 public API cases for workers0/1/2
with exact outputs and reader cleanup/reuse, then both file-inclusive corpus measurements. English
B1 workers1/2 improved8.65/7.10%, greedy B8 improved28.25/28.06%, and timestamp B8 improved26.34/25.95%;
all11 pairs improve. This reproduces a sizeable English gain with a different magnitude from the first
run. Japanese B8 regresses1.24–2.23%; timestamp workers2 is slower in every pair. Retain default0.
English inputs are73 FLAC files; Japanese inputs are12 WAV files. This difference is not evidence for
language-specific selection or a universal codec policy.

102M saved worker0/1 outputs, identical on all15 cases, then failed during worker2. CUDA error700 was
reported while creating a TensorRT encoder context, surfacing through the serial `_decode` path;
`create_execution_context` returnedNone, followed by an AttributeError and native teardown segfault.
An earlier asynchronous failure remains possible; no root cause is established. The original log is
`outputs/runtime-2776966/102m_api.log` (traceback at265 onward). There was no case-level checkpoint in
that validator, so the exact triggering API case is not known.102M timing and the final LSE diagnostic
never ran. Do not report this allocation as a successful public verification or adopt the option broadly.

[Validated partial1B timings and API status](runtime-audio-prefetch-partial-2776966.json). The next
`diagnose_audio_prefetch_api.pbs` retains the48 passed tests by hash, runs independent LSE checks in
separate processes, then tests102M worker2 with CUDA_LAUNCH_BLOCKING=1 and explicit case synchronization
against saved worker0 output. It follows with the original asynchronous0/1/2 sequence and the missing
102M timing if those checks succeed. Case start/end and partial outputs are saved locally. Debug sync
is correctness diagnosis only, never timing evidence. Runtime computation is unchanged; a small guard
now reports failed TensorRT context creation directly instead of masking it with AttributeError. This
guard is not a fix for CUDA error700. Public prefetch remains opt-in/default0 and pending validation.

## Diagnostic2777145: asynchronous failure reproduced at long_prompt

Frozenaa586c8, F/exit1. Fresh102M worker2 with CUDA_LAUNCH_BLOCKING and case synchronization
passes all15 API cases against the saved worker0 output. The ordinary asynchronous0/1/2 sequence
again passes workers0/1 and worker2 through translation8, then fails at long_prompt. The traceback in
`outputs/runtime-2777145/102m_async_api.log` identifies the encoder's profile0/lane0 context creation,
with preceding CUDA700; the new guard reports this directly. The root cause is still undetermined,
and the synchronous pass does not establish asynchronous safety. Missing102M timings remain absent.

`diagnose_prefetch_cuda_matrix.pbs` now isolates: fresh worker2 without blocking, three worker0 model
lifecycles,0/1/2 with legacy frontend warm-up,0/1/2 with eager frontend, and0/1/2 with case-only
synchronization (no global launch blocking). Each process records its own exit and partial case;
expected reproductions do not prevent other independent controls. A separate Compute Sanitizer
memcheck run uses stream-ordered race tracking, following the [NVIDIA documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
Installed diagnostic tool:2025.3.1.0 from CUDA13.0.1. Sanitizer and synchronization change scheduling;
none of these passes is speed evidence. Runtime and defaults are unchanged by this matrix.
