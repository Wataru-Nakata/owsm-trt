# Public beam preset composition (pending measurement)

`scripts/profile_beam_presets.py` compares four already implemented settings through the public
session factory, with unchanged checkpoint/engines and resident audio:

|Variant|beam_grouped_cross / beam_compact_tail|beam_fused_lse|
|---|---|---|
|reference|False / False|False|
|lse|False / False|True|
|grouped_tail|True / True|False|
|combined|True / True|True|

This H200/1B experiment treats the previously verified grouped-row/tail preset as one factor and
full-vocabulary LSE as the other. It introduces no new runtime kernels or defaults. The decoder
profile still dominates the1B samples; the existing grouped preset addresses that path. Actual
combined gains must be measured, not obtained by multiplying earlier independent percentages.
Both factors are N-class: full tokens, timestamp IDs, segments, text, invalid/empty/monotonic segment
indicators and WER/CER are retained against the default. Reference time annotations are unavailable.

B1 reports separate timestamped utterance latencies. B2/B3 report whole corpus streaming requests,
including fixed beam batches and the last partial batch, not single-utterance p95. Eleven repetitions
alternate forward/reverse mode order after every used shape is warmed. All four session pools stay
resident; peak memory is therefore a combined benchmark footprint, not deployment memory per mode.
New captures during steady timing and failed decoder trials are errors. Cold warmup, dispatch,
normalizer flags, host check intervals and the last request's tail transition are separate artifacts.

Extra no-timestamp, translation, long prompt, silence/noise, repeated30s speech and110s checks record
all output differences outside timed samples. `profile_beam_presets.pbs` stages code, environment,
models, engines and input audio on `$PBS_LOCALDIR`, publishes results at exit, and requires the
successful public-LSE verification artifacts in its frozen snapshot. No performance result is yet
available. Existing defaults remain unchanged; do not describe these settings as an exact mode.
