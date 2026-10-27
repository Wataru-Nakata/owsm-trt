# Public beam preset composition

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
successful public-LSE verification artifacts in its frozen snapshot. The completed result follows.
Existing defaults remain unchanged; do not describe these settings as an exact mode.


## Measured composition2777500

Frozenddaf677, H200/1B, F/exit0. All six workload/corpus conditions finish11 alternating repeats
without new warm captures, silent fallback or within-mode output changes. No new kernels or
search rules are introduced in this composition benchmark.

|Corpus|Request scope|LSE vs default|Grouped/tail vs default|Combined vs default|Combined vs grouped/tail|
|---|---|---:|---:|---:|---:|
|English73|B1 individual|+1.043%|+6.684%|+7.987%|+1.221%|
|Japanese12|B1 individual|+1.143%|+7.582%|+8.722%|+1.060%|
|English73|B2 whole corpus|+0.784%|+20.667%|+21.846%|+0.977%|
|Japanese12|B2 whole corpus|+0.858%|+25.998%|+27.052%|+0.837%|
|English73|B3 whole corpus|+0.580%|+17.169%|+17.994%|+0.704%|
|Japanese12|B3 whole corpus|+0.772%|+20.543%|+21.031%|+0.405%|

Values are directly measured RTFx changes, not multiplied standalone gains. Combined versus
default improves all11 repeat totals in every condition. The added LSE benefit over grouped/tail
is smaller and includes negative repeats at B3 (English minimum−0.214%, Japanese−0.518%). Earlier
wide-row reports used separately timed fixed groups in manifest order. This public whole-list
benchmark uses the production duration sort, so group composition and useful tail length differ;
its B2/B3 percentages are not direct replications of those earlier group-latency percentages.

|Corpus / scope|Default p50→combined (ms)|Default p95→combined (ms)|Default→combined RTFx|
|---|---:|---:|---:|
|English B1 individual|24.500→22.824|52.134→47.799|240.07→259.25|
|Japanese B1 individual|34.829→32.274|80.701→73.513|328.39→357.03|
|English B2 whole73|1695.867→1392.603|1708.432→1398.304|283.34→345.23|
|Japanese B2 whole12|411.336→324.006|415.029→324.873|382.59→486.09|
|English B3 whole73|1266.708→1076.689|1285.746→1081.079|378.54→446.65|
|Japanese B3 whole12|289.049→238.782|291.619→240.118|544.71→659.27|

All B1 corpus outputs match. At B2/B3, grouped/tail and combined change English utterance
1272-141231-0005's segment end5.80→5.82s; all73 texts and WER45/1150 stay equal. Japanese12 outputs
and CER26/592 match. LSE alone matches all baseline tokens/times/segments and adds no differences
when composed with grouped/tail. English B3 decoder calls increase766→770 for grouped/tail and
combined; all other condition totals match. N-class differences are retained, not hidden by text.

The silence test changes hallucinated text/times under grouped/tail in both languages; combined
matches grouped/tail. Both baseline and changed silence outputs extend beyond the2s synthetic
input. Noise, no-timestamp, translation, long prompt, repeated30s speech and110s cases otherwise
match. These controls have no reference time annotations, so baseline agreement is not timestamp
accuracy. Long-form joined token IDs remain unavailable; only its text/segments are compared.

Actual R10/R15 dispatch changes from TensorRT to guarded Gluon, with shared cross-KV and extra
tail graphs. Cold costs are not independent: English B2 grouped/tail first warmup is49.36s versus
11.04s reference, whereas later combined warmup reuses JIT and takes10.27s. These include a full
corpus request and are not per-utterance startup latency. Four session pools remain resident, so
allocation deltas/peak are not standalone deployment memory. Other GPUs/checkpoints are unmeasured.

The measured options remain individually opt-in/defaultFalse. Example for the verified H200/1B shape:

```python
model = OWSM.from_pretrained(
    "espnet/owsm_v4_medium_1B",
    beam_grouped_cross=True, beam_compact_tail=True, beam_fused_lse=True,
)
results = model.transcribe(audio_files, beam_size=5, batch_size=3, timestamps=True)
```

[All paired results, output differences, quality, cold/dispatch and hashes](beam-presets-results-2777500.json).
Raw full corpus tokens are in `outputs/runtime-2777500/`. Stageout_status1 is recorded separately
from exit0; required artifacts were verified. This loaded-waveform benchmark does not measure file
prefetch, encoder lookahead or colder filesystem behavior.
