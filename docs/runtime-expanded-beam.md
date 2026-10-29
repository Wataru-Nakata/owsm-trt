# Expanded multilingual beam preset timing

This compares the existing public grouped cross-attention, compact-tail and fused-LSE preset against
the optimized default on H200/1B, beam5 with timestamps. Upper-bound stopping and beam kernels are
enabled in both variants. No new decoder kernel or default is introduced. The preset remains N-class:
the [13,232-input B3 audit](runtime-large-regression.md) retains238 changed inputs and mixed language
edit-count changes. Exact output on a timing sample does not establish full-corpus exactness.

All14 conditions and224 unique input IDs are complete across four successful allocations.
Each language has32 deterministic duration/dataset-stratified speech inputs, excluding prior screening
overlap, nonspeech and recordings over30s. The selection is unchanged from the original full-language
attempt. This is neither an independent quality holdout nor a corpus-weighted throughput estimate.

Each condition retains seven alternating default/preset repeats in one successful allocation.
B1 measures individual requests; B3 measures one duration-sorted32-input request, including the final
B2 batch. B3 p50/p95 are therefore whole32-input request latency. Loading, initial warmup/capture and
diagnostic decoder counts are separate from warm wall timing. Repeats are not extra audio inputs.

|Languages|Validated job artifact|Unique inputs|Conditions|
|---|---|---:|---:|
|Spanish, Chinese|[Job2787704](runtime-expanded-beam-preset-2787704.json)|64|4|
|German, French|[Job2792018](runtime-expanded-beam-preset-2792018.json)|64|4|
|English, Japanese|[Job2792040](runtime-expanded-beam-preset-2792040.json)|64|4|
|Russian|[Job2792068](runtime-expanded-beam-preset-2792068.json)|32|2|

Every listed job completed PBS F/exit0 and publication exit0. Stageout_status1 is recorded separately;
required artifacts were verified. Frozen3bb1362 has the same runtime tree0852 as113ee39. Each B1 default
output matches its saved full-corpus default control. Across jobs, checkpoint/engines remain identical;
the physical allocation and frontend/cache history can differ. Paired gains compare variants within
a condition, not unrelated jobs, and are not multiplied by earlier optimization percentages.

## Warm paired results

|Language / request|RTFx change|Default → preset p50 ms|Default → preset p95 ms|Positive repeat totals|
|---|---:|---:|---:|---:|
|German / B1 individual|+8.87%|33.359 →30.732|62.832 →57.605|7/7|
|German / B3 whole32|+22.92%|809.628 →656.739|813.945 →662.998|7/7|
|English / B1 individual|+9.16%|31.993 →29.335|82.337 →74.833|7/7|
|English / B3 whole32|+21.30%|762.562 →629.484|767.509 →629.848|7/7|
|French / B1 individual|+8.94%|38.290 →35.162|99.832 →91.149|7/7|
|French / B3 whole32|+23.60%|956.338 →774.442|960.872 →775.244|7/7|
|Japanese / B1 individual|+7.64%|28.740 →26.723|53.869 →49.736|7/7|
|Japanese / B3 whole32|+23.36%|687.770 →557.836|690.911 →558.840|7/7|
|Russian / B1 individual|+9.28%|42.587 →38.958|108.704 →98.897|7/7|
|Russian / B3 whole32|+22.62%|1071.791 →873.375|1075.842 →879.021|7/7|
|Spanish / B1 individual|+9.26%|43.597 →40.083|96.866 →88.250|7/7|
|Spanish / B3 whole32|+24.22%|1064.957 →857.295|1067.735 →860.149|7/7|
|Chinese / B1 individual|+8.80%|31.732 →29.214|78.485 →71.653|7/7|
|Chinese / B3 whole32|+21.90%|787.023 →645.328|788.086 →647.178|7/7|

98/98 paired repeat totals improve in the completed conditions. All samples and outliers
are retained. RTFx changes use pooled warm wall time. B1 percentiles pool32 inputs ×7 repeats, so
their spread includes duration variation and does not describe224 independent inputs per language.

## Output and decoder checks

|Changed condition|Token sequences|Timestamp sequences|Texts|Default → preset edit count / reference units|
|---|---:|---:|---:|---:|
|deu/beam_stream_ts_b3|2|0|2|33 →31 /575 (WER)|
|eng/beam_stream_ts_b3|1|1|0|49 →49 /764 (WER)|
|rus/beam_stream_ts_b3|2|1|1|68 →67 /648 (WER)|

All unlisted conditions have identical full tokens, timestamp sequences, text and segments.
Differences are retained in each raw `profile/LANGUAGE/CASE/tokens.json`, with file hashes and quality
counts in the linked job artifacts. No human timestamp references are available: generated boundary
agreement is not timestamp accuracy. These sampled changes do not reverse the full-corpus N findings.

|Language / request|Default → preset post-prefill decoder replays|Prefill selections per diagnostic pass|
|---|---:|---:|
|deu/beam_single_ts|1140 →1140|32|
|deu/beam_stream_ts_b3|494 →494|11|
|eng/beam_single_ts|1108 →1108|32|
|eng/beam_stream_ts_b3|470 →470|11|
|fra/beam_single_ts|1356 →1356|32|
|fra/beam_stream_ts_b3|606 →606|11|
|jpn/beam_single_ts|920 →920|32|
|jpn/beam_stream_ts_b3|417 →417|11|
|rus/beam_single_ts|1575 →1575|32|
|rus/beam_stream_ts_b3|688 →688|11|
|spa/beam_single_ts|1564 →1564|32|
|spa/beam_stream_ts_b3|672 →672|11|
|zho/beam_single_ts|1116 →1116|32|
|zho/beam_stream_ts_b3|478 →478|11|

Counts come from an untimed diagnostic pass and exclude selection0 following prefill. Zero draft-call
counts in beam wall records are not beam decoder counts. Effective routes are Gluon for both B1
variants and TensorRT→Gluon at B2/B3. Tail compaction is inactive at B1; B2/B3 retain additional tail
graphs and last-request transition metadata. These are not kernel timelines or per-phase GPU timings.

Per-job model loading and per-condition cold warmup/capture remain in the artifacts, outside warm
latency. Both variant pools are resident, so allocator peaks describe the combined experiment, not
standalone deployment memory. Five-second GPU utilization samples span all phases and variants.

## Reproduction

Run only inside a compute allocation with staged model, engines and corpus:

```bash
python scripts/profile_expanded_greedy.py --model MODEL --data LOCAL_CORPUS \
  --out LOCAL_RESULTS --comparison beam_preset --languages spa zho --repeat 7 --per-language 32
python scripts/summarize_expanded_greedy.py outputs/runtime-JOB \
  --out docs/runtime-expanded-beam-preset-JOB.json
```

Use the other subsets `deu fra`, `eng jpn`, and `rus` analogously. The PBS driver accepts colon-separated
`PROFILE_LANGUAGES`; omission preserves full-language selection. Twenty-four small metadata/aggregation
fixtures pass. The temporary source worktree was removed after immutable snapshots were saved.

Jobs2787353 and2787385 were quota-interrupted with11/14 and9/14 completed condition records. Their raw
samples remain preserved and are excluded from the successful subset results. No per-condition timing
samples are spliced across jobs. The final collection retains per-job provenance and does not report a
global multi-job speedup or a new production default.
