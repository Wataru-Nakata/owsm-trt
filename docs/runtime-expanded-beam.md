# Expanded multilingual beam preset timing

This compares the existing public grouped cross-attention, compact-tail and fused-LSE preset against
the optimized default on H200/1B, beam5 with timestamps. Upper-bound stopping and beam kernels are
enabled in both variants. It introduces no new decoder kernels or default changes. The preset is N-class:
the [13,232-input B3 audit](runtime-large-regression.md) retains238 changed inputs and mixed language
edit-count changes. Exact output on a timing sample does not establish exactness on the full corpus.

Seven-language coverage is in progress. The completed job below covers Spanish and Chinese only
(64 unique input IDs, four conditions); the remaining ten language/shape conditions are not yet complete.
The deterministic32-input-per-language sample is the same as the earlier expanded timing selection,
balanced by duration/dataset and excluding prior screening overlap, nonspeech and recordings over30s.
This is not an independent quality holdout or a corpus-weighted throughput estimate.

Each condition has seven alternating reference/candidate repeats after all used shapes are warmed.
B1 measures individual requests; B3 sends32 inputs in one duration-sorted request, including the final
B2 batch. B3 p50/p95 therefore describe the entire32-input request. Audio loading, model loading,
warmup/capture and diagnostic decoder counts are outside the warm timing samples.

## Completed Spanish/Chinese comparison

[Job2787704](runtime-expanded-beam-preset-2787704.json), frozen3bb1362, completed PBS F/exit0 and
publication exit0. Stageout_status1 is recorded separately; all required artifacts were validated.
The executable runtime tree0852 is identical to113ee39. All64 B1 default outputs also match the
saved full-corpus B1 default control, with the same checkpoint/engines and executable runtime.

|Language / request|RTFx change|Default → preset p50 ms|Default → preset p95 ms|Positive repeat totals|
|---|---:|---:|---:|---:|
|Spanish / B1 individual|+9.26%|43.597 →40.083|96.866 →88.250|7/7|
|Spanish / B3 whole32|+24.22%|1064.957 →857.295|1067.735 →860.149|7/7|
|Chinese / B1 individual|+8.80%|31.732 →29.214|78.485 →71.653|7/7|
|Chinese / B3 whole32|+21.90%|787.023 →645.328|788.086 →647.178|7/7|

All four conditions retain identical full tokens, timestamp sequences, text and segments across
variants. All28 paired repeat totals improve; no outliers are removed. These are incremental gains
over the optimized default for this existing preset, not whole-version gains or an additional factor
to multiply by earlier optimization percentages.

|Language / request|Post-prefill decoder replays, both variants|Prefill selections per diagnostic pass|
|---|---:|---:|
|spa / B1|1564|32|
|spa / B3|672|11|
|zho / B1|1116|32|
|zho / B3|478|11|

Decoder counts come from a separate untimed diagnostic pass. They exclude selection0 following
prefill; zero draft-call counts in beam wall records are not beam decoder counts. The effective routes
are Gluon for B1 in both variants and TensorRT→Gluon for B2/B3. The grouped preset uses additional
tail graphs at B2/B3; compact-tail is inactive at B1. Dispatch and last-request tail transitions are
retained in the artifact. They are not a complete kernel timeline or per-phase GPU timing.

Model load in this job is8.296s, excluding environment/corpus staging.
Per-condition cold warmup and capture metadata remain separate. Both variant pools are resident;
allocator peaks are the combined experiment footprint, not standalone deployment memory. Five-second
GPU utilization samples span loading, warmup, timing and diagnostics, not individual variants.

## Reproduction and allocation scope

Run on a compute node with staged model, engines and corpus:

```bash
python scripts/profile_expanded_greedy.py --model MODEL --data LOCAL_CORPUS \
  --out LOCAL_RESULTS --comparison beam_preset --languages spa zho --repeat 7 --per-language 32
python scripts/summarize_expanded_greedy.py outputs/runtime-2787704 \
  --out docs/runtime-expanded-beam-preset-2787704.json
```

The PBS driver accepts`PROFILE_LANGUAGES=spa:zho`. Omitting the subset preserves the full seven-language
selection. Twenty-four small metadata/aggregation fixtures pass. The temporary code worktree used to
freeze the original runtime was removed after snapshot creation.

Short language-subset jobs replace a full14-condition run after two account-cap interruptions.
Jobs2787353 and2787385 ended via quota cancellation with11/14 and9/14 completed condition records;
their raw samples remain preserved but are not used in the completed-job results above. Each new
condition must finish all seven paired repeats in one successful allocation. Across-language coverage
will span separate allocations, with per-job source/GPU/cold/memory provenance. Frontend cache history
can differ between subsets; per-language paired changes must not be presented as one global speedup.
No human timestamp references are available, so generated boundary agreement is not timestamp accuracy.
