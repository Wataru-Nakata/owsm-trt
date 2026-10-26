# Wider-row Gluon experiment

R=10/15 (B=2/3, K5) currently dispatches to TensorRT. Prior R16 Gluon measurements in the
source were slower than TensorRT; changing the limit alone is not assumed to help. Recent
shared cross-KV and grouped-query work was measured only at R5. This isolated N-class
experiment checks whether their memory reuse makes R10/15 competitive.

`profile_beam.py --gluon-max-rows 16` creates a separate candidate backend; ordinary routing
remains R<=8 and <=16 tokens. Existing Gluon tiles support up to32 tokens; this experiment
uses10/15, padded MT16, HL2. GEMM units/persistent CTA count, resident-grid requirements and
self-KV layout do not change. Cross-attention loops cover more than one head unit per CTA
when needed. Tests cover distinct utterance KV, offset0 and offsets through70 in capacity128,
fp32 decoder logits/new-KV, exact shared/expanded results, graph replay and changed inputs.

The allocation first runs expanded GPU tests and real-checkpoint fp32 checks at R10/15.
It compares three isolated candidate wrappers with the current TensorRT reference: shared
independent queries at automatic splits, grouped queries with four slices, and eight slices.
English73/Japanese12, B2/B3, seven alternating repetitions, K5/timestamps; all actual dispatch,
full token/time/segment output, WER/CER, diagnostics, cold setup and uninstrumented wall
measurements are saved. Different floating-point arithmetic is N-class and may change outputs.
There are no reference timestamp annotations; differences are agreement, not correctness errors.

The same allocation independently repeats selective checkpoint mapping (three pairs/model)
and includes an unmodified production-settings worker for1B/102M, recording actual torch.load
flags. Startup validation and decoder performance are separate. No runtime routing limit is
changed by the experiment. The first completed comparison is below; production routing remains unchanged pending selection and public API validation.

Reproduce inside an allocation with `scripts/profile_gluon_rows.pbs`; prepared code/engines/audio
are staged locally and only coarse progress/final artifacts are published. Submission uses the
standing serial1 GPU authorization and a fresh account-capacity/queue/duplicate check.

## First validation attempt:2773118

Source80fd2e8 stopped after14passing tests at shared/expanded graph-replay comparison B2/K3
(capacity128): one row's logits differed. No load/beam benchmark ran. Inspection found that the
test retained graph/input/output tensors but not both closures owning externally allocated
workspaces. CUDA graph replay requires those addresses to remain valid. The test now retains
both owners; all exact comparisons remain. This is a suspected cause pending GPU revalidation,
not an assertion that the kernel is correct. Raw failure: `outputs/runtime-2773118/tests.log`.

## Completed comparison:2773135

Source9ad235c, H200, same1B checkpoint/engines.39 tests pass; retaining both test workspace
owners resolves the prior exact graph-replay failure without kernel changes or relaxed assertions.
All12 real-weight B2/B3 checks pass. No trial fallback or diagnostic numerical error occurs.
Seven alternating repeats per dataset/shape; English73 and Japanese12; B2/B3 call latency is
per batch, and RTFx is aggregate audio throughput. Partial final English batches are included.

|Candidate|B2 English RTFx gain|B2 Japanese|B3 English|B3 Japanese|
|---|---:|---:|---:|---:|
|shared, independent queries|5.85%|5.74%|3.55%|3.66%|
|grouped, four key slices|19.16%|19.99%|10.91%|11.16%|
|grouped, eight key slices|13.56%|14.44%|7.58%|8.07%|

Grouped4 improves every paired group's median. Median within-group CV is0.11–0.46% for that
comparison. English B2 p50 is45.38→38.04ms, B3 48.05→43.04ms; Japanese70.49→58.36ms and
75.17→67.05ms. Decoder-call counts match exactly, so the gain is not fewer search steps.
The reference uses TensorRT at R10/R15 and the candidate uses Gluon at every cache bucket.

All text matches (73/73 English,12/12 Japanese), with unchanged English WER45/1150 and Japanese
CER26/592. All Japanese tokens/times/segments match. Every tested wider-row variant differs
from the reference in the same two English endpoint tokens:10.48→10.50s and5.80→5.82s.
Thus71/73 English full token/segment sequences match. No new invalid/nonmonotonic/unpaired
timestamps occur; reference boundary annotations are unavailable. This is an N-class candidate,
not an exact-output default replacement.

B3 distinct cross-KV storage is496,336,896→82,722,816 bytes (about496→83MB). There remain four
decoder graphs, with R15 replacing the TRT closures. Initial compilation/capture remains
material: the English grouped4 R15 session reported13.33s including newly compiled shapes,
versus0.78s reference; this is excluded from steady-state timing and is not pure capture cost.
Combined-variant peak allocator figures do not isolate process-memory savings.

An eager/instrumented example still attributes47.68ms to44 decoder calls versus5.68ms selection
and7.11ms encoder. These are not internal CUDA-graph timings; missing graph phase attribution
remains explicit. Candidate35 encoder overlap and selection work warrant separate comparisons.

[Compact results with hashes/dispatch/quality](beam-wider-results-2773135.json). Full outputs,
wall samples, diagnostics and phase records: `outputs/runtime-2773135/`. Reproduce using
`scripts/profile_gluon_rows.pbs` or its saved immutable snapshot. Job exited0; PBS stageout
status1 was recorded and all expected published artifacts verified. Final capacity176+0/192.

Four slices are fastest among these settings. The next serial allocation measures two slices
(the B3/H16 group has96 attention units at NS2) and decoder initialization independently.
No routing limit or public default has been changed; an eventual fast mode needs public API
validation, including silence/noise and shape/fallback limits.

## Two-slice follow-up (2773320)

A separate seven-repeat, alternating comparison uses source37e9348 and the same1B engine hashes.
All four R10/R15 real-shape checks pass;19 load/storage tests pass. Against each run's unchanged
TensorRT baseline, grouped2 improves RTFx16.91%/18.09% for English/Japanese B2 and16.69%/17.11%
for B3. English p50 B3 is48.28→41.28ms; Japanese is74.70→63.09ms. Decoder-call counts are unchanged.
All text matches, WER45/1150 and CER26/592 are unchanged. English token/time/segments match71/73,
Japanese12/12. The same two English segment ends shift+20ms as grouped4/8; no reference timestamps
are available. This remains an opt-in N result. Two slices are the next B3 candidate; four remain
the next B2 candidate. Cross-job absolute times are not an isolated two-versus-four-slice comparison.

[Compact follow-up metrics](runtime-results-2773320.json); full artifacts and eager phases:
`outputs/runtime-2773320/grouped2_{english,japanese}`. No production routing change in this job.
