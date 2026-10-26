# Combined beam optimization validation

Final branch: `beam-timestamp-optimized`. Original `v0.2` and all intermediate/experimental branches
are preserved. Remote `main` was rechecked on2026-10-26 and still points to892c29fb9b01b22c3e56e6848af818c5470c9288.

The final comparison reference is `4b3b581`: it already includes strict score-bound stopping,
completion-check tuning and candidate/finish kernel fusion. The candidate additionally uses
valid-prefix self-KV gathers and shared cross-KV on supported Gluon shapes. Decoder CTA/prefetch
settings and grouped sibling-query attention were measured but rejected. Grouped-query code is
kept only on `beam-grouped-cross`, not in this final runtime.

The combined one-GPU allocation uses the same loaded1B fp16 checkpoint and engines per comparison,
K5/timestamps, English73 (5 repetitions) and Japanese12 (3 repetitions), B1 thenB3, plus six English
edge/long-form cases. All measured shapes warm up before timing, and baseline/candidate order
alternates. Detailed traces/diagnostics are separate from uninstrumented wall samples. This final
comparison is needed because isolated speedups cannot be added or multiplied into a measured gain.

Standalone correctness coverage:221 base tests plus12 cross-KV/Ring/fallback tests. Public
`beam_kernels=False` selects reference candidate/finish operations, PyTorch KV reordering and
expanded cross-KV; `beam_upper_bound=False` separately disables strict-bound stopping. The scope
of the final ablation is KV changes; it does not re-measure original ESPnet or repeat earlier bound
and search-fusion comparisons. Final results are recorded below.

Resources retain gai51740/R9920261300/rt_HG, one node/one GPU, at most1hour. Standing user approval
allows serial jobs within192 account GPUs; queue, duplicates and allocated+eligible queued+new
capacity are checked immediately before every submission. Temporary code/environment/audio/logs
stay on PBS_LOCALDIR, with compact phase progress and final artifacts on shared storage.

## Final measured result

Job `2769367.pbs1` completed in5m39s, exit0. All221 base tests and12 additional cross-KV/Ring/fallback
tests passed. Software:torch2.14.1, Triton3.8.0, TensorRT11.4.0.106, CUDA13, H200132SM. The checkpoint,
engine and reference-runtime hashes are preserved in [the result manifest](beam-results-2769367.json).
Raw outputs, histories, traces and logs are in `outputs/beam-2769367/`.

| Input | B | Transcribe p50 (ms), reference → final | p95 (ms), reference → final | RTFx, reference → final | RTFx gain |
|---|---:|---:|---:|---:|---:|
| English73 | 1 | 27.74 → 25.43 | 59.49 → 54.42 | 212.54 → 232.19 | 9.25% |
| English73 | 3 | 53.71 → 49.44 | 112.86 → 103.51 | 289.03 → 314.74 | 8.90% |
| Japanese12 | 1 | 37.93 → 34.70 | 90.93 → 82.56 | 299.82 → 328.20 | 9.46% |
| Japanese12 | 3 | 83.36 → 76.13 | 142.26 → 128.78 | 426.80 → 468.19 | 9.70% |

Every group improved by paired median (73/73 EnglishB1,25/25 EnglishB3,12/12 JapaneseB1,4/4 JapaneseB3).
Median within-group CV is0.06–0.47%; these gains exceed measured variation. B3 English includes24
full groups and one final B1 group, under completion-check interval4 for both modes. B1 uses interval1.
Timing is per transcribe call; RTFx sums input duration divided by total warmed wall time. This is
an observed combined gain, not a product of isolated gains or a comparison to original ESPnet.

All token IDs, timestamp IDs, segments, text and decoder call counts match in all four runs.
The final baseline records also match the earlier job2768878 with the same4b3b581 reference.
English WER remains45/1150=3.91%; Japanese CER remains26/592=4.39%. English has74 output segments,
Japanese12; empty, invalid/out-of-input-range, unpaired and nonmonotonic cases are all0 in both modes.
These validity/agreement checks do not measure reference timestamp boundary accuracy; annotated
boundary errors remain unavailable. All18 comparisons for the six English extra cases match:
no timestamps, English-to-English translation, long prompt, silence/noise,30s repeated speech,110s audio.

| Input | B | Decoder calls p50 (unchanged) | Decode wall / selection p50 (ms), reference → final |
|---|---:|---:|---:|
| English | 1 | 21 | 1.100 → 1.003 |
| English | 3 | 28 | 1.589 → 1.447 |
| Japanese | 1 | 30.5 | 1.084 → 0.988 |
| Japanese | 3 | 46 | 1.589 → 1.444 |

The amortized per-selection number includes prefill, selection and host work; it is not decoder
kernel-only time. Prefill contributes one selection without a single-token decoder call.
B1 has0 decoder calls after logical completion. B3 has median6 extra calls per English utterance,
7.5 per Japanese utterance, including waiting for longer batch members; p95 is37.4/46.6 and max84/62.
Completion-check chunk overshoot alone is at most3. Candidate stop masks are
English6 exhausted/67 strict bound and Japanese1 exhausted/11 strict bound. Continuous beam batching
and active-group compaction remain unverified, potentially relevant to these measured tails.

Peak allocated memory with both comparison sessions resident is6.80GB atB1,10.56GB EnglishB3 and
9.45GB JapaneseB3. These are not isolated per-mode memory peaks. Distinct B1 cross-KV storage is
27.57MB versus165.45MB; the prior stage verified that saving directly. Initial EnglishB1 candidate
session construction/JIT/capture took2.66s; reference construction runs later and benefits from
compiled kernels, so cold costs are reported rather than claimed as a fair speedup. Whole-job GPU
utilization averaged33.62% over63 five-second samples (0–100%), including setup/tests/load; per-mode
utilization is unavailable. Eager instrumented decoder/reorder kernel sums are17.70/0.71ms atB1
and48.34/3.13ms atB3; they are not graph-internal phase attribution or end-to-end wall fractions.

B1 uses Gluon with shared cross KV; B3/R15 uses the existing TensorRT shape fallback with expanded
KV. Other models/GPUs, large batches, annotated timestamp quality and language coverage beyond
these manifests are unmeasured. The60-candidate ledger preserves rejected and unverified proposals
without fabricated performance estimates. The admission audit counted176 allocated+0 queued+1new=177;
monitoring observed no192-GPU overage, with184 allocated and0 queued at verified completion.

## Reproduction

Inside an allocated compute job, stage the same environment/model/engine/audio to node-local storage
and save the reference runtime before timing:

```sh
git show 4b3b581:src/owsm_trt/runtime.py > /local/baseline_runtime.py
python scripts/profile_beam.py --model /local/owsm_v4_medium_1B --manifest /local/manifest.json \
  --beam 5 --timestamps --batch 1 --chunk 1 --repeat 5 --upper-bound on \
  --fused-rank --fused-finish --kv-reorder prefix --share-cross-kv \
  --baseline-runtime /local/baseline_runtime.py --phases --trace --disk \
  --extra-checks --long-audio /local/long_110s.wav --out /local/final_b1
```

For B3 use `--batch 3 --chunk 4`; for Japanese use its manifest, `--lang jpn --repeat 3`, and omit
English extra cases. `EXPERIMENT=final` in `scripts/profile_beam.pbs` reproduces the four-run workflow
from an immutable source snapshot; resource and account admission checks are still required.
