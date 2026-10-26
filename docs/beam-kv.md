# Beam KV reorder experiments

Reference: `4b3b581`, with strict stopping, B=1 completion checks and fused candidate/finished-state
kernels enabled. Work branch: `beam-kv-reorder`; prior branches are preserved.

Candidate 14 is compared in two parts. `BeamSession(..., kv_reorder="full")` replaces four PyTorch
gather/copy launches for K/V with two Triton launches, preserving the whole cache. `"prefix"` gathers
only positions below the device-side `pos`, then copies those positions back and zeros the unused tail.
The zeroing matters because a masked attention weight of zero multiplied by a stale NaN value is NaN.
The selected token at `pos` has no cache entry yet. Neither mode reads `pos` on the host.

All parent reads finish into separate scratch before copy-back, so duplicate parents and cycles are
safe. One fixed workspace sized for the largest bucket is reused by every graph; no address swapping,
extra parity graphs or cross-beam KV sharing is introduced. The PyTorch reference remains selectable
as `"torch"`; `"prefix"` is the CUDA default after the measurement below. CPU or missing-Triton execution falls back to it.

Before GPU measurement, 71 small interpreter tests pass and two real-decoder/CUDA graph tests are
skipped. Coverage includes identity, all-identical/duplicated/permuted parents, fp16/fp32, empty/full
prefixes, tile boundaries, repeated workspace use, NaN tails and fallback/invalid options.
The GPU tests also compare exact decoder logits, tokens and diagnostic state across bucket growth
and repeated decodes with different primer lengths. No CPU inference or audio processing runs on login.

The prepared `EXPERIMENT=kv` PBS run compares both modes against the already optimized reference,
on the same loaded 1B fp16 engines in each run. English 73 inputs use five repetitions; Japanese 12
use three. B=1 uses chunk 1; B=3 uses chunk 4. Both variants get English extra/long-form checks,
separate eager traces and disk-inclusive timing. Coarse phase updates go to one shared `progress.log`;
all detailed logs, temporary data and caches stay in `$PBS_LOCALDIR` until publication.

Standing authorization covers serial one-GPU jobs with `gai51740`, `R9920261300`, `rt_HG`, one node,
maximum one hour each. A fresh successful query must confirm allocated plus eligible queued GPUs,
including this job, are at most 192, with no duplicate. Queue validity is checked again before qsub.
GPU job `2768878.pbs1` completed in 9m44s, with 220 tests passing (including both real-decoder
CUDA graph tests). The admission audit counted 184 allocated + 0 queued + 1 new = 185 GPUs;
the final audit counted 176 allocated, 0 queued. No capacity overage was observed.

## Measured results

H200, same loaded fp16 1B checkpoint/engines, K=5, timestamps enabled. Latencies below are warmed
transcribe calls; B=3 English includes one final B=1 group. Numbers compare each variant against
its own alternating reference, not separate runs against each other.

| Variant/input | B | Reference p50 → candidate (ms) | Reference p95 → candidate (ms) | RTFx gain |
|---|---:|---:|---:|---:|
| full/English | 1 | 27.57 → 26.71 | 59.08 → 57.15 | 3.12% |
| full/English | 3 | 53.74 → 50.44 | 112.93 → 105.15 | 7.60% |
| prefix/English | 1 | 27.73 → 26.62 | 59.23 → 57.02 | 3.91% |
| prefix/English | 3 | 53.69 → 49.45 | 112.85 → 103.59 | 8.71% |
| prefix/Japanese | 1 | 37.83 → 36.25 | 90.52 → 86.11 | 4.53% |
| prefix/Japanese | 3 | 83.28 → 76.25 | 142.21 → 128.82 | 9.65% |

All groups improved by paired median. For prefix, median within-group CV is 0.08–0.37% for English,
0.07–0.36% for Japanese; paired median speedups are 3.84% / 8.50% English, 4.30% / 8.92% Japanese.
Every tested token, timestamp, segment, transcript and decoder call count matches. English WER
remains 45/1150 (3.91%); Japanese CER remains 26/592 (4.39%). Both variants also match on the six
English extra cases, including 30s repeated speech and 110s audio. Reference timestamp errors remain
unmeasured: these are baseline agreement, not annotated timestamp accuracy.

`prefix` is adopted; `full` remains available for ablation. `beam_kernels=False` retains the PyTorch
reference. The fixed scratch stores two largest-cache tensors; the benchmark's reported peak VRAM
(~6.94 GB B=1, ~10.69 GB English B=3) includes both resident sessions, so it does not measure isolated
per-variant VRAM. Cold/JIT/capture costs, hashes and complete statistics are in
[the compact result](beam-results-2768878.json); raw output is `outputs/beam-2768878/`.

The eager instrumented prefix trace reports decoder/reorder kernel sums of 18.70/0.70 ms for B=1,
48.35/3.12 ms for B=3. This supports investigating decoder execution next; those are not CUDA graph
phase times or end-to-end proportions. Decoder tuning continues as a separate ablation.

Reproduce in an allocation with the same staged checkpoint/engine and a `4b3b581` runtime snapshot:

```sh
python scripts/profile_beam.py --model /local/owsm_v4_medium_1B --manifest /local/manifest.json \
  --beam 5 --timestamps --batch 1 --chunk 1 --repeat 5 --upper-bound on \
  --fused-rank --fused-finish --kv-reorder prefix --baseline-runtime /local/baseline_runtime.py \
  --phases --trace --disk --extra-checks --long-audio /local/long_110s.wav --out /local/results
```

For B=3 use `--batch 3 --chunk 4`; Japanese uses `--lang jpn --repeat 3` and its manifest.
