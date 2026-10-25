# Beam kernel fusion follow-up

Reference: `090886a`, including the validated strict score bound and B=1 host checks every step.
Work branch: `beam-search-fusion`; the previous `beam-timestamp-profile` branch is preserved.

## Adopted changes

* Candidate 02: a Triton kernel updates best finished hypotheses, finished-length history, end_detect,
  the strict score bound, active/done rows, decoder input tokens and optional diagnostic history.
  One CTA owns an utterance, including all its beam slots. Equal maxima keep the first slot and best
  finished scores update only on strict improvement. The reference PyTorch implementation remains.
* Candidate 06 / the score-addition portion of 07: a blockwise kernel evaluates timestamp constraints
  from the small per-row state, adds parent scores and invalidates forbidden/non-finite candidates.
  It records active NaN/+Inf inputs separately; all-negative-infinity rows remain exhaustion.
  Full-vocabulary **PyTorch log-softmax and global top-K are unchanged**, preserving their reduction
  and tie behavior. This is not fused logsumexp, a changed normalizer or a hierarchical top-K.

Both kernels are enabled by default following the GPU evaluation below (adoption commit `4695e0b`).
`OWSM.from_pretrained(..., beam_kernels=False)` selects the reference operations while keeping the
strict score bound. Low-level `BeamSession` flags `fused_rank` / `fused_finish` independently control
the kernels. CPU or missing-Triton paths fall back to PyTorch; no GPU JIT-failure fallback is claimed.
The profiler accepts `--fused-rank` / `--fused-finish` and records actual enabled flags in `cold.json`;
without those flags it explicitly selects the reference operations for ablations.

## Verification before submission

The existing synthetic tests plus the new small kernel tests pass with **63 passed, 4 CUDA-only
skipped** under `TRITON_INTERPRET=1`. Tests compare full candidate tensors and finished states with
zero tolerance. Cases cover timestamp states, block boundaries, ties, invalid slots, active/inactive
NaN/+Inf, all -inf, no EOS, all EOS, strict-bound toggling, forced EOS, partial completion, diagnostic
history and end_detect. GPU-only tests additionally replay graphs across cache buckets.
The interpreter reports four NumPy warnings for the reference-compatible `-inf - -inf` comparisons;
these compare false and state parity passes. This is a correctness sanity check, not a GPU benchmark.
Ruff, PBS shell syntax and diff-whitespace checks pass.

```bash
# Small synthetic states only; no audio/model inference on a login node.
TRITON_INTERPRET=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  uv run pytest -q -p no:cacheprovider tests/test_beam.py tests/test_beam_kernels.py
```

## GPU experiment

The allocation is `gai51740`, reservation `R9920261300`, `rt_HG`, one node, one GPU, maximum one hour.
The user approved this experiment, then explicitly authorized subsequent one-GPU work without
repeated confirmation when it fits within the account's 192-GPU limit. This supersedes the earlier
per-job confirmation requirement for this scope. Keep this work to one active GPU job at a time.
Job `2766606.pbs1` was submitted on 2026-10-25 at 18:55:56 UTC after a fresh check found
184 allocated + zero queued + this GPU = 185. The scheduler confirmed the requested settings.
The job completed with exit status 0 after **14 min 25 s**. The final observation found 176 allocated
and zero queued GPUs; no over-capacity event was observed during this job. Monitoring stopped after
verified completion. A transient capacity-query timeout before submission caused no qsub; the
collector now uses PBS's documented `qstat -E` batching. The user's standing one-GPU authorization
remains recorded for subsequent work within the account gate.

The immutable snapshot contains `090886a:src/owsm_trt/runtime.py` as `baseline_runtime.py`.
`EXPERIMENT=fusion` selects the added branch in `scripts/profile_beam.pbs`. It stages the source,
environment, existing 1B fp16 checkpoint/engines and inputs on `$PBS_LOCALDIR`, then:

1. Runs beam, kernel, model and long-form tests inside the compute allocation.
2. Benchmarks finish only, rank only and both, each against the previous optimized runtime on the
   **same loaded engines**. Every run alternates baseline/changed order over five repetitions.
3. Uses all 73 English inputs at B=1/chunk=1 and B=3/chunk=4, warming the exact measured groups.
4. Checks combined kernels on 12 Japanese inputs at both settings over three repetitions.
5. Collects separate eager traces, disk-inclusive timing and English extra/long-form checks for the
   combined variant. Saves tokens, timestamps, segments, quality, diagnostics, environment and hashes.

Example inside the allocation (repeat for the other variant/batch settings):

```bash
uv run python scripts/profile_beam.py \
  --model /local/model --manifest /local/manifest.json \
  --beam 5 --timestamps --batch 1 --chunk 1 --repeat 5 --limit 73 \
  --upper-bound on --baseline-runtime /local/baseline_runtime.py \
  --fused-rank --fused-finish --phases --trace --out /local/results/both_b1
```

Submission requires a fresh successful queue/capacity check and no duplicate beam job. Account
allocated plus eligible queued requests plus this GPU must not exceed 192. Held jobs are not treated
as immediately runnable; concurrent changes by other submitters still prevent an absolute account-wide
guarantee. Only measured improvements with full token parity become defaults.

## Measured results and adoption

Raw results are in `outputs/beam-2766606/`; the [compact result record](beam-results-2766606.json)
contains hashes, full timing statistics, within-input variability, cold setup, dispatch/captures,
quality and resource observations. Snapshot `27ac5d5` used explicit opt-in flags. No timed encoder
captures occurred and no repetitions were excluded. All runs used the same checkpoint/engine hashes
as the preceding evaluations. The table compares the already optimized `090886a` reference with both
kernels enabled, within each run; it does not compare unrelated runs or add isolated speedups.

| Dataset | Batch | Call p50, before → after (ms) | Call p95, before → after (ms) | RTFx, before → after | RTFx gain |
|---|---:|---:|---:|---:|---:|
| English | 1 | 33.68 → 28.01 | 72.19 → 59.71 | 174.73 → 210.27 | 20.34% |
| English | 3 | 62.66 → 54.90 | 130.91 → 114.16 | 248.24 → 283.56 | 14.23% |
| Japanese | 1 | 46.90 → 38.42 | 109.39 → 91.48 | 241.64 → 295.46 | 22.27% |
| Japanese | 3 | 95.93 → 83.96 | 163.80 → 143.42 | 370.15 → 422.89 | 14.25% |

English has five repetitions: 365 calls per B=1 mode and 125 calls per B=3 mode. B=3 includes a
single-input tail after 24 full groups per repeat; every call in that run explicitly uses chunk 4.
B=1 uses chunk 1. Japanese has three repetitions (36 B=1 / 12 B=3 calls per mode) at the same intervals.
Latencies are batch-call wall times on loaded audio, including feature/H2D, encoder, prefill, search,
synchronization and output formatting. Disk-inclusive and eager diagnostic measurements are separate.

Independent English ablations, each paired with the reference in its own process:

| Kernels | Batch | RTFx gain | Median paired group speed gain | Faster input groups |
|---|---:|---:|---:|---:|
| Finished-state only | 1 | 10.34% | 10.26% | 73/73 |
| Finished-state only | 3 | 7.12% | 7.09% | 25/25 |
| Candidate mask/score only | 1 | 7.38% | 7.13% | 73/73 |
| Candidate mask/score only | 3 | 5.86% | 5.88% | 25/25 |
| Both | 1 | 20.34% | 20.33% | 73/73 |
| Both | 3 | 14.23% | 14.33% | 25/25 |

Every English input group improves in every isolated ablation. With both kernels, median within-group
transcribe CV is 0.68% → 0.49% at B=1 and 0.35% → 0.33% at B=3, much smaller than the measured gains.
Japanese improves for 12/12 B=1 and 4/4 B=3 groups. Japanese B=1 has one reference group with CV 17.50%;
raw samples remain included. Its paired median gain is 19.59%, versus the aggregate RTFx gain of 22.27%.
No speed or numerical claim is made for unmeasured GPUs, models, beam widths or larger batches.

All variants match **all token IDs, timestamp IDs, text and segment boundaries**: English 73/73,
Japanese 12/12 at both batch sizes. Decoder call counts match the reference for every group and repeat.
The speed gain comes from work per step. Median decode milliseconds per selection changes from
1.373 → 1.114 for English B=1 and 1.880 → 1.616 for English B=3; this metric includes prefill/host work
amortized over selections and is not decoder-kernel-only time. WER stays 45/1150 = 3.91%; Japanese CER
stays 26/592 = 4.39%. The English set produces 74 segments, Japanese 12; both have zero empty/invalid
segments, unpaired timestamps and nonmonotonic times under the profiler's checks. Reference-time
accuracy was not evaluated. The six English extra cases also match for baseline, bound OFF and ON.

The allocated-job suite passes **145 tests in 48.15 s**, including CUDA candidate/state comparisons
with zero tolerance and graph replay across buckets. After enabling defaults and adding API forwarding,
the CPU/interpreter sanity suite passes **65 tests, 4 CUDA-only skipped**. That sanity environment uses
Python 3.12 / torch 2.8.0, below the project's declared minimum; it is not a replacement for the allocated
GPU suite (torch 2.14.1, Triton 3.8.0, TensorRT 11.4.0.106, CUDA 13.0, H200 with 132 SMs).
The GPU snapshot precedes the default/API-routing change; no second GPU run is claimed for that change.

The separate eager diagnostic trace sums finish/history kernels to 0.053 ms at B=1 and 0.102 ms at B=3,
selection to 2.248 / 4.672 ms, reorder to 1.685 / 8.104 ms and decoder to 18.688 / 48.374 ms. These are
instrumented eager sums, not graph-phase measurements. About 77–78% of these kernel sums now belongs
to the decoder; there is no claimed decoder-kernel improvement. Five-row calls still dispatch to Gluon,
fifteen-row calls to TensorRT under the existing row limit.

The rank kernel adds a persistent candidate buffer of about 0.95 MiB for five rows or 2.86 MiB for
fifteen rows, plus small block-error flags. Comparison sessions remain resident during measurement,
so peak allocated memory (English about 6.29 GiB at B=1 and 9.25 GiB at B=3) is not isolated per-mode
memory. The first fixed B=1 session setup/JIT/capture takes 8.34 s; subsequent variants reuse compiler
caches, and the reference is constructed afterwards. These cold times are not a fair cold-start ablation.
Whole-job GPU utilization averages 43.67% across five-second samples, including setup, tests, compilation
and all modes; per-mode utilization remains unavailable.

Candidates **02 and 06 are adopted**. Candidate 07 is implemented only for mask/score addition;
full-vocabulary logsumexp remains PyTorch's and has not been replaced. Further decoder, cache or
approximate-search changes are outside the measured adoption in this follow-up.
