# Beam kernel fusion follow-up

Reference: `090886a`, including the validated strict score bound and B=1 host checks every step.
Work branch: `beam-search-fusion`; the previous `beam-timestamp-profile` branch is preserved.

## Changes prepared for measurement

* Candidate 02: a Triton kernel updates best finished hypotheses, finished-length history, end_detect,
  the strict score bound, active/done rows, decoder input tokens and optional diagnostic history.
  One CTA owns an utterance, including all its beam slots. Equal maxima keep the first slot and best
  finished scores update only on strict improvement. The reference PyTorch implementation remains.
* Candidate 06 / the score-addition portion of 07: a blockwise kernel evaluates timestamp constraints
  from the small per-row state, adds parent scores and invalidates forbidden/non-finite candidates.
  It records active NaN/+Inf inputs separately; all-negative-infinity rows remain exhaustion.
  Full-vocabulary **PyTorch log-softmax and global top-K are unchanged**, preserving their reduction
  and tie behavior. This is not fused logsumexp, a changed normalizer or a hierarchical top-K.

Both are opt-in `BeamSession` flags (`fused_rank`, `fused_finish`), with PyTorch fallback on CPU or
without Triton. There is no new default speed claim before the GPU comparison. The profiler accepts
`--fused-rank` / `--fused-finish` and records actual enabled flags in `cold.json`.

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

## Prepared GPU experiment

The requested allocation is one additional job: `gai51740`, reservation `R9920261300`, `rt_HG`,
one node, one GPU, maximum one hour. Prior job approvals are consumed; this job needs explicit
approval under the AGENTS.md supplied in the conversation. No submission has occurred yet.

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
guarantee. Only measured improvements with full token parity will become defaults. GPU results and
adoption decisions remain pending.
