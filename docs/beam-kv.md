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
as `"torch"`, initially the default. CPU or missing-Triton execution falls back to it.

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
GPU performance and adoption are pending. The broader optimization continues after this ablation.
