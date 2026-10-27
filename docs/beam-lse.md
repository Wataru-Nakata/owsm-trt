# Blockwise full-vocabulary normalization experiment

Candidate07, N class; disabled by default and no high-level API option. The prior eager/instrumented
B3 sample measured selection0.126ms per call versus decoder1.08ms (`beam-local-topk.md`). This
motivates a small isolated reduction/memory-traffic experiment, not a graph-phase attribution or
predicted speedup. Decoder remains dominant and prior local top-K did not yield a meaningful gain.

`BeamSession(fused_lse=True)` computes fp32 block maxima/sums across every vocabulary row, then
combines them in the existing candidate-mask/parent-score kernel. It avoids materializing the
full log-probability tensor. The same global torch.topk, EOS placement, beam slots, strict score
bound, end_detect, cache handling and history updates remain. Timestamp masks are applied after
the full-vocabulary normalizer, including logits of disallowed tokens. A new reduction order and
Triton exp/log change rounding; token or timestamp near ties may differ. This is not an exact-output
default, even if a test set happens to match. NaN/+Inf on live parents must reject the entire row and
set the existing numerical error flag; all-Inf negative rows are exhausted without an error. Dead
parents and finished groups cannot be revived. No per-step host synchronization is added.

Scratch is `[B*K,2,ceil(V/1024)]` fp32 (about5.7KiB atB3/K5/V50002). Each vocabulary block merges
the small partial table; this redundant merge trades arithmetic for removing a full intermediate
write/read. It may be slower than torch.log_softmax. Shared-prefill support is tested independently
but is not enabled in the main ablation. Unsupported CPU/Triton configurations use the original path.

`tests/test_beam_lse.py` checks strided prefill, vocab8/2053/50002, finite/masked large logits,
near ties, sparse/all-invalid candidates, NaN/+Inf, inactive/done rows, numerical-error flags,
CUDA-graph replay, bucket boundaries, forced limits, prompts and repeated decoding. Scores use
explicit fp32 tolerances; masks/error flags are exact. Toy decoded output must match, while real
output differences are recorded by the profiler rather than hidden behind text-only checks.
Existing beam/naive/greedy/timestamp regressions run in the allocated job too.

`profile_beam_lse.pbs` stages the environment, unchanged engines/checkpoints, code and audio on
`$PBS_LOCALDIR`. It compares a frozen pre-change `baseline_runtime.py` to the candidate on1B/102M,
English73/Japanese12, B1/B3 K5 timestamps,11 alternating repeats, with original check intervals1/4.
It records phases separately from uninstrumented warm wall time, exact full-token/time/segment
comparisons, WER/CER, decoder steps, actual dispatch, hashes/captures and extra no-timestamp,
translation, prompt, noise/silence and110s checks. No reference timestamp annotations are available;
segment differences are baseline differences, not accuracy against ground truth.

```sh
python scripts/profile_beam.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --beam 5 --timestamps --batch 1 --chunk 1 --repeat 11 \
  --upper-bound on --fused-rank --fused-finish --fused-state --kv-reorder prefix --share-cross-kv \
  --fused-lse --baseline-runtime /local/baseline_runtime.py --phases --out /local/results
```

GPU correctness/performance is pending. Keep defaults and README speed claims unchanged unless
full comparison and independent replication justify adoption. Do not combine predicted or measured
gains with unrelated audio/frontend variants.
