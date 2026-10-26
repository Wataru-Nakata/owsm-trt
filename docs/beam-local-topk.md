# Parent-local beam top-K experiment

After the wider-row decoder comparison,2773135's eager/instrumented B3 example measures
selection at0.126ms per call (5.68ms across45 calls), versus47.68ms for44 decoder calls.
This is motivation for a small candidate08 ablation, not a CUDA-graph phase breakdown or
a promised speedup. Decoder remains the largest component.

`BeamSession(local_topk=True)` keeps full-vocabulary normalization and the existing timestamp
mask/score/validity logic, then selects up toK candidates per parent and globalK from that
smaller pool. Returned indices map back to the original parent/token layout. EOS is considered
only within globalK; no top2K replenishment or changed stopping rule is added. When V<K,
the parent pool uses V. No epsilon or score change is used.

The chosen top-K score multiset is exact, but PyTorch topk's tied token/parent ordering can
change. This is an explicitly opt-in compatibility experiment, not an exact-output default.
Ties involving EOS/timestamps may change subsequent search and outputs. The default isFalse
and the high-level API has no new option. The profiler `--local-topk` requires a separate
reference runtime and records all output differences.

60 small CPU selection tests pass: distinct scores, distinct one-ULP neighbors, sparse valid
sets, all-invalid and ties, B1/B3, K2/5/8, V3/17. The initial near-tie fixture accidentally
rounded distinct mathematical values to equal fp32 values; it now asserts representable
uniqueness rather than weakening index equality. Selected scores/indices and duplicate
exclusion are checked. Existing invalid-parent/EOS/mask/error/capacity state tests now cover
both selectors. Large-vocabulary graph tests and GPU performance remain pending.

## GPU decision: not adopted

Job2773407, source983ac74, H200/1B, same checkpoint/engines and unchanged decoder routing.
177 state/selection/graph tests pass (9.72s). Nine alternating repetitions per condition:

|Input|B|RTFx change|Median paired group change|Groups improved|
|---|---:|---:|---:|---:|
|English73|1|+0.076%|+0.052%|37/73|
|English73|3|+0.062%|-0.097%|9/25|
|Japanese12|1|+0.216%|+0.110%|6/12|
|Japanese12|3|+0.004%|-0.068%|1/4|

All tokens/timestamp tokens/segments/text and decoder-call counts match. Extra no-timestamp,
translation, long prompt, silence/noise,30s repeated speech and110s recording outputs also match
with bound stopping both ON/OFF. This observed agreement does not remove the theoretical tie
ordering caveat. There is no consistent improvement beyond measurement variability, so the
production default remains global top-K. The isolated flag remains for reproduction only.
[Metrics and hashes](beam-topk-results-2773407.json); full wall/phase/diagnostic/raw token artifacts:
`outputs/runtime-2773407/`. No GPU speedup is claimed for this change.
