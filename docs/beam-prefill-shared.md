# Shared prefill normalization experiment

Candidate11, opt-in `BeamSession(prefill_shared=True)` / `profile_beam.py --prefill-shared`.
No public factory option or default change. The first selection normalizes the original[B,V]
prefill logits once per utterance and reads them directly in the candidate kernel, avoiding
replication into[B*K,V] and repeated normalization. Raw logits retain their original prompt-row
stride; log probabilities are contiguous. Later decoder steps use the ordinary path.

The candidate buffer remains[B,K*V], and the same global torch.topk sees the same layout. The
active-parent mask, timestamp masks, invalid/NaN/Inf handling, global EOS collection and all
stopping rules are unchanged. Prefix KV replication/reordering is unchanged. CPU or disabled
fused ranking falls back to the original implementation. E intent; actual full-token and state
agreement must be measured because normalization batch shape changes.

43 CPU artificial-state tests pass (two GPU-only skips), and14 small Triton interpreter cases
verify ties, exhausted rows, NaN/Inf and strided long-prompt inputs. Prepared GPU checks cover
vocabulary8/2053/50002, K2/5/8, only-two-valid-timestamp candidates, ended groups, and graph/reuse/
forced-limit/bucket-boundary behavior. GPU results are reported below; this is not enabled by default.

`scripts/profile_beam_prefill.pbs` compares the committed unchanged runtime snapshot with this
single option, on unchanged engines/checkpoints. It uses11 alternating repetitions for1B/102M,
English73/Japanese12, beam5 timestamps B1/B3, the production host-check intervals1/4, and separate
no-timestamp, translation, prompt, synthetic and110s checks. The change is a one-time per-request
saving; no speedup is assumed until full-request timing exceeds observed variation.


## Results (2773753)

Frozena3efdc9, unchanged runtime referenceb90283a, H200.180 GPU/state checks pass in9.57s,
including all65 new prefill cases;31 naive beam/greedy/timestamp regressions pass in5.46s.
All eight real-model conditions match complete tokens, timestamp tokens, segment boundaries/text,
quality metrics and decoder-call counts. Actual candidate dispatch reports`prefill_shared=True`.
No-timestamp/translation/prompt/synthetic/30s extra checks match. The110s check matches joined
text/segments; its API does not retain raw window tokens.

|Model/corpus|Batch|Aggregate RTFx change|Median paired-group gain|Positive groups|Repeat-total gain range|
|---|---:|---:|---:|---:|---:|
|1B English73|1|+0.482%|+0.464%|72/73|+0.334–+0.664%|
|1B English73|3|+0.160%|-0.015%|11/25|-0.659–+1.076%|
|1B Japanese12|1|+0.315%|+0.222%|12/12|+0.114–+0.631%|
|1B Japanese12|3|+0.179%|+0.008%|2/4|-0.209–+0.839%|
|102M English73|1|-0.013%|+0.033%|43/73|-0.553–+0.720%|
|102M English73|3|+0.319%|+0.316%|20/25|-0.166–+0.882%|
|102M Japanese12|1|-0.220%|-0.212%|3/12|-0.631–+0.375%|
|102M Japanese12|3|+0.113%|-0.161%|1/4|-0.504–+1.787%|

Each input/group has11 alternating samples; rows of different audio lengths are not pooled to
estimate variation. Per-group CV medians range0.16–1.09%.1B B1 has a small consistent benefit,
but other shapes are mixed or within variation. This does not justify a broad default change or
a new model-specific dispatch policy. The runtime/profiler opt-in remains for reproduction,
with no high-level factory option. No GPU-memory saving or cold-start improvement is claimed.
Shared prefill introduces its own kernel variants (including the prompt row stride); warmed
latencies do not account for first-use compilation.

[Full paired samples, summaries, dispatch and hashes](beam-prefill-results-2773753.json); raw
outputs: `outputs/runtime-2773753`. English/Japanese quality stays unchanged for both models.
