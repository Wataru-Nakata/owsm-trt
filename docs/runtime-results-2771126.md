# Throughput and CTC validation: 2771126

H200, same checkpoints/engines within each comparison; seven alternating repeats.
Verification-window budget is N-class because fp16 shapes/dispatch change. Default remains256.
Only128 is offered as a measured opt-in; smaller budgets are slower overall and some change text.

| Corpus | Batch | Timestamps | Budget | RTFx change | Matching tokens | WER/CER |
|---|---:|---|---:|---:|---:|---:|
| draft_english | 3 | False | 256 | +0.00% | 73/73 | 3.913% |
| draft_english | 3 | False | 128 | -0.31% | 73/73 | 3.913% |
| draft_english | 3 | False | 64 | -4.05% | 73/73 | 3.913% |
| draft_english | 3 | False | 32 | -21.12% | 71/73 | 3.913% |
| draft_english | 3 | True | 256 | +0.00% | 73/73 | 3.913% |
| draft_english | 3 | True | 128 | +0.07% | 73/73 | 3.913% |
| draft_english | 3 | True | 64 | -4.30% | 73/73 | 3.913% |
| draft_english | 3 | True | 32 | -20.06% | 73/73 | 3.913% |
| draft_english | 8 | False | 256 | +0.00% | 73/73 | 3.913% |
| draft_english | 8 | False | 128 | +1.12% | 73/73 | 3.913% |
| draft_english | 8 | False | 64 | -10.03% | 72/73 | 3.913% |
| draft_english | 8 | False | 32 | -34.96% | 73/73 | 3.913% |
| draft_english | 8 | True | 256 | +0.00% | 73/73 | 3.913% |
| draft_english | 8 | True | 128 | +3.58% | 73/73 | 3.913% |
| draft_english | 8 | True | 64 | -5.30% | 73/73 | 3.913% |
| draft_english | 8 | True | 32 | -21.54% | 73/73 | 3.913% |
| draft_japanese | 3 | False | 256 | +0.00% | 12/12 | 4.899% |
| draft_japanese | 3 | False | 128 | +0.69% | 12/12 | 4.899% |
| draft_japanese | 3 | False | 64 | -2.50% | 12/12 | 4.899% |
| draft_japanese | 3 | False | 32 | -17.56% | 12/12 | 4.899% |
| draft_japanese | 3 | True | 256 | +0.00% | 12/12 | 4.899% |
| draft_japanese | 3 | True | 128 | +0.60% | 12/12 | 4.899% |
| draft_japanese | 3 | True | 64 | +2.38% | 12/12 | 4.899% |
| draft_japanese | 3 | True | 32 | -12.95% | 12/12 | 4.899% |
| draft_japanese | 8 | False | 256 | +0.00% | 12/12 | 4.899% |
| draft_japanese | 8 | False | 128 | +2.79% | 12/12 | 4.899% |
| draft_japanese | 8 | False | 64 | -5.91% | 12/12 | 4.899% |
| draft_japanese | 8 | False | 32 | -24.93% | 12/12 | 4.899% |
| draft_japanese | 8 | True | 256 | +0.00% | 12/12 | 4.899% |
| draft_japanese | 8 | True | 128 | +6.92% | 12/12 | 4.899% |
| draft_japanese | 8 | True | 64 | -1.01% | 12/12 | 4.899% |
| draft_japanese | 8 | True | 32 | -43.21% | 12/12 | 4.899% |

B3 budgets256/128 choose identical windows; their small differences are timing variation.
B8 budget128 caps the window at16 rather than32. Timestamped decoder calls rise49→56 for English
and16→17 for Japanese, but smaller calls outweigh the extra calls: p50 191.48→183.24ms and
49.57→46.46ms. All tokens/timestamps/segments/text match at128 in these corpora. No global equality
guarantee is made for other inputs/checkpoints; timestamp reference annotations are unavailable.

Budget32/B8 without timestamps uses fixed batches under the existing primer-length rule. Actual
dispatch reveals B8×window4 still uses TensorRT: the current Gluon routing ceiling is16 total tokens
although its implementation hard capacity is32. No routing ceiling or kernel precision was changed.
English text differences: B3/noTS/budget32 on1272-135031-0023 and1272-141231-0007;
B8/noTS/budget64 on1272-141231-0007. Normalized WER remains unchanged; full differences are retained.

## CTC-only v4 1B, frontend graph

| Corpus | Case | p50 ms eager → graph | p95 ms eager → graph | RTFx change |
|---|---|---:|---:|---:|
| ctc_english | single_greedy | 5.631 → 5.394 | 5.906 → 5.697 | +4.35% |
| ctc_english | stream_greedy_b1 | 330.940 → 329.924 | 333.382 → 334.243 | +0.07% |
| ctc_japanese | single_greedy | 5.839 → 5.529 | 6.293 → 6.105 | +5.15% |
| ctc_japanese | stream_greedy_b1 | 55.694 → 54.916 | 58.758 → 56.093 | +2.19% |

All CTC features, tokens and text match exactly. The streaming case uses existing two encoder
lanes; English throughput improvement is negligible. No CTC decoding or padding rules changed.

Artifacts: `outputs/runtime-2771126/draft_analysis.json`, `analysis.json`, per-case raw samples,
tokens, quality and actual dispatch counts. Source/environment/hash metadata are beside them.
The public `draft_token_budget=128` path is undergoing an independent replication before final acceptance.
