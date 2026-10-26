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
forced-limit/bucket-boundary behavior. GPU status is pending, not an adoption claim.

`scripts/profile_beam_prefill.pbs` compares the committed unchanged runtime snapshot with this
single option, on unchanged engines/checkpoints. It uses11 alternating repetitions for1B/102M,
English73/Japanese12, beam5 timestamps B1/B3, the production host-check intervals1/4, and separate
no-timestamp, translation, prompt, synthetic and110s checks. The change is a one-time per-request
saving; no speedup is assumed until full-request timing exceeds observed variation.
