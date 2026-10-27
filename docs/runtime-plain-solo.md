# Plain single-row Ring solo scheduling: prepared experiment

The existing Ring decoder has two budgets: a smaller one while an encoder can run concurrently,
and a larger `solo` budget with the same cross-attention split count. Drafted verification already
selects them according to overlap. Plain `GreedySession.decode` constructs its single-token calls
without `solo`, even in serial fixed-batch transcription. The earlier program sweep changed the
underlying concurrent budget (and therefore NS); this experiment reuses the existing solo budget
and preserves arithmetic.

`profile_plain_solo.py` changes construction-time dispatch only for R1/S1 `kind="decode"`.
Prefill, drafted verification, multi-row calls, scalar EOS guarding, host-check frequency and
search are unchanged. The experiment disables encoder lookahead and times separate single
requests for plain ASR, timestamp ASR and translation. This is not a policy for overlapping
encoding; any later runtime integration must retain the concurrent schedule in that path.
No runtime default has changed and no speed result is available yet.

The proposed `profile_plain_solo.pbs` runs on the compute node with both1B and102M, English73 /
Japanese12,11 alternating repeats. Real-checkpoint synthetic-state checks require exact logits,
new-KV and cache at offsets3/63/127/255 and identical NS. GPU EOS/restart tests cover both budgets,
argmax/logits and CUDA graphs. Full token/time/text/segments and actual call counts must match.
Wall samples exclude these diagnostics; cold captures and co-resident allocator peaks are separate.
Per-request p50/p95 pool differing durations; paired per-input results will also be analyzed.
