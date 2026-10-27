# Plain single-row Ring solo scheduling: first measured comparison

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
The first measured result is below; public-path integration still needs independent validation.

The proposed `profile_plain_solo.pbs` runs on the compute node with both1B and102M, English73 /
Japanese12,11 alternating repeats. Real-checkpoint synthetic-state checks require exact logits,
new-KV and cache at offsets3/63/127/255 and identical NS. GPU EOS/restart tests cover both budgets,
argmax/logits and CUDA graphs. Full token/time/text/segments and actual call counts must match.
Wall samples exclude these diagnostics; cold captures and co-resident allocator peaks are separate.
Per-request p50/p95 pool differing durations; paired per-input results will also be analyzed.


## First result2776121

Frozen4b54e3a,34 routing/GPU EOS/graph/restart tests and eight real-checkpoint state checks per
model/corpus passed. All tokens, timestamp IDs, segments, text and decoder calls matched across
all12 conditions. Eleven alternating per-request repeats gave aggregate RTFx changes:

|Model/corpus|Plain ASR B1|Timestamp ASR B1|Translation B1|
|---|---:|---:|---:|
|1B English73|+5.53%|+6.29%|+5.35%|
|1B Japanese12|+6.29%|+6.25%|+6.00%|
|102M English73|+12.45%|+13.06%|+12.42%|
|102M Japanese12|+14.39%|+14.54%|+16.11%|

Every per-input median improved (73/73 English,12/12 Japanese in each condition). All11 paired
whole-corpus totals improved in every condition. Pooled single-request p95 decreased5.79–6.76%
on1B and12.74–14.20% on102M. These percentiles pool mixed durations/repeats; per-input medians
are preserved in the report. This is plain decoding, including translation; it is not a gain for
ASR already using CTC-drafted verification. Both modes retain the accepted single-row EOS guard.

Select the existing solo schedule for serial plain calls, with the concurrent schedule retained
when encoder lookahead is enabled. The next job must verify actual public default/rollback,
lookahead exclusion, drafted behavior, reuse and independent request timings before adoption.
[Per-input results, quality and environment](runtime-plain-solo-2776121.json); raw artifacts in
`outputs/runtime-2776121/`. PBS Stageout_status1 coexisted with complete artifacts and exit0.
No reference translation/boundary quality or isolated cold/VRAM improvement is claimed.


## Public integration: independent validation pending

`GreedySession(..., plain_solo=False)` adds an optional single-row serial budget. The public
factory enables it with `greedy_kernels=True` only when encoder lookahead is disabled and the
actual backend supports distinct solo calls. Drafted verification retains its own existing
concurrency choice; other rows/backends and `greedy_kernels=False` keep the previous plain path.
No per-token host branch/synchronization is added; the setting binds the constructed graph.

`validate_plain_solo.pbs` checks both budgets' EOS/restart GPU behavior and actual public/default
rollback, including encoder-lookahead exclusion, then independently repeats all12 request cases
through `--public-policy`. Separate diagnostics assert and save every actual plain call's program
budget and NS. It also fills the recently adopted row policy's W2 graph/cache coverage. The first
profiler result above is not yet an independently verified public-default result.


## Incomplete public validation2776203

Frozen658fa45 passed58 focused tests and seven additional Ring W2 GPU cases. Its1B public API
artifact has all13 requests equal, including lookahead exclusion, and the final success message
was printed. However, the validator process did not exit before its250s timeout (job exit124).
102M API and all independent timing processes were never reached. This is a failed/incomplete
job, not a completed adoption result or a new performance measurement.

The validator now explicitly drops lingering session/window loop references and clears graphs
before model destruction and interpreter shutdown, with CUDA synchronization and phase logs.
A faulthandler watchdog will capture remaining stalls. Retained lazy-session/TRT graph ownership
at shutdown is a hypothesis, not an established root cause. The fix changes only validation
lifetime management; it does not bypass process cleanup or weaken output checks. Retry required.
Raw complete/partial artifacts: `outputs/runtime-2776203/`.
