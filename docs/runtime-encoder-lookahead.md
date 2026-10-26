# Fixed-batch encoder lookahead experiment

Candidate35 addresses the sequential fixed-batch path used by beam and translation. After the
wider-row comparison, the illustrative eager B3 encoder takes7.11ms against47.68ms decoder;
these are instrumented phases, not a prediction of end-to-end overlap or speed.

The opt-in `OWSM.from_pretrained(..., encoder_lookahead=True)` preserves duration sorting, fixed batches,
primer, beam width and decoder shapes. It stages exactly one subsequent encoder batch on a
separate CUDA stream while the current batch decodes. Two utterance-sized cross-KV/CTC buffer
sets are retained per session and reused. Events protect producer completion and slot reuse;
encoder contexts use lane1 scratch, encoder graphs retain their separate cuBLAS capture stream,
and both streams join the caller on success or exception. Active cross KV is copied/expanded on
the decode stream before prefill. Partial final batches keep their own original shape/session.

CPU, single-batch calls and existing drafted greedy streaming keep their paths. Greedy ASR with
a long prompt and enabled drafts also retains its previous draft semantics. The option is
False by default, and creates no staging resources when disabled. This is not continuous beam
refill; ended hypotheses stay within their original group. No claim about single-request latency.

Eight lightweight CPU routing/order tests pass. Eight GPU tests are prepared for tiny real
encoders/decoders: greedy/beam, graph/eager, B1/B3, mixed final batch, long primer, repeat reuse,
ordinary encode after staging and exception recovery. These16 tests pass in2773541 (7.23s); stronger exact encoder-KV snapshots also pass in2773664.

`scripts/profile_encoder_lookahead.py` alternates reference/lookahead whole-list calls with the
same checkpoint/engines. It first warms every encountered shape, requires exact tokens/time/
segments/text before and throughout timing, and separates cold capture from warm wall time.
Initial targets:1B English73/Japanese12, beam5TS B1/B3 and translation B1/B8. Translation quality
is unmeasured because these manifests contain source transcripts, not target translations.
A separate12-input profiler timeline checks overlap; GPU kernel sums must not be confused with
wall time. Staging bytes/capture counts and shared-process peak allocation are recorded; this
peak does not isolate the memory footprint of either variant. Initial results follow below.


## First same-engine comparison (2773541)

Frozen1a4a0bf, H200, seven alternating whole-list repetitions, warmed shapes. All12 conditions
match full tokens, timestamp tokens, segments and text before and throughout timing. Same1B
checkpoint/engine hashes across English/Japanese;102M uses its own unchanged checkpoint/engines.
The complete input list is one timed request; these are throughput results, not single-audio latency.

|Model/input list|Case|Whole-list p50 before→after(ms)|RTFx change|
|---|---|---:|---:|
|english1b|beam_b1|1965.80→1859.89|+5.69%|
|english1b|beam_b3|1258.52→1183.47|+6.34%|
|english1b|translation_b1|1568.78→1445.00|+8.57%|
|english1b|translation_b8|461.95→442.41|+4.42%|
|japanese1b|beam_b1|463.39→442.98|+4.61%|
|japanese1b|beam_b3|281.95→272.53|+3.46%|
|japanese1b|translation_b1|338.28→320.16|+5.66%|
|japanese1b|translation_b8|141.48→141.69|-0.14%|
|english102m|beam_b1|960.70→917.62|+4.69%|
|english102m|beam_b3|441.07→419.45|+5.15%|
|english102m|translation_b1|522.59→482.59|+8.29%|
|english102m|translation_b8|156.25→148.90|+4.93%|

English has73 inputs, Japanese12. Translation references are unavailable; full output agreement
is checked but no target-language WER/CER is claimed. Japanese translation B8 has only two
batches and is neutral. The102M beam B1 paired range includes a-6.67% outlier despite its positive
median; independent replication is required. The option remains off by default; the independent follow-up is below.

Cold observations are **not an isolated startup comparison**: the reference always warms first,
so initial compiler/frontend caches differ. They are preserved as setup observations only, with
no claimed cold improvement. Candidate staging bytes/encoder graph counts and the shared-process
peak are in the artifact. A69MB12-input CUDA trace is preserved; it will be parsed on a compute
node to verify inter-stream kernel overlap without treating kernel sums as wall time.

[All initial metrics, hashes and settings](runtime-lookahead-results-2773541.json); raw outputs:
`outputs/runtime-2773541`. The follow-up2773582 repeats all English1B conditions with nine pairs,
tests two-input B1 requests, and measures interaction with the already-verified grouped/tail
beam option. Separate improvements must not be added or multiplied.


## Independent follow-up and opt-in API (2773582)

Frozen39c62f7, same H200/1B engines/checkpoint, nine alternating whole-list repetitions for73
English inputs. Full tokens/times/segments/text match in all seven additional conditions.

|Condition|Whole-list p50 before→after(ms)|RTFx gain|Paired range|
|---|---:|---:|---:|
|lookahead_repeat/beam_b1|1963.43→1852.74|+5.97%|+5.58–+6.68%|
|lookahead_repeat/beam_b3|1256.38→1184.68|+6.05%|-2.26–+8.06%|
|lookahead_repeat/translation_b1|1515.31→1399.53|+8.27%|+7.22–+8.45%|
|lookahead_repeat/translation_b8|455.67→439.78|+3.61%|+1.84–+4.82%|
|lookahead_two/beam_b1|46.58→45.23|+2.98%|+1.98–+3.40%|
|lookahead_two/translation_b1|37.28→35.61|+4.67%|+4.33–+5.44%|
|lookahead_wide/beam_b3|1075.69→1007.70|+6.75%|+6.34–+7.28%|

`repeat` is73 inputs, `two` is two inputs (11 repetitions), `wide` is73 with both grouped cross
attention and tail compaction enabled. The wide result is the **additional** lookahead comparison
against that same wide/tail configuration; it is not a product of separate speedup claims.
The repeated ordinary beam B3 median improves but one pair regresses2.26%; no sample was removed.
The102M B1 replication is reported below. Cold observations retain unequal first-use
compiler caches and cannot establish a startup improvement.

The separately recorded12-input trace contains61,914 GPU kernel events with stream IDs. After
unioning each stream's intervals, distinct streams overlap for11.93ms out of288.68ms with active
kernels (4.13%). This confirms concurrent kernel work in the traced run. It does not attribute
all events to encoder/decoder phases or equate the kernel sum with wall time. The69MB trace was
parsed inside the next compute allocation, not on the login node.

Warm throughput gains and exact observed outputs support an explicit option. The default stays
False because extra staging/captures have cold and memory costs and small batches can be neutral.
The final production-option and strengthened exact encoder-KV checks pass as reported below.

```python
model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", encoder_lookahead=True)
results = model.transcribe(recordings, beam_size=5, timestamps=True, batch_size=3)
```

[Follow-up timings and overlap evidence](runtime-lookahead-results-2773582.json); full results:
`outputs/runtime-2773582/lookahead_{repeat,two,wide}`. Translation and fixed plain ASR also use
this path; existing drafted ASR streaming and single-batch calls retain theirs.


## Public API acceptance and102M tail latency (2773664)

Frozenacd09b3. All45 focused tests pass, including bitwise raw and beam-expanded cross-KV checks
for staged/sequential GPU encoders with graph/eager execution, partial batches, long primers,
repeated reuse and exception recovery. The actual `from_pretrained(encoder_lookahead=True)`
option matches False on seven mixed speech/silence/noise cases: single requests before/after
reuse, B3 beam with/without timestamps, a long prompt, plain greedy timestamps and translation.
Staging is observed only on eligible calls. No fallback or output difference is found.

The independent102M English73 beam5TS B1 comparison uses11 alternating pairs. Whole-list p50
is954.38→911.27ms (+4.73% RTFx), but p95 is968.33→1002.89ms. Three pairs regress (minimum-9.92%,
maximum+5.86%, median+4.73%). Full tokens/times/segments/text match73/73 and WER stays94/1150.
The earlier outlier is therefore **not dismissed as resolved**: the positive median coexists
with worse tail latency. Keep this option off by default; it does not establish a universal
single-request or p95 improvement. No samples are discarded.

[All verification and paired measurements](runtime-adoption-results-2773664.json); raw public API
outputs: `outputs/runtime-2773664/public_api`; timings: `outputs/runtime-2773664/repeat102m`.
