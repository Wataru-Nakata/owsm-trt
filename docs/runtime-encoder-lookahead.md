# Fixed-batch encoder lookahead experiment

Candidate35 addresses the sequential fixed-batch path used by beam and translation. After the
wider-row comparison, the illustrative eager B3 encoder takes7.11ms against47.68ms decoder;
these are instrumented phases, not a prediction of end-to-end overlap or speed.

The private `OWSM._encoder_lookahead=True` experiment preserves duration sorting, fixed batches,
primer, beam width and decoder shapes. It stages exactly one subsequent encoder batch on a
separate CUDA stream while the current batch decodes. Two utterance-sized cross-KV/CTC buffer
sets are retained per session and reused. Events protect producer completion and slot reuse;
encoder contexts use lane1 scratch, encoder graphs retain their separate cuBLAS capture stream,
and both streams join the caller on success or exception. Active cross KV is copied/expanded on
the decode stream before prefill. Partial final batches keep their own original shape/session.

CPU, single-batch calls and existing drafted greedy streaming keep their paths. Greedy ASR with
a long prompt and enabled drafts also retains its previous draft semantics. The flag is private,
False by default, and creates no staging resources when disabled. This is not continuous beam
refill; ended hypotheses stay within their original group. No claim about single-request latency.

Eight lightweight CPU routing/order tests pass. Eight GPU tests are prepared for tiny real
encoders/decoders: greedy/beam, graph/eager, B1/B3, mixed final batch, long primer, repeat reuse,
ordinary encode after staging and exception recovery. These16 tests pass in2773541 (7.23s); stronger exact encoder-KV snapshots are prepared for follow-up.

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
median; independent replication is required. Defaults remain private/off while this is evaluated.

Cold observations are **not an isolated startup comparison**: the reference always warms first,
so initial compiler/frontend caches differ. They are preserved as setup observations only, with
no claimed cold improvement. Candidate staging bytes/encoder graph counts and the shared-process
peak are in the artifact. A69MB12-input CUDA trace is preserved; it will be parsed on a compute
node to verify inter-stream kernel overlap without treating kernel sums as wall time.

[All initial metrics, hashes and settings](runtime-lookahead-results-2773541.json); raw outputs:
`outputs/runtime-2773541`. The follow-up2773582 repeats all English1B conditions with nine pairs,
tests two-input B1 requests, and measures interaction with the already-verified grouped/tail
beam option. Separate improvements must not be added or multiplied.
