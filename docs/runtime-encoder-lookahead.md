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
ordinary encode after staging and exception recovery. GPU correctness/performance is pending.

`scripts/profile_encoder_lookahead.py` alternates reference/lookahead whole-list calls with the
same checkpoint/engines. It first warms every encountered shape, requires exact tokens/time/
segments/text before and throughout timing, and separates cold capture from warm wall time.
Initial targets:1B English73/Japanese12, beam5TS B1/B3 and translation B1/B8. Translation quality
is unmeasured because these manifests contain source transcripts, not target translations.
A separate12-input profiler timeline checks overlap; GPU kernel sums must not be confused with
wall time. Staging bytes/capture counts and shared-process peak allocation are recorded; this
peak does not isolate the memory footprint of either variant. No speed claim yet.
