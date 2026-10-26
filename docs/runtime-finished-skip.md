# Skip completed plain single-row decoder calls (experiment)

Plain greedy/translation sessions check completion on the host every four graph replays.
The opt-in `GreedySession(skip_finished=True)` reuses its EOS flag for a uniform guard at the
start of Ring R1/S1 calls. Every CTA reads the same immutable scalar before any shared-memory
producer/barrier starts. A completed call leaves outputs/cache/workspaces untouched. The outer
selection still writes EOS and advances state as before; host polling and replay counts do not
change. The flag is reset before trials/capture and each request. No per-step host reads are added.

Default is False. Only the capable single-row Ring backend receives the flag; B>1, TensorRT,
other kernels, beam search and drafted verification retain their existing calls. GPU fallback is
reported and invalidates the benchmark. This is whole-call skipping after completion, not
per-row beam compaction. It does not change timestamp rules or prematurely stop live hypotheses.

Fourteen CPU tests cover prefill EOS, later EOS, capacity truncation, bucket changes, session
reuse, lazy/eager construction and unsupported contracts. Four GPU tests passed on H200 in job2773811: active arithmetic and cache equality, all public/workspace tensors unchanged under the
guard, then restart repeatedly, with and without CUDA graphs/argmax. The uniform Gluon early return compiled on Triton3.8 and all four contracts passed.

`scripts/profile_finished_skip.pbs` runs those tests first, then independent 1B/102M English73
and Japanese12 processes. Each uses the same model/engines for alternating nine warmed reference/
skip whole-list measurements: plain ASR B1, timestamp ASR B1, translation B1, and B3 fallback control.
Full tokens, timestamp tokens, text and segments must match; engine/checkpoint hashes and actual
dispatch are saved. Separate diagnostic graph runs count all decoder calls and calls entered
with EOS already set, with one host read after decode. Instrumented runs must reproduce both
outputs and replay counts; their times are excluded. Cold observations have unequal compiler
caches and the allocator peak is shared, not an isolated memory comparison. English extras cover
silence/noise, long prompt, a repeated 30s window, drafted B1/B3, encoder lookahead, and 110s audio;
long-form checks compare joined text/segments, not unavailable raw per-window tokens.

`--request-latency` additionally collects alternating single-request B1 samples with the same
warmed sessions. This is a separate optional verification pass; whole-list p95 is not per-request
p95. Mixed-duration request percentiles and per-input timing variation must be reported separately.

No broadly validated default speedup claim; see the partial result and remaining checks below. Temporary environments/code/audio/logs stay on node-local
storage. Only compact results and coarse progress are published to the shared project.


## Partial result and diagnostic fix:2773811

The first run passed40 focused tests including all4 Ring GPU contracts and31 naive regressions.
All1B English73 cases/extras matched; completed Japanese12 ASR cases and translation wall samples
also matched. English B1 RTFx changed+4.30%/+4.85%/+4.58% (plain/timestamped/translation);
Japanese completed ASR cases changed+3.87%/+2.47%. These are whole-list p50 ratios from9 alternating
pairs, not per-request latency claims. English B3 (one guarded B1 tail) was-0.13%.
Diagnostic call totals stayed1700/1882/1696 for English B1, with93/130/98 already-EOS entries.

The job exited1 during Japanese translation's separate diagnostic capture. Logs show old CUDA
graph and TensorRT context destruction inside capture, before cudaErrorStreamCaptureInvalidated.
The instrumented step closure captured its owning session, forming a cycle that could be collected
inside a later capture. The profiler now captures only its counter/done tensors and explicitly
releases diagnostic sessions outside capture. Runtime/kernel code is unchanged by this fix.
Japanese translation diagnostics, Japanese B3 and all102M conditions remain incomplete, so the
optimization remains opt-in. A rerun adds independent single-request latency measurements.
[Preserved partial evidence](runtime-finished-skip-partial-2773811.json); full logs remain under
`outputs/runtime-2773811/`. Long-form extras compare public joined segments only.
