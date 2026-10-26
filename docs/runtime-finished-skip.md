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
reuse, lazy/eager construction and unsupported contracts. Four GPU tests are prepared, not yet
measured: active arithmetic and cache equality, all public/workspace tensors unchanged under the
guard, then restart repeatedly, with and without CUDA graphs/argmax. Compilation of the uniform
Gluon early return is still unverified on the installed GPU toolchain.

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

No measured speed improvement yet. Temporary environments/code/audio/logs stay on node-local
storage. Only compact results and coarse progress are published to the shared project.
