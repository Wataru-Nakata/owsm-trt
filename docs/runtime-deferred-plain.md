# Deferred plain decoder setup

Job2771972's ASR cold record still prepared four plain decoder graphs even though the request only
used drafted verification. The experiment adds `GreedySession(lazy_plain=True)` to defer those
closures (including backend trial) and graphs until the first plain `decode` request. The high-level
experiment flag `OWSM.greedy_lazy_plain` defaults False pending GPU comparisons. Beam is unchanged.

Cold preparation zeros the shared plain token/offset/position/done state before trial/capture. It runs
before the new request's prefill and rules initialization, so no live prefix needs preserving. Drafted
requests and language-ID prefill never prepare plain steps. The same plain calls/graphs are reused
thereafter; no decoder arithmetic or selection semantics change. Cache buffers remain allocated.

`profile_runtime_cold.py --mode plain --limit 12 --repeat 5 --cases asr prompt lid translation stream
stream8` alternates eager/deferred construction with one resident model/engine. The first cycle is
separate from warmed-JIT session-cold samples. Every reconstruction also reuses the session across
all tasks and requires exact results. This is not full model load/build timing. `profile_runtime.py
--variants reference graph_plain graph_deferred --lazy-drafts` compares the accepted current path
with deferral in steady state; all verification shapes are explicitly warm outside those samples.

Initial job2772355 failed before benchmarks because the new artificial test used EOS=-1 to force
long decoding; drafted padding embeds EOS and requires an in-vocabulary ID. Commitf5a503f changes
the fixture to valid EOS1 with a suppressed logit and asserts full untimed generation lengths.
The runtime implementation is unchanged. The failed job provides no performance result. The retry
runs new tests first with verbose first-failure reporting and disables core dumps.

Three lightweight CPU mock-state checks passed. GPU results and adoption remain pending; no speed
claim is made here yet. Test coverage targets rows1/3, graph on/off, timestamp on/off/mixed primers,
long prefill, cache buckets16/32/64, repeated draft/plain switching, and construction only once.
