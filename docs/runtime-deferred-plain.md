# Deferred plain decoder setup

Job2771972's ASR cold record still prepared four plain decoder graphs even though the request only
used drafted verification. The experiment adds `GreedySession(lazy_plain=True)` to defer those
closures (including backend trial) and graphs until the first plain `decode` request. The high-level
internal flag `OWSM.greedy_lazy_plain` follows public `lazy_drafts=True` after the replication below. Beam is unchanged.

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

Three lightweight CPU mock-state checks passed. The first GPU comparison follows below; adoption awaits replication. Test coverage targets rows1/3, graph on/off, timestamp on/off/mixed primers,
long prefill, cache buckets16/32/64, repeated draft/plain switching, and construction only once.


## First comparison:2772365

H200, sourcef5a503f. All140 tests passed in49.72s. All16 steady-state cases (English8,Japanese4,
102M English4) preserve tokens, timestamp tokens, segments and text. Same engine/checkpoint hashes
are verified between cold and warm artifacts. Five session reconstructions per mode/task gave:

| Model/request | Eager ms | Deferred ms | Reduction |
|---|---:|---:|---:|
| 1B ASR |336.49|317.61|5.61%|
| 1B long prompt |344.35|336.35|2.32%|
| 1B stream B8,12 inputs |1955.41|1957.89|-0.13%|
| 102M ASR |141.08|135.78|3.76%|
| 102M long prompt |147.23|140.08|4.86%|
| 102M stream B8,12 inputs |1088.88|851.19|21.83%|

ASR paired reductions are positive in all five repeats (1B2.12–6.34%,102M2.88–8.51%);102M B8 is
14.82–23.22%. Prompt/1B streaming gains are smaller or variable. Translation was effectively flat
on1B and6.55% slower on102M in this run; it still constructs plain graphs before generation, so this
is a control that needs replication. Language ID was also a control: pipeline uses
`decode(max_new_tokens=1)` and the first implementation unnecessarily prepared all four graphs.
Commitf782942 skips that preparation when only prefill selection is required; new tests exercise the
actual one-token decode rather than raw prefill. That correction is not measured by2772365.

For drafted ASR/streaming, plain graph/closure counts fell4→0; language-ID and translation counts
were4→4. Direct ASR retained about0.50MiB/0.26MiB less on1B/102M. Streaming allocation depends on
allocator/context retention; no fixed VRAM saving is inferred from differences of medians.
Steady-state RTFx changes range-0.62% to+0.77%; no warm speedup is claimed. The cold frontend runs
eagerly and engines/JIT are resident. These are not model-load/build or disk-inclusive timings.
Artifacts: outputs/runtime-2772365, analysis-graph_plain.json, deferred_cold_analysis.json.

The next job repeats five cold task types seven times and validates the public rollback/candidate
on both models. `--reuse-first-only` tests all cross-task reuse on the first cycle; every subsequent
fresh request still requires exact output. It avoids repeatedly paying untimed setup for all tasks
in every sample. Original2772365 performed reuse after every construction (60 timed rows/model,
12 separate first observations), and remains unchanged. Default remainedOFF in that measurement snapshot; adoption follows below.


## Replication and adoption:2772457

Sourcee8a2d000, same H200/checkpoints/engines. Nine focused GPU tests pass; public serial rollback
versus the future deferred default matches all10 requests on each model, including plain, drafted,
stream, beam, prompt and language-ID reuse. Seven timed fresh reconstructions per mode/task plus
ten separate first observations per model all match. The actual one-token LID path leaves zero
plain graphs, as intended afterf782942.

| Model/request | Eager ms | Deferred ms | Reduction |
|---|---:|---:|---:|
| 1B ASR |283.63|267.59|5.65%|
| 1B long prompt |283.29|269.49|4.87%|
| 1B language ID |274.34|254.20|7.34%|
| 1B translation |290.52|293.81|-1.13%|
| 1B stream B8,12 inputs |1932.37|1902.80|1.53%|
| 102M ASR |111.96|104.33|6.82%|
| 102M long prompt |112.97|105.45|6.65%|
| 102M language ID |107.85|99.06|8.15%|
| 102M translation |113.30|112.57|0.64%|
| 102M stream B8,12 inputs |1055.08|829.18|21.41%|

ASR medians improve in both independent jobs, although individual paired samples include timing
outliers. LID paired reductions are3.92–17.51% on1B and3.00–12.31% on102M.102M B8 improves in every
pair,19.69–24.57%, replicating the earlier21.83% median. Translation controls fluctuate around no
gain;1B B8 is also not a consistent gain. Earlier102M translation regression does not reproduce.
No steady-state improvement is claimed. Existing eager fallback remains `lazy_drafts=False`;
standalone GreedySession keeps explicit `lazy_plain=True`. Other GPUs/models are unmeasured.

Adopted: high-level `lazy_drafts=True` also defers unused plain calls. This includes single-selection
language ID; a later ordinary decode still prepares once before its prefill. No cache layout,
decoder arithmetic, stopping or output rules change. Artifact: outputs/runtime-2772457, both API
and cold JSONs; measured snapshot used the same internal flag set explicitly before any session.
