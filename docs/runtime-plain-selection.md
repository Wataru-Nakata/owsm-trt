# Plain greedy / translation selection fusion

Job2771972's English translation trace has24 ring decoder calls (15.94ms GPU sum),25 argmax calls
(0.343ms) and many per-token boolean/mask/state kernels. All PyTorch-native GPU kernels together
sum1.545ms/846 calls in that trace; that includes frontend/other work, not selection alone or wall time.
This motivates measuring mask/argmax/state fusion in the non-drafted path.

The candidate uses vocabulary-block maxima followed by a combined final argmax/done/token/history/
timestamp update. It keeps mask-before-argmax, first-index ties, first surviving NaN, all-invalid
index0, done-row EOS behavior and unchanged pos. Unlike drafted choose, the baseline is PyTorch
argmax, so nonfinite handling has independent tests. Decoder logits and arithmetic are untouched.
Small vocabularies, non-fp32 logits, CPU/missing Triton retain PyTorch. GreedySession(fused_select=True)
remains an explicit low-level switch. OWSM now enables this path with greedy_kernels=True;
False restores PyTorch plain selection as well as the serial drafted path.

GPU tests cover arbitrary states, time-boundary masks, ties/near-ties, NaN/Inf/all-invalid logits,
EOS, repeated captured calls, final output slot, long prefill and bucket16/32/64 transitions.
`profile_greedy_select.py` reports synthetic selection-only graph CUDA events separately.
`profile_runtime.py graph_plain` compares actual same-engine requests with accepted `graph_lazy`,
including plain greedy with/without timestamps, translation, drafted/beam regressions and102M.
All returned tokens/segments must match.

Job2772060 passed131 tests and the10-request public API rollback comparison. The benchmark then
failed before timing because its PBS wrapper named the model-root config.yaml instead of the file
under exp/. No selection/runtime performance result was produced. The wrapper path was corrected
against the staged model tree for the retry; runtime/kernel code is unchanged.


## H200 same-engine result:2772077

Source8d293ac completed131 tests, a10-request public API rollback regression, all9 synthetic
selection shapes, and13 runtime conditions. All tokens, timestamp IDs and segments matched.
Public-API regression in this job predates adoption of plain fusion; its new default wiring is
checked in the follow-up job. Timing compares graph_plain against accepted graph_lazy, both with
frontend graphs and existing drafted optimizations; eleven alternating repeats after all-shape warmup.

| Model / language | Plain no timestamps RTFx gain | Plain timestamps | Translation |
|---|---:|---:|---:|
|1B English|8.85%|9.31%|9.17%|
|1B Japanese|8.52%|8.79%|8.52%|
|102M English|26.60%|26.97%|27.07%|

Each table condition uses12 utterances. Within-input timing CV medians were0.12–0.31%, and the
paired median gains agreed with the aggregate gains. Drafted timestamp requests(73 English inputs)
and long prompts were nearly unchanged. Beam showed-0.39% in this run; its code and configuration
are unaffected by plain fusion. These checks establish output agreement, not speedup of unaffected paths.

For1B English plain timestamps, across the12×11 mixed-duration requests, median latency was
26.12→24.00ms and p95 was88.52→80.49ms.102M:10.07→7.90ms and35.97→27.99ms. These percentiles
include differences in input length; per-input variation is retained separately in the raw report.

Synthetic R1/V50002 selection CUDA-event medians: without rules20.37→4.15µs; timestamp rules
64.98→5.09µs; notime rules65.74→5.24µs. This is selection only, not model latency. Workspace is
300bytes per row. First-use JIT is material: the first synthetic fused configuration's combined
compile/capture took486.75ms versus48.61ms for Torch; later configurations reuse compiled kernels.
It is excluded from warm timing. Runtime peak-memory samples retain all variant pools and are not
standalone deployment measurements. There are no reference timestamp annotations; equality is
against the same engine, not timestamp accuracy against a human reference.

Artifacts: outputs/runtime-2772077/{analysis-graph_lazy.json,selection.json,tests.log,api1b/public_api.json},
plus english/japanese/base102m case wall/token/cold/dispatch JSON. The checkpoint and engine hashes,
software versions and GPU details are in each dataset's environment.json.


## Independent replication and public default validation:2772119

Source43183c0 validated the adopted public default against greedy_kernels=False on both1B and102M:
all10 requests per model matched, including plain/translation, drafted/streaming, beam, long prompt,
language identification and session reuse. The test asserts the expected plain-selection workspace.

Eleven alternating repeats independently reproduced RTFx gains:1B English8.89/9.05/8.88%,1B Japanese
9.30/8.92/8.67%,102M English26.41/26.78/26.93% (plain/timestamped plain/translation respectively).
All returned tokens/segments match. English drafted streaming B1/B8 changed-0.24/-0.39%, and beam
changed-0.68%; these untouched paths have no performance improvement claim. They retain exact outputs.
Artifacts: outputs/runtime-2772119 API1B/API102M, analysis-graph_lazy.json and raw dataset records.


On the12-input plain ASR subsets, both timestamp modes retain identical reference-error counts:
1B English WER10/303(3.30%),1B Japanese CER29/592(4.90%),102M English WER33/303(10.89%).
These are small-subset checks, not model-wide accuracy estimates. Translation has no matching
translation reference in this corpus and has no quality score; its baseline output agreement is recorded.
