# Blockwise full-vocabulary normalization experiment

Candidate07, N class; verified public opt-in `beam_fused_lse=True`, disabled by default.
H200/1B and102M independent measurements and public routing checks are below. The prior eager/instrumented
B3 sample measured selection0.126ms per call versus decoder1.08ms (`beam-local-topk.md`). This
motivates a small isolated reduction/memory-traffic experiment, not a graph-phase attribution or
predicted speedup. Decoder remains dominant and prior local top-K did not yield a meaningful gain.

`BeamSession(fused_lse=True)` computes fp32 block maxima/sums across every vocabulary row, then
combines them in the existing candidate-mask/parent-score kernel. It avoids materializing the
full log-probability tensor. The same global torch.topk, EOS placement, beam slots, strict score
bound, end_detect, cache handling and history updates remain. Timestamp masks are applied after
the full-vocabulary normalizer, including logits of disallowed tokens. A new reduction order and
Triton exp/log change rounding; token or timestamp near ties may differ. This is not an exact-output
default, even if a test set happens to match. NaN/+Inf on live parents must reject the entire row and
set the existing numerical error flag; all-Inf negative rows are exhausted without an error. Dead
parents and finished groups cannot be revived. No per-step host synchronization is added.

Scratch is `[B*K,2,ceil(V/1024)]` fp32 (about5.7KiB atB3/K5/V50002). Each vocabulary block merges
the small partial table; this redundant merge trades arithmetic for removing a full intermediate
write/read. It may be slower than torch.log_softmax. Shared-prefill support is tested independently
but is not enabled in the main ablation. Unsupported CPU/Triton configurations use the original path.

`tests/test_beam_lse.py` checks strided prefill, vocab8/2053/50002, finite/masked large logits,
near ties, sparse/all-invalid candidates, NaN/+Inf, inactive/done rows, numerical-error flags,
CUDA-graph replay, bucket boundaries, forced limits, prompts and repeated decoding. Scores use
explicit fp32 tolerances; masks/error flags are exact. Graph state/output must match eager execution of the same blockwise arithmetic. Torch-versus-blockwise
branch/output differences are diagnosed separately and real output differences are recorded by the
profiler rather than hidden behind text-only checks.
Existing beam/naive/greedy/timestamp regressions run in the allocated job too.

`profile_beam_lse.pbs` stages the environment, unchanged engines/checkpoints, code and audio on
`$PBS_LOCALDIR`. It compares a frozen pre-change `baseline_runtime.py` to the candidate on1B/102M,
English73/Japanese12, B1/B3 K5 timestamps,11 alternating repeats, with original check intervals1/4.
It records phases separately from uninstrumented warm wall time, exact full-token/time/segment
comparisons, WER/CER, decoder steps, actual dispatch, hashes/captures and extra no-timestamp,
translation, prompt, noise/silence and110s checks. No reference timestamp annotations are available;
segment differences are baseline differences, not accuracy against ground truth.

```sh
python scripts/profile_beam.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --beam 5 --timestamps --batch 1 --chunk 1 --repeat 11 \
  --upper-bound on --fused-rank --fused-finish --fused-state --kv-reorder prefix --share-cross-kv \
  --fused-lse --baseline-runtime /local/baseline_runtime.py --phases --out /local/results
```

The completed correctness/performance and independent public verification are recorded below.
The N-class reduction remains off by default. Do not combine predicted or measured
gains with unrelated audio/frontend variants.

## First attempt2776917: internal side-beam mismatch, no performance result

Frozenc4820cb stopped at the first graph/state comparison:24 mask/error/full-normalizer tests passed,
but8/90 entries of the final `seq` side-beam history differed from torch normalization. The assertion
for returned output and preceding active/done/best sequence checks passed in that case. The old test
incorrectly required all internal branches to stay exact across an explicitly N-class normalizer; a
rounding-induced near tie is plausible but not yet proven. No naive regression or real-model timing ran.

Keep the original failure log in `outputs/runtime-2776917/tests.log`. A per-step compute-node
diagnostic now compares candidates and selected scores on identical inputs, verifies exact masks/error
flags and bounds any selected-score gap by the measured numerical perturbation. Save the two search
trajectories and final outputs, including any branch differences. The graph/bucket/limit/reuse test
now requires exact full state against eager execution of the same blockwise arithmetic. The24 direct
torch-reference numerical tests remain, and full real-model output comparisons are still required.
This separates graph/state correctness from the already-declared N behavior without concealing either.
The runtime reduction implementation has not been changed; no speed or adoption claim is made.

## Diagnostic2777145: numerical branch difference confirmed; timing pending

The same-input diagnostic passes exact mask/error checks. Maximum candidate perturbation is
1.1920928955078125e-7; selected scores have zero gap under the same-input torch comparison.
The two independently accumulated searches retain different side-beam histories while their
returned output matches. This supports rounding propagation through tied paths in this synthetic
fixture; it does not establish real-model equality or quality.149 LSE/beam tests now pass, including
exact graph-versus-eager state with the same arithmetic, and31 naive beam/greedy/timestamp regressions
pass. [Per-step trajectories and numeric checks](beam-lse-diagnostic-2777145.json). The allocation's
later independent audio API diagnosis fails, so the job is not globally successful. Full-model LSE
performance remains unmeasured and the default stays off.


## First full result2777366: public opt-in replication prepared

Frozen6a15d1f finishes F/exit0 with149 focused and31 naive checks. The same checkpoint/engine
comparison uses11 alternating repeats; native frontend capture is common to both variants.

|Model / corpus|B|RTFx gain|Paired group median|Improved groups|Repeat-total gain range|
|---|---:|---:|---:|---:|---:|
|1B English73|1|1.158%|1.127%|72/73|0.830..1.369%|
|1B English73|3|0.547%|0.841%|24/25|-1.199..1.242%|
|1B Japanese12|1|1.217%|1.222%|12/12|0.852..1.503%|
|1B Japanese12|3|0.871%|0.777%|4/4|0.547..2.268%|
|102M English73|1|2.199%|2.147%|73/73|1.649..2.934%|
|102M English73|3|2.674%|2.460%|25/25|1.865..3.227%|
|102M Japanese12|1|2.282%|2.006%|12/12|1.360..3.217%|
|102M Japanese12|3|2.422%|2.016%|4/4|1.773..5.406%|

All full tokens, timestamp IDs, segments, text and decoder call counts match on the corpus cases.
All additional no-timestamp, translation, prompt, silence/noise,30s repeated speech and110s cases
also match with bound stopping on/off. WER/CER is unchanged (1B English45/1150 words, Japanese26/592
characters;102M94/1150 and43/592). Timestamp quality is baseline agreement, with no time annotations.
Pooled request p95 improves0.93–1.33% for1B and1.80–2.57% for102M. Aggregate peak allocated memory
is essentially unchanged, but both sessions are resident; this is not standalone memory accounting.
Extra partial-normalizer scratch remains5.7KiB forB3/K5/V50002. Cold/capture components are recorded,
not attributed as clean standalone JIT differences. Eager instrumented event phases are not CUDA
graph-node attribution. [All samples, hashes, quality, cold/dispatch and comparisons](beam-lse-results-2777366.json).

`beam_fused_lse=False` is prepared as an explicit public opt-in, forwarded to beam sessions only.
CUDA beam kernels must be enabled/available; otherwise the existing normalizer is retained. The
first measured agreement does not turn this N-class reduction into an exact guarantee. Keep the
default off. The independent `validate_beam_lse_api.pbs` tests public constructor routing, K2/3/5/8,
repeat/session eviction, greedy, translation, long prompts and disabled-kernel fallback, then repeats
all eight measured corpus conditions through `--verify-api`. Expanded public verification remains
pending until that run completes; no extra default or combined optimization gain is claimed.


## Independent public verification2777441: adopt opt-in, retain defaultOFF

Frozen9552af9 finishes F/exit0 with36 public-option/routing tests. The prior149 kernel/state and31
naive regression checks are retained by source/artifact hash; there is no reduction-kernel change.
Both checkpoints pass all12 public cases: K2/3/5/8, no timestamps, translation, long prompt, repeated
beam/session reuse, greedy direct/stream/translation and disabled-kernel fallback. Actual session
flags and fallback dispatch are checked. All eight full-corpus cases also pass the independent
public API output/dispatch comparison and six extra cases. All returned tokens, timestamp IDs,
text/segments and decoder call totals match the same-engine reference; WER/CER stays unchanged.

|Model / corpus|B|RTFx gain|Paired group median|Improved groups|Repeat-total gain range|
|---|---:|---:|---:|---:|---:|
|1B English73|1|1.175%|1.191%|73/73|0.829..1.371%|
|1B English73|3|0.765%|0.768%|25/25|0.420..1.074%|
|1B Japanese12|1|1.256%|1.104%|12/12|0.988..1.757%|
|1B Japanese12|3|0.876%|0.871%|4/4|0.424..1.918%|
|102M English73|1|2.145%|1.913%|73/73|1.869..3.384%|
|102M English73|3|2.234%|2.099%|25/25|1.776..3.319%|
|102M Japanese12|1|2.540%|2.518%|12/12|2.155..2.891%|
|102M Japanese12|3|2.728%|2.419%|4/4|2.199..5.033%|

Every one of the11 repeat totals improves in each condition. These are loaded-waveform,
warm single/fixed-batch measurements; thread prefetch and grouped/tail options are not enabled.
`beam_fused_lse=True` is accepted as a measured opt-in for these H200 shapes. The default remains
False because changes in reduction order can affect other near ties, checkpoints or inputs.
Unsupported/disabled beam kernels retain torch.log_softmax. Greedy decoding is unaffected.
No general bitwise guarantee, reference-time accuracy, standalone memory advantage or combined
speedup is implied. The separate four-preset experiment measures interaction with grouped/tail.

[All samples, quality, public dispatch, hashes and cold measurements](beam-lse-api-results-2777441.json)
retain the raw analysis path/hash; full API outputs are in `outputs/runtime-2777441/`. PBS reports
Stageout_status1 separately, although exit_status0 and all expected artifacts were verified.

```python
model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", beam_fused_lse=True)
result = model.transcribe(audio, beam_size=5, timestamps=True, batch_size=1)
```
