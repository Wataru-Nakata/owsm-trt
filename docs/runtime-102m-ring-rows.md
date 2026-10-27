# 102M existing Ring row route (measured opt-in N route)

The actual 102M multi-row route in job2773841 was TensorRT (`no kernel for rows`); the default
Gluon implementation rejects non-power-of-two D=384/F=1536. The existing Ring implementation
supports that width and exposes `decoder_kernel="ring:rr"`, and job2774140 measured its multi-row timing/quality on the English/Japanese corpora below. This is evaluation of an existing route, not a new decoder.

`scripts/profile_ring_programs.py --sweep rows` compares the unchanged auto backend with a
separate routing object that shares the existing Ring weights/encoder/TensorRT engines. Only
the row route changes, within the existing 32-token and 8-row limits. Reference failure/routing
state stays independent. B1 single-row calls keep the same Ring kernel in both modes. Unsupported
larger windows still use TensorRT. A separate diagnostic pass records actual drafted dispatch
and plain/beam decoder calls (post-output position/step reads, no per-step host reads). These
counters are excluded from the warmed wall samples.

This is N: different matrix/attention arithmetic can change logits and near-tie tokens. Outputs
are recorded without requiring cross-variant equality; repeat nondeterminism, failed kernel
trials and new steady-state graph capture are errors. ASR WER/CER and token/timestamp/segment
agreement are reported separately. Translation and reference timestamp annotations are unavailable.
The auto default is unchanged; measured quality differences prevent an E/compatibility claim.

Prepared `profile_102m_ring_rows.pbs`: D384 kernel checks including R3/R5/R8 S1 and R8 S4, plus
real-checkpoint checks against fp32; English73/Japanese12, nine alternating whole-list repeats
for plain B3, timestamp B3, translation B8, timestamp beam5 B1, and drafted timestamp B8. Wall,
cold load/capture and dispatch instrumentation are separate. Both variants' sessions coexist,
so allocator peaks and compiler-cache-biased cold observations are not isolated deployment gains.


## Gate failure2774096: no speed measurement

Frozen41a7d86 passed19 D384/scheduling kernel tests, but the real102M checkpoint failed an
additional row check: max absolute logit difference divided by reference range0.0744 (gate<0.02),
argmax agreement0.88. The process stopped before any corpus timing or quality summary; Japanese
was not run. The old script saved shape details only after all checks, so this artifact cannot
distinguish R8 S1 from R8 S4. No speed/output-compatibility claim or production routing change
is made from this failed job. Logs remain in `outputs/runtime-2774096/`.

`diagnose_ring_rows.py` is prepared to persist every R1/R3/R5/R8 S1 and R8 S4 check, using both
the64-frame random input of the existing gate and the full encoder time dimension. It compares
CPU/GPU fp32, GPU fp16, Ring and (only at supported encoder shapes) the same TensorRT engines,
including per-layer new-KV error and Ring replay determinism. Additional real-audio prefill
states cover R3/R5/R8. It does not relax the production gate, run performance measurements or
interpret a numerical diagnosis as adoption. Source/provenance and partial cases are retained.


## Numerical diagnosis2774118 and isolated quality experiment

All13 cases completed with finite, repeat-exact Ring output. The original failure was R8 S1,
T64 (7.443% scaled max logit error, argmax0.875). GPU fp16 also exceeded2% there (2.215%).
At the actual encoder length374, random R8 S1 errors were Ring3.692%, Torch fp16 2.914%,
TensorRT39.235%; CPU/GPU fp32 agreed within0.009%. Random R8 S4 also exceeded2% in all three
fp16 implementations. Errors accumulated through later layers. This is evidence of sensitivity
of these synthetic states to fp16 arithmetic; it does not prove all Ring differences harmless.

On actual-audio prefill states R3/R5/R8, Ring's scaled max errors were0.179%/0.220%/0.096%, with
all argmaxes matching CPU fp32. TensorRT errors were0.290%/0.290%/0.239%. These three single-step
checks do not establish corpus quality. [Complete diagnosis](runtime-ring-numerics-2774118.json).

The new `--row-numerics` profiler-only option enables an explicitly N quality experiment with
matching checkpoint, engine, runtime/kernel source, GPU and software evidence. It records the
known R8 random-state gate failure as **failed**, rather than raising only for that exact diagnosed
value; new numerical values, other shape failures, nonfinite data, barrier failures, source drift
and failed runtime trials still abort. `_check_kernel` and production defaults remain unchanged.
`scripts/profile_102m_ring_quality.pbs` completed the full English/Japanese comparison in2774140. The previously passed19 GPU
tests are preserved as prior evidence rather than rerun without a kernel change.


## H200 / 102M result2774140

Frozen df8815b, nine alternating warmed whole-list pairs, English73 and Japanese12, same
checkpoint/engines. Exit0; the scheduler reported Stageout_status1, but the exit marker and
all expected artifacts were published. All within-route repeats were deterministic. The known
synthetic numerical gate failure remains recorded as failed; no new runtime fallback occurred.

| Condition | English RTFx change | Japanese RTFx change | Changed full outputs (English/Japanese) |
|---|---:|---:|---:|
| Plain ASR B3 | +34.97% | +46.33% | 2/73, 0/12 |
| Plain timestamp ASR B3 | +35.65% | +46.89% | 3/73, 1/12 |
| Translation B8 | +8.37% | +21.04% | 2/73, 7/12 |
| Timestamp beam5 B1 | +10.51% | +11.27% | 1/73, 1/12 |
| Drafted timestamp ASR B8 | +0.29% (noise) | +7.97% | 0/73, 0/12 |

All paired whole-list gains were positive except English drafted ASR (-3.40..+8.74%).
Separate diagnostic decoder counts matched between routes. All timestamp token IDs matched.
English beam changed token segmentation for one output with identical text/segments; Japanese
beam changed one text character while aggregate CER stayed43/592 (7.26%).

English plain ASR WER worsened99/1150 to100/1150 (8.61% to8.70%), with and without timestamps.
Japanese timestamp ASR CER worsened50/592 to51/592 (8.45% to8.61%); plain ASR stayed51/592.
Beam WER stayed94/1150 (8.17%), and drafted WER/CER were unchanged. Japanese translation was
poor in **both** routes, including repetitions and mixed-language output: four of12 outputs
reached374 tokens in both. No reference translations or time boundaries are available, so the
translation timings do not establish acceptable translation quality or timestamp accuracy.

The existing public option is `decoder_kernel="ring:rr"` on `OWSM.from_pretrained(...)`.
It remains opt-in N. Default `auto` continues using TensorRT for102M multi-row calls. This is
an evaluation of the existing decoder, not a new kernel or an output-compatible replacement.
The measured shape envelope is H200/102M R<=8 and R*S<=32; larger windows fall back to TensorRT.
Other GPUs, long recordings and silence/noise were not evaluated in this job.

Reproduce the immutable N comparison with the matching frozen source and diagnosis:

```bash
python scripts/profile_ring_programs.py --model /local/owsm_v4_base_102M \
  --manifest /local/manifest.json --lang eng --sweep rows \
  --row-numerics /local/row_evidence.json --repeat 9 --out /local/results/english
```

Use the Japanese12 manifest and `--lang jpn` for the second process. Run inside a compute
allocation. `--row-numerics` rejects mismatched source/checkpoint/engines; do not copy the
exception to changed source. Whole-list p50/p95 and variation are in the linked report; these
are not per-request latency quantiles. Cold observations have compiler-cache order bias,
allocator peaks include both resident variants, and GPU utilization samples are process-wide.
No isolated cold-start/VRAM or reference timestamp improvement is claimed.
[Report, output differences and environment](runtime-102m-ring-results-2774140.json).
Raw outputs, wall samples and per-call diagnostics: `outputs/runtime-2774140/`.
