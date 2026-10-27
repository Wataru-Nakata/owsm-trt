# Wider102M Ring (verified N opt-in; auto unchanged)

Candidate23, N investigation. The actual-graph trace2777712 shows TensorRT decoder GEMMs,
attention and norm kernels dominate102M R15 beam and R8 translation. The measured existing
`ring:rr` route supports at most8rows; default auto retains TensorRT there because this is a
numerically different implementation. No existing gate or routing limit is raised here.

`diagnose_ring_rows.py --wide-one-token` directly examines the existing kernel at R10/15/16,
S1 on H200132SM with102M D384/F1536/L6/H6. Each shape gets a separate explicit Ring instance
with60/90/96programs (one per row/head), all below132SM. No dynamic routing attribute mutation
or concurrent use is needed. The kernel accepts at most32tokens; these cases fit its existing
MTP16/HL2 layouts. Workspaces scale with M/R/H, and attention loops traverse row/head work
with the program stride. Both the original24program and proposed row/head schedules use NS1;
no cross-attention padding or KV layout changes are proposed. This source audit motivates the
experiment; it is not evidence of safe occupancy, correct output or speed on wider rows.

Six compute-only synthetic tests compare original24programs and row/head schedules, graph
replays, per-row offsets including0, cache restart, all cache writes and argmax, plus existing
fp32 error tolerances. Cross lengths64/374 and R10/15/16 are covered. The real102M checkpoint
then has nine numerical comparisons: random KV at64/374 and real audio/prefill for each R.
CPU/GPU fp32, GPU fp16 and applicable same-engine TensorRT references, exact repeat output,
preserved cache prefixes and per-layer new-KV errors are retained. Real input examples are
English; corpus-level beam quality and Japanese are not part of this diagnostic.

A random-state failure remains a failure in the report. A finite/argmax gate failure is not
silently waived or turned into a speed result. This job performs no latency comparison and
changes no public option or default. Any later N benchmark must review these numerical
results and evaluate full token/text/timestamp differences and quality before adoption.

```sh
python scripts/diagnose_ring_rows.py --model /local/owsm_v4_base_102M \
  --manifest /local/manifest.json --wide-one-token --out /local/numerics
```

Run with the staged `scripts/diagnose_ring_wide.pbs` on an allocated compute node. Numerical results are below; performance and corpus quality remain pending.


Job2777869 (585f842) passed6synthetic tests and completed9numerical cases. Every Ring result
was finite, repeat-exact and preserved the existing cache prefix. P24vs60/90/96 synthetic
logits, complete cache writes and argmax matched exactly through graph reuse. Exit0 and all
artifacts were saved; scheduler Stageout_status1 is retained separately.

On real-audio prefills, scaled maximum logit error against CPU fp32 was0.0842%/0.1007%/0.1232%
forR10/15/16, with100%argmax agreement. At random full-length KV, Ring exceeded the existing2%
gate atR10(3.889%) andR15(2.033%); these remain **failed**. TensorRT errors for the same cases
were3.263%/5.321%, and GPUfp16 was3.854%/1.283%. CPU/GPUfp32 agreed within0.0018%over allcases.
This indicates numerical sensitivity of those synthetic states; it does not establish harmless
corpus differences. The64-frame Ring random cases passed the existing gate, although argmax
agreement was90%/93.3%forR10/15. No gate was relaxed.

The next step is an explicitly N-class, profiler-only comparison of existing auto dispatch and
wide one-token Ring forR10/15/16. Only those exact shapes change; smaller rows, prefill, wider
windows and unsupported configurations retain the base backend. Full outputs/quality must be
reported, and any nonfinite/error/fallback aborts. No default or public option is changed.
[Complete numerical evidence and source hashes](runtime-ring-wide-numerics-2777869.json).


First corpus comparison2777897 (ea9d0ec), same engines/checkpoint and11alternating warmed pairs:

| Condition | English RTFx change | Japanese RTFx change | Changed full outputs English/Japanese |
|---|---:|---:|---:|
| Beam5 timestamps B2 | +26.33% | +35.24% | 1/73, 0/12 |
| Beam5 timestamps B3 | +24.52% | +35.41% | 1/73, 0/12 |
| Translation B10 | +23.90% | +21.68% | 5/73, 7/12 |
| Translation B16 | +15.99% | +0.09% unchanged R12control | 2/73, 0/12 |

Every measured changed-shape condition improved in all11paired corpus totals. B1beam is an
unchanged control (+0.88%English/+0.45%Japanese); these small shifts are not kernel gains.
B2/B3/B10/B16 samples measure the whole corpus request, not individual utterance latency.
English B3 p50/p95 was431.35/437.60ms→346.41/349.89ms; Japanese95.12/97.62→70.66/70.96ms.
All warm captures/fallback/determinism checks and10route guards passed. Exit0; Stageout_status1
is retained separately from the complete published artifacts.

Both beam shapes retain all timestamp tokens and valid segment boundaries. The one English
change is an inserted comma in1272-141231-0004. WER remains94/1150(8.17%); Japanese CER remains
43/592(7.26%). Beam decoder calls change1032→1036 and752→756 in English,261→257 and173→169
in Japanese; speed is therefore not purely a fixed-call kernel ratio. No reference time
annotations exist, and these small corpora do not establish general quality parity.

Translation changes include text and repetition differences; no translation references are
available. English silence changes one repeated hallucinated transcript to another; both
reference/candidate have empty segment lists. Noise/speech in that group match. Other extras
(no timestamps, beam translation, long prompt, mixed30swindow) match for both languages;
Japanese silence/noise also match. No claim of silence or translation quality improvement.

Independent replication and public opt-in integration/verification remain pending. Auto and
production routing stay unchanged. Both variant pools coexist, cold/JIT order is biased and
allocation-wideGPU telemetry includes all phases. The result is an N experiment, not bitwise
replacement of TensorRT or new model arithmetic.
[Full differences, quality, timing and environment](runtime-ring-wide-results-2777897.json).


Independent frozen-code replication2777935 completed10route tests, all10conditions and all
extras. Every baseline/candidate output exactly matches2777897, including all previously
reported N differences. Changed-shape cases again improved in all11paired totals:

| Condition | English repeat RTFx change | Japanese repeat RTFx change |
|---|---:|---:|
| Beam5 timestamps B2 | +27.32% | +36.05% |
| Beam5 timestamps B3 | +24.09% | +35.64% |
| Translation B10 | +24.38% | +21.98% |
| Translation B16 | +16.21% | +0.16% unchanged R12control |

Output/quality/decoder-count differences remain as above. B1beam and Japanese B16 are unchanged
controls and their small timing shifts are not gains. Public `decoder_kernel="ring:wide"`
integration is prepared, with26CPU routing/fallback checks passed; GPU API verification remains
pending. It starts from the auto route (including its existing single-row numerical check),
adds only the reviewed R10/15/16 S1 calls and retains logged auto fallback on unsupported
configurations or setup errors. Auto/default and existing `ring:rr` remain unchanged.
[Independent paired results](runtime-ring-wide-results-2777935.json).


Public integration2777975 (ffef64b) passed26routing/fallback tests, all10timing conditions
matched both modes of frozen2777897 exactly, and13APIcases per language matched the experimental
wrapper exactly. These include partial batches, draft verification, lookahead, long prompts,
silence/noise,30s windows,110s long-form and reuse. Long-form parity compares joined text/segments;
other cases include full token IDs. No new fallback or warm capture occurred.

The public route independently measures English B2/B3beam RTFx+26.54%/+24.24% and Japanese
+36.16%/+35.54%, all11pairs positive in each condition. English translationB10/B16 is+24.12%/
+16.47%; JapaneseB10 is+21.49%. The output/quality differences above are unchanged. Unchanged
B1beam/JapaneseB16controls have no speedup claim. This result supports the explicit N opt-in;
it does not make it output-compatible with auto or establish translation/silence quality.

```python
model = OWSM.from_pretrained("espnet/owsm_v4_base_102M", decoder_kernel="ring:wide")
results = model.transcribe(audio_paths, beam_size=5, timestamps=True, batch_size=3)
```

The default remains `auto`. Existing `ring:rr` is unchanged and is a different option: `ring:wide`
retains auto for rows<=8 and for all rows/windows outside exactly R10/15/16 S1decode. Thus beam5
B1 has no new route; beam5B2/B3 can use the wider kernel. Other GPU/model shapes retain auto,
with an explicit warning. No combined gain with LSE, encoder lookahead, prefetch or frontend
options is claimed. Only H200132SM/102M is measured; unsupported hardware fallback is covered
by CPU routing guards, not real-GPU performance testing.
[Public timing, frozen-output parity and26APIcases](runtime-ring-wide-api-2777975.json).
