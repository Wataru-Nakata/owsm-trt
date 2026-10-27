# Wider102M Ring diagnosis (no production routing change)

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
