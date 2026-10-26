# 102M existing Ring row route (unmeasured N experiment)

The actual 102M multi-row route in job2773841 was TensorRT (`no kernel for rows`); the default
Gluon implementation rejects non-power-of-two D=384/F=1536. The existing Ring implementation
supports that width and exposes `decoder_kernel="ring:rr"`, but multi-row timing/quality for
this corpus has not been measured. This is evaluation of an existing route, not a new decoder.

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
No default changes or performance claims are made before measurement.

Prepared `profile_102m_ring_rows.pbs`: D384 kernel checks including R3/R5/R8 S1 and R8 S4, plus
real-checkpoint checks against fp32; English73/Japanese12, nine alternating whole-list repeats
for plain B3, timestamp B3, translation B8, timestamp beam5 B1, and drafted timestamp B8. Wall,
cold load/capture and dispatch instrumentation are separate. Both variants' sessions coexist,
so allocator peaks and compiler-cache-biased cold observations are not isolated deployment gains.
