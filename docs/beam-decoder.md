# Decoder configuration ablations

Reference `b95cd8b` includes strict stopping, completion checks, candidate/finish kernels and the
measured prefix KV gather. Further experiments target the dominant decoder cost, starting with
existing Gluon configuration options. They are profiler-only and do not change runtime defaults.

On H200, R=5 (B=1, K=5) uses Gluon; R=15 remains TensorRT under the existing routing limits. Compare
132→128 persistent CTAs, attention prefetch enabled→disabled, and their combination. The current
1B GEMM tiling requires at least 128 programs and the persistent grid must fit available SMs.
Neither variant changes precision, arithmetic tiling, beam width or routing limits. Changes in
execution scheduling still require actual token comparison before adoption; speed is unmeasured.

`profile_beam.py --gluon-programs 128` and/or `--no-gluon-prefetch` build a separate backend wrapper
for candidate sessions, retaining the reference wrapper and its dispatch/failure state. Both share
immutable decoder weights and the same TensorRT engines; they execute serially. A focused synthetic
CPU test checks isolation, shared weights and the upper SM-count guard. Invalid/missing Gluon
configuration raises instead of silently creating a no-op comparison. Real dispatch/fallback is
saved per bucket, and requested/reference options are in `environment.json`.

The prepared `EXPERIMENT=decoder` job tests 221 cases, then alternates each candidate against the
reference for English73 (5 repetitions) and Japanese12 (3 repetitions), B=1/K=5/timestamps. Each
English run has separate eager profiling and six additional compatibility cases. All outputs stay
node-local until final publication, except coarse phase updates. Standing one-GPU authorization
and a fresh successful <=192 GPU admission audit remain required. No new GPU measurement or
runtime-default change is claimed by this preparation.

## Result: retain existing settings

Job `2768958.pbs1` passed 221 tests and completed all six comparisons. Every bucket dispatched to
Gluon, without fallback. English73/Japanese12 tokens, timestamps, segments and decoder call counts
all match; the six additional English cases match in every variant. WER/CER remain unchanged.

| Candidate | English RTFx change | Japanese RTFx change |
|---|---:|---:|
| 128 CTAs (reference132) | -0.17% | -0.33% |
| attention prefetch off | -1.71% | -1.70% |
| both | -1.79% | -1.84% |

There is no demonstrated gain to adopt. Defaults remain 132 CTAs on H200, attention prefetch on.
No routing expansion or precision change was attempted. The admission check counted176 allocated,
0 queued +1 new =177 <=192. [Complete statistics](beam-results-2768958.json) and raw artifacts
`outputs/beam-2768958/` preserve the negative result. The next experiment targets cross-KV storage
and reads shared across beam rows, rather than retuning these settings.
