# Shared cross-attention KV ablation

Candidate19, enabled on supported CUDA shapes after the results below; explicit override `BeamSession(..., share_cross_kv=True)` / profiler `--share-cross-kv`.
Reference `3bc1152` includes the adopted prefix self-KV gather; decoder defaults are unchanged.

A Gluon decoder reads physical cross KV `[L, B, H, T, Dk]` for `R=B*K` utterance-major query rows.
Only its source address changes from row `r` to utterance `r//K`. Each query keeps its existing
independent attention, reduction order, cache and logits. The whole encoder time axis stays intact.
BeamSession aliases encoder/prefill cross KV to these buffers and omits K-fold expansion. The
cross-attention prefetch uses the same mapping. No query fusion, pruning or precision change.

The backend advertises support only for the existing Gluon row/token limits and successful kernels.
Thus H200 B=1/K=5 uses sharing; B=3/K=5 keeps TensorRT and expanded KV. If a shared kernel's trial
fails during session construction, the TRT closures allocate expanded buffers and expose a refresh
callback, run after every encode. This fallback is visible in per-bucket dispatch and may use more
memory; it must not be counted as a successful sharing measurement.

CPU checks cover buffer aliasing, unsupported backends and repeated encoder updates after a forced
trial failure across all buckets. Five allocation-only tests compare exact Gluon logits/self-KV and
CUDA graph replays, two utterance/beam layouts, prefetch on/off, repeated primer lengths and bucket
growth. Existing Ring tests exercise the unchanged shared attention helper signature. These GPU
checks are pending before the next job. Focused CPU checks:32 passed,6 skipped (the five new GPU
checks plus an existing CUDA-only case). No model/audio processing was run on the login node.

The next one-GPU allocation runs English73/Japanese12 at B=1 and B=3, same loaded checkpoint/engines,
with alternating baseline/candidate order. B=3 is a fallback regression/control (English's one-item
tail can still share KV). Actual sharing, fallback reasons, distinct KV-buffer bytes, kernel dispatch,
cold costs, outputs and independent eager traces are saved. Performance/adoption remain unverified.

## Measured result and adoption

Job `2769045.pbs1` completed in5m47s; 221 base tests plus12 additional tests passed. All73 English
and12 Japanese tokens/timestamps/segments/text and decoder call counts match the previous adopted
path. Six extra English cases match, including30s and110s audio. English baseline records also
match the previous job's baseline. No trial failures occurred. WER45/1150 and CER26/592 are unchanged;
reference timestamp annotation errors remain unmeasured.

| Input | B | Reference p50 → shared (ms) | Reference p95 → shared (ms) | RTFx change |
|---|---:|---:|---:|---:|
| English | 1 | 26.53 → 25.41 | 56.90 → 54.40 | +4.29% |
| Japanese | 1 | 36.26 → 34.69 | 86.31 → 82.53 | +4.68% |
| English | 3 | 49.43 → 49.44 | 103.55 → 103.57 | +0.08% (control) |
| Japanese | 3 | 76.58 → 76.27 | 129.02 → 128.86 | +0.22% (control) |

Every B=1 group improves by paired median (English4.53%, Japanese4.59%), above median within-group
CV of0.09–0.54%. At B=3 the actual decoder stays TRT; do not interpret its tiny change as a gain.
Distinct cross-KV buffers shrink from165,445,632 to27,574,272 bytes at B=1, a137,871,360-byte saving.
This includes removal of both K-fold decoder copies and the redundant encoder/prefill allocation;
it is a measured tensor-storage saving, not isolated process peak VRAM.

The runtime default is now sharing when the backend supports this shape. `share_cross_kv=False`
selects expanded buffers directly; the public `beam_kernels=False` switch disables sharing alongside
the other beam kernels. H200/Gluon R=5 is measured; R=15 retains TRT. Other GPU/shape performance
is unmeasured. The Ring attention-helper API and greedy numerical checks remain compatible.

The admission audit counted176 allocated+0 queued+1 new=177; monitoring observed at most185 GPUs
and a final184. [Complete results](beam-results-2769045.json), including hashes, dispatch, capture
and diagnostic data, correspond to raw `outputs/beam-2769045/`. Reproduce the prior command with
`--share-cross-kv --baseline-runtime <3bc1152 runtime>` and otherwise the same prefix/fused options.
The next separate experiment groups sibling beam queries into one cross-attention tile.
