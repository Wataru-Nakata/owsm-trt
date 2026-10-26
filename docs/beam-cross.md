# Shared cross-attention KV ablation

Candidate19, opt-in `BeamSession(..., share_cross_kv=True)` / profiler `--share-cross-kv`.
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
