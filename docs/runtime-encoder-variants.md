# Encoder head and packaged FP8 experiments

Candidates41/43, independent from the adopted decoder and lookahead changes. No production
routing or default changes. `scripts/profile_encoder_variants.py` compares four encoder variants:

* Existing cached fp16 encoder (reference).
* Fresh ordinary fp16 encoder, to expose TensorRT rebuild variation.
* Fresh fp16 encoder with the external CTC draft head omitted. It keeps the same encoder and
  cross-KV projection and returns zero int32 CTC IDs for the existing binding contract. It only
  runs with `ctc_draft=False`; no internal encoder block or padded attention frames are removed.
* Existing `fp8=True` API with the checkpoint-matched packaged scales. This is an independent A
  quality experiment. An older audio-calibrated cached engine has a different calibration digest
  and is deliberately not substituted for the packaged-scale build.

All variants use byte-identical decoder/verify engines and the same checkpoint. Head removal is
E intent but rebuilding TensorRT can change arithmetic/tactics, so observed outputs are compared
as N. Experimental engines live only in explicitly named compute-local directories; the normal
factory cache and defaults are unchanged. Do not load the no-head variant with greedy CTC drafts.

The experiment records encoder-only CUDA graph replay times for B1/3/8 and short/long inputs,
full raw cross-KV equality and max/mean absolute differences, separately warmed whole-list wall
p50/p95/RTFx, full tokens/time/segments/text, WER/CER and independent beam diagnostics. It rotates
variant timing order. Encoder timings exclude feature/H2D and capture, and are not end-to-end
latencies. Shared-process peak memory includes all variants. Unequal first-use compiler caches
make cold observations unsuitable for claiming startup gains. No output equivalence is assumed;
each variant must remain stable across its own repeated calls.

Input scope: English73, Japanese12; beam5 timestamps B1/B3 and plain translation B1/B8, plus
silence/noise, long prompt, no-timestamp beam and110s long-form functional checks. Translations,
synthetic inputs and timestamp boundaries have no reference annotations here. Each beam diagnostic
run is separate from the uninstrumented wall measurements. Drafted greedy performance is outside
this no-head experiment.

Three small real-model graph tests check exact cross KV, shape/dtype binding and that CTC is never
called. `scripts/profile_encoder_variants.pbs` runs them, builds/measures inside a1GPU allocation,
and publishes results plus a few engine archives. Engines/ONNX/temporary files remain node-local
until publication. Current status: prepared; GPU measurements pending. No speedup is claimed.
