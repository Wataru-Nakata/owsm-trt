# Avoid unused decoder ONNX export when reusing engines

`build_engines(..., decoders_from=...)` copies existing decoder/verify engines, but previously
exported their unused decoder ONNX graph and external weights first. FP8 builds use this route.
The implementation extracts `export_encoder_onnx` from the unchanged joint exporter and introduces
`skip_decoder_export=True` on `build_engines`. True skips only that unused export when decoder
engines are supplied; ordinary full builds and CTC-only builds retain their behavior. This is now the default; False retains the original joint export. No engine tag/options/graph precision changes.

Five lightweight routing tests pass, including both omitted-option routes: copied engine bytes, plugin metadata and profile entries
are preserved, and a normal build still exports both graphs. Actual ONNX numerical checks now
include the encoder-only entry point in the existing untrimmed encoder test; all three variants passed on the compute node.

`scripts/profile_encoder_export.pbs` runs the actual ONNX and FP8-scale tests, then profiles the
1B packaged-FP8 encoder export in fresh processes: three alternating full/encoder-only pairs,
with two first workers excluded. Each worker loads the same checkpoint/scales before timing.
Every worker's encoder ONNX and external-data hashes must match exactly. Decoder export file
bytes are recorded. This measures an export component, **not** model loading, total TensorRT
build time or steady-state inference. Warmed shared OS/JIT caches and per-process compiler costs
are disclosed; no speed gain is assumed.

A separate ninth worker builds a real TensorRT FP8 encoder with `skip_decoder_export=True`,
forbids decoder-export calls, verifies copied decoder/verify hashes, then runs B1/B3 encoder
bindings and checks finite cross KV and valid CTC IDs. Its build time is an observation without
an isolated build-speed comparison. Temporary ONNX/engine files are removed only from directories
created by the worker under `$PBS_LOCALDIR`; small hashes, timings and logs are published.


Job2773776 (frozen2c2670e, H200, torch2.14.1/CUDA13.0/TensorRT11.4.0.106) passed
7 tests, all nine encoder ONNX/external-data hash comparisons, and the real TRT build smoke.
Three timed fresh-process alternating pairs gave full-export p50 **50.579s** versus encoder-only
**26.636s**, a **47.34% export-component reduction**. The individual paired reductions were
46.76%,46.71%,47.46%. The skipped decoder graph/external weights total745,192,536 bytes.
Export GPU peak was unchanged at4,489,243,136 bytes. The smoke's89.61s total build is not a paired
build-speed result. No new inference engine is required for existing caches, and no steady-state
inference gain is claimed. The packaged FP8 quality limitations remain unchanged.

[Raw timings, hashes and smoke evidence](runtime-decoder-export-results-2773776.json).
Raw logs are in `outputs/runtime-2773776/`; reproducer: `scripts/profile_encoder_export.pbs` with
an immutable source snapshot, one allocated H200 and the same cached 1B checkpoint/decoder engines.
The wall benchmark excludes first workers and model load; file hashes are measured outside timing.
