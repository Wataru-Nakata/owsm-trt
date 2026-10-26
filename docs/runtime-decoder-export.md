# Avoid unused decoder ONNX export when reusing engines

`build_engines(..., decoders_from=...)` copies existing decoder/verify engines, but previously
exported their unused decoder ONNX graph and external weights first. FP8 builds use this route.
The prototype extracts `export_encoder_onnx` from the unchanged joint exporter and introduces
`skip_decoder_export=False` on `build_engines`. True skips only that unused export when decoder
engines are supplied; ordinary full builds and CTC-only builds retain their behavior. The default
remains False until compute-node validation and timing. No engine tag/options/graph precision changes.

Three lightweight routing tests pass: copied engine bytes, plugin metadata and profile entries
are preserved, and a normal build still exports both graphs. Actual ONNX numerical checks now
include the encoder-only entry point in the existing untrimmed encoder test; these are prepared
for compute-node execution, not yet run on this revision.

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
