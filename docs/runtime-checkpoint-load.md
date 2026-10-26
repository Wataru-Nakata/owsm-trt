# Private checkpoint mapping experiment

In2772365, model-load wall time with cached engines was about7.0–7.1s for1B in the later processes
(first observed12.2s), and1.88s for102M (first4.50s). These are prior observations, not new gains.
The cached-engine path calls `_load_frontend`, which deserializes the whole3.9GiB1B checkpoint to
extract small frontend buffers, then `_load_decoder`, which reloads the checkpoint and only copies
decoder tensors. Encoder tensors are not used by either helper when engines already exist.

`load_checkpoint(..., memory_map=True)` is now selected by the cached-engine frontend and megakernel decoder helpers. ZIP checkpoints use private mapping
when the PyTorch mapping configuration is private; accessing tensor contents faults in those pages.
Legacy files, file-like inputs, unsupported/private-option platforms or shared-map configurations use
ordinary loading. `weights_only=True` and load errors are preserved. No global PyTorch mapping options
are changed. The general loader default, full Torch model loading and engine builds retain ordinary loading; pipeline has no new option.

Small tests verify exact values/dtypes/strides, shared storage aliases, private writes leaving the
checkpoint unchanged, ZIP/legacy/file-like loading, actual mapped dispatch, shared-map fallback and
unchanged weights-only rejection. Nine initial tests plus the explicit dispatch test passed on CPU;
these use only tiny synthetic files. No real model/audio was loaded on the login node.

The implementation follows [torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)
and [private mapping semantics](https://docs.pytorch.org/docs/2.14/notes/serialization.html#torch.serialization.get_default_mmap_options).
PyTorch documents mapped storage loading and MAP_PRIVATE as the default. Existing model loading
copies selected tensors into module parameters, so this experiment avoids reading unneeded tensors;
it does not skip required model weights or change arithmetic.

`scripts/profile_model_load.py` uses a fresh subprocess per mode/sample, alternates eager/mapped
five times after a separate first cycle, and keeps Python import/CUDA initialization outside the
reported model-load wall time. OS page cache/JIT are warm and model/engines are node-local; no cache
is dropped. Changed engine files or engine/checkpoint hashes fail the comparison. Full worker wall
also includes validation/hash IO and is not a startup metric. Separate diagnostic processes time
checkpoint/frontend/decoder helper host calls; those timings are inclusive and overlap.

Each worker compares English/Japanese ASR, timestamps, translation, beam and language ID after load
(three inputs/language except language ID). Same-input outputs must exactly match across all modes
and processes. CPU peak RSS includes imported libraries/driver, and GPU peak is recorded before
validation; neither should be confused with steady-state inference speed. The profiler forces the requested mode even when production helpers explicitly request mapping.
Results and measured limits follow.

## Result: adopt selective loading

Job2773040, frozen346b71c, H200, Torch2.14.1/Triton3.8/TRT11.4/CUDA13.
19 focused tests passed in8.06s. Both model runs contain five alternating pairs of fresh processes,
two first-observation processes and two separately instrumented processes. All26 public requests
per worker (English/Japanese ASR, timestamps, translation, K5 timestamps and language ID) match
exactly across all14 workers/model. Checkpoint and engine hashes match; no decoder failures/builds.

| Model | Ordinary → mapped load median | Reduction | Paired reduction range | CPU peak RSS median | GPU peak allocated |
|---|---:|---:|---:|---:|---:|
|1B|6750.30 →4608.47ms|31.73%|29.43–33.74%|6634.52 →4251.66MiB|1946.13MiB, unchanged|
|102M|1865.11 →1610.96ms|13.63%|8.77–15.03%|1660.73 →1465.14MiB|696.78MiB, unchanged|

1B load range was6551–7130ms ordinary and4341–5032ms mapped;102M1834–1896ms and1591–1673ms.
All five paired samples improve. Instrumented torch.load calls verify `mmap=True`, CPU and
`weights_only=True` for both mapped loads; ordinary calls omit mmap. In the separate1B diagnostic,
frontend helper time fell1212→62ms and decoder2725→1607ms; helper timings overlap checkpoint
loading and must not be summed as independent phases.

First observed mapped loads were8843ms/3826ms versus subsequent ordinary6977ms/1888ms.
Those observations include uneven initial setup and are excluded from the paired startup claim.
No uncached-disk, Lustre-performance, engine-build, other-model/GPU or steady-state benefit is claimed.
CPU RSS includes imported libraries/driver; mapping reduces resident pages, not checkpoint size.

Raw results: `outputs/runtime-2773040/{load1b,load102m}/model_load.json`, tests/logs/environment,
plus `load_analysis.json`. Reproduce inside an allocation with:

```sh
python scripts/profile_model_load.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --extra-manifest /local/extra_manifest.json \
  --repeat 5 --out /local/results/load1b
```

The commit enables mapping only in the two selective helpers. Full model loading is unchanged.
Private mappings preserve source bytes and aliasing; malformed/restricted payload errors still
propagate. The profiler retains an ordinary-load comparison after adoption.
