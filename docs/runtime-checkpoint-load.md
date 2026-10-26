# Private checkpoint mapping experiment

In2772365, model-load wall time with cached engines was about7.0–7.1s for1B in the later processes
(first observed12.2s), and1.88s for102M (first4.50s). These are prior observations, not new gains.
The cached-engine path calls `_load_frontend`, which deserializes the whole3.9GiB1B checkpoint to
extract small frontend buffers, then `_load_decoder`, which reloads the checkpoint and only copies
decoder tensors. Encoder tensors are not used by either helper when engines already exist.

`load_checkpoint(..., memory_map=True)` is an opt-in experiment. ZIP checkpoints use private mapping
when the PyTorch mapping configuration is private; accessing tensor contents faults in those pages.
Legacy files, file-like inputs, unsupported/private-option platforms or shared-map configurations use
ordinary loading. `weights_only=True` and load errors are preserved. No global PyTorch mapping options
are changed. The ordinary default remains unchanged pending measurement; pipeline has no new option.

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
validation; neither should be confused with steady-state inference speed. Script/PBS defaults do
not change public runtime settings. Performance and real-model results remain unmeasured.
