# Warm frontend input reuse (unmeasured E experiment)

`OWSM._features` currently pads/copies audio into a temporary CUDA batch and then copies that batch
into the cached frontend graph input. The prototype fills an existing graph input directly, avoiding
the temporary GPU allocation and full-batch device copy. It preserves zero padding, trimming,
pinning, source dtype conversion, STFT/mel/normalization and returned-feature ownership.

The private `_frontend_direct_input` switch defaults to `False`. `FrontendReplay.try_waves` returns
`None` for cache misses, empty/unsupported batches, CPU, gradients, autocast and changed graph keys;
these requests retain the existing preparation/capture path. No graph is captured by the new method.
The cache key includes shape, default input dtype, canonical CUDA device and fused frontend flags.
Every input fill waits for the prior replay/output clone. The completion event is recorded even when
pinning or copying raises after queued writes. Exclusive native capture stream lifetime is unchanged.

The focused test has one CPU eligibility check and13 GPU checks covering global/utterance MVN,
fused/unfused kernels, B1/3/8, mixed CPU/GPU and fp32/fp64 sources, empty/cropped/padded input,
retained outputs across streams, cache/flag/autocast misses and a failure after queued buffer writes.
Existing frontend graph lifetime tests also run in the compute job.

`scripts/profile_frontend_direct.pbs` stages the environment, model, engines and audio on
`$PBS_LOCALDIR`. Separate feature/H2D measurements compare31 alternating warm calls for B1/3/8
with exact feature equality. Request measurements use the default graph batch limit1 and11
alternating repeats: 1B/102M English73/Japanese12, plain/drafted ASR with/without timestamps,
translation, B1 streaming, beam B1/B3, unchanged wider-batch controls and English110s. B3/8 graph
capture is explicitly enabled only in separate feature and API checks; no default expansion occurs.
Twelve additional API checks include encoder lookahead, GPU audio tensors, long prompts,
silence/noise,30s inputs and explicit graph limit8. Full outputs must agree with the existing path;
long-form exposes joined text/segments, with no joined token sequence.

Raw warm wall samples, cold captures, resident-pool memory, actual decoder dispatch, output
comparison and WER/CER are saved separately. Both session pools are resident during comparison;
peak memory is not isolated deployment memory. There are no human timestamp or translation
references. No speed improvement or public/default adoption is claimed before GPU validation.

Reproduce inside an allocated compute job:

```sh
python scripts/profile_frontend_direct.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --long-audio /local/long_110s.wav \
  --repeat 11 --out /local/results
```
