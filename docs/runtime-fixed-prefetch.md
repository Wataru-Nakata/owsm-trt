# Fixed-batch file prefetch (pending GPU measurement)

Candidate40 extension. Earlier same-engine file-inclusive greedy measurements establish that
CPU file loading can be material (`runtime-audio-prefetch.md`), especially English FLAC. Beam and
translation still load each fixed batch on the calling thread before decode, with no file overlap.
This experiment applies the existing bounded loader to that path through `audio_prefetch_workers`.
It does not infer a beam/translation speedup from the greedy figures.

With workers1/2 and an all-file list spanning multiple batches, the next batch may load while the
current fixed batch decodes. Duration sorting, batch boundaries, primers, partial batches, result
order, arithmetic and model engines are unchanged. At most two submitted batches include the current
one; CPU workers own separate file handles. All feature/H2D/model work stays on the caller thread.
Single batches, mixed/loaded inputs and CPU autocast load synchronously; CTC-only, long-form and
encoder-lookahead routes retain their existing loading. Default0 remains synchronous.

Twenty-one tiny model-free routing/error tests cover beam/translation/long-prompt paths, sorted
and restored order, partial batches, exclusions, ordered reader/consumer errors and subsequent reuse.
Together with the existing bounded-loader/options/lookahead-gate tests,60 CPU checks pass. Real
resampling, CUDA encoder lookahead and real-model API checks run only in an allocated job.

`scripts/profile_fixed_prefetch.pbs` stages code/environment/audio/models/unchanged engines on
`$PBS_LOCALDIR`, runs focused checks and all15 public API cases per checkpoint/worker setting,
including an injected second-fixed-batch read failure and reuse. It then measures11 alternating
file-inclusive comparisons of workers0/1/2 on1B/102M, English73/Japanese12, timestamped beam5 B1/B3
and translation B1/B8. All are whole-corpus requests: B1 means sequential one-utterance batches,
not individual-request p95. Loading, worker startup/join and duration sorting stay inside timing.
The OS file cache is warm; no cold shared-storage/network gain is implied. Outputs and decoder
calls must match, actual reader threads/dispatch are recorded outside timing, and new warm captures
or silent fallback fail the run. Aggregate memory includes all benchmark session pools.

```sh
python scripts/profile_audio_prefetch.py --fixed --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --lang eng --repeat 11 --out /local/results
```

No speed claim is available until the GPU comparison finishes. This extension does not enable
other optional normalization/grouped/tail/frontend-batch changes, so their gains are not multiplied.
