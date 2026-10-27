# Fixed-batch file prefetch

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
including an injected second-fixed-batch read failure and reuse, against saved pre-extension
outputs from2777327. It then measures11 alternating
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

The first completed GPU comparison is below; independent replication is pending. This extension does not enable
other optional normalization/grouped/tail/frontend-batch changes, so their gains are not multiplied.


## First public comparison2777564

Frozenb182cb3, H200, F/exit0,85 focused tests pass. Both checkpoints pass all15 public API cases
for each worker count0/1/2 against saved2777327 outputs, including the second-fixed-batch read
failure and subsequent reuse. All16 workload/corpus conditions match full tokens, timestamp IDs,
text/segments and decoder call totals. Worker dispatch is verified; no warm capture or fallback.

|Model / corpus|Beam TS B1 workers1/2|Beam TS B3 workers1/2|Translation B1 workers1/2|Translation B8 workers1/2|
|---|---:|---:|---:|---:|
|1B English73|+5.61/+5.38%|+8.46/+8.46%|+8.90/+7.85%|+22.92/+23.27%|
|102M English73|+10.96/+10.73%|+24.43/+23.28%|+25.85/+24.77%|+58.02/+57.35%|
|1B Japanese12|+0.56/+0.23%|-1.01/+2.93%|+0.70/+0.80%|+0.54/+0.35%|
|102M Japanese12|+0.56/+0.69%|+1.26/+0.61%|+0.32/+0.05%|+16.79/+15.82%|

These are RTFx changes for full-file-list requests, including duration sorting, reads and worker
startup/join. Every English repeat pair improves in all conditions. English workers1 whole-request
p95 improves7.28–18.61% for1B and9.57–36.44% for102M. More workers do not consistently help.

Japanese aggregates are outlier-sensitive and do not support a broad gain.102M translation B8
worker1's +16.79% aggregate has paired median+0.051% and7/11 positive pairs; worker2 has paired
median−0.125%,4/11 positive. One reference outlier makes a pair appear+186% faster.1B Japanese
beam B3 worker1 has paired median+0.387% but one−13.13% pair, aggregate−1.01% and p95+7.35%.
All pairs and output evidence remain in the artifact; do not infer language/codec routing policy.

The extension remains under the existing opt-in worker count and default0. Keep encoder-lookahead,
CTC-only and long-form exclusions. Independent same-engine replication follows before publishing
these as repeatable fixed-path gains. Warm node-local English FLAC versus Japanese WAV does not
predict cold filesystem or arbitrary codec behavior. Translation equality is relative to baseline,
not reference translation quality. ASR WER/CER is unchanged; reference timestamps are unavailable.

[All samples, quality, public API, hashes, cold/dispatch and coarse GPU telemetry](runtime-fixed-prefetch-2777564.json).
PBS Stageout_status1 is separate from verified exit0 and complete artifacts. Raw outputs remain in
`outputs/runtime-2777564/`. No single-file, loaded-input or combined-opt-in speedup is claimed.


## Independent replication2777617: accept fixed-batch opt-in, retain default0

Frozen9f2c6ce finishes F/exit0. All16 full token/time/text/segment and decoder-call comparisons
match again with11 alternating file-inclusive repeats; actual reader dispatch is verified. The85
tests and15 public API cases/model are retained by hash from2777564, not reported as rerun. The
separate singleton-header guard does not change ordering or execution for these multi-input calls.

|Model / corpus|Beam TS B1 workers1/2|Beam TS B3 workers1/2|Translation B1 workers1/2|Translation B8 workers1/2|
|---|---:|---:|---:|---:|
|1B English73|+5.58/+5.92%|+8.76/+8.95%|+8.90/+9.15%|+22.12/+22.64%|
|102M English73|+10.76/+11.15%|+22.89/+22.33%|+24.65/+24.73%|+57.69/+56.17%|
|1B Japanese12|+4.15/+4.25%|+0.58/+0.26%|+0.40/+0.34%|+1.32/+1.43%|
|102M Japanese12|+0.89/+0.97%|+1.51/+0.88%|+0.21/-0.46%|+0.21/+0.01%|

All English pairs improve again. For English translation B8 worker1,1B whole73-file p50 falls
558.83→456.28ms and p95 569.48→467.89ms;102M p50 falls276.57→177.91ms and p95 288.54→178.75ms.
Japanese remains unsuitable for a blanket recommendation:1B beam B1's +4.15% aggregate is
outlier-heavy (paired median+0.534%);102M translation B8's earlier +16.79% collapses to+0.21%,
paired median+0.167%. Worker2's102M Japanese translation B1 regresses in10/11 pairs.

Accept this extension of the existing worker-count opt-in for sequential fixed beam/translation
batches, keep default0 and all exclusions. Select worker counts using actual file-inclusive
workloads; no automatic language/model/codec policy is added. Two independent English repetitions
support the measured improvement, while Japanese negative/noisy samples are retained. This is
not a single-request, loaded-waveform, cold-storage or composed-preset speed claim.

[Replication pairs, quality, prior-check hashes, dispatch, cold and telemetry](runtime-fixed-prefetch-2777617.json).
Full raw outputs are in `outputs/runtime-2777617/`. Stageout_status1 is separate from exit0 and
verified complete artifacts. The singleton-header experiment in that allocation is separate.
