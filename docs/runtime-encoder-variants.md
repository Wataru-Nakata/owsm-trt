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
until publication. Results are recorded below; no new production encoder variant is adopted.


## Measured results (2773701)

Frozenb90283a, H200/1B, three tiny graph tests pass in3.72s. Same checkpoint and byte-identical
decoder/verify engines throughout; seven rotating-order warm whole-list samples per variant.
Every variant is deterministic across its own repeats. Rebuilding the ordinary encoder alone
changes some translation outputs; no-head changes must therefore be evaluated as N.

|Corpus/case|Cached fp16 p50(ms)|No-head RTFx change|Packaged FP8 RTFx change|
|---|---:|---:|---:|
|English73 beam5TS B1|1962.93|+0.30%|+0.29%|
|English73 beam5TS B3|1255.39|-0.38%|+0.71%|
|English73 translation B1|1511.32|+0.45%|+0.31%|
|English73 translation B8|456.69|+0.09%|+3.15%|
|Japanese12 beam5TS B1|457.94|-0.45%|-0.14%|
|Japanese12 beam5TS B3|281.46|+0.11%|+0.52%|
|Japanese12 translation B1|337.30|+0.52%|+1.07%|
|Japanese12 translation B8|138.57|+0.36%|-69.92%|

No-head is **not adopted**. Whole-request gains are small/mixed, with output changes on one
English B3 utterance, one Japanese B8 translation, and synthetic silence. English B1 and both
Japanese beam cases match full tokens; English B3 WER is44/1150 versus45/1150, Japanese beam CER
remains26/592. This small corpus cannot establish a quality advantage. The no-head encoder file
is1,937,186,444bytes versus1,472,834,884 for the rebuilt ordinary encoder; no isolated GPU-memory
saving is established. Extra engine/build cost and no broad speed gain do not justify an API.
The isolated exporter and artifacts remain for reproduction; production code is unchanged.

FP8 uses packaged digest`s7897e857ee72`, not the old audio-calibrated cached engine. Beam gains
are nearly neutral and full English beam tokens match68/73 at B1 and67/73 at B3; WER is44/1150
and42/1150 respectively. Japanese beam tokens match10/12 and12/12, CER26/592 at both. These
are changed outputs from an existing A option, not an exact acceleration. Translation has no
reference targets here; full text matches64/73 and65/73 English,4/12 and6/12 Japanese.

In Japanese translation B8, `ja_jp-11052728666022746352` repeats “renowned” until the374-token
generation limit (reference36tokens). Whole-list p50 rises138.57→460.74ms, with all seven pairs
regressing68.96–70.43% in RTFx. This is a reproducible observed failure, not a noise outlier.
FP8 stays experimental/defaultOFF; this study does not support a general beam/translation gain.

Encoder-only GPU event times (7samples,30graph replays each, excluding features/capture):

|Shape/input|Cached fp16|Rebuilt fp16|No head|FP8|
|---|---:|---:|---:|---:|
|B1 short|2.685ms|2.667ms|2.643ms|2.538ms|
|B1 long|3.616ms|3.595ms|3.554ms|3.404ms|
|B3 short|5.868ms|5.960ms|5.835ms|5.249ms|
|B3 long|8.861ms|8.859ms|8.634ms|8.146ms|
|B8 short|12.922ms|12.798ms|12.825ms|10.520ms|
|B8 long|20.165ms|20.004ms|19.786ms|18.493ms|

Raw KV remains finite. No-head matches cached KV bitwise only on the two B1 cases; other
shapes and all rebuilt/FP8 cases have numerical differences. Faster encoder execution alone
does not imply faster or equivalent final decoding, as the repetition shows.

Long prompt and three no-timestamp speech inputs match all variants. Silence differs as above;
noise matches. The110s recording matches text and38 segment boundaries/texts. Joined long-form
results have empty token lists, so this does **not** verify raw per-window token IDs. No reference
timestamp annotations are available. No normalizer, decoder precision, beam rules or padding
was intentionally changed.

[Full samples, differences, hashes and diagnostics](runtime-encoder-results-2773701.json). Raw
outputs and three engine archives: `outputs/runtime-2773701`. The separate normal/no-head builds
take188.71/186.52s; packaged FP8 build/load118.59s. These observations have unequal compiler
cache state and are not an isolated cold-start speed comparison.
