# Expanded regression corpus and resumable evaluation

The English73/Japanese12 sets used for tuning are screening sets. Their agreement does not establish
coverage of uncommon inputs. Large output regressions are now a separate acceptance step from the
small, repeated latency benchmark. The1B/beam5/timestamp/B1 and B3 upper-bound OFF/ON audits each completed with
zero output differences across13,232 inputs. The search-kernel OFF/ON audit also completed at B1;
other large-corpus ablations and timing remain pending.
Job2780055 prepared the corpus. Its
[corpus manifest](runtime-large-regression-corpus-2780055.json) records13,204 real short-corpus inputs
(29.798hours),28 synthetic cases and87 long-context chapters, packed into27 verified shards
(5,194,571,795 bytes including the index). The initial audit selects13,232 inputs;41 ordinary corpus
recordings exceed30seconds and use the long-form API, whose token IDs are unavailable. The87 long
chapters reuse test-clean speech and are reserved for the separate long-context case.
The prepared `beam_long` case runs the87 chapters through beam5 `transcribe_long`, independently
of the greedy `long` case. Its GPU results are pending; the completed B1 audit selected
the13,232 ordinary/stress inputs, including41 recordings longer than one window.

The [complete B1 result](runtime-large-regression-2780523.json) combines the preserved partial runs
with job2780523 (PBS F/exit0, final publication exit0). All104 chunks pass hash, source/config,
selected-input ordering and recomputed output-difference checks. Text and segments match on13,232/13,232
inputs; full token and timestamp-token sequences match on13,191/13,191 available outputs. The41
long-form outputs have no joined token IDs and are excluded from that denominator. Both variants
use the same checkpoint/engines, current search kernels and frozen8151703 source: this isolates
the upper bound, not the entire change from892c29f. Job walltime is not a steady-state speed result.

Speech-only error rates are identical between OFF and ON; generated nonspeech is excluded:

| Language | Speech inputs | Metric | OFF = ON |
|---|---:|---|---:|
| English | 5,878 | WER | 3.73% |
| Japanese | 3,966 | CER | 7.76% |
| German | 862 | WER | 6.90% |
| French | 289 | WER | 11.44% |
| Russian | 356 | WER | 7.86% |
| Spanish | 908 | WER | 4.84% |
| Chinese | 945 | CER | 11.92% |

The complete artifact also retains102 distinct diagnostic inputs, identical on both variants:
83 have invalid/out-of-range segment times, all28 nonspeech probes generate text,5 have repeated
fourgram indicators, and1 has an empty segment (categories overlap). There are no unpaired or
nonmonotonic timestamp-token/segment sequences in the available diagnostics. These are baseline
quality/data findings, not upper-bound regressions or human timestamp errors. Raw range flags108
include25 floating-point boundary false positives; corrected flags83 retain the20ms allowance.
Three of the range flags are longer-than-window FLEURS inputs. In particular,
`fleurs_es_test--es_419-10844880591943993561` is30.3seconds but ends at59.46seconds with1,058 predicted
characters against397 reference characters; `fleurs_en_validation--en_us-8126899585335816719` is31.74seconds
but ends at41.66seconds. The30.6second German input `fleurs_de_test--de_de-9045243611409192005`
has an empty final segment ending31.16seconds. These cases expose short-tail window behavior in
both variants. The retained outputs reproduce the issue; changing windowing or restricting times
would require a separate quality/compatibility evaluation. The existing `cut_at_pauses` option was
not enabled in this audit and is not assumed to solve it.
The [complete B3 result](runtime-large-regression-2780819.json) uses frozen773f827 source and
language/API-preserving input groups. Job2780636 produced a [validated9,600-input checkpoint](runtime-large-regression-2780636-partial.json);
job2780819 completed the remaining3,632 inputs with identical source/config/engines/corpus (PBS F/exit0,
publication exit0). All104 chunks pass strict validation. Text/segments match on13,232 pairs and
full token/timestamp sequences match on13,191 available pairs;41 long-form outputs lack joined IDs.
This is an upper-bound-only E check at each fixed batch configuration. B3's common baseline flags
include81 invalid/out-of-range segments,28 nonspeech text outputs,6 repeated-fourgram inputs,
1 empty segment and1 unpaired timestamp sequence, across101 distinct diagnostic inputs.

The separate [complete cross-batch comparison](runtime-batch-numerics-13232.json) uses identical
runtime trees, checkpoint/engine hashes, Torch/CUDA and GPU type, but changes batch shapes and
default decoder routing together (N). Comparing upper-bound OFF outputs between B1 and B3 gives:

| Output field | Equal / available pairs |
|---|---:|
| Full token sequence |12,946 /13,191|
| Timestamp-token sequence |13,132 /13,191|
| Segments |12,993 /13,232|
| Text |13,044 /13,232|

There are245 differing inputs:240 in three-input groups and5 in two-input groups. All35 ordinary
single-input groups and41 long-form single-input groups match. Speech-only error-count changes
(B3 minus B1) are Japanese+1/111,347 characters, English+6/111,768 words, German−5/18,706 words,
French−4/6,759 words, Russian+3/6,501 words, Spanish+2/23,314 words and Chinese+196/35,649 characters.
Chinese CER is11.9246% at B1 and12.4744% at B3. These are batch/dispatch differences, not upper-bound
regressions or an isolated kernel experiment; they do not establish causality for a particular change.

A retained [Chinese corner case](runtime-batch-corner-cases.json),
`fleurs_zh_test--cmn_hans_cn-11229861275679868227` (19.3seconds), accounts for+200 edit errors:
B1 returns63 tokens, one complete segment and14/53 character edits; B3 returns374 tokens,
478 raw text characters, a repeated fourgram58 times, one opening timestamp with no closing pair,
no complete segment and214/53 edits. Both bound variants agree within each batch. The other944
Chinese inputs net−4 edits. The exact input group and both outputs were tested again in
[focused reproduction2780984](runtime-batch-corner-2780984.json), using identical checkpoint, engines,
software and runtime source tree. All48 runs pass recomputed controls: three repeats per configuration
are stable, and search-kernel/bound OFF/ON agree at each fixed route/batch. Auto B1 and B3 reproduce
both original outputs exactly. Actual dispatch and full raw outputs are retained.

| Decoder configuration | Target tokens | Character edits /53 | Timestamp IDs | Repeats |
|---|---:|---:|---|---:|
| Auto B1 (Gluon) |63|14|201,1147|Stable3/3|
| Auto B3 (TRT shape fallback) |374|214|200 only|Stable3/3|
| TensorRT-only B1 |63|14|200,1147|Stable3/3|
| TensorRT-only B3 |374|214|200 only|Stable3/3|

The B3 repetition persists without an auto decoder switch or the new search kernels/bound.
The TensorRT-only B1 result has the same text as auto B1 with a20ms start-time difference. This
rules out decoder-route switching alone as the explanation; it does not yet separate feature/encoder/
cross-projection batch effects from decoder/prefill batch effects. No truncation, timestamp clamping,
route default or model arithmetic was changed. `reproduce_batch_corner.py` and its PBS driver retain
the48-call controls, exact group, dispatch, original-output differences and stability checks. Its small
metadata fixtures pass. The public API sorts by duration, so this is not a decoder-row permutation test.
The prepared `isolate_batch_corner.py` diagnostic crosses conditioning computed at B1/B3 with
prefill/decoding at B1/B3, keeping TensorRT and the original duration order. It copies fixed cross-KV
without rerunning the encoder during the decode controls, records conditioning differences and stop
diagnostics, and requires native1/1 and3/3 outputs to match the prior public API controls. Conditioning
includes frontend, encoder and cross projections; mixed combinations are N interventions. Its small
matrix/accounting tests pass; GPU results are pending and no runtime behavior changes are proposed.

For13,044 inputs with identical text and ordered segment labels/count, all26,516 boundary differences
have median/p95 zero. The52 nonzero boundary displacements have median20ms, p95120ms and maximum740ms;
the maximum is the synthetic English10second silence probe. The188 changed-text/segmentation inputs
are excluded from this boundary comparison, not treated as aligned. No human timing accuracy is inferred.
All28 nonspeech probes generate text at both batch sizes; Japanese inserted-character totals are231/256
and English inserted-word totals197/204 for B1/B3. They remain excluded from speech WER/CER.
The earlier [6,656-input partial comparison](runtime-batch-numerics-6656.json) is preserved as a checkpoint.

Reproduce this saved-output comparison without loading models or audio:

```bash
python scripts/compare_regression_batches.py \
  outputs/runtime-large-audit/owsm_v4_medium_1B_beam_bound_beam_b1 \
  outputs/runtime-large-audit/owsm_v4_medium_1B_beam_bound_beam_b3 \
  --out outputs/runtime-large-audit/cross-batch-results.json
```

The comparison verifies matching runtime trees, engines/checkpoints, corpus keys and per-record
configuration digests, then groups results by actual B3 call size/API. Missing joined token IDs
are excluded from token denominators. Nonspeech insertions are separate from speech WER/CER.
Boundary displacement statistics include only outputs with identical text and ordered segment
texts/count; changed text or segmentation is counted as excluded, not assigned an ordinal time error.
All and nonzero displacements have separate medians/nearest-rank p95 values. These are differences
between outputs, not reference timing accuracy. Six small metadata fixtures cover this accounting
and rejection of corrupt/config-incompatible or internally nonmatching audit records.

Job2780055 was stopped after a fresh account query showed200 GPUs (the session cancelled only its
own1GPU job). PBS exit271 is not a successful full audit. Its
[validated partial result](runtime-large-regression-2780055-partial.json) contains3,584 Japanese
inputs in28 complete chunks, with exact tokens/timestamp tokens/segments/text and zero differing
outputs. `complete=false` and `interrupted=true` are retained. Resume validation checks every chunk
hash, selected key, source/config digest and output difference. Job2780199 resumed the remaining
9,648 inputs with the exact same8151703 code/checkpoint/engines/corpus. It too was stopped after a
fresh query showed193 GPUs. Strict resume validation then covered4,224 outputs in33 chunks, all
returned tokens/timestamps/segments/text equal, leaving9,008 inputs for2780523. Neither stopped job
establishes full-corpus acceptance or a new latency result.

The [4,224-input diagnostic catalogue](runtime-large-regression-cases-4224.json) retains30 originally
flagged inputs, identically flagged on both variants. Rechecking the saved boundaries found13 false
positives at the existing20ms endpoint allowance: binary rounding makes `8.30 > 8.28 + .02` true.
The shared diagnostic now tolerates1ns at that boundary and rejects nonfinite segment times.
Original flags remain recorded; corrected diagnostics leave17 distinct inputs: all14 Japanese
nonspeech probes generated text,8 inputs had out-of-range/invalid segment times, and2 had
repeated-fourgram indicators (these groups overlap). Generation, CER/WER and output agreement are
unchanged. Three JVS flags involve0.587–0.611second recordings with17–43 character
references. Their CRC32 values and sizes match the existing original ZIP members, but audio/transcript correspondence
has not been reviewed. They remain in the audit; no difficult cases were removed to improve scores.
The nonspeech cases are excluded from speech CER/WER and counted separately. These are baseline
quality/data diagnostics, not newly introduced upper-bound differences or human timing annotations.

First job2779853
exited1 after53seconds during preparation: the dev-other manifest remained but its source directory
was absent. No inference ran and no complete corpus was published. The retry checks all source
paths first and records every missing input in `preflight.json`. It can restore dev-other locally
from the [official OpenSLR12 archive](https://www.openslr.org/12/), checking the
[published MD5](https://openslr.trmal.net/resources/12/md5sum.txt), each selected archive member and
its transcript before packing. Other missing datasets fail visibly instead of silently reducing coverage.
The verified download is retained as one shared archive; extracted audio stays node-local and is
deleted after shard publication. This repair and its small archive/metadata tests are implemented;
successful preparation is recorded in2780055 and full B1 verification in2780523.
Second job2780026 also stopped before inference: the first inventory treated all JVS nonpara30
transcript lines as recordings. The original ZIP member index confirms2,700 recorded files across
the selected90 speakers (30 each), while the1,800 extra transcript entries have no audio member in
that ZIP. Inventory now uses the original ZIP's member list, retains the unused transcript IDs in
provenance, and still fails if an actually recorded input is missing on disk. The adjacent original
`jvs_ver1.zip` is therefore required as well as the extracted JVS directory. A byte-only fixture
checks the distinction between unrecorded text and a genuinely missing recording.
Cold preparation has a20-minute limit; the remaining allocation determines the inference budget
(at most2600seconds), with time reserved for publishing results. If too little time remains even
for model startup, the completed corpus is retained and the audit is explicitly deferred.

The available manifests contain10,504 short recordings across English, Japanese, Chinese, German,
Spanish, French and Russian, plus87 concatenated LibriSpeech chapters. They include2,620 LibriSpeech
test-clean,2,864 dev-other,394 English FLEURS,266 Japanese FLEURS,1,000 JVS parallel100,945 Chinese,
862 German,908 Spanish,289 French and356 Russian FLEURS. The1,000 JVS items are only10 speakers
reading100 sentences. To add text/speaker diversity, preparation reads nonpara30 from90 JVS speakers
(up to2,700 further recordings), excluding the91–100 calibration partition. Actual counts, distinct
texts, known speakers, original sample rates, hours and duration buckets are recorded during packing.
These existing datasets are not claimed to be an untouched holdout. The87 chapters reuse test-clean
speech, so they are long-context coverage, not87 independent sources. The earlier Japanese12 overlap
is explicitly marked; reference segment boundaries/target translations are unavailable.
The local JVS release includes phoneme alignment files; its `README.txt` explicitly describes them
as automatically generated. They are not human timestamp annotations and are not used as timing
ground truth in this pass. The installed JVS `whisper10`/`falset10` subsets are additional candidates
for later speaking-style coverage; their full completeness and model outputs remain unchecked.

`prepare_large_regression.py` runs only inside an allocated job. One writer packs original encoded
audio plus per-sample JSON into512-sample WebDataset tar shards under `$PBS_LOCALDIR`, verifies
each member hash/count, then atomically publishes completed shards to `.cache/regression-expanded-v1`.
The input index, source-manifest hashes and a final manifest make the corpus reusable. Audio decoding,
resampling, model work and tests stay on compute nodes; no temporary waveform files go to Lustre.
Future reuse copies each shared shard once and verifies its staged bytes before marking the local
corpus complete. Good/corrupt-copy fixtures cover that check; no staging speedup is claimed, and
the completed B1 continuation retained the original8151703 implementation.

Twenty-eight generated silence/noise/tone probes (English/Japanese primers) complement real speech;
they do not substitute for natural noise/music/conversation coverage. Prompt stress is a separate
`beam_prompt` case limited to complete inputs within the model's input window; the public long-form
API does not accept an external prompt. Longer inputs remain covered by the other cases, without
trimming audio or silently dropping the prompt.

The first priority is1B, beam5, timestamps=True, B1: same-engine existing upper-bound OFF versus
current ON. B3 follows to cover mixed lengths and partially finished groups. Its
evaluation groups preserve input order and share a language and API: a longer-than-window neighbour
does not switch an ordinary short input to long-form semantics or remove its token IDs. Groups may
therefore have1/2/3 inputs; each future record stores its group keys, API and requested batch size.
This grouping change has small fixture coverage and the complete B3 result above. The completed B1
continuation stayed frozen at8151703.
Each call in this audit handles one such group. Large multi-batch request behavior, row refill,
encoder lookahead and file-prefetch coverage require separate cases; this audit does not establish them.

Subsequent ablations also include `beam_kernels` (upper bound ON in both, search/state/cache kernels OFF/ON)
and `beam_combined` (upper bound and beam kernels both OFF/ON), to cover bugs shared by both sides
of the upper-bound-only comparison. These use the existing PyTorch reference path, the same decoder
backend/engine, and separate session pools. The [complete search-kernel B1 result](runtime-large-regression-2780923.json)
from job2780923/frozen9e87cfaf passes all104 chunk/config/corpus/order/output checks (PBS F/exit0,
publication exit0):13,232 text/segment pairs and13,191 available token/timestamp pairs match exactly.
This compares the combined public `beam_kernels` switch while keeping the upper bound ON in both
variants. It does not separately attribute each component's parity or establish a new speed gain.
The [candidate-output repeat check](runtime-large-regression-repeat-2780923.json) also matches the
candidate outputs from the earlier B1 bound audit on every available field, with identical runtime
source tree, engines/checkpoint and software. B3 search kernels and combined bound/kernel ablations
remain pending. B3 search-kernel job2781006 is running with the same frozen9e87cfaf source as B1,
after a fresh account/queue/no-duplicate guard.
Other ablations are102M serial64 versus24 (E), and the explicit1B NS4/102M NS8 option versus NS2 (N),
with plain/drafted ASR, translation and long-form cases. N differences remain visible, not treated
as exact or silently accepted as accuracy improvements. New default adoption waits for relevant
large-corpus checks as well as the repeated performance benchmark.

`audit_large_regression.py` stores both complete outputs, timestamp IDs, segments, text, WER/CER
counts, invalid/empty/unpaired/nonmonotonic times and diagnostic repetition indicators. Results are
published only as complete128-sample compressed chunks, with corpus/checkpoint/engine/code hashes.
A subsequent job verifies config and finished chunks and resumes the remaining input IDs. A
45-minute pass ending successfully is **not** a completed evaluation unless its `complete` flag is
true. Each1GPU job remains limited to1hour and requires fresh account<=192/queue/duplicate checks.
On a capacity overrun, only this session's job is stopped; completed chunks remain reusable.
During a resumed job, monitoring uses `progress.json` with the current job ID; an earlier run's
`status.json` must not override newer progress. Future final status files also record their job ID.
The frozen8151703 continuation retains its older final-status format.

The audit is one pass per variant, including I/O/cold graphs/warm calls. Its elapsed time is not a
steady-state latency or speedup result. `summarize_large_regression.py` aggregates agreement and
quality separately by language, dataset and duration. Long-form returns joined text/segments only;
empty joined token lists do not count toward token agreement. Repetition indicators are heuristics,
not ground-truth errors. Inputs longer than a model window use long-form, avoiding reference text
being compared to an intentionally trimmed30-second input.
The summary verifies published chunk hashes and recomputes output differences. Generated nonspeech
probes retain insertion/empty-output diagnostics but are excluded from speech WER/CER. Mixed WER/CER
counts never receive a combined error rate.
Segment boundary order/overlap is checked even for long-form outputs. Token-level timestamp flags
report their available-record count; missing long-form token IDs are not evidence of valid token timing.
Summaries recompute invalid segment counts from saved boundaries/durations with the corrected
floating-point comparison; `recorded_invalid_segments` retains the original per-run diagnostic.

The preceding small public-split repeat2778800 was explicitly stopped when a fresh account query
reported198 GPUs. PBS exit271 and partial publication are recorded in
`outputs/runtime-ring-split-api/capacity-action.json`. Progress reached many completed phases, but
missing artifacts prevent accepting the job as full validation. Prototype results2778533 remain
valid; no public-split default adoption or independent speed replication is claimed from2778800.

The proposed frontend direct-input retry was withdrawn before submission: earlier2771513 had
already tested the same basic idea without a broad gain. Its temporary runtime prototypefc02d74
was reverted in4f6041e. Expanded coverage takes priority over repeating that small screening test.
