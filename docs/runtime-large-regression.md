# Expanded regression corpus and resumable evaluation

The English73/Japanese12 sets used for tuning are screening sets. Their agreement does not establish
coverage of uncommon inputs. Large output regressions are now a separate acceptance step from the
small, repeated latency benchmark. The1B/beam5/timestamp/B1 upper-bound OFF/ON audit completed with
zero output differences across13,232 inputs; other large-corpus ablations and timing remain pending.
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
B3 job2780636 used frozen773f827 source and language/API-preserving input groups. It ended
normally at the audit budget with a [validated partial result](runtime-large-regression-2780636-partial.json):
9,600 text/segment pairs and9,584 available token/timestamp pairs match,75 chunks pass strict
validation, and `complete=false` is retained. Job2780819 resumes the remaining3,632 inputs with
the identical773f827 source/config/engines/corpus. Full B3 acceptance remains pending.

A separate [partial cross-batch comparison](runtime-batch-numerics-6656.json) checks6,656 matched
inputs with identical runtime source trees, checkpoint/engine hashes, Torch/CUDA and GPU type.
Upper-bound OFF/ON agrees within each batch. Comparing B1 against B3 instead changes batch shapes
and default decoder routing together:98 inputs differ, including79 texts and21 available timestamp
sequences. Of6,648 available full token sequences,6,550 match; the8 long-form outputs have no IDs.
On this matched subset, Japanese CER edit totals are8,637 versus8,638 per111,347 reference characters;
English WER edit totals are2,490 versus2,495 per49,162 reference words. This is an N comparison,
not an upper-bound regression or an isolated kernel experiment. Full cross-batch results and
attribution of individual differences remain pending; no time-boundary accuracy is inferred.

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
This grouping change has small fixture coverage; B3 GPU results remain pending. The completed B1
continuation stayed frozen at8151703.
Each call in this audit handles one such group. Large multi-batch request behavior, row refill,
encoder lookahead and file-prefetch coverage require separate cases; this audit does not establish them.

Subsequent ablations also include `beam_kernels` (upper bound ON in both, search/state/cache kernels OFF/ON)
and `beam_combined` (upper bound and beam kernels both OFF/ON), to cover bugs shared by both sides
of the upper-bound-only comparison. These use the existing PyTorch reference path, the same decoder
backend/engine, and separate session pools. They are pending large-corpus E checks, not results.
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
