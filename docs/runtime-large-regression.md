# Expanded regression corpus and resumable evaluation

The English73/Japanese12 sets used for tuning are screening sets. Their agreement does not establish
coverage of uncommon inputs. Large output regressions are now a separate acceptance step from the
small, repeated latency benchmark. No large-corpus results have been measured yet. First job2779853
exited1 after53seconds during preparation: the dev-other manifest remained but its source directory
was absent. No inference ran and no complete corpus was published. The retry checks all source
paths first and records every missing input in `preflight.json`. It can restore dev-other locally
from the [official OpenSLR12 archive](https://www.openslr.org/12/), checking the
[published MD5](https://openslr.trmal.net/resources/12/md5sum.txt), each selected archive member and
its transcript before packing. Other missing datasets fail visibly instead of silently reducing coverage.
The verified download is retained as one shared archive; extracted audio stays node-local and is
deleted after shard publication. This repair and its small archive/metadata tests are implemented;
successful full preparation and inference remain pending.
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

Twenty-eight generated silence/noise/tone probes (English/Japanese primers) complement real speech;
they do not substitute for natural noise/music/conversation coverage. Prompt stress is a separate
`beam_prompt` case limited to complete inputs within the model's input window; the public long-form
API does not accept an external prompt. Longer inputs remain covered by the other cases, without
trimming audio or silently dropping the prompt.

The first priority is1B, beam5, timestamps=True, B1: same-engine existing upper-bound OFF versus
current ON. B3 follows to cover mixed lengths and partially finished groups. Supported subsequent
ablations also include `beam_kernels` (upper bound ON in both, search/state/cache kernels OFF/ON)
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

The preceding small public-split repeat2778800 was explicitly stopped when a fresh account query
reported198 GPUs. PBS exit271 and partial publication are recorded in
`outputs/runtime-ring-split-api/capacity-action.json`. Progress reached many completed phases, but
missing artifacts prevent accepting the job as full validation. Prototype results2778533 remain
valid; no public-split default adoption or independent speed replication is claimed from2778800.

The proposed frontend direct-input retry was withdrawn before submission: earlier2771513 had
already tested the same basic idea without a broad gain. Its temporary runtime prototypefc02d74
was reverted in4f6041e. Expanded coverage takes priority over repeating that small screening test.
