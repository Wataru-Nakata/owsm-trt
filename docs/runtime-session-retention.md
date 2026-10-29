# Session retention under mixed batch requests

Candidate22 also covers session eviction. The production encoder-decoder factory retains two
sessions, keyed by utterance batch and beam width. A B1 beam request followed by a five-input B3
request uses `(1,5)`, `(3,5)` and `(2,5)`. Repeating this sequence recreates all three sessions under
the two-entry LRU. The paired H200/1B experiment below verifies an E-intent three-entry alternative
on24 inputs. `OWSM.from_pretrained(..., session_cache_size=3)` exposes the retention count as a
positive integer. Its default remains2; CTC-only caching is separate and unchanged. The public
integration has25 focused CPU checks and a complete13,232-input H200/1B beam audit.

`scripts/profile_session_retention.py` compares the existing two-entry policy with an opt-in
three-entry control. It selects12 inputs each from English and Japanese in the expanded corpus,
uses two distinct six-input cycles, and alternates the policies over seven repeats. Each cycle is
one single-input request followed by one five-input request at B3, with beam5 and timestamps.
Every full output must match across policies and repeated calls. This is a mixed-shape workload;
reference timings intentionally include session setup and graph capture caused by eviction.
They must not be presented as a decoder kernel improvement or a gain for an ordinary fixed-shape
warm request.

Model loading, audio reads and initial warmup are separate. A diagnostic pass counts session
creations without keeping evicted sessions alive: six creations over two cycles are expected
for the reference and none for the candidate. Timed calls do not install this observer. Candidate
captures must stay constant after warmup; output changes or decoder fallback invalidate the run.
Both session pools remain resident, so reported allocator peaks describe the combined experiment,
not standalone deployment memory. A separate memory assessment is required before adoption.

Focused CPU fixtures cover positive cache limits, three-shape reuse, LRU refresh and rejection
before model loading. They allocate no model tensors; GPU validation is reported separately below.
The CTC-only session cache uses a separate policy and is unchanged.

Run only inside an allocated compute job, with the environment, model, engines and corpus staged
on node-local storage:

```bash
python scripts/profile_session_retention.py --model MODEL --data LOCAL_CORPUS \
  --languages eng jpn --repeat 7 --out LOCAL_RESULTS
```

The initial measurements used a private runtime field; the public option uses the same LRU count.

[Job2792200](runtime-session-retention-2792200.json), frozen5fdd30f, finished with PBS/publication
exit0. All24 full token/time/text/segment outputs match across policies and every warm/timed/diagnostic
call. All14 paired repeat totals improve. The separate observer confirms six session creations over
two cycles with retention2 and zero with retention3. Candidate graph counts remain constant after
both six-input groups have warmed up.

| Language | Cycle p50, 2 →3 sessions | Cycle p95, 2 →3 sessions | Pooled RTFx, 2 →3 sessions |
|---|---:|---:|---:|
| English |3221.94 →225.21ms|3419.50 →263.66ms|21.11 →305.74|
| Japanese |4240.05 →174.41ms|4401.78 →198.37ms|13.10 →318.43|

Each cycle contains two API requests and six inputs. These14.48×/24.31× workload gains eliminate
repeated setup/capture; they are not decoder-kernel gains, single-call p95s, or gains for fixed-shape
requests already fitting the cache. Both pools were resident during the paired run. Its approximately
12.6GiB allocator peak cannot establish the extra deployment memory required by retention3.
Initial setup/JIT and shape warmup are retained separately in the result, without comparing their
ordered reference/candidate measurements as an independent cold-start speedup.

Standalone memory assessment runs the same profiler twice in fresh processes,
once with `--isolated-limit 2` and once with `--isolated-limit 3`, using `--repeat 3`. Setting
`SESSION_MEMORY_ONLY=1` in the PBS wrapper runs these processes sequentially in one allocation.
It records resident and peak PyTorch allocated/reserved bytes, plus synchronized whole-device used
memory samples. Device samples include CUDA context and library allocations; they are not process
peak measurements. These sequential runs must not be used as a paired latency comparison.

After normal PBS/publication completion, validate all24 full outputs against the paired run and both
isolated processes, then compare memory:

```bash
PYTHONPATH=scripts python scripts/summarize_session_memory.py RAW_MEMORY_RESULTS \
  --control RAW_PAIRED_RESULTS/retention --out MEMORY_SUMMARY.json
```

Memory outside these model/beam/request shapes remains unmeasured. The isolated mode changes only
benchmark instrumentation; the production default is still two retained sessions.

[Job2792242](runtime-session-memory-2792242.json), frozen afce443, completes both isolated processes
with PBS/publication exit0. All24 outputs match each other and the earlier paired run. Candidate
resident allocated bytes stay constant across its six measured cycles per language.

| Language | Additional resident allocated | Additional reserved | Additional sampled device usage |
|---|---:|---:|---:|
| English |557.76MiB|576MiB|710MiB|
| Japanese |557.27MiB|618MiB|750MiB|

These differences use final synchronized observations. The allocator, library/context allocations
and device samples have different scopes, as described above. They are measured costs of retention3
for this workload, not a general memory bound.

`audit_large_regression.py --mode beam_sessions --case beam_b3` compares public cache sizes2/3
with the optimized default search in both. It preserves the established corpus/grouping and records
session creation counts and maximum retained entries per process. Resume keeps observations per job;
counts describe that process, not an uninterrupted execution across allocations. It is an E audit,
so any output difference fails the run. This pass includes41 ordinary long-form inputs, which lack
joined token IDs; their text/segments remain comparable.

The [completed public audit2792704](runtime-large-regression-2792704.json), frozen da80175, verifies
all104 chunks: text/segments13,232/13,232 and available token/timestamp sequences13,191/13,191 match.
The reference also matches the earlier optimized-default B3 outputs from2781006 on every available
field, with identical inputs/grouping/checkpoint/engines. Current reference/candidate decoder routes
match at B1/B2/B3. Historical2781006 did not record dispatch metadata, so historical route equality is
not claimed; all runtime files except the API/cache changes in `pipeline.py` are byte-identical.

Account-cap interruptions stopped2792366 at10,496 inputs/82 chunks (PBS271, raw exit unavailable)
and2792572 before any additional chunk (PBS271/raw143). The normal final allocation2792704
(PBS/publication exit0) adds2,736 inputs/22 chunks, preserving every earlier chunk hash. Session
creation counts through published chunks are79→3 in the initial process and22→3 in the final
process. These are process-local factory counts, not decoder calls or an uninterrupted latency run.
The zero-progress interrupted process contributes no published session-observation snapshot.
No speed claim is derived from this one-pass multi-allocation audit; use the paired measurements above.

`check_session_reuse.py` runs a separate mixed-task regression on12 English/Japanese inputs.
Eight named cases alternate drafted/plain greedy, streaming batches, translation, a long prompt and
timestamp/no-timestamp beam requests. Three request orders exercise retained and evicted sessions
without clearing the caches between tasks or languages. All96 API calls and276 output visits are
retained and compared against the first result for the same case/input. `SESSION_REUSE_CHECK=1` in
the PBS wrapper runs this check; it cannot be combined with the isolated-memory mode.

[Job2792857](runtime-session-reuse-2792857.json), frozen86e19d2, finishes with PBS/publication exit0.
All96 calls/276 output visits match, covering46 named-case/input records across two policies and
three orders. Both policies exercise `(1,1)`, `(2,1)`, `(3,1)`, `(1,5)`, `(2,5)` and `(3,5)` session
keys; maximum retained entries are2/3. Ordinary primers have3/4 tokens, and both prompt cases have
124. The greedy prompt exceeds its draft-window limit, exercising the long-primer path.
The12 inputs are a subset of the earlier paired corpus, not additional independent speech data.
This check adds H200/1B compatibility coverage, with no new timing or translation-quality claim.
CTC-only models, other GPU/models and cache counts other than2/3 remain outside this GPU validation.
