# Frontend warm-up stream reuse: controlled workspace experiment

Independent102M B8 measurements2776464 reproduced throughput gains, but observed288MiB/64MiB
remaining after nine/two candidate graphs were released. `capture_graph` creates a fresh side
stream for each warm-up. PyTorch documents persistent workspace per cuBLAS handle/stream pair
([CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cublas-workspaces)); this is a
plausible explanation for the observed32MiB increments, not yet a controlled attribution.

The optional `warmup_stream` helper parameter permits frontend warm-up on its existing dedicated
capture stream. The frontend already serializes shared workspace/replays using its ready event,
waits for the previous replay before a cold capture, and returns independent cloned outputs.
The caller-to-side and side-to-caller dependencies stay in place. Existing encoder/decoder callers
retain a fresh side stream. No private workspace-clear API, global cuBLAS setting, graph-pool sharing,
operator or arithmetic change is introduced. Both frontend and public pipeline defaults remain off
for this experiment; batch graph limit remains1.

`scripts/profile_frontend_warmup.pbs` runs19 tests covering both warm-up policies, mixedB1/B3/B8,
normalization, fused/unfused, output lifetime across streams, cache saturation and autocast fallback.
Each model/corpus/batch1or8 memory experiment runs in a fresh process, with just one variant cache
resident at a time. This avoids earlier batches consuming the finite PyTorch stream pool before a
later memory comparison. Common eager expected features and the resident model remain fixed.
Both variants capture the same batch shapes. Per-shape cold calls exclude equality checks; retained,
reserved, peak and post-release allocated deltas are recorded without clearing global workspaces.

Separate1B/102M English73/Japanese12 B1 timestamped single requests and B8 drafted timestamp ASR
alternate11 times with warm captures. Both variants enable graphs up toB8; the sole difference is
warm-up stream ownership. Require exact features/full outputs/decoder calls, no new warm captures
and no decoder fallback. A memory/cold improvement must not be reported as a warm-speed gain.
No result or adoption is claimed before this job completes. B8 public exposure and disk-inclusive
I/O work remain follow-ups after this memory question is resolved.


## First result2776645: public validation and reverse-order memory check pending

Frozenaa1d9c4 completed with exit0.19 tests, eight warm request conditions and eight isolated
feature-memory conditions passed, with exact features/full outputs/decoder-call totals. Reusing
the frontend stream reduced retained PyTorch allocation atB1 English from596.5–596.9 to52.5–52.9MiB,
andB8 English from381.9–383.2 to93.9–95.2MiB. JapaneseB1 fell301.7→45.7MiB; B8 fell119.8→55.8MiB.
Legacy post-release residuals were544/288MiB (EnglishB1/B8) and256/64MiB (Japanese); reuse showed
zero incremental residual. These are allocation deltas with a fixed model/expected features, not
whole-process VRAM. Reference ran first, so a reverse-order check is required before generalizing
standalone workspace savings.

Typical cold feature/H2D sums fell EnglishB1≈52→42–45ms, B8≈32–35→26–29ms; JapaneseB1≈22–25→17ms
and102M B8≈14.5→9.6ms.1B Japanese B8 reference593ms is an outlier, not evidence of a59x cold gain.
Warm RTFx changed−0.64% to+0.70%, with mixed paired signs; no warm-speed gain is claimed.

Prepare default frontend stream reuse, leaving other capture callers unchanged, and validate the
public path independently. A bounded public batch-limit option may expose the separately replicated
102M B8 gain, with default1 retained. [First measurements](runtime-frontend-warmup-2776645.json);
raw artifacts `outputs/runtime-2776645/`. The earlier throughput gain and this memory reduction
are separate ablations and must not be multiplied into a new speed claim.


## Public integration prepared

The frontend now selects dedicated-stream warm-up by default; the shared `capture_graph` helper's
other callers retain their original behavior. `frontend_graph_max_batch` is a validated public
integer1..8 option, with default1. Both constructor and `from_pretrained` reject invalid settings
before loading. The independent job `validate_frontend_api.pbs` checks legacy/default/limit8 through
plain, drafted, beam, translation, long-prompt, lookahead, silence/noise, reuse and110s calls on both
models, then repeats isolated memory with the candidate first and batch-limit throughput with
stream reuse on both variants. These public-path results are pending.

The same allocation independently verifies the two-line public mono-file loader integration against
a preserved legacy loader, with43 focused tests including the15 already-passing CPU option checks.
This is a separate file-inclusive ablation; it must not be merged into the loaded-input frontend gain.

## Public verification2776797: adopted, batch limit1 retained

Frozen61fef6c passed43 focused tests and all9 public API cases for each model in all three modes.
Features, sampled full output tokens/time IDs/text/segments and decoder call totals match; long-form
compares text/segments because joined token IDs are unavailable. No new warm capture or decoder
fallback occurred. Reverse-order isolated memory runs (candidate first) reproduce all earlier
allocation deltas: English B1 saves544MiB, English B8 saves288MiB; Japanese B1/B8 save256/64MiB.
The reused stream leaves zero incremental post-release allocation. These are retained PyTorch
allocation deltas with a fixed model/features, not whole-process VRAM. Cold times now have mixed
signs; no startup or warm-speed gain is attributed to stream reuse.

With stream reuse enabled on both sides, public `frontend_graph_max_batch=8` versus default1 gave
102M timestamped B8 throughput+5.99% English/+2.66% Japanese; all11 pairs improve, paired medians
+5.26/+2.16%, and whole-list p95 falls55.84→51.13ms /40.57→38.56ms. This independently reproduces
the earlier102M improvement.1B gains+0.35/+1.11% have mixed signs; B1 control changes−0.77..+0.55%
also have mixed signs. Keep default1 and expose1..8 as an explicit memory/performance tradeoff.
English B8 retained feature allocation is93.9–95.2MiB versusB1 52.5–52.9MiB in separate batch sweeps;
shape distribution and cache history affect memory. No broad/default B8 or combined speed claim.

[Environment, paired samples, reverse-order memory and public checks](runtime-frontend-api-2776797.json).
Raw artifacts and reproducer: `outputs/runtime-2776797/`, `scripts/validate_frontend_api.pbs`.
H200/v4 1B/102M only; other hardware, checkpoints and frontend CTC batch expansion are unmeasured.


## Expanded lifecycle regression2777265; exclusive capture correction pending

The earlier nine-case public check did not cover enough successive model lifecycles. The expanded
15-case validator fails on the third102M model at long_prompt, including three worker0 instances
without audio prefetch and with case synchronization. Legacy fresh warm-up streams and eager
frontend controls both pass15 cases for workers0/1/2, matching the saved baseline. Fresh worker2
also passes. [Matrix evidence](runtime-cuda-matrix-2777265.json); raw logs `outputs/runtime-2777265/`.
No timing claim follows from these diagnostic controls. Compute Sanitizer fails much earlier at
single_file, with378 reports including driver queries and TRT stream-ordered allocation accesses;
those reports do not establish the cause of the third-model failure.

A newly constructed `torch.cuda.Stream` is selected from a finite round-robin pool, so another
session can capture on the frontend's stream. PyTorch2.14.1 graph destruction
[clears that stream's cached cuBLAS workspaces](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/cuda/CUDAGraph.cpp#L348-L350).
The [upstream issue193402](https://github.com/pytorch/pytorch/issues/193402) describes a surviving
graph retaining an address freed by another graph's destruction. This is a concrete lifetime
hypothesis here, not yet a confirmed diagnosis of the API failure.

The proposed correction creates one exclusive native CUDA stream per frontend replay object,
wraps it with ExternalStream, and retains it until the last replay completes and all its graphs
are released. Warm-up reuse and the32-shape bound stay unchanged. No global workspace clear or
new arithmetic is introduced. `diagnose_frontend_lifetime.py` exercises pooled-stream graph churn
against frozen old code and the correction, including release. `validate_frontend_lifetime.pbs`
runs that reproducer,94 focused tests, six successive102M model lifecycles and three1B lifecycles,
then the missing102M prefetch timings and memory comparison. Validation is pending; the earlier
memory reduction alone is insufficient evidence of safe model/session reuse.


## Verified correction2777327

Frozen6a15d1f completed F/exit0. The minimal old-code control recorded a pooled-stream alias,
destroyed unrelated graphs on that pool, then failed with CUDA illegal address during replay.
The exclusive stream did not alias any of96 pooled stream requests (32 distinct streams), preserved
all feature outputs after the same graph churn, and completed finalizer cleanup. This independently
reproduces the workspace lifetime mechanism rather than inferring it from a later TRT exception.

94 focused tests pass. The expanded public API compares every output with the saved pre-change
worker0 baseline:102M runs workers0/1/2 twice in one process (six model lifecycles);1B runs all three.
Each lifecycle covers15 cases, including long prompts, beam/translation, mixed/loaded files,
stereo/resampling, CPU autocast, injected read failure/reuse and110s recording. All agree exactly.
The separate streaming-exception regression first reproduces outstanding GPU work in all six old-loop
conditions; the new exception drain passes, without adding normal-path host synchronization.

A reverse-order memory replication with the exclusive stream preserves the previous102M English
savings: B1 retained delta596.51→52.51MiB (544MiB less), B8 381.94→93.94MiB (288MiB less).
Candidate post-release delta is0 in both; legacy residuals are544/288MiB. Exact features, same shapes,
zero fallback. These remain incremental PyTorch allocation measurements, not whole-process peaks
or new latency gains. Keep the exclusive capture stream and warm-up reuse; batch limit default1.
[Tests, lifecycle controls, API/environment and memory evidence](runtime-frontend-lifetime-2777327.json).
Raw results are `outputs/runtime-2777327/`; reproduce with `validate_frontend_lifetime.pbs` and
`diagnose_frontend_lifetime.py --baseline-frontend <frozen0946b43/frontend_graph.py>`.
