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
