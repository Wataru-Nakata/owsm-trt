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
