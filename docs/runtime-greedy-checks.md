# Plain greedy completion check intervals

After the measured plain mask/argmax/state fusion, the host still checks completion every4 steps
in GreedySession. This can execute up to3 decoder calls after a post-prefill EOS, or4 when EOS is
selected directly from prefill. A more frequent check also costs synchronization; its net effect
must be measured. Beam already has its separately measured B1/B3 policy.

`scripts/profile_greedy_checks.py` keeps the accepted public optimizations, checkpoint, engines,
precision and decoder dispatch fixed and compares intervals4/2/1. Session.chunk is changed only
inside this benchmark; the production greedy default stays4. All73 English inputs and12 Japanese
inputs are used with plain greedy, timestamped plain greedy and translation;102M uses English.
Warm synchronized wall samples alternate variant order over11 repeats. First-use/capture costs and
audio loading are recorded separately, and every returned token/segment must match interval4.

A separate diagnostic pass records actual decoder calls, host checks and the first EOS selection
index per row. Selection0 is prefill, whose decoder call is excluded from the search-call count.
The batch's excess is actual decoder calls minus its last row's first EOS index; a length-limited
request with no EOS has no excess claim. GPU state is copied only after each diagnostic request,
never added to the timed per-token loop. Artificial CPU checks cover prefill EOS, later EOS and
length-limit accounting. GPU speed and output agreement are unmeasured until the job completes.

The same allocated job optionally collects Ring warp-stall/source counters after the speech runs,
using the unchanged warmed kernel. This follows2772119's low eligible-warp/issue-rate observations.
The optional profiler status is recorded separately and cannot be treated as request timing.

## Result:2772221; default interval4 retained

In all9 dataset/case combinations, all returned tokens, timestamp IDs, segments and text matched across intervals. Eleven alternating
repeats used English73, Japanese12 and102M English73, three cases per dataset.

| Dataset/case | Interval2 RTFx gain | Interval1 RTFx gain |
|---|---:|---:|
|1B English plain|1.28%|2.33%|
|1B English timestamps|3.30%|4.18%|
|1B English translation|1.74%|3.43%|
|1B Japanese plain|1.61%|2.34%|
|1B Japanese timestamps|1.41%|1.47%|
|1B Japanese translation|1.10%|0.77%|
|102M English plain|1.10%|0.15%|
|102M English timestamps|1.60%|0.03%|
|102M English translation|1.13%|-0.71%|

Interval1 removes all observed post-EOS calls. For example,1B English timestamped plain decoding
falls1882→1752 decoder calls, removing130 extras, while host checks rise472→1752 in the separate
73-request diagnostic pass. It is not uniformly better: English timestamp p95 rises38.00→38.56ms,
102M translation p95 rises13.13→13.72ms, and Japanese translation's paired median change is-0.72%
despite its positive aggregate RTFx.102M within-input CV medians are about0.7–1.3%; several aggregate
gains are small relative to variation or disagree with paired medians.

Therefore the production default stays4. Low-level GreedySession(chunk=1 or2) remains available
for a measured workload. This experiment does not claim a broadly faster new default. Artifacts:
outputs/runtime-2772221/{analysis.json,english,japanese,base102m}; each case's actual_dispatch files
contain separately collected EOS/call/check records, not backend-dispatch counts.
