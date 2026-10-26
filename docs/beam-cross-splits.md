# Cross-attention slice and query layout sweep

Reference:4d8466a (the adopted KV version). This is a separate opt-in N experiment: changing the
cross-attention key slices changes fp32 softmax reduction order, so exact output compatibility is
measured, not assumed. Default runtime behavior is unchanged; this remains opt-in after validation.

The previous grouped-query experiment kept NS=1 at R=5. That reduced cross-attention units from
80 to16 on the1B model. The new sweep compares NS=2/4/8 both with independent queries and grouped
sibling queries. Grouping shares K/V loads; key splitting restores more independent units. The
persistent grid still uses132 CTAs on H200 and the existing row/token routing limits. All sequence
positions, search rules, weights and dtypes remain unchanged.

Job2770255 completed on H200 with221 existing tests and22 cross-attention tests passing. English73
uses5 alternating repetitions, Japanese12 uses3; each configuration shares loaded checkpoint/engines
with its own reference. Cold load/JIT/capture and warmup are excluded. Resource settings were one
rt_HG GPU, gai51740/R9920261300, at most one hour. Code/environment/audio/logs were staged locally.

| Query layout | Slices | English RTFx change | Japanese RTFx change | Main English tokens |
| --- | --- | --- | --- | --- |
| Independent | 2 | -10.07% | -10.81% | 72/73 |
| Independent | 4 | -12.72% | -12.91% | 73/73 |
| Independent | 8 | -15.88% | -16.75% | 73/73 |
| Grouped | 2 | -1.73% | -1.77% | 72/73 |
| Grouped | 4 | +2.18% | +2.64% | 73/73 |
| Grouped | 8 | +5.95% | +6.41% | 73/73 |

Japanese tokens match12/12 in all configurations. Main text matches85/85, with WER45/1150 and
CER26/592 unchanged. Slice2 changes one English timestamp path. No reference time annotations
are available; timestamp agreement is not timing accuracy. Decoder calls are unchanged.

Grouped/slice8 reduces English transcribe p50/p95 from25.28/54.17 to23.89/50.80 ms, and Japanese
from34.56/82.04 to32.93/76.75 ms. English median within-input CV is0.21% reference and0.18% variant.
The eager decoder kernel sum for the profiled English input is16.11 ms; this is not graph wall time.
Both modes' observed shared-allocator peak is6.85 GB; these are not isolated memory footprints.

**Compatibility limitation:** slice8 changes the two-second silence transcript from
“I'm sorry, I'm sorry.” to “I'm sorry, but I'm not going to.” The two-second seeded noise output
is unchanged. Both silence variants already produce timestamps outside the two-second input.
No-timestamps ASR, translation, long prompt,30-second repeated speech and110-second audio match.
Slice4 matches all additional cases but has a smaller measured gain. These N variants are not
enabled by default. The opt-in profiler switches are `--gluon-group-queries --gluon-cross-splits 8`;
low-level callers can construct `GluonDecoder(..., group_cross_queries=True, cross_splits=8)`.
Routing remains <=8 rows / <=16 tokens; this is a measured B1/K5 mode, not a B3 throughput claim.

Reproduce using the common model/manifest/environment setup in [beam profiling](beam-search.md):

```sh
uv run python scripts/profile_beam.py --model espnet/owsm_v4_medium_1B \
  --manifest /path/to/english.json --beam 5 --timestamps --batch 1 --repeat 5 \
  --baseline-runtime /path/to/runtime-4d8466a.py --upper-bound on --chunk 1 \
  --fused-rank --fused-finish --kv-reorder prefix --share-cross-kv \
  --gluon-group-queries --gluon-cross-splits 8 --out outputs/grouped8
```

Detailed results and hashes are in
[the result manifest](beam-results-2770255.json); raw tokens, traces and logs are retained under
`outputs/beam-2770255`. Independent splitting and grouped/slice2 are rejected for speed regression.
The account was briefly observed at193 GPUs before this job was submitted, due to other jobs;
submission waited until185 allocated GPUs and no eligible queued GPUs, for186 including this job.
