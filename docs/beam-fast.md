# Further timestamp-beam improvements

This round starts at4d8466a, which already includes strict-bound stopping, fused rank/finish,
valid-prefix KV gathering and shared cross KV. No prior improvement is counted again.

The default now also fuses parent/index handling, token-history gathering and timestamp state.
[The isolated E comparison](beam-state-fusion.md) measures4.30% /4.24% additional English RTFx at
B1/B3 and5.07% /4.02% on Japanese, with all tokens/segments/calls identical and275 GPU tests.
This preserves full-vocabulary PyTorch log-softmax and global top-K, and uses fixed reusable outputs.

Two independent N options remain off by default:

```python
from owsm_trt import OWSM

model = OWSM.from_pretrained(
    "espnet/owsm_v4_medium_1B",
    beam_grouped_cross=True,
    beam_compact_tail=True,
)
results = model.transcribe(
    audio, lang="eng", beam_size=5, timestamps=True, batch_size=3,
)
```

`beam_grouped_cross` groups sibling queries and uses eight key slices on the Gluon path for K5.
B1 uses it throughout decoding; with compaction, a remaining group can use it too. Larger full
batches retain TensorRT routing. The backend has independent dispatch/failure state and shares the
existing weights/engines. Greedy continues using the original backend.

`beam_compact_tail` switches to the single remaining utterance's decoder rows at an existing host
completion check. Selection keeps the original B*K layout and beam width. It adds one graph per
group/cache bucket and uses layer-strided KV views, without a cache-copy transition. Unsupported
shared-cross decoders retain the original execution. Both options can change floating-point order.

The isolated grouped8 comparison improves B1 by about6-8%, but changes a two-second silence
transcript. The isolated tail comparison improves B3 by8-10%, while two English segment ends shift
by20 ms; text remains identical. See [attention sweep](beam-cross-splits.md) and [tail comparison](beam-tail.md).
These are measured baseline differences, not errors against reference timestamps; no time annotation
is available. Gains from separate runs must not be added or multiplied.

The final combined comparison uses one loaded checkpoint and engine set per language/batch,
alternating reference and candidate; English73 x5 and Japanese12 x3. It additionally exercises the
production OWSM session factory on all inputs and compares its tokens/segments with the timed mode.
Cold load/JIT/capture, diagnostic eager traces and disk-inclusive measurements are stored separately
from synchronized warm wall time. Eager kernel sums do not attribute CUDA graph nodes. Peak memory
is the process containing both variants, not independent per-mode memory. Final combined results follow below.

Reproduction (inside an allocated compute job; prepare the baseline from4d8466a):

```sh
git show 4d8466a:src/owsm_trt/runtime.py > /tmp/runtime-4d8466a.py
uv run python scripts/profile_beam.py --model espnet/owsm_v4_medium_1B \
  --manifest /path/to/english.json --beam 5 --timestamps --batch 3 --chunk 4 --repeat 5 \
  --baseline-runtime /tmp/runtime-4d8466a.py --upper-bound on \
  --fused-rank --fused-finish --fused-state --kv-reorder prefix --share-cross-kv \
  --compact-tail --gluon-group-queries --gluon-cross-splits 8 --verify-api \
  --out outputs/combined-fast
```

Use `--batch 1 --chunk 1` for single-utterance latency. Main quality coverage is English73 and
Japanese12; other languages, GPUs and checkpoints remain unmeasured. Extra silence/noise,
no-timestamp, translation, long prompt,30-second repeated speech and110-second recording checks
use B1, so they do not establish B3 compaction quality for those cases. No engine rebuild,
quantization, reduced beam width, altered timestamp constraints or multi-GPU inference is included.

## Final combined result

Job2770624, source2976e99, reference runtime4d8466a. H200 (132 SM), CUDA13.0,
Torch2.14.1, Triton3.8.0, TensorRT11.4.0.106, same fp16 checkpoint/engines. Nsight was unavailable.

| Inputs | B | p50 before → after (ms) | p95 before → after (ms) | RTFx before → after | RTFx gain |
| --- | --- | --- | --- | --- | --- |
| English73 | 1 | 25.45 → 22.78 | 54.14 → 48.02 | 231.74 → 259.15 | +11.83% |
| Japanese12 | 1 | 34.82 → 31.22 | 82.27 → 72.69 | 328.00 → 368.54 | +12.36% |
| English73 | 3 | 49.84 → 45.93 | 103.97 → 87.52 | 314.64 → 360.90 | +14.70% |
| Japanese12 | 3 | 77.64 → 64.91 | 129.72 → 114.22 | 459.98 → 534.93 | +16.29% |

B3 is batch latency. Within-input CV medians (reference/candidate) are0.55%/0.41% English B1,
0.48%/0.14% Japanese B1,2.25%/0.21% English B3 and0.94%/2.55% Japanese B3. These are sample
measurements, not guaranteed speedups. English decoder row-calls decrease13270→10210 at B3 and
Japanese3075→2545; logical decoder step counts are unchanged. B1 row counts do not change.

All main text matches; English WER45/1150 (3.91%) and Japanese CER26/592 (4.39%) are unchanged.
B1 tokens/timestamps/segments match85/85, B3 matches83/85. The two English B3 changes remain the
same +20 ms segment-end differences listed in the tail report. Main segment counts match and
there are no invalid, unpaired or nonmonotonic timestamps. The existing silence text difference
persists in the extra B1 check. All four **production API vs timed candidate** comparisons match
all tokens, timestamps, segments and text, including the final partial English batch.

All293 GPU tests pass (222 basic/API routing,71 state/cross/tail/backend). The baseline token
outputs also match the previously adopted state-fusion outputs, so the reference did not change
between the two jobs. Cold initialization is separate: first English B3 candidate construction
24.27 s vs0.63 s reference includes new tail-kernel JIT and12 additional graphs; after compiler
cache warmup, Japanese B3 construction is0.80 s vs0.72 s. First B1 construction is12.58 s vs10.68 s.
The shared process peaks at6.85 GB B1,11.18 GB English B3 and10.02 GB Japanese B3; these are not
isolated variant footprints. Coarse whole-job GPU utilization samples are in the result manifest.

[Complete metrics and hashes](beam-results-2770624.json); raw outputs/traces/logs:
`outputs/beam-2770624`. All jobs were serial, one rt_HG GPU on gai51740/R9920261300, with fresh
capacity/queue/duplicate checks. Final submission observed176 allocated +8 eligible queued +1 new
=185 GPUs; completion observed184 allocated and no queued jobs. No job remains active for this round.
