# 102M Ring row program scheduling: first measured comparison

Job2775955 attributed about25% of R3 and39% of R5/R8 instrumented kernel time to cross-attention.
The unchanged Ring row budget24 assigns30 or48 heads to24 CTAs. Test existing `row_programs`
settings through a profiler-only adapter; no runtime default, kernel arithmetic or API change:

|Rows|Reference|heads|tiles|
|---|---:|---:|---:|
|2|24|24|32|
|3|24|24|32|
|4|24|24|32|
|5|24|30|32|
|6|24|36|48|
|7|24|42|48|
|8|24|48|48|

Both candidate tables cover all row/head pairs in one pass and retain the reference cross-split
count (2 atR2,1 atR3..8). Single-row calls keep the same original Ring kernel, including its
accepted EOS guard. Unsupported large windows keep the existing TensorRT fallback. This is E
intent **within the same opt-in Ring:rr route**; its previously measured N differences versus
`auto` remain. It is not an experiment relaxing the failed random-KV fp32 gate.

`profile_ring_row_programs.py` requires exact Ring-relative logits/new-KV/self-cache/argmax for
allR2..8 at windows1/4 on the real102M checkpoint, finite data and unchanged split counts. GPU
unit tests cover the same shapes under repeated graphs at offsets3/31/3. Full English73/Japanese12
outputs must match for B3 plain ASR/timestamps, B8 translation/drafted timestamps and beam5 B1.
Eleven alternating whole-list repeats share checkpoint/engine/weights with separate sessions.
Any output difference, runtime trial failure or warm capture aborts. Counters are collected
separately from wall timing. The first measured result is below; independent public-path verification is next.

Run `scripts/profile_ring_row_programs.pbs` in the approved serial compute allocation, with
node-local source/environment/audio/cache. Cold compiler-cache order and all variants' combined
VRAM are reported separately and cannot establish standalone deployment improvements. No reference
translations or timestamp boundaries are available. Initial stage profiles are fixed-state
attribution, not a prediction of this experiment's request gains.


## First result2775986

Frozen33d7edf, H200,30 GPU/CPU tests passed and42 real-checkpoint shape checks per corpus were
finite/bitwise exact to Ring. All tokens/text/timestamps/segments and decoder call counts matched
on English73/Japanese12, with11 alternating warmed whole-list repeats. Aggregate RTFx changes:

|Condition|English heads / tiles|Japanese heads / tiles|
|---|---:|---:|
|Plain ASR B3|+0.39% / +1.50%|+1.30% / +2.87%|
|Timestamp ASR B3|+0.84% / +1.32%|+1.35% / +2.72%|
|Translation B8|+18.21% / +18.62%|+21.62% / +22.46%|
|Timestamp beam5 B1|+17.99% / +17.67%|+18.68% / +19.16%|
|Drafted timestamp ASR B8|+1.29% / +1.69%|+16.13% / +16.33%|

All beam/translation and Japanese drafted pairs were positive. English drafted timing was noisy
(heads-0.41..+7.31%,6/11 positive); no robust gain there. Heads leaves R2..4 at the original24,
so its B3 changes are measurement variation. Tiles adds a small B3 gain but some timestamp pairs
regress; it is unnecessary to the clear R5/R8 result. Select the smaller **heads** policy for
independent API/edge/latency validation: change only R5..8, retain scalar/default-auto routes.

This is same-Ring scheduling parity, not a new accuracy comparison against auto/TRT. Prior102M
Ring WER/CER differences and poor translation output remain; matching those outputs does not
improve translation quality. Do not multiply this gain by the earlier Ring-route result: the
combined current public route needs its own measurement against auto. [Full first result](runtime-ring-row-programs-2775986.json).
Raw artifacts: `outputs/runtime-2775986/`; Stageout_status1 retained despite complete exit0 output.
