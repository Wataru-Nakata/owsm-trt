# 102M Ring row program scheduling: prepared experiment

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
separately from wall timing. No speed result is available yet; current defaults remain24.

Run `scripts/profile_ring_row_programs.pbs` in the approved serial compute allocation, with
node-local source/environment/audio/cache. Cold compiler-cache order and all variants' combined
VRAM are reported separately and cannot establish standalone deployment improvements. No reference
translations or timestamp boundaries are available. Initial stage profiles are fixed-state
attribution, not a prediction of this experiment's request gains.
