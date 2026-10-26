# Existing ring SM-budget experiment (prepared, unmeasured)

Job2770946's instrumented110s trace attributes47.57ms to67 ring decoder calls. After the greedy
selection improvements, the existing ring program budgets are the next measured component to tune.
This experiment changes settings of the existing kernel; it does not introduce a new megakernel.

Compare the1B/H200 default concurrent/solo32/64 against16/64,48/64,32/96. Concurrent settings affect
encoder overlap as well as decoder time, so both single requests and whole-manifest B1 streaming are
measured. The sole change in32/96 is the solo budget. Every persistent CTA still participates in the
existing barriers; GPU SM count must be at least96. Ring tile/warp/ring-capacity parameters are unchanged.

This is an N experiment. The current kernel derives cross-attention split count from the concurrent
budget:1/2/3 slices at16/32/48 programs for1B. Changing that count can alter fp16 outputs. Solo96 retains
the baseline split count2. Tiny-model tests compare logits/new-KV/cache and existing solo bitwise
checks; each real-checkpoint setting is also checked against the existing fp32 reference before
benchmarking. Any kernel failure/fallback or repeated-output nondeterminism aborts; N differences
are retained with WER/CER and time-token/segment comparisons rather than hidden.

All variants share checkpoint/engine/weights and have isolated captured sessions. Selection mode is
fixed across variants. Alternating7-repeat wall samples exclude warmup and instrumentation; a separate
pass counts actual cache/window/solo/programs/cross-split dispatch. Both session pools and all variant
pools are additional benchmark-resident memory, not deployment VRAM. Disk reads are an aggregate
outside request latency. Long-form has no public joined token IDs, so its comparison is text/segments.

`scripts/profile_ring_programs.pbs` stages the environment, source, model, cached engines and English73/
Japanese12/110s audio into node-local storage. It uses the established serial1GPU<=1h policy, with fresh
queue/group/duplicate/account-cap192 checks at submission. No measurement or default adoption yet.
