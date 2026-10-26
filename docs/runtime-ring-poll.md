# Ring barrier polling experiment

Job2772221 measured zero local-memory requests/sectors in the two profiled Ring calls, so a spill
reduction is not justified by those samples. PC-sampled long-scoreboard stalls include3060/6134
samples on CCTL.IVALL for plain R1/W1 and8773/14552 for W8 solo. Representative SASS places these
after LDG.E.STRONG.GPU in the grid-counter polling loop. Much of the barrier category instead
lands at warp-specialization wait/register-allocation machinery; those samples are not removable
wall-time fractions. Raw evidence is outputs/runtime-2772221/{ring_stalls.ncu-rep,stall_analysis.json}.

The opt-in `RingDecoder(relaxed_poll=True)` polls the same32-bit device-scope counter with
`ld.global.relaxed.gpu.b32`, then performs one `ld.global.acquire.gpu.b32` before the existing CTA
barrier and any dependent work. Release arrivals, grid participation, timeout/error handling,
final counter reset, decoder arithmetic, row layout and program counts stay unchanged.

The PTX specification defines a strong read followed by an acquire on the same location as an
acquire pattern; observation order extends through atomic RMW operations. This is the basis for
the change, not a claim that relaxed polling alone synchronizes payloads. The final acquire is
side-effecting inline assembly and must remain in generated code. See [PTX9.0 sections8.8–8.9](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html#memory-consistency-model-release-and-acquire-patterns).

The shared helper's default remains acquire-on-every-poll; the Gluon path and ordinary Ring
construction retain that default. This is E intent and remains experimental pending GPU validation.
Fourteen GPU tests cover4/32/64-CTA payload publication with repeated graph replay, timeout flags,
and exact logits/KV/predictions at windows1/8/32, solo/concurrent program counts and offset changes.
Only test collection and static checks have run before submission; no GPU speed claim is available.

`scripts/profile_ring_poll.pbs` runs those tests plus existing regression checks, then same-engine
alternating reference/relaxed comparisons on1B English73/Japanese12, plain/drafted/translation,
streaming and110-second long-form cases. Plain/translation use12 inputs per case. Long-form compares
text/segments because joined token IDs are unavailable. Optional Nsight counters for the changed
polling loop are collected afterward, independently of warm speech wall time. Any fallback or output
difference fails the compatibility experiment; cold JIT/capture costs remain separate.
