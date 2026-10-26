# Ring barrier polling experiment (not adopted)

Job2772221 measured zero local-memory requests/sectors in the two profiled Ring calls, so a spill
reduction is not justified by those samples. PC-sampled long-scoreboard stalls include3060/6134
samples on CCTL.IVALL for plain R1/W1 and8773/14552 for W8 solo. Representative SASS places these
after LDG.E.STRONG.GPU in the grid-counter polling loop. Much of the barrier category instead
lands at warp-specialization wait/register-allocation machinery; those samples are not removable
wall-time fractions. Raw evidence is outputs/runtime-2772221/{ring_stalls.ncu-rep,stall_analysis.json}.

The experimental `RingDecoder(relaxed_poll=True)` in commit33e230f polls the same32-bit device-scope counter with
`ld.global.relaxed.gpu.b32`, then performs one `ld.global.acquire.gpu.b32` before the existing CTA
barrier and any dependent work. Release arrivals, grid participation, timeout/error handling,
final counter reset, decoder arithmetic, row layout and program counts stay unchanged.

The PTX specification defines a strong read followed by an acquire on the same location as an
acquire pattern; observation order extends through atomic RMW operations. This is the basis for
the change, not a claim that relaxed polling alone synchronizes payloads. The final acquire is
side-effecting inline assembly and must remain in generated code. See [PTX9.0 sections8.8–8.9](https://docs.nvidia.com/cuda/archive/13.0.1/parallel-thread-execution/index.html#memory-consistency-model-release-and-acquire-patterns).

The shared helper's default remains acquire-on-every-poll; the Gluon path and ordinary Ring
construction retain that default. This has E intent. The measured result below does not justify adoption; the branch was removed
from the current runtime after validation, and remains reproducible from commit33e230f/snapshot.
Fourteen GPU tests cover4/32/64-CTA payload publication with repeated graph replay, timeout flags,
and exact logits/KV/predictions at windows1/8/32, solo/concurrent program counts and offset changes.
All14 tests and131 existing regression checks passed in the allocated GPU job.

`scripts/profile_ring_poll.pbs` runs those tests plus existing regression checks, then same-engine
alternating reference/relaxed comparisons on1B English73/Japanese12, plain/drafted/translation,
streaming and110-second long-form cases. Plain/translation use12 inputs per case. Long-form compares
text/segments because joined token IDs are unavailable. Optional Nsight counters for the changed
polling loop are collected afterward, independently of warm speech wall time. Any fallback or output
difference fails the compatibility experiment; cold JIT/capture costs remain separate.


## Measured result:2772303

H200/1B,11 alternating repeats, same checkpoint/engines. All145 tests passed in58.88s, all13 runtime
conditions matched, and historical baseline tokens also match the preceding accepted-kernel runs.
RTFx changes (English/Japanese): drafted single-utterance plain-0.16/+0.27%, timestamps-0.07/+0.20%;
non-drafted plain+0.17/-0.12%, timestamps+0.11/+0.42%, translation+0.05/-0.16%; B1 streaming+1.23/+0.30%.
English110s long-form changed-0.23%. Most changes are within timing variation; the streaming gain
was not consistent across corpora. There is no broadly improved latency path to adopt.

Nsight confirms the changed mode was actually selected. Static CCTL.IVALL sites across the two
profiled kernels dropped48→29, but sampled long-scoreboard stalls remained similar (6064 and14499
samples; ratios4.60/8.17 per active issue). Local-memory requests remained zero. Reducing polling
instructions did not remove enough work on the critical execution path to yield reliable speedup.
No102M follow-up is needed for a candidate that failed the initial performance criterion.

Runtime/helper changes and their experiment-only tests/wrapper were removed; default synchronization
is unchanged. Reproduction uses commit33e230f and the immutable source snapshot in
outputs/runtime-ring-poll. Measurements: outputs/runtime-2772303/{analysis.json,tests.log,
historical_baseline_comparison.json,ring_poll_stalls.ncu-rep}, raw dataset records and counter CSV.
