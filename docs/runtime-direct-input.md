# Direct frontend input: measured and not adopted

Job2771513 (H200, source2f2d189, same resident checkpoint/engine) passed116 tests and completed
23 runtime cases: English12, Japanese5, base102m4, CTC English2. Features matched exactly for B1/B8
in all4 datasets; returned tokens, timestamp tokens, text and segments matched in every case.
Long-form comparison covers text/segments only because the public joined result has no token IDs.
The corpus remains English73/Japanese12 plus the110-second recording; no reference time annotations.

The candidate loaded waveform data directly into frontend graph input, initialized only the padding
tail, and removed a temporary GPU batch and full input copy. It retained output clones and serialized
workspace use. All results below compare against the already adopted frontend graph, not eager mode.
Seven warmed alternating repeats were collected with resident models and already loaded audio.

| Model/case | Incremental RTFx | Median within-input CV of candidate |
|---|---:|---:|
| 1B English single greedy | +0.14% | 0.95% |
| 1B English single timestamps | +0.35% | 0.72% |
| 1B English streaming B1 | +1.14% | 1.17% |
| 1B English streaming B8 | -2.58% | 0.85% |
| 1B English110s | -0.08% | 0.29% |
| 1B Japanese single greedy | -0.09% | 0.94% |
| 1B Japanese streaming B1 | +0.14% | 0.60% |
| 102M English single greedy | +0.76% | 2.82% |
| 102M English single timestamps | +0.03% | 1.21% |
| CTC1B English single | -0.01% | 1.24% |
| CTC1B English streaming B1 | -0.87% | 1.72% |

CV is descriptive variation, not a confidence interval. B8 mostly uses the unchanged eager frontend
and can encounter a final smaller batch. These observations do not establish a broad speed gain.
The candidate is not adopted and its runtime path has been removed from the active implementation;
source2f2d189 and its immutable archive preserve reproducibility. Existing frontend_graph behavior
and the previously accepted gains are retained.

The preceding job2771494 stopped after114 passed/2 failed tests, before benchmarking. Both failures
were assertions in the fp16 AMP eager fallback: silent padding can underflow the existing logmel
clamp and produce NaNs during per-utterance mean subtraction. The corrected test explicitly compares
NaN/Inf masks and exact remaining values. Job2771513 confirmed identical masks/values. Normal fp32
graph tests had already passed; no runtime change was made to conceal the AMP behavior.

Artifacts: `outputs/runtime-2771513/{analysis-graph.json,tests.log,source.json}` and per-dataset
`frontend.json`, `environment.json`, per-case wall/tokens/cold/summary JSON. Failed-job evidence is
retained in `outputs/runtime-2771494`. Both serial1GPU jobs exited and their monitors stopped.
Reproduce the experiment using source2f2d189 and `scripts/profile_frontend_direct.pbs` from that
snapshot within an allocated compute job, then aggregate using the current script:

```bash
python3 scripts/summarize_runtime.py outputs/runtime-2771513 --reference graph
```
