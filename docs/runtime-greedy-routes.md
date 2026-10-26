# Greedy Gluon routing experiment 2771405

Frozen02e2479, H200/1B, same engine/checkpoint, nine alternating repeats. Default routing remains16.
Gluon hard capacity32 was inspected, the R8W4 numerical/KV test was added (9 shape tests pass),
and the real checkpoint passed the existing fp32 R8W4 check before each language run.

| Corpus | TS | Budget | Route16 RTFx | Route32 RTFx | Increment from route alone |
|---|---|---:|---:|---:|---:|
| draft_english | False | 256 | 2792.8 | 2793.3 | +0.02% |
| draft_english | False | 128 | 2868.5 | 2904.2 | +1.25% |
| draft_english | False | 32 | 1887.5 | 1952.6 | +3.44% |
| draft_english | True | 256 | 2406.0 | 2445.9 | +1.66% |
| draft_english | True | 128 | 2528.6 | 2551.7 | +0.91% |
| draft_english | True | 32 | 1934.4 | 1925.6 | -0.45% |
| draft_japanese | False | 256 | 3520.8 | 3518.7 | -0.06% |
| draft_japanese | False | 128 | 3587.1 | 3651.6 | +1.80% |
| draft_japanese | False | 32 | 2684.4 | 2867.2 | +6.81% |
| draft_japanese | True | 256 | 3143.2 | 3141.8 | -0.04% |
| draft_japanese | True | 128 | 3362.4 | 3386.1 | +0.71% |
| draft_japanese | True | 32 | 1785.6 | 1850.8 | +3.65% |

At budget128 the incremental gain is only about0.7–1.8%. Budget32 is still substantially slower
than the reference even when its R8W4 calls dispatch to Gluon. One English timestamp/segment path
changes at32@32 (1272-141231-0005); all text and normalized WER/CER remain equal. Other evaluated
settings match all exposed output fields. Reference boundary annotations are unavailable.

No default/API routing change is adopted from this experiment. Other32-token row/window shapes
and wider quality coverage would be needed before generalizing the route. The implementation changes
numerical behavior above16 tokens (the existing hi/lo representation reduces to one half), so this
is N-class even though the weights and target equations are unchanged.

Artifacts: `outputs/runtime-2771405/draft_analysis.json`, per-language environment/shape_check,
per-case tokens/wall/cold/actual_dispatch, source/exit/test logs.
