# Public runtime API acceptance: 2771267

H200, frozen commitcaa7126, one GPU,3min40s.118 tests passed, including fp16/bf16 autocast fallback,
global/utterance MVN, cross-stream feature lifetimes, invalid API options, lazy setup, model and long-form.
The public constructor received `draft_token_budget=128`; default remains256.11 alternating repeats.

| Corpus | Timestamps | p50 ms 256 → 128 | p95 ms 256 → 128 | Aggregate RTFx gain |
|---|---|---:|---:|---:|
| draft_english | False | 166.391 → 164.247 | 170.788 → 167.479 | +1.22% |
| draft_english | True | 192.254 → 184.658 | 194.532 → 186.592 | +4.20% |
| draft_japanese | False | 44.439 → 42.966 | 45.209 → 43.712 | +3.33% |
| draft_japanese | True | 49.917 → 46.623 | 50.366 → 47.181 | +6.82% |

Both budgets match all tokens, timestamps, text and segments within these corpora. Each budget also
reproduces the previous window sweep2771126, with identical checkpoint and engine hashes. English
normalized WER3.913%, Japanese CER4.899%, unchanged across budgets. No reference time annotation is
available; timing agreement is with the baseline, not ground-truth boundary error. This remains an
opt-in numerical configuration because other windows/inputs can change fp16 results.

Reproduce inside a compute allocation:
```bash
uv run python scripts/profile_draft_windows.py --model <1B-model> --manifest <manifest> --lang eng --batch 8 --budgets 128 256 --repeat 11 --out <result-dir>
```

Use:
```python
model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B", draft_token_budget=128)
results = model.transcribe(audio_list, batch_size=8, timestamps=True)
```

The default improvements are frontend_graph=True and lazy_drafts=True, independently switchable to
False. Pointwise fusion remains disabled. Other GPUs and unlisted checkpoints/shapes are unmeasured.
All performance samples use resident models and already loaded audio. Private frontend graph memory
is additional (16 added captures increased allocation by531.5MiB,17 entries total); see the cold/VRAM report.

Artifacts: `outputs/runtime-2771267/draft_analysis.json`, `api_vs_sweep.json`, tests.log,
per-language environment and per-case wall/tokens/cold/actual_dispatch JSON. All five runtime jobs
used the authorized1GPU serial configuration; queue/group/duplicates/account cap192 were checked
before each submission. This validation and its monitor have stopped; no task jobs remain active.
