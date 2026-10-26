# Ring memory scheduling ablations

Previous Nsight diagnostics recorded low eligible warps and memory waits, with cross attention and
FFN stages taking measurable portions of instrumented time (see runtime-ring-stages.md). SM budget,
L2 lookahead and relaxed grid polling did not produce consistent request gains. The next comparison
uses existing Ring settings, with no runtime default or kernel implementation change:

- `profile_ring_programs.py --sweep attention`: default vs `prefetch_attention=False`. E intent;
  arithmetic, programs and attention split count stay unchanged. Any output difference fails.
- `--sweep chunk`: default chunk2 vs chunk1/chunk4. N; the two FP32 accumulators receive different
  K groups, so numerical output differences must be recorded. This is an independent experiment.

The target is H200, v4 medium1B, R1 windows1..32, both solo and concurrent calls. Each alternative
first passes the existing real-weight fp32 sanity check at W8 and solo W32. Uninstrumented requests
alternate for11 repeats, separately from first-use/capture. English73/Japanese12, normal and timestamp
ASR, translation, plain decoder, B1 streaming and the available110s recording are planned. Same
checkpoint/TensorRT engine hashes and actual dispatch are recorded. Long-form compares returned
text/segments because joined token IDs are unavailable. Human timestamp reference is unavailable.

These settings already exist; this work must not be reported as introducing a new prefetch/ring
kernel. Measured results follow below. No defaults change without repeatable benefit
and the applicable numerical/output checks. Script: scripts/profile_ring_memory.pbs.


## Measured result:2772457, defaults retained

H200/1B,11 alternating repeats. All shape checks pass. Turning attention prefetch off is slower on
both corpora: direct ASR-1.2%, timestamps-1.75%, plain/translation approximately-3.4..-3.7% RTFx.
English B1 streaming+1.48% does not reproduce in Japanese(-0.94%);110s long-form-2.98%. All exposed
outputs match. Keep attention prefetch enabled.

Chunk1 is slower by2.2–2.7% for direct ASR and8.8–9.8% for plain/translation. Chunk4 is slower by
2.3–3.7% for direct ASR and3.2–4.3% for plain/translation; English long-form-7.72%. Both also regress
B1 streaming in both corpora. Chunk1 changes token/text outputs on2/12 Japanese translation inputs;
other sampled cases match, including chunk4. Translation has no reference transcript here, so this
is a baseline difference, not a measured translation-quality error. Keep chunk2; no new runtime
branch or high-level tuning option is warranted. The existing constructor options remain unchanged.

Artifacts: outputs/runtime-2772457/{analysis.json,english_attention,japanese_attention,english_chunk,
japanese_chunk}, environment/checkpoint/engine hashes, raw timing/dispatch/token/quality records and
shape checks. These are rejected existing-setting ablations, not newly introduced kernel features.
