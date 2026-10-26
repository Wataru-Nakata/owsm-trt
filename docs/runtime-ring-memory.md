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
kernel. No speed or quality result is available yet. No defaults change without repeatable benefit
and the applicable numerical/output checks. Script: scripts/profile_ring_memory.pbs.
