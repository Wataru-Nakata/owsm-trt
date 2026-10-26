# Block-parallel greedy selection experiment

The existing graph trace from job2770946 (`english/long_greedy/operators_graph.json`) attributes
5.04 ms across67 calls to `_choose`, versus47.57 ms to the ring decoder. These are instrumented GPU
kernel sums, not an estimate of achievable end-to-end speedup. B8 has different TensorRT/encoder
costs; it is included as a separate shape rather than extrapolated from B1.

The candidate splits the existing2048-token vocabulary loop across CTAs, then reduces partial maxima.
Timestamp state/masking and each block's argmax are unchanged. The final reduction picks the first
maximal block, preserving the existing smallest-index tie rule and token0 for all-invalid logits.
`skip_notime` still leaves the decoder's existing prediction untouched. No logits are normalized,
recomputed, rounded or pruned. Private graph-stable scratch belongs to each verification window;
small vocabularies retain serial execution. `GreedySession(split_choose=True)` and the internal
`OWSM.greedy_split_choose` switch are experimental and defaultFalse pending actual verification.

Tests compare serial and split paths on irregular vocabulary lengths, ties across block boundaries,
near ties, all-invalid logits, infinities/NaNs, first/end timestamp states, skipped rows and graph
replays. Random valid states also compare with the PyTorch reference. The NaN behavior is checked
against the existing kernel, not newly specified as a numerical policy.

`profile_greedy_choose.py` measures synthetic identical-logit graph replay CUDA events separately
from model latency. `profile_runtime.py --variants reference graph graph_split` compares selection
on the same checkpoint/engine with separate captured sessions and alternating warmed trials.
Compare `graph_split` with `graph` for the incremental selection effect; `reference` is eager frontend.
Job2771592 passed109 GPU tests and17 runtime cases (English8,Japanese5,102M4), with all
sampled tokens/timestamp tokens/text/segments matching. Long-form token IDs remain unavailable.
Against graph/serial selection, timestamped single-request RTFx improved4.18% English,4.47% Japanese
on1B and10.03% on102M; the110s recording improved6.13%. Timestamped B8 improved0.78% English and
1.64% Japanese on1B,4.96% on102M. Plain B1 streaming changed-0.60%/-0.31%/-0.86% respectively;
the extra launch on rows whose prediction was already computed warrants a fused follow-up.
These are7 alternating repeats on H200, not other-GPU claims. English single-TS p50 went6.328 to6.080ms,
Japanese7.960 to7.605ms,102M2.515 to2.314ms; long-form74.696 to70.438ms (resident model/audio).

Synthetic event time at R1W1 went44.68 to5.01us (8.91x kernel-only), R1W16 46.30 to9.38us,
R1W64 49.13 to24.71us, R8W16 47.49 to35.07us. This factor must not be reported as whole-runtime
speedup. Workspace300bytes atR1W1,4800bytes atR1W16,76800bytes atR16W16. Runtime comparisons keep
both session pools resident, so their peak VRAM is not standalone deployment memory. Cold captures
and all samples are retained in `outputs/runtime-2771592`, especially `selection.json` and
`analysis-graph.json`. The default remains unchanged while fusion/acceptance is verified.
