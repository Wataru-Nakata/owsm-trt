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
No GPU result exists yet for this candidate. No performance or compatibility claim is made.
