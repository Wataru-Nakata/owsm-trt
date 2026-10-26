# Plain greedy / translation selection fusion (experimental)

Job2771972's English translation trace has24 ring decoder calls (15.94ms GPU sum),25 argmax calls
(0.343ms) and many per-token boolean/mask/state kernels. All PyTorch-native GPU kernels together
sum1.545ms/846 calls in that trace; that includes frontend/other work, not selection alone or wall time.
This motivates measuring mask/argmax/state fusion in the non-drafted path.

The candidate uses vocabulary-block maxima followed by a combined final argmax/done/token/history/
timestamp update. It keeps mask-before-argmax, first-index ties, first surviving NaN, all-invalid
index0, done-row EOS behavior and unchanged pos. Unlike drafted choose, the baseline is PyTorch
argmax, so nonfinite handling has independent tests. Decoder logits and arithmetic are untouched.
Small vocabularies, non-fp32 logits, CPU/missing Triton retain PyTorch. GreedySession(fused_select=True)
and internal OWSM.greedy_fused_select are opt-in; default remains False.

GPU tests cover arbitrary states, time-boundary masks, ties/near-ties, NaN/Inf/all-invalid logits,
EOS, repeated captured calls, final output slot, long prefill and bucket16/32/64 transitions.
`profile_greedy_select.py` reports synthetic selection-only graph CUDA events separately.
`profile_runtime.py graph_plain` compares actual same-engine requests with accepted `graph_lazy`,
including plain greedy with/without timestamps, translation, drafted/beam regressions and102M.
All returned tokens/segments must match. No new speed claim until measurement.
