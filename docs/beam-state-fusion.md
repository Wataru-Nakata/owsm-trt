# Exact beam state update fusion

Reference4d8466a; opt-in `BeamSession(..., fused_state=True)` / profiler `--fused-state`.
This candidate combines index decomposition, active-parent validity, score copy, token-history
parent gather/write and timestamp-state updates into one kernel. Full-vocabulary log-softmax and
global PyTorch top-K remain unchanged, including their tie choices. Self-KV gathering and finished
hypothesis handling retain the previously validated kernels.

One CTA owns each utterance and all K source/destination rows. All parent history and timestamp
state loads precede a block barrier before in-place stores, covering duplicate parents and cycles.
The barrier contract is documented by [Triton](https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html).
There is no device-to-host state read or per-step allocation for the parent/token/valid outputs.
The initial specialization supports K<=8, capacity<=512 on CUDA with Triton; other cases keep the
existing operations. Diagnostics remain optional and the decode return type does not change.

Small interpreter checks cover K5/B3 identity/duplicates/cycles, timestamps/no timestamps, limits
0/15/511, repeated updates, inactive/done rows and invalid scores. Existing boundary tests and the
new state checks initially pass53 tests with3 GPU cases skipped; separate NaN/+Inf/all-inf cases
also pass. Graph/bucket tests plus a real decoder check are allocation-only. Performance and adoption
are pending; the attention split sweep is a separate experiment, not enabled by this state option.

The state allocation compares English73/Japanese12 at B1/B3 on the same loaded engines as the
reference, including extra cases, separate eager profiling and exact diagnostics. The profiler now
attributes the fused metadata kernel separately as `state_update`; graph-internal timing remains
unavailable. CPU-only timings are not performance evidence.
