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
`OWSM.greedy_split_choose` switch were initially experimental; the accepted public default is described below.

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

## Fused follow-up

A second candidate combines block-max reduction with draft acceptance. It still writes every `pred`
for diagnostics, preserves the existing decoder predictions of skipped notime rows, and returns the
same accepted-count/next-token pair. This removes the extra reduction launch introduced by split
selection: serial choose+accept and fused partial-choose+reduce-accept both have two launches.
`GreedySession(fuse_accept=True, split_choose=True)` is experimental; the normal default is unchanged.
The follow-up includes exact acceptance runs, intermediate EOS, irregular window sizes, graph reuse,
and the same timestamp/nonfinite cases. `profile_greedy_choose.py --fused` times choose+accept for all
three modes, rather than mixing a choose-only baseline with acceptance-inclusive candidates.

Job2771634 passed127 tests and all17 runtime cases. Fused selection retained exact sampled outputs.
Against graph/serial, timestamped single RTFx improved4.15% English/5.18% Japanese on1B and8.99% on102M;
110s improved6.05%. Synthetic choose+accept R1W1 was46.50us serial,6.66us split,5.75us fused.
English plain B1 streaming nevertheless changed-2.01% with candidate CV2.21%; all variants used174
calls and two fused trials were high outliers. This does not justify declaring the plain-mode issue
resolved. Data: `outputs/runtime-2771634/analysis-graph.json`, selection.json, wall/tokens/cold records.
Job2771706 additionally passed the synthetic1s/30s silence and30s noise comparison (4 API cases);
its `edge_recipe.json` defines the node-local input generation. These are baseline agreement checks,
not annotated WER/timestamp accuracy. The normal default is still unchanged.

## Reuse notime prediction (measured, experimental)

When every primer contains the same notime token used by `_rules_start` and the backend already
writes the correctly masked argmax, the extra choose kernel can be omitted entirely. Mixed batches,
TensorRT closures without an argmax and timestamp requests keep normal selection. The request's
primer is already on the host; no per-step device synchronization is added.

`reuse_notime=True` is experimental. With lazy draft setup, first use captures only the requested
mode's graphs. The other mode is captured only if later needed, before timestamp-state setup and
prefill. Capture warmup resets the shared pinned input to zero offsets; otherwise a previous request's
large offsets could be replayed against smaller cache buckets. Tests explicitly inject that condition,
retain graph identities, switch short/long/notime/timed/mixed requests and compare streaming outputs.
Cold captures/memory and actual chosen graphs are recorded separately. No speed claim yet.

## Short-primer flag initialization (prepared)

For drafted requests whose primer fits a verification window, online timestamp count/last/first and
previous-token state live in `_Verify.inp` and the CPU row state. Reinitializing the corresponding
base-session buffers is unnecessary. The candidate `cache_draft_flags=True` updates just per-row
notime flags, caching their host tuple across matching requests. Plain decode and long prefill
invalidate that cache and still initialize all state. Mixed batches copy their distinct flags.
No decoder logits, selection or accepted tokens change by design. Tests include plain-decode
interposition between identical draft primers to catch stale flag caching. This option remains
experimental and disabled pending a separate same-engine ablation against notime reuse.

## Accepted default after replication

Job2771808 passed134 regression tests and4 mode/lifetime tests. English73 utterances, Japanese12,
base102M English73 and the110s recording were measured across17 API/model cases,11 alternating
repeats per variant. All returned tokens, timestamp tokens, text and segments matched the serial
baseline; long-form exposes only text/segments. Translation uses12 inputs and long-prompt uses6.
No reference timestamp annotations are available. All engine/checkpoint hashes, WER/CER, wall samples,
cold counts and actual dispatch are retained in `outputs/runtime-2771808/analysis-graph.json` and
its dataset directories. Timings use resident audio/model; disk read and model load are separate.

Against graph/serial selection, fused RTFx changed:

| Case | 1B English | 1B Japanese | 102M English |
|---|---:|---:|---:|
| Single, timestamps | +4.29% | +4.82% | +10.55% |
| Single, plain | -0.08% | -0.27% | -0.21% |
| Streaming B1, plain | -0.16% | +0.11% | +0.05% |
| Streaming B8, timestamps | +0.72% | +1.62% | +5.38% |
| 110s long-form | +5.62% | unmeasured | unmeasured |

This reproduces the timestamp/long-form gains across three jobs; plain-mode changes are small.
`OWSM.from_pretrained(greedy_kernels=True)` now enables split selection plus fused acceptance by
default; False restores serial selection. Standalone GreedySession keeps explicit switches. This
changes the surrounding greedy verification kernels, not the decoder arithmetic, engine or search.
CPU/missing Triton and vocabularies<=2048 keep the reference path. H200 is the measured GPU.

Notime-only graph reuse did not add a consistent gain: English plain streaming B1 was-0.51% versus
serial, Japanese B8-0.14%; fused without reuse was-0.16%/+1.27%. Both match tokens. First-use capture
creates only the requested mode, but switching modes retains another graph set. The extra mode is
therefore **not** enabled by the accepted default. `reuse_notime` stays experimental.
Per-case VRAM includes all variant session pools, not standalone deployment usage. Synthetic
selection workspace remains300bytes atR1W1/76.8KB atR16W16, separate from graph metadata.

## Short-primer measurements and narrower follow-up

Job2771857 passed135 regression and8 state/mode tests, with exact outputs for the17 cases. Against
notime reuse alone, cached initialization improved single-request RTFx1.92%/1.42% (plain/timestamp)
English1B,1.58%/0.95% Japanese1B and4.43%/4.17% English102M. Candidate within-input median CVs were
0.4–0.6% for1B and1.2–1.3% for102M. English plain B1 streaming changed-0.93%/-1.94% (1B/102M), with
CV1.28%/7.09%; no streaming benefit is established. Artifact: `outputs/runtime-2771857/analysis-graph.json`.

The follow-up limits this candidate to fixed-batch `decode_drafted` short primers. Streaming retains
its prior initialization. `profile_runtime.py`'s `graph_flags` compares it with the accepted fused
selection **without** the unadopted notime graph mode. It is still disabled by default until that
independent replication completes. Existing mixed/long-primer/plain-decode state tests remain.
