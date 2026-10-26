# Decoder initialization experiments

After mapped loading,2773040's separately instrumented1B decoder helper still took1607ms
(102M256ms). It constructs all randomly initialized parameters and then overwrites them from
the checkpoint. The new private opt-in `_load_decoder(..., empty_init=True)` constructs on
meta, allocates empty CPU storage, regenerates the sole nonpersistent positional buffer and
uses the original strict copy-based state load. It avoids random initialization; production
retains ordinary initialization until measured. No checkpoint dtype/storage alias is assigned
to parameters, so original conversion/copy semantics remain. Missing/unexpected/shape errors
are preserved. A guard rejects unexpected buffers rather than leaving them uninitialized.

This follows PyTorch's [meta-device loading recipe](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)
and `Module.to_empty` behavior inspected in the installed PyTorch source. Unlike the assign
example, this experiment keeps copying state into independent parameter storage. The position
buffer is not checkpoint state and is explicitly reconstructed. Skipping unused initialization
changes global random-number consumption; inference is deterministic and no training path is changed.

Nine tests cover ZIP/legacy, fp16/fp32/fp64 input conversion to the usual parameter dtype,
exact parameters/position/logits/new-KV, writable parameter independence, and strict errors.
All nine decoder tests and ten checkpoint tests passed in allocation2773320 (19 tests,4.27s).

The existing fresh-process profiler can compare `--experiment decoder_init`, with both sides
using adopted private mapping. Initialized/empty workers alternate; Python/CUDA init and
instrumented helper times stay separate. Actual engine/checkpoint hashes and all English/Japanese
ASR/TS/translation/beam/LID outputs must match. The measured meta-mode regression is reported below.

Prepared driver: `scripts/profile_decoder_load.pbs`, tests first, then five alternating fresh-process
pairs for1B and102M plus first/instrumented observations. Engines/checkpoints/audio and temporary
outputs remain node-local. Submission is serial after the active decoder-shape allocation ends
and a fresh queue/capacity/duplicate audit succeeds.

```sh
python scripts/profile_model_load.py --model /local/owsm_v4_medium_1B \
  --manifest /local/manifest.json --extra-manifest /local/extra_manifest.json \
  --experiment decoder_init --repeat 5 --out /local/results/load1b
```

The prepared allocation also compares the existing grouped cross-attention setting with two
key slices for B2/B3 beam5. The1B decoder has16 heads: at B3 this gives96 attention work units
on132 H200 SMs, while four/eight slices have192/384 units. This motivates a measurement, not a
speed claim. Arithmetic reductions differ; all output differences remain N-class results.
This is separate from startup initialization and leaves production routing unchanged.

## Measured decision: not adopted

Job2773320, frozen source37e9348, H200/CUDA13.0, Torch2.14.1/Triton3.8.0/TRT11.4.0.106.
Five alternating fresh-process pairs per model, plus two excluded first workers and two separately
instrumented workers. Both modes use private mapped checkpoint loading and the same cached engines.

| Model | Ordinary initialization median | Meta/empty median | Change in load time | CPU peak RSS change |
| --- | --- | --- | --- | --- |
|1B|4079.16ms|5299.34ms|+29.91% (slower)|+68.83MiB|
|102M|1629.04ms|4428.75ms|+171.86% (slower)|+63.67MiB|

All five pairs regressed:1B20.53–40.51%,102M164.87–223.12%. GPU peak allocation is unchanged.
All728 public API calls (26 requests ×14 workers ×2 models), including full tokens and timestamps,
match; checkpoint/engine hashes are unchanged. The separately instrumented decoder helper is
also slower (1B1449.77→3230.60ms,102M259.60→3163.94ms). These inclusive helper times are not
independent phase sums. No cause was isolated; skipping random initialization alone is insufficient
to justify this meta construction path. Production keeps `empty_init=False`.

Full records and raw outputs: `outputs/runtime-2773320/load1b`, `load102m`, `load_analysis.json`.
The first observations, Python/CUDA initialization, validation and hash I/O remain excluded from
the reported load samples. No cold-disk, engine-build or steady-state gain is claimed.

## Direct parameter allocation follow-up (prepared, unmeasured)

The mapped/ordinary helper still measured1449.77ms for1B after the meta experiment. A separate
`_load_decoder(..., skip_init=True)` now uses standard Linear/Embedding subclasses whose
`reset_parameters` is empty. Their normal PyTorch constructors allocate storage and establish
attributes; no global initializer monkeypatch or meta dispatch is used. The loader immediately
performs the original strict, copying state load. Missing weights fail before a decoder is returned.
Norms and nonpersistent positional buffers initialize normally. Full models and default Decoder
construction retain ordinary nn.Linear/Embedding and initialization order.

This skips random draws and therefore changes global RNG consumption; it does not change
loaded weights, parameter dtype, checkpoint storage ownership or numerical forward operations.
Defaults remain False. The prior meta path remains a separate reproduction option; using both
raises an error. Nineteen small model tests are prepared: both modes' ZIP/legacy/dtype/strict/error
and forward checks plus a direct-RNG/position-buffer guard. They require a compute allocation.

`scripts/profile_decoder_skip.pbs` first runs these tests and focused naive beam/greedy/timestamp
regressions, then `profile_model_load.py --experiment decoder_skip --repeat 5` on1B/102M in fresh
processes with the same node-local cached engines. All English/Japanese public outputs must match.
This is a new ablation, not an attribution of the previous meta slowdown; no improvement is assumed.


## Direct allocation adopted after2773582

Source39c62f7, H200, five alternating fresh-process pairs for each model, with warmed node-local
files/cached engines. Two first and two instrumented workers per model remain outside timing.

|Model|Initialized p50|Direct skipped p50|Load reduction|Paired reductions|
|---|---:|---:|---:|---:|
|1B|4694.93ms|3283.58ms|30.06%|28.15–32.71%|
|102M|1593.71ms|1400.16ms|12.14%|5.92–12.61%|

All728 public API requests match complete results; checkpoint/engine hashes match.29 storage/
initialization tests and17 focused naive beam/greedy/timestamp regressions pass. GPU peak allocation
is unchanged; CPU peak RSS is essentially unchanged (1B-1.38MiB,102M+8.72MiB), so no memory saving
is claimed. Separately instrumented decoder helpers take1540.77→234.78ms and228.02→45.88ms;
these inclusive host timings are not independent phase sums.

Production megakernel setup now explicitly requests `skip_init=True` at both decoder-loading call
sites. The private helper keeps its ordinary default for reference/diagnostic use; full-model Torch
loading and engine building retain normal construction. Strict state copying and initialized position
buffers remain. `empty_init=True` still reproduces the rejected meta experiment. RNG consumption
changes as documented above; loaded inference outputs match. The final unmodified production-settings worker is verified below.

[All load samples and hashes](runtime-decoder-skip-results-2773582.json); raw workers/output records:
`outputs/runtime-2773582/load1b`, `load102m`. These gains are additional to mapped loading in this
comparison, but should not be multiplied by the mapping percentages from another job. No single-
request steady-state, disk-cold or engine-build speed improvement is attributed to this change.


## Production dispatch and independent replication (2773664)

Frozenacd09b3, H200, three fresh-process timing pairs per model. The unmodified factory uses
`_UninitializedLinear` for megakernel setup; the explicit reference uses ordinary `Linear`. Both
1B and102M pass this class check. All572 public API calls across22 fresh workers match complete
outputs and engine/checkpoint hashes.45 focused tests pass in10.30s, including the storage/init
and strengthened lookahead checks.

|Model|Initialized p50|Skipped p50|Load reduction|Paired reductions|
|---|---:|---:|---:|---:|
|1B|4556.31ms|3199.52ms|29.78%|26.61–32.32%|
|102M|1617.49ms|1354.59ms|16.25%|10.99–17.47%|

GPU peaks are unchanged. This remains warmed node-local cached-engine loading, excluding engine
build, disk-cold loading and steady-state inference. No additional production change was needed.
[Verification and samples](runtime-adoption-results-2773664.json); raw outputs:
`outputs/runtime-2773664/load1b`, `load102m`.
