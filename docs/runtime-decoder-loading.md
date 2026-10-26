# Decoder initialization experiment

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
Model tests belong in an allocation and have not run yet.

The existing fresh-process profiler can compare `--experiment decoder_init`, with both sides
using adopted private mapping. Initialized/empty workers alternate; Python/CUDA init and
instrumented helper times stay separate. Actual engine/checkpoint hashes and all English/Japanese
ASR/TS/translation/beam/LID outputs must match. No speed or memory improvement is yet measured.

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
