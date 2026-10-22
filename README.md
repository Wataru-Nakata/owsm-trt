# owsm-trt

Fast inference for [OWSM](https://www.wavlab.org/activities/2024/owsm/) (Open Whisper-style Speech Models, ESPnet)
with TensorRT. You can `pip install` it, and it does not depend on ESPnet.

```bash
pip install owsm-trt
```

```python
from owsm_trt import OWSM

model = OWSM.from_pretrained("espnet/owsm_v4_medium_1B")   # first call builds TensorRT engines (~4 min, cached)
print(model.transcribe("speech.wav", lang="eng").text)

model.transcribe(["a.wav", "b.wav", "c.wav"], batch_size=8)  # batched
model.transcribe("speech.wav", timestamps=True).segments     # utterance timestamps
model.transcribe("speech.wav", lang="eng", task="st_deu")    # speech translation
model.detect_language("speech.wav")                          # [("eng", 0.999), ...]
model.transcribe_long("podcast.wav").segments                # > 30 s, with timestamps
```

```bash
owsm-trt build --model espnet/owsm_v4_medium_1B          # optional: pre-build engines
owsm-trt transcribe --lang eng speech.wav
owsm-trt transcribe --long --json podcast.wav
owsm-trt bench --batch-size 8
```

## Speed

Test setup:
* model: `espnet/owsm_v4_medium_1B`
* GPU: one H200
* data: 73 LibriSpeech dev-clean utterances (481 s of audio)
* decoding: greedy unless noted
* RTFx: seconds of audio transcribed per second (median of 3 runs)

| setup | batch | RTFx | WER |
|---|---:|---:|---:|
| ESPnet `Speech2Text`, beam 5 (its default), fp32 | 1 | 15 | 4.00% |
| ESPnet `Speech2Text`, greedy, fp32 | 1 | 25 | 3.91% |
| owsm-trt, `backend="torch"`, fp16 | 1 | 116 | 3.74% |
| owsm-trt, `backend="torch"`, fp16 | 8 | 459 | 3.91% |
| **owsm-trt, TensorRT, fp16** | 1 | **220** | 3.91% |
| **owsm-trt, TensorRT, fp16** | 8 | **711** | 3.74% |

That is about 9× ESPnet's greedy decoding and 15× its default beam search for a single stream, and 28–47× with
batching. WER is unchanged: the 0.17-point differences come from 1–2 words out of 1150. Transcripts match the fp32
reimplementation on 72/73 utterances.

## How it works

* **No ESPnet.** OWSM v3.1 / v4 (E-Branchformer encoder + Transformer decoder) is reimplemented in plain PyTorch
  (`owsm_trt/modeling.py`), and the official checkpoints load with `strict=True`. In fp32 it matches ESPnet's
  `Speech2Text`: the encoder output differs by at most 7e-5, and greedy transcripts are identical
  (`scripts/parity_espnet.py`).
* **TensorRT engines are built on first use.** The encoder and a single decoder step are exported to ONNX and compiled
  into strongly-typed fp16 engines.
  * There are separate optimization profiles for batch-1 decoding, batched decoding, and prompt prefill. Kernels
    tuned for batch 16 waste most of the GPU at batch 1.
  * Engines depend on the GPU and the TensorRT version, so they are cached under `~/.cache/owsm_trt` (set
    `OWSM_TRT_CACHE` to move them).
* **Decoding work is minimized.**
  * Cross-attention keys/values are projected once per utterance.
  * The self-attention KV cache is static and comes in length buckets (64/128/256/512), so a step only reads as
    much cache as the sequence needs.
  * Each greedy step (engine, cache write, argmax, token feedback) is one CUDA-graph replay, and the host checks for
    `<eos>` every 4 steps.
  * Batched inputs are sorted by duration, so a batch does not wait on one long utterance.
* **Same API without TensorRT.** `backend="torch"` runs the same decoding loop in PyTorch with CUDA graphs. It is useful
  where TensorRT is unavailable and is still ~5× faster than ESPnet.

## Supported models

The ESPnet `s2t` models with an E-Branchformer encoder and a Transformer decoder. These were checked:

| model | fp32 parity with ESPnet greedy | TensorRT fp16 (H200, batch 1 / 8) |
|---|---|---|
| `espnet/owsm_v4_base_102M` | 8/8 transcripts identical | RTFx 392 / 1321 (earlier build) |
| `espnet/owsm_v4_small_370M` | 5/5 identical | — |
| `espnet/owsm_v4_medium_1B` | **73/73 identical** | RTFx 220 / 711, WER 3.91% / 3.74% |
| `espnet/owsm_v3.1_ebf` | 3/3 identical | RTFx 196 / 571, WER 3.65% |

Other v3.1 E-Branchformer sizes use the same code paths. CTC-only models (OWSM-CTC) and the Transformer-encoder OWSM
v1–v3 are not supported; loading them fails with a clear error.

## Requirements

You need an NVIDIA GPU, Python ≥ 3.10, torch ≥ 2.6, and PyTorch and TensorRT built for the **same CUDA major
version**. The PyPI defaults (`torch`, `tensorrt`) are both CUDA 13. If your torch is a CUDA 12 build, run
`pip install tensorrt-cu12`; owsm-trt warns at load time when the two do not match.

Tested with torch 2.14 + TensorRT 11.4 on an H200. The ONNX export was also verified with torch 2.6 and 2.8.

## Limitations (v0.1)

* Only greedy decoding (ESPnet's default is beam search with `beam_size=5`; on the test above greedy is not worse).
* Within a batch, prompts must have the same length (always true unless `prompt=` text is used).

## Development

```bash
uv sync                                   # dev environment
uv run pytest                             # CPU unit tests (KV cache, buckets, fused projections, profiles)
```

Scripts in `scripts/`:
* `parity_espnet.py`: fp32 parity against official ESPnet. It runs in `tools/espnet_ref`, which is only used for
  reference and is never a dependency.
* `eval_wer.py`: WER, speed, and transcript agreement for one backend configuration.
* `profile_phases.py`: CUDA-event timing of the frontend, encoder, prefill, and per-bucket decode steps.
* `trt_layers.py`: per-layer TensorRT profile plus an engine-inspector dump.
* `bench_espnet.py`: ESPnet `Speech2Text` baseline.
* `smoke_features.py`: language ID, timestamps, translation, prompts, and long-form.

On a cluster with slow shared storage (e.g. Lustre), `scripts/build_env_tar.sh` builds a relocatable environment on
local disk and packs it into one `.tar.zst`. `scripts/gpu_eval.pbs` unpacks it into `$PBS_LOCALDIR` and runs the
stages listed in `STAGES`.
