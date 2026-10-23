"""GPU job driver for long-form experiments (``scripts/gpu_eval.pbs`` stage ``pyfile``; argv[1] = model).

Submitted by scripts/long_submit.sh, which freezes src/ and scripts/ next to a copy of this file, so the
job runs the code as it was at submission. ``LONG_PLAN`` (colon-separated) picks the steps:

* ``seq``: sequential transcribe_long (ESPnet's decode_long): warm-up pass, timed pass, saved as
  long_seq.json in the job's output directory; with ``dump`` also in the plan, the warm-up pass
  dumps every window (drafts_long.json) for scripts/draft_sim.py
* ``seqold``: the same without guessed timestamps in the drafts (pipeline.TIMESTAMP_GUESSES = False)
* ``nodraft``: sequential without CTC drafts
* ``b<N>[s<S>]``: windows cut at pauses, batch size N (cuts searched in the last S seconds of a
  window), compared with this job's long_seq.json (or LONG_REF)
* ``r<N>``: ESPnet's windows, the current windows of all recordings decoded together (batch size N)
* ``prof``: GPU time of one 30 s encoder run and of single decoder calls per window size (batch 1)
* ``fp32``: PyTorch fp32 on the first LONG_FP32_LIMIT (8) recordings, sequential with guessed
  timestamps, without, and without drafts: the outputs should be the same
``LONG_LIMIT``: only the first N recordings. ``LONG_DATA``: the manifest (default: the test-clean chapters).
"""

import os
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "src"))  # the frozen package, ahead of the job's snapshot
sys.path.insert(0, str(here))

import json  # noqa: E402

import torch  # noqa: E402
from dump_drafts import dump_long  # noqa: E402
from eval_long import evaluate  # noqa: E402

import owsm_trt  # noqa: E402
from owsm_trt import OWSM, pipeline  # noqa: E402
from owsm_trt.audio import load_audio  # noqa: E402


def timed(fn, iters: int = 50) -> float:
    """Mean milliseconds of ``fn`` on the current stream."""
    fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def profile(model: OWSM, wav: torch.Tensor) -> None:
    sess = model._session(1)
    chunk = wav[: model.cfg.num_samples].to(model.device)
    feats = model._features([chunk])
    print(f"prof: features {timed(lambda: model._features([chunk])):.3f} ms, encoder "
          f"{timed(lambda: sess.encode(feats)):.3f} ms (30 s window, batch 1)")  # fmt: skip
    for c in (128, 256):
        costs = {w: timed(sess._verify[c, w].graph.replay) for w in sess.windows if (c, w) in sess._verify}
        print(
            f"prof: decoder call, cache bucket {c}: " + ", ".join(f"w{w} {ms:.3f}" for w, ms in costs.items())
        )


@torch.inference_mode()
def main() -> None:
    model_dir = sys.argv[1]
    plan = os.environ.get("LONG_PLAN", "seq").split(":")
    out = Path(f"outputs/gpu_eval-{os.environ['PBS_JOBID'].split('.')[0]}")
    out.mkdir(parents=True, exist_ok=True)
    print("package:", owsm_trt.__file__, "plan:", plan)
    data = os.environ.get("LONG_DATA", ".cache/longform/test_clean/manifest.json")
    items = json.load(open(data))
    if os.environ.get("LONG_LIMIT"):
        items = items[: int(os.environ["LONG_LIMIT"])]
    waves = [load_audio(it["path"]) for it in items]
    t0 = time.perf_counter()
    model = OWSM.from_pretrained(model_dir)
    print(f"loaded in {time.perf_counter() - t0:.0f}s")
    ref = os.environ.get("LONG_REF")
    seq = str(out / "long_seq.json")
    for step in plan:
        if step == "seq":
            warm = True
            if "dump" in plan:
                t0 = time.perf_counter()
                dump_long(model, items, waves, str(out / "drafts_long.json"))  # also the warm-up pass
                print(f"dump pass {time.perf_counter() - t0:.1f}s")
                warm = False
            evaluate(model, items, waves, 1, save=seq, compare=[ref] if ref else [], label="seq", warmup=warm)
        elif step == "seqold":
            pipeline.TIMESTAMP_GUESSES = False
            evaluate(model, items, waves, 1, save=str(out / "long_seqold.json"), compare=[seq], label=step)
            pipeline.TIMESTAMP_GUESSES = True
        elif step == "nodraft":
            model.ctc_draft = False
            model._sessions.clear()
            evaluate(model, items, waves, 1, save=str(out / "long_nodraft.json"), compare=[seq], label=step)
            model.ctc_draft = True
            model._sessions.clear()
        elif step.startswith("b"):
            b, _, search = step[1:].partition("s")
            pipeline.PAUSE_SEARCH_SECONDS = float(search or 15)
            evaluate(model, items, waves, int(b), repeat=3, save=str(out / f"long_{step}.json"),
                     compare=[ref or seq], label=step, cut_at_pauses=True)  # fmt: skip
            pipeline.PAUSE_SEARCH_SECONDS = 15.0
        elif step.startswith("r"):
            evaluate(model, items, waves, int(step[1:]), repeat=3, save=str(out / f"long_{step}.json"),
                     compare=[ref or seq], label=step)  # fmt: skip
        elif step == "prof":
            profile(model, waves[0])
        elif step == "fp32":
            n = int(os.environ.get("LONG_FP32_LIMIT", 8))
            ref32 = OWSM.from_pretrained(model_dir, backend="torch", dtype="float32")
            first = str(out / "long_fp32_guesses.json")
            for name in ("guesses", "noguess", "nodraft"):
                pipeline.TIMESTAMP_GUESSES = name != "noguess"
                ref32.ctc_draft = name != "nodraft"
                ref32._sessions.clear()
                evaluate(ref32, items[:n], waves[:n], 1, save=str(out / f"long_fp32_{name}.json"),
                         label=f"fp32 {name}", compare=[first] if name != "guesses" else [], warmup=False)  # fmt: skip
            pipeline.TIMESTAMP_GUESSES = True
            del ref32
            torch.cuda.empty_cache()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
