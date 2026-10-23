"""GPU job driver for long-form experiments (``scripts/gpu_eval.pbs`` stage ``pyfile``; argv[1] = model).

Submitted by scripts/long_submit.sh, which freezes src/ and scripts/ next to a copy of this file, so the
job runs the code as it was at submission. ``LONG_PLAN`` (colon-separated) picks the steps:

* ``seq``: sequential transcribe_long (ESPnet's decode_long): warm-up pass, timed pass, saved as
  long_seq.json in the job's output directory; with ``dump`` also in the plan, the warm-up pass
  dumps every window (drafts_long.json) for scripts/draft_sim.py
* ``b<N>``: the batched mode with batch size N, compared with this job's long_seq.json (or LONG_REF)
* ``nodraft``: sequential without CTC drafts
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
from owsm_trt import OWSM  # noqa: E402
from owsm_trt.audio import load_audio  # noqa: E402


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
        elif step == "nodraft":
            model.ctc_draft = False
            model._sessions.clear()
            evaluate(
                model, items, waves, 1, save=str(out / "long_nodraft.json"), compare=[seq], label="nodraft"
            )
            model.ctc_draft = True
            model._sessions.clear()
        elif step.startswith("b"):
            b = int(step[1:])
            evaluate(model, items, waves, b, repeat=3, save=str(out / f"long_b{b}.json"),
                     compare=[ref or seq], label=step)  # fmt: skip
        sys.stdout.flush()


if __name__ == "__main__":
    main()
