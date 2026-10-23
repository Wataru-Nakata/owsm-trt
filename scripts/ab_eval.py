"""A/B speed and output check of two versions of the package on one GPU (gpu_eval.pbs stage ``pyfile``).

    PYFILE=scripts/ab_eval.py AB_SRC=outputs/ab/v02/src AB_CONFIGS=b1:b16:long1:long16 [AB_DATA=...]
    [AB_LONG_DATA=...] [AB_REF=...|none] [AB_ROUNDS=2] [AB_EXTRA="..."]

Runs scripts/eval_wer.py (``b<N>``: batch size N) or scripts/eval_long.py (``long<N>``) of the job's
snapshot with ``owsm_trt`` imported from ``AB_SRC`` ("A", e.g. a ``git archive`` of the previous release)
and from the job's snapshot ("B"), alternating A, B, A, B, ... per configuration, each in a process of
its own; both read and build engines in the same cache (their export versions differ). Prints a summary
of RTFx, WER and identical transcripts per configuration.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(os.environ["PYTHONPATH"]).parent / "scripts"  # gpu_eval.pbs sets it to its snapshot's src


def run(src: str, model: str, config: str, ref: str, save: Path, extra: list[str]) -> dict:
    env = dict(os.environ, PYTHONPATH=str(Path(src).resolve()))
    if config.startswith("long"):
        data = os.environ.get("AB_LONG_DATA", ".cache/longform/test_clean/manifest.json")
        cmd = [sys.executable, str(SCRIPTS / "eval_long.py"), "--batch-size", config[4:]]
    else:
        data = os.environ.get("AB_DATA", ".cache/testdata/librispeech_dummy/manifest.json")
        cmd = [sys.executable, str(SCRIPTS / "eval_wer.py"), "--batch-size", config[1:], "--dtype", "float16"]
        if ref != "none":
            cmd += ["--compare", ref]
    cmd += ["--model", model, "--backend", "tensorrt", "--data", data, "--save", str(save), *extra]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = out.stdout + out.stderr
    res = {"ok": out.returncode == 0}
    for key, pat in (
        ("wer", r"WER ([\d.]+)%"),
        ("rtfx", r"RTFx (\d+)"),
        ("same", r"identical to \S+: (\d+/\d+)"),
    ):
        m = re.search(pat, text)
        res[key] = m.group(1) if m else None
    if not res["ok"] or res["rtfx"] is None:
        print(text[-4000:])
    return res


def _text(v) -> str:
    return v["text"] if isinstance(v, dict) else v


def main() -> None:
    model = sys.argv[1]
    src_a = os.environ["AB_SRC"]
    src_b = os.environ["PYTHONPATH"]  # gpu_eval.pbs: the job's snapshot of this checkout
    configs = os.environ.get("AB_CONFIGS", "b1:b16").split(":")
    ref = os.environ.get("AB_REF", "none")
    rounds = int(os.environ.get("AB_ROUNDS", "2"))
    extra = os.environ.get("AB_EXTRA", "").split()
    job = os.environ.get("PBS_JOBID", "x").split(".")[0]
    out = Path(os.environ.get("AB_OUT", f"outputs/gpu_eval-{job}/ab"))
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for c in configs:
        for r in range(rounds):
            for name, src in (("A", src_a), ("B", src_b)):
                res = run(src, model, c, ref, out / f"{name}_{c}.json", extra)
                print(f"{c} round {r} {name}: {res}", flush=True)
                summary.append((c, name, res))
        try:
            a, b = (json.load(open(out / f"{n}_{c}.json")) for n in ("A", "B"))
        except FileNotFoundError:
            continue
        diff = [k for k in a if _text(a[k]) != _text(b.get(k, ""))]
        print(f"{c}: A and B transcripts identical {len(a) - len(diff)}/{len(a)}")
        for k in diff[:5]:
            print(f"  {k}\n    A: {_text(a[k])[:300]}\n    B: {_text(b.get(k, ''))[:300]}")
    print("summary (config, version, RTFx per round, WER, identical to ref):")
    for c in configs:
        for name in ("A", "B"):
            rs = [r for cc, n, r in summary if cc == c and n == name]
            print(
                f"  {c:7s} {name}: RTFx {[r['rtfx'] for r in rs]} WER {rs[-1]['wer']} same {rs[-1]['same']}"
            )


if __name__ == "__main__":
    main()
