"""FP8 calibration study driver (gpu_eval.pbs pyfile stage, PYFILE=scripts/fp8_study.py; MODEL as argv[1]).
Test sets from scripts/prepare_multilingual.py; scripts/fp8_table.py scores the saved transcripts.

STUDY_STEPS (':'-separated): measure | base | fp8:<scales file>[@margin][#skip-regex] | lang  (see below)
STUDY_SETS: test sets (default: all), STUDY_BATCH (16), STUDY_REPEAT (2).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

model = sys.argv[1]
src = Path(os.environ["PYTHONPATH"])
scripts = src.parent / "scripts"
ML = ".cache/multilingual"
TESTS = {
    "en": ".cache/testdata/librispeech_test_clean/manifest.json",
    "ja": f"{ML}/jvs_test.json",
    "de": f"{ML}/fleurs_de_de_test.json",
    "es": f"{ML}/fleurs_es_419_test.json",
    "zh": f"{ML}/fleurs_cmn_hans_cn_test.json",
}
CALIB = [
    f"{ML}/fleurs_{lang}_validation.json"
    for lang in ("de_de", "es_419", "cmn_hans_cn", "ja_jp", "fr_fr", "en_us", "ru_ru")
]
CALIB += [f"{ML}/jvs_calib.json", f"{ML}/librispeech_dev_other.json"]
job = os.environ.get("PBS_JOBID", "x").split(".")[0]
out = Path(f"outputs/gpu_eval-{job}")
tag = Path(model).name
sets = os.environ.get("STUDY_SETS", "en:ja:de:es:zh").split(":")
batch = os.environ.get("STUDY_BATCH", "16")
repeat = os.environ.get("STUDY_REPEAT", "2")


def run(cmd, label):
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    text = r.stdout + r.stderr
    keep = [
        x
        for x in text.splitlines()
        if re.search(
            r"^(WER|CER) |RTFx|drafts:|built |FP8 encoder|FP8 scales|recordings,|Traceback|copied", x
        )
    ]
    print(f"--- {label}: " + " | ".join(keep[-8:]), flush=True)
    if r.returncode:
        print(text[-3000:], flush=True)


tokens = os.environ.get("STUDY_STEPS", "measure:base").split(":")
steps = []
while tokens:  # "fp8" takes the next token as its spec (qsub -v: ':' separates)
    t = tokens.pop(0)
    steps.append(f"fp8:{tokens.pop(0)}" if t == "fp8" else t)
for step in steps:
    if step == "measure":
        run(
            [
                str(scripts / "fp8_calibrate.py"),
                "measure",
                "--model",
                model,
                "--data",
                *CALIB,
                "--per-manifest",
                os.environ.get("STUDY_PER_MANIFEST", "100"),
                "--out",
                f"outputs/fp8/rows_{tag}.json",
            ],
            "measure",
        )
    elif step == "lang":  # Mandarin: <zho> or <cmn>?
        for lang in ("zho", "cmn"):
            run(
                [
                    str(scripts / "eval_wer.py"),
                    "--model",
                    model,
                    "--data",
                    TESTS["zh"],
                    "--batch-size",
                    batch,
                    "--repeat",
                    "1",
                    "--lang",
                    lang,
                    "--metric",
                    "cer",
                ],
                f"zh as {lang}",
            )
    else:
        extra, label = [], "fp16"
        if step == "pk":  # the scales the package ships (from_pretrained(fp8=True))
            extra, label = ["--fp8"], "fp8 packaged"
        if step.startswith("fp8:"):
            spec = step[4:]
            skip = spec.split("#", 1)[1] if "#" in spec else ""
            spec = spec.split("#", 1)[0]
            margin = spec.split("@", 1)[1] if "@" in spec else "1"
            scales = spec.split("@", 1)[0]
            extra = ["--fp8-scales", scales, "--fp8-margin", margin] + (["--fp8-skip", skip] if skip else [])
            label = f"fp8 {Path(scales).name} x{margin}" + (f" skip {skip}" if skip else "")
        for s in sets:
            save = out / f"{label.replace(' ', '_').replace('/', '_')}_{s}.json"
            run(
                [
                    str(scripts / "eval_wer.py"),
                    "--model",
                    model,
                    "--data",
                    TESTS[s],
                    "--batch-size",
                    batch,
                    "--repeat",
                    repeat,
                    "--save",
                    str(save),
                    *extra,
                ],
                f"{label} {s}",
            )
