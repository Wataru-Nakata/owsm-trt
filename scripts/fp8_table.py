"""Score scripts/fp8_study.py's saved transcripts (outputs/gpu_eval-*/<label>_<set>.json) and gather RTFx from the logs."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
from eval_wer import characters, edit_distance, normalize  # noqa: E402

ML = ".cache/multilingual"
TESTS = {
    "en": ".cache/testdata/librispeech_test_clean/manifest.json",
    "ja": f"{ML}/jvs_test.json",
    "de": f"{ML}/fleurs_de_de_test.json",
    "es": f"{ML}/fleurs_es_419_test.json",
    "zh": f"{ML}/fleurs_cmn_hans_cn_test.json",
}
CER = {"ja", "zh"}
refs = {k: {it["id"]: it["text"] for it in json.load(open(v))} for k, v in TESTS.items()}
jobs = sys.argv[1:]
res = defaultdict(dict)
for job in jobs:
    d = Path(f"outputs/gpu_eval-{job}")
    log = (d / "pyfile.log").read_text() if (d / "pyfile.log").exists() else ""
    for line in log.splitlines():
        m = re.match(r"--- (.+) (en|ja|de|es|zh): .*RTFx (\d+)", line)
        if m:
            res[m.group(1).replace(" ", "_").replace("/", "_")][m.group(2) + "_rtfx"] = int(m.group(3))
        m = re.match(r"--- (.+) (en|ja|de|es|zh): .*tokens, (\d+) calls", line)
        if m:
            res[m.group(1).replace(" ", "_").replace("/", "_")][m.group(2) + "_calls"] = int(m.group(3))
    for f in d.glob("*_[a-z][a-z].json"):
        label, s = f.stem.rsplit("_", 1)
        if s not in refs:
            continue
        hyps = json.load(open(f))
        units = characters if s in CER else normalize
        e = sum(edit_distance(units(refs[s][k]), units(h)) for k, h in hyps.items())
        n = sum(len(units(refs[s][k])) for k in hyps)
        res[label][s] = 100 * e / n
for label, r in sorted(res.items()):
    cells = []
    for s in ("en", "ja", "de", "es", "zh"):
        if s in r:
            cells.append(
                f"{s} {r[s]:5.2f}" + (f" ({r.get(s + '_rtfx', '-')})" if s + "_rtfx" in r else "      ")
            )
    print(f"{label.replace('owsm_v4_medium_1B_', '').replace('owsm_ctc_v4_1B_', ''):30s} " + "  ".join(cells))
