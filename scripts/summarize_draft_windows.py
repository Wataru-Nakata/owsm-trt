"""JSON-only summary of numerical verification-window experiments."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    report = {}
    for root in sorted(args.run.glob("draft_*")):
        if not (root / "summary.json").is_file():
            continue
        cases = json.loads((root / "summary.json").read_text())
        for name, modes in cases.items():
            ref = modes["256"] if "256" in modes else modes["256@16"]
            cold = json.loads((root / name / "cold.json").read_text())
            dispatch = json.loads((root / name / "actual_dispatch.json").read_text())
            wall = json.loads((root / name / "wall.json").read_text())
            for budget, mode in modes.items():
                mode["rtfx_speedup"] = mode["rtfx"] / ref["rtfx"] - 1
                mode["p50_speedup"] = ref["wall_ms"]["p50"] / mode["wall_ms"]["p50"] - 1
                mode["windows"] = cold[budget]["windows"]
                mode["stream"] = cold[budget]["stream"]
                mode["actual_dispatch"] = dispatch[budget]
                mode["calls_per_repeat"] = [r["decoder_calls"] for r in wall if str(r["budget"]) == budget]
            print(
                root.name,
                name,
                {
                    b: dict(
                        gain=round(m["rtfx_speedup"] * 100, 2),
                        tokens=m["agreement"]["tokens"]["equal"],
                        text=m["agreement"]["text"]["equal"],
                        quality=m["quality"]["rate"],
                    )
                    for b, m in modes.items()
                },
            )
        report[root.name] = cases
    (args.run / "draft_analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
