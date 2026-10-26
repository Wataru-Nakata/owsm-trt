"""Summarize completed whole-runtime cases from JSON; partial failures remain explicit."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    status = args.run / "exit_status.txt"
    report = {
        "exit_status": int(status.read_text()) if status.exists() else None,
        "source": json.loads((args.run / "source.json").read_text()),
        "datasets": {},
    }
    for root in sorted(args.run.iterdir()):
        if not (root / "summary.json").is_file():
            continue
        summaries = json.loads((root / "summary.json").read_text())
        result = {
            "environment": json.loads((root / "environment.json").read_text()),
            "frontend": json.loads((root / "frontend.json").read_text()),
            "cases": {},
        }
        for name, summary in summaries.items():
            if "not_evaluated" in summary:
                result["cases"][name] = summary
                continue
            records = json.loads((root / name / "wall.json").read_text())
            groups = defaultdict(lambda: defaultdict(list))
            for row in records:
                groups[row["variant"]][tuple(row["ids"])].append(row["wall_ms"])
            for v, s in summary.items():
                s["rtfx_speedup"] = s["rtfx"] / summary["reference"]["rtfx"] - 1
                pairs = [
                    statistics.median(groups["reference"][k]) / statistics.median(vals) - 1
                    for k, vals in groups[v].items()
                ]
                s["paired_median_speedup"] = statistics.median(pairs)
                s["within_input_cv_median"] = statistics.median(
                    statistics.pstdev(vals) / statistics.mean(vals) for vals in groups[v].values()
                )
                s["all_equal"] = all(x["equal"] == x["total"] for x in s["agreement"].values())
            result["cases"][name] = {
                "summary": summary,
                "cold": json.loads((root / name / "cold.json").read_text()),
            }
            print(
                root.name,
                name,
                {v: round(s["rtfx_speedup"] * 100, 2) for v, s in summary.items() if v != "reference"},
            )
        report["datasets"][root.name] = result
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
