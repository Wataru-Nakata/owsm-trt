"""Summarize existing nvidia-smi CSV samples; includes cold/idle time, not a steady-state attribution."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(path: Path) -> dict:
    columns = {'utilization.gpu [%]': [], 'memory.used [MiB]': []}
    malformed = 0
    with path.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            try:
                values = {key: float(row[key].split()[0]) for key in columns}
            except (KeyError, AttributeError, IndexError, ValueError):
                malformed += 1
                continue
            for key, value in values.items():
                columns[key].append(value)
    return dict(source=str(path), malformed_rows=malformed,
        metrics={key: dict(samples=len(values), mean=statistics.mean(values),
                           median=statistics.median(values), maximum=max(values)) if values else None
                 for key, values in columns.items()},
        scope='Entire sampled allocation: includes load/JIT/capture/diagnostics/idle and all resident variant pools. '
              'Five-second utilization snapshots are not a per-variant steady-state metric or kernel wall time.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(json.dumps(summarize(args.csv), indent=2) + '\n')
