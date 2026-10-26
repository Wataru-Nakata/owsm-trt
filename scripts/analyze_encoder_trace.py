"""CUDA kernel overlap from a Chrome/Kineto trace; parse large traces on a compute node."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def analyze(events: list[dict]) -> dict:
    streams, names = defaultdict(list), defaultdict(Counter)
    missing = 0
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        stream = event.get("args", {}).get("stream")
        if stream is None:
            missing += 1
            continue
        key = str((event.get("pid"), stream))
        streams[key].append((event["ts"], event["ts"] + event["dur"]))
        names[key][event.get("name", "unknown")] += event["dur"]
    points = []
    for intervals in streams.values():
        for start, end in merge(intervals):
            points.extend(((start, 1), (end, -1)))
    active, previous, union, overlap = 0, 0, 0, 0
    for time, delta in sorted(points):
        elapsed = time - previous
        if active:
            union += elapsed
        if active > 1:
            overlap += elapsed
        active += delta
        previous = time
    return dict(
        kernel_events=sum(map(len, streams.values())), missing_stream_events=missing,
        active_kernel_union_ms=union / 1000, inter_stream_kernel_overlap_ms=overlap / 1000,
        overlap_fraction_of_active_kernel_time=overlap / union if union else None,
        streams={key: dict(kernel_events=len(intervals),
                           duration_sum_ms=sum(end - start for start, end in intervals) / 1000,
                           active_union_ms=sum(end - start for start, end in merge(intervals)) / 1000,
                           top_kernels_ms={name: time / 1000 for name, time in names[key].most_common(8)})
                 for key, intervals in streams.items()},
        note="Only positive-duration GPU kernel events with stream IDs; copies and CPU events excluded. "
             "Each stream is unioned before measuring concurrent streams. Kernel sums are not wall time. "
             "This establishes inter-stream compute overlap, not automatic encoder/decoder attribution. "
             "Missing graph-node kernel events make the overlap estimate incomplete.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = analyze(json.loads(args.trace.read_text())["traceEvents"])
    result["trace"] = str(args.trace)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
