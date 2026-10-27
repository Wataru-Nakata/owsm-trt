"""Output diagnostics shared by beam profiling and saved-output regression summaries."""

import math


def invalid_segments(segments: list[dict], duration: float) -> int:
    limit = duration + .02
    # Keep the existing 20ms endpoint allowance; binary rounding at its boundary is not a violation.
    return sum(not math.isfinite(s['start']) or not math.isfinite(s['end'])
               or s['start'] < 0 or s['end'] < s['start']
               or (s['end'] > limit and not math.isclose(s['end'],limit,rel_tol=0,abs_tol=1e-9))
               for s in segments)
