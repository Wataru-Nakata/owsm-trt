"""Opt-in host diagnostics, kept separate from uninstrumented latency measurements."""

import gc
import resource
import time
from contextlib import AbstractContextManager

import torch


class HostLatency(AbstractContextManager):
    def __init__(self):
        self.events = []
        self.samples = []

    def _gc_event(self, phase, info):
        # No CUDA calls, object inspection or explicit collection inside the callback.
        self.events.append((phase, time.perf_counter_ns(), info['generation'],
                            info.get('collected', 0), info.get('uncollectable', 0)))

    def __enter__(self):
        gc.callbacks.append(self._gc_event)
        return self

    def __exit__(self, *exc):
        gc.callbacks.remove(self._gc_event)

    def measure(self, call):
        torch.cuda.synchronize()
        before = resource.getrusage(resource.RUSAGE_SELF)
        counts = gc.get_count()
        start_cpu = time.process_time_ns()
        start = time.perf_counter_ns()
        try:
            result = call()
            torch.cuda.synchronize()
        finally:
            end = time.perf_counter_ns()
            end_cpu = time.process_time_ns()
            after = resource.getrusage(resource.RUSAGE_SELF)
            self.samples.append(dict(start_ns=start, end_ns=end, cpu_ms=(end_cpu-start_cpu)/1e6,
                gc_counts_before=counts, gc_counts_after=gc.get_count(),
                voluntary_switches=after.ru_nvcsw-before.ru_nvcsw,
                involuntary_switches=after.ru_nivcsw-before.ru_nivcsw,
                minor_faults=after.ru_minflt-before.ru_minflt,
                major_faults=after.ru_majflt-before.ru_majflt))
        return result, (end-start)/1e6

    def report(self, rows):
        if len(rows) != len(self.samples):
            raise ValueError('host diagnostic and wall sample counts differ')
        active, collections = {}, []
        for phase, ns, generation, collected, uncollectable in self.events:
            if phase == 'start':
                active[generation] = ns
            elif generation in active:
                collections.append(dict(start_ns=active.pop(generation), end_ns=ns,
                    generation=generation, collected=collected, uncollectable=uncollectable))
        samples = []
        for row, observed in zip(rows, self.samples):
            overlapping = [event for event in collections
                           if event['start_ns'] < observed['end_ns'] and event['end_ns'] > observed['start_ns']]
            samples.append(row | observed | dict(gc=overlapping, gc_overlap_ms=sum(
                min(e['end_ns'], observed['end_ns'])-max(e['start_ns'], observed['start_ns'])
                for e in overlapping)/1e6))
        return dict(scope='Instrumented host diagnostic only; wall includes callback/measurement overhead. '
                          'Process CPU time includes all threads; GC overlap is not a corrected latency. '
                          'No GC thresholds, enabled state or runtime scheduling changed.',
                    gc_enabled=gc.isenabled(), gc_threshold=gc.get_threshold(),
                    samples=samples, collections=collections, unmatched_gc_starts=active)
