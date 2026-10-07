"""Opt-in stage timings. CUDA events are resolved once per batch, not per stage."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import time

import torch


class StageProfiler:
    def __init__(self, device):
        self.device = torch.device(device)
        self.pending = []
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def stage(self, name, cpu=False):
        cuda = self.device.type == "cuda" and not cpu
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(self.device))
        else:
            start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                end.record(torch.cuda.current_stream(self.device))
                self.pending.append((name, start, end))
            else:
                self.seconds[name] += time.perf_counter() - start
                self.calls[name] += 1

    def finish_batch(self):
        if self.pending:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.pending:
                self.seconds[name] += start.elapsed_time(end) / 1000
                self.calls[name] += 1
            self.pending.clear()

    def summary(self):
        self.finish_batch()
        return {name: {"seconds": seconds, "calls": self.calls[name]}
                for name, seconds in self.seconds.items()}


def stage(profiler, name, cpu=False):
    return profiler.stage(name, cpu) if profiler is not None else nullcontext()


def memory_stats(device):
    if torch.device(device).type != "cuda":
        return {"cuda_allocated_mib": None, "cuda_peak_mib": None}
    return {"cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 2**20}
