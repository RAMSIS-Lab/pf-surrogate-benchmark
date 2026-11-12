"""Timing helper: median of 3 timed passes after warm-up, synchronized on CUDA."""
import time, statistics
import torch


def time_fn(fn, device, n_warm=2, n_rep=3):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_rep):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)
