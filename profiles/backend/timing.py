"""CUDA timing and memory utilities for benchmark backends."""

from __future__ import annotations

import time

import torch


def timed(function, repeats, warmup):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def peak_allocated_gib():
    return torch.cuda.max_memory_allocated() / 2**30


def reset_memory_stats():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
