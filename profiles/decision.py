"""Benchmark exact top-mass route-mask decisions."""

from __future__ import annotations

import argparse
import json

import torch

from attention_backends.sparse import _row_mask


def measure(heads, blocks, target, repeats):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mass = torch.rand(heads, blocks, blocks, device="cuda")
    mass /= mass.sum(-1, keepdim=True)
    _row_mask(mass, target, 1.0)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        mask = _row_mask(mass, target, 1.0)
    end.record()
    torch.cuda.synchronize()
    return {
        "heads": heads,
        "blocks_per_axis": blocks,
        "decision_ms": start.elapsed_time(end) / repeats,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "mask_mib": mask.numel() / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--blocks", type=int, nargs="+", default=(780, 1950))
    parser.add_argument("--target", type=float, default=.90)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps([
        measure(args.heads, blocks, args.target, args.repeats)
        for blocks in args.blocks
    ], indent=2))


if __name__ == "__main__":
    main()
