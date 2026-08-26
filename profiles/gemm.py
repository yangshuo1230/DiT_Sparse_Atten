"""Benchmark BF16 attention-shaped batched GEMMs."""

from __future__ import annotations

import argparse
import json

import torch


def measure(batch, m, n, head_dim, repeats):
    q = torch.randn(batch, m, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, head_dim, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        torch.bmm(q, k)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = torch.bmm(q, k)
    end.record()
    torch.cuda.synchronize()
    milliseconds = start.elapsed_time(end) / repeats
    operations = 2 * batch * m * n * head_dim
    return {
        "batch": batch,
        "shape": f"{m}x{head_dim} @ {head_dim}x{n}",
        "microseconds": milliseconds * 1000,
        "tflops": operations / (milliseconds * 1e-3) / 1e12,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tiles", type=int, nargs="+", default=(16, 32, 64, 128, 256))
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps([
        measure(args.batch, tile, tile, args.head_dim, args.repeats)
        for tile in args.tiles
    ], indent=2))


if __name__ == "__main__":
    main()
