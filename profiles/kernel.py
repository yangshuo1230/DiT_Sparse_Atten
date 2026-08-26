"""Microbenchmark full QxK matrix-tile masks with independent head routes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch


def timed(function, repeats):
    function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def benchmark(tokens, heads, head_dim, tile, keep, query_chunk, repeats):
    from attention_backends.sparse import _empty_route_stats, sparse_chunk

    if tokens % tile:
        raise ValueError(
            f"tokens={tokens} must be exactly divisible by tile={tile}")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(1, tokens, heads, head_dim, device=device, dtype=dtype,
                    generator=generator)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    matrix_tiles = math.ceil(tokens / tile)
    selected = max(1, round(matrix_tiles * keep))
    route = torch.zeros((heads, matrix_tiles, matrix_tiles), dtype=torch.bool)
    for head in range(heads):
        for query_tile in range(matrix_tiles):
            indices = torch.randperm(
                matrix_tiles, generator=generator, device=device)[:selected]
            route[head, query_tile, indices.cpu()] = True
    route = route.to(device)

    def dense():
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))

    def sparse():
        stats = _empty_route_stats(heads, matrix_tiles, device)
        outputs = []
        for start in range(0, tokens, query_chunk):
            end = min(start + query_chunk, tokens)
            outputs.append(sparse_chunk(
                q, k, v, route, tile, start, end, None,
                stats))
        return torch.cat(outputs, 1)

    dense_seconds = timed(dense, repeats)
    sparse_seconds = timed(sparse, repeats)
    return {
        "tokens": tokens,
        "heads": heads,
        "head_dim": head_dim,
        "matrix_tile_shape": f"{tile}x{tile}",
        "query_chunk": query_chunk,
        "keep_fraction_per_query_tile_and_head": keep,
        "theoretical_matmul_speedup": 1 / keep,
        "dense_seconds": dense_seconds,
        "matrix_sparse_seconds": sparse_seconds,
        "empirical_speedup": dense_seconds / sparse_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--tokens", type=int, default=3648)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rows = [
        benchmark(args.tokens, args.heads, args.head_dim, tile, keep,
                  args.query_chunk, args.repeats)
        for tile in (16, 32, 64)
        for keep in (.5, .625)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "profile_matrix_tiles.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    with (args.output / "profile_matrix_tiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
