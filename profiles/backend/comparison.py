"""Dense, PyTorch-sparse, and Triton-sparse attention comparison backend."""

from __future__ import annotations

import math

import torch

from attention_backends.sparse import (
    _dense_route,
    _empty_route_stats,
    _spatial_layout,
    _spatial_offsets,
    sparse_chunk,
)
from profiles.backend.metrics import summarize
from profiles.backend.timing import peak_allocated_gib, reset_memory_stats, timed


def _grid_for_tokens(tokens):
    frames = max(1, tokens // 3600)
    if frames * 3600 != tokens:
        raise ValueError("Token counts must be a multiple of 3,600")
    return frames, 60, 60


def _dense_output(q, k, v, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2)


def _sparse_output(q, k, v, route, layout, chunk, scale, use_triton):
    stats = _empty_route_stats(q.shape[2], route.shape[-1], q.device)
    offsets = _spatial_offsets(layout, q.device)
    outputs = []
    for start in range(0, q.shape[1], chunk):
        end = min(start + chunk, q.shape[1])
        outputs.append(sparse_chunk(
            q, k, v, route, layout.tile_tokens, start, end, scale,
            stats, offsets, use_triton=use_triton, collect_stats=True))
    return torch.cat(outputs, 1), stats


def run_comparison(tokens, heads, head_dim, tile, keep, query_chunk,
                   repeats, warmup, **_):
    reset_memory_stats()
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn((1, tokens, heads, head_dim), device=device,
                    dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    layout = _spatial_layout(_grid_for_tokens(tokens), tile, tokens)
    if layout is None:
        raise ValueError(f"No exact spatial layout exists for {tokens} tokens")
    scale = head_dim**-0.5
    route, dense_reference = _dense_route(
        q, k, v, layout, query_chunk, .90, keep, scale)
    route_mask = route["mask"].to(device)
    dense_sdpa = _dense_output(q, k, v, scale)

    pytorch_output, pytorch_stats = _sparse_output(
        q, k, v, route_mask, layout, query_chunk, scale, False)
    triton_output, triton_stats = _sparse_output(
        q, k, v, route_mask, layout, query_chunk, scale, True)

    def run_dense():
        _dense_output(q, k, v, scale)

    def run_pytorch():
        _sparse_output(q, k, v, route_mask, layout, query_chunk, scale, False)

    def run_triton():
        _sparse_output(q, k, v, route_mask, layout, query_chunk, scale, True)

    dense_seconds = timed(run_dense, repeats, warmup)
    pytorch_seconds = timed(run_pytorch, repeats, warmup)
    triton_seconds = timed(run_triton, repeats, warmup)

    result = {
        "tokens": tokens,
        "heads": heads,
        "head_dim": head_dim,
        "tile_tokens": layout.tile_tokens,
        "tile_shape": [layout.tile_h, layout.tile_w],
        "keep_fraction": float(route_mask.float().mean().item()),
        "dense_seconds": dense_seconds,
        "pytorch_sparse_seconds": pytorch_seconds,
        "triton_sparse_seconds": triton_seconds,
        "triton_speedup_vs_pytorch": pytorch_seconds / triton_seconds,
        "triton_speedup_vs_dense": dense_seconds / triton_seconds,
        "pytorch_speedup_vs_dense": dense_seconds / pytorch_seconds,
        "peak_allocated_gib": peak_allocated_gib(),
        "quality": {
            "dense_bootstrap_vs_sdpa": summarize(dense_reference, dense_sdpa),
            "pytorch_sparse_vs_sdpa": summarize(pytorch_output, dense_sdpa),
            "triton_sparse_vs_sdpa": summarize(triton_output, dense_sdpa),
            "triton_vs_pytorch": summarize(triton_output, pytorch_output),
        },
    }
    del q, k, v, route, route_mask, dense_reference, dense_sdpa
    del pytorch_output, pytorch_stats, triton_output, triton_stats
    reset_memory_stats()
    return result


def profile_stages(tokens, heads, head_dim, tile, keep, query_chunk, **_):
    """Measure route metadata, output, and statistics stages separately."""
    from profiles.backend.timing import reset_memory_stats, timed

    from attention_backends.sparse import (
        _group_indices,
        _triton_sparse_layout,
    )

    reset_memory_stats()
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn((1, tokens, heads, head_dim), device=device,
                    dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    layout = _spatial_layout(_grid_for_tokens(tokens), tile, tokens)
    scale = head_dim**-0.5
    route, _ = _dense_route(q, k, v, layout, query_chunk, .90, keep, scale)
    route_mask = route["mask"].to(device)
    query_tiles = query_chunk // layout.tile_tokens
    group_mask = route_mask[:, :query_tiles].permute(1, 0, 2).reshape(
        query_tiles * heads, -1)

    metadata_seconds = timed(
        lambda: _group_indices(group_mask, layout.tile_tokens, tokens), 3, 1)
    compact_seconds = timed(
        lambda: _triton_sparse_layout(group_mask, layout.tile_tokens, tokens), 3, 1)
    return {
        "tokens": tokens,
        "metadata_seconds": metadata_seconds,
        "compact_tile_list_seconds": compact_seconds,
        "metadata_fraction_of_pytorch_sparse": metadata_seconds / timed(
            lambda: _sparse_output(
                q, k, v, route_mask, layout, query_chunk, scale, False), 3, 1),
    }
