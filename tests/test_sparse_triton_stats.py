import pytest
import torch

from attention_backends.sparse import (
    _empty_route_stats,
    _spatial_layout,
    _spatial_offsets,
    sparse_chunk,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton sparse statistics require CUDA",
)
def test_triton_directional_statistics_match_pytorch():
    torch.manual_seed(0)
    tokens, heads, head_dim, tile = 3600, 4, 128, 64
    q = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    layout = _spatial_layout((1, 60, 60), tile, tokens)
    tiles = tokens // layout.tile_tokens
    route = torch.rand(heads, tiles, tiles, device="cuda") < 0.625
    route[:, torch.arange(tiles), torch.arange(tiles)] = True

    reference_stats = _empty_route_stats(heads, tiles, "cuda")
    triton_stats = _empty_route_stats(heads, tiles, "cuda")
    offsets = _spatial_offsets(layout, "cuda")

    reference = sparse_chunk(
        q, k, v, route, layout.tile_tokens, 0, 240, head_dim**-0.5,
        reference_stats, offsets, use_triton=False, collect_stats=True)
    actual = sparse_chunk(
        q, k, v, route, layout.tile_tokens, 0, 240, head_dim**-0.5,
        triton_stats, offsets, use_triton=True, collect_stats=True)

    torch.testing.assert_close(
        actual.float(), reference.float(), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(
        triton_stats["mass"], reference_stats["mass"], atol=2e-3, rtol=2e-3)
    for axis in ("y", "x"):
        for prefix in ("q", "k"):
            torch.testing.assert_close(
                triton_stats[f"{prefix}_{axis}"],
                reference_stats[f"{prefix}_{axis}"],
                atol=2e-3,
                rtol=2e-2,
            )
