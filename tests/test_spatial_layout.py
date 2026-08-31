import torch

from attention_backends.spatial import build_spatial_layout


def test_spatial_layout_is_reversible_and_keeps_microtiles_whole():
    layout = build_spatial_layout((2, 4, 8), block_size=8, max_microtile_tokens=8)
    assert layout.microtile_shape == (2, 4)
    assert sorted(layout.permutation_cpu.tolist()) == list(range(64))
    value = torch.arange(64).reshape(1, 64, 1, 1)
    torch.testing.assert_close(layout.restore(layout.reorder(value)), value)

    tile_h, tile_w = layout.microtile_shape
    area = tile_h * tile_w
    for offset in range(0, layout.tokens, area):
        assert offset // layout.block_size == (offset + area - 1) // layout.block_size


def test_720p_layout_preserves_token_and_block_counts():
    layout = build_spatial_layout((16, 45, 80), block_size=128)
    assert layout.microtile_shape == (1, 16)
    assert layout.tokens == 57600
    assert layout.blocks == 450
    assert layout.neighbors_cpu.shape == (450, 4)
    assert (layout.neighbors_cpu >= 0).any()
